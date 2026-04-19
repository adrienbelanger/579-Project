"""
model_lstm.py  --  LSTM-based Actor-Critic + PPO model

New classes
-----------
ActorCriticLSTM    CNN feature extractor  ->  LSTMCell  ->  actor / critic heads
PPOModelLSTM       PPOModel subclass that carries (h, c) across rollout steps
"""


from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

import config
from model import ActorCritic, EpisodePoint, Model, PPOModel
import config

# ---------------------------------------------------------------------------
# 1.  Config
# ---------------------------------------------------------------------------

# ActorCriticLSTM

class ActorCriticLSTM(nn.Module):
    """
    CNN  ->  LSTMCell  ->  actor / critic heads.

    Differences from ActorCritic (the MLP version):
      - The final nn.Linear(2592, hidden_size) of the CNN is kept, but its
        output is fed into an LSTMCell rather than directly to the heads.
      - forward() takes and returns an explicit hidden state (h, c) so that
        PPOModelLSTM can thread it across timesteps.

    Shapes
    ------
    obs          : (batch, C, H, W)  float32, already /255 normalised
    h_in, c_in   : (batch, hidden_size)
    logits     : (batch, action_dim)
    value      : (batch,)
    h_out, c_out: (batch, hidden_size)
    """

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.cfg = config.ActorCriticConfig()
        lstm_cfg = config.PPOConfig_LSTM()
        hidden = lstm_cfg.lstm_hidden_size

        self.cnn = nn.Sequential(
        # architecture follows conv of 16 -> Relu -> Flatten -> Linear
            nn.Conv2d(self.cfg.input_channels, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, hidden),
            nn.ReLU(),
        )

        # LSTM cell (one step at a time, matching rollout loop)
        self.lstm = nn.LSTMCell(hidden, hidden)

        # Output heads 
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

        # Orthogonal init (same scheme as ActorCritic)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, self.cfg.base_init_gain)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
        # actor / critic heads get their own smaller gains
        nn.init.orthogonal_(self.actor.weight,  self.cfg.policy_init_gain)
        nn.init.orthogonal_(self.critic.weight, self.cfg.value_init_gain)
        # LSTM: use orthogonal for input->hidden and hidden->hidden weights
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, self.cfg.base_init_gain)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    # ------------------------------------------------------------------
    def obs_to_tensor(self, observation: Any) -> torch.Tensor:
        '''Takes raw game observations and transforms them into usable tensors.
        
            Args:
                observation: Game observation of any shape, will be compressed to match expected tensor.
        '''
        if isinstance(observation, torch.Tensor):
            x = observation.float()
        else:
            x = torch.as_tensor(np.asarray(observation)).float()

        if x.ndim == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != self.cfg.input_channels and x.shape[-1] == self.cfg.input_channels:
            x = x.permute(0, 3, 1, 2)
        return x

    def forward(
        self,
        observation: Any,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One-step forward pass.

        Returns
        -------
        logits  : (batch, action_dim)
        value   : (batch,)
        h_out   : (batch, hidden_size)
        c_out   : (batch, hidden_size)
        """
        x = self.obs_to_tensor(observation).to(self.actor.weight.device) / 255.0
        features = self.cnn(x)                      # (batch, hidden)
        h_out, c_out = self.lstm(features, (h_in, c_in))
        logits = self.actor(h_out)                  # (batch, action_dim)
        value  = self.critic(h_out).squeeze(-1)     # (batch,)
        return logits, value, h_out, c_out

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero hidden state for the start of an episode."""
        hidden = config.PPOConfig_LSTM().lstm_hidden_size
        h = torch.zeros(batch_size, hidden, device=device)
        c = torch.zeros(batch_size, hidden, device=device)
        return h, c


# PPOModelLSTM
class PPOModelLSTM(PPOModel):
    """
    PPO with an LSTM actor-critic.

    Differences from PPOModel
    - _rollout  threads (h, c) across the T=128 timesteps.
                On episode end (done), the corresponding actor's (h, c)
                is zeroed so the LSTM starts fresh for the next episode.
                Returns the stored per-step hidden states so _update can
                replay them.

    - _update   feeds sequences "in time order" through the LSTM to
                recompute log-probs and values, then applies the PPO loss.
                We do NOT randomly shuffle across the time dimension,
                only across the N actors (independent sequences).
    """

    def __init__(self, action_dim: int) -> None:
        self.cfg = config.PPOConfig_LSTM()
        self.env_cfg = config.EnvConfig()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        self.network = ActorCriticLSTM(action_dim=action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_eps,
        )
        self._trained = False

    # Rollout  (overrides PPOModel._rollout)
    def _rollout(
        self,
        env: gym.vector.VectorEnv,
        obs: np.ndarray,
        episode_returns: np.ndarray,
        total_timesteps: int,
        h: torch.Tensor,   # (N, hidden)  <- extra args vs base class
        c: torch.Tensor,   # (N, hidden)
    ) -> tuple:             # type: ignore[override]
        """
        Collect T steps from N parallel actors.

        Returns (on top of the base class tuple):
          H_stored   (T, N, hidden)  h state entering each step
          C_stored   (T, N, hidden)  c state entering each step
          h, c       updated carry for the next rollout call
        """
        N, T = self.cfg.actors, self.cfg.horizon
        O, A, R, D, LP, V = [], [], [], [], [], []
        H_stored, C_stored = [], []     # hidden states entering each step
        episode_points: list[EpisodePoint] = []
        frames_per_env_step = self.env_cfg.frameskip * self.env_cfg.MaxAndSkipObservation

        for _ in range(T):
            # Store the hidden states that enter this step,  needed for
            # truncated BPTT replay in _update.
            H_stored.append(h.detach().clone())
            C_stored.append(c.detach().clone())

            with torch.no_grad():
                logits, values, h, c = self.network(obs, h, c)

            dist    = Categorical(logits=logits)
            actions = dist.sample()

            next_obs, rewards, terminated, truncated, _ = env.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated)

            rewards_np = np.asarray(rewards, dtype=np.float32)
            episode_returns += rewards_np
            total_timesteps += int(rewards_np.shape[0])

            for actor_idx, done in enumerate(dones):
                if done:
                    episode_points.append(
                        EpisodePoint(
                            total_timesteps=total_timesteps,
                            game_frames=total_timesteps * frames_per_env_step,
                            episode_return=float(episode_returns[actor_idx]),
                        )
                    )
                    episode_returns[actor_idx] = 0.0
                    # Reset LSTM state for this actor on episode end
                    h[actor_idx].zero_()
                    c[actor_idx].zero_()

            O.append(self.network.obs_to_tensor(obs))
            A.append(actions)
            R.append(torch.as_tensor(rewards, dtype=torch.float32).to(self.device))
            D.append(torch.as_tensor(dones,   dtype=torch.float32).to(self.device))
            LP.append(dist.log_prob(actions))
            V.append(values)
            obs = next_obs

        return (
            torch.stack(O),           # (T, N, C, H, W)
            torch.stack(A),           # (T, N)
            torch.stack(R),           # (T, N)
            torch.stack(D),           # (T, N)
            torch.stack(LP),          # (T, N)
            torch.stack(V),           # (T, N)
            torch.stack(H_stored),    # (T, N, hidden)  <- new
            torch.stack(C_stored),    # (T, N, hidden)  <- new
            obs,
            episode_returns,
            total_timesteps,
            episode_points,
        )

    # Training loop  (overrides PPOModel.train)
    def train(self, env: gym.vector.VectorEnv, seed: int) -> list[EpisodePoint]:
        N, T = self.cfg.actors, self.cfg.horizon
        updates = self.cfg.total_timesteps // (N * T)
        if updates <= 0:
            raise ValueError(f"PPOConfig.total_timesteps must be more than actors * horizon (here higher than {self.cfg.actors * self.cfg.horizon}).") # added since we gt 0 trainign before

        obs, _ = env.reset(seed=seed)
        episode_returns = np.zeros(N, dtype=np.float32)
        total_timesteps = 0
        episode_points: list[EpisodePoint] = []

        # Initial hidden state: zeros at the start of training
        h, c = self.network.init_hidden(N, self.device)

        self.network.train()
        for update_i in tqdm(range(updates), desc="      Training", unit="update"):
            alpha = 1.0 - (update_i / updates)
            lr   = self.cfg.learning_rate * alpha if self.cfg.anneal_learning_rate else self.cfg.learning_rate
            clip = self.cfg.clipping      * alpha if self.cfg.anneal_clipping      else self.cfg.clipping
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            (O, A, R, D, LP, V,
             H_stored, C_stored,
             obs, episode_returns, total_timesteps,
             new_points) = self._rollout(
                env, obs, episode_returns, total_timesteps, h, c
            )
            # Update h, c carry. they are already reset inside _rollout
            # on any done, so we just grab the last step's outputs.
            # Re-run a no-grad step to get the final carry cleanly.
            with torch.no_grad():
                _, _, h, c = self.network(obs, H_stored[-1], C_stored[-1])
                # Re-zero actors whose last step was a done.
                last_dones = D[-1].bool()
                h[last_dones] = 0.0
                c[last_dones] = 0.0

            episode_points.extend(new_points)

            with torch.no_grad():
                _, next_v, _, _ = self.network(obs, h, c)

            adv, ret = self._gae(R, D, V, next_v)
            self._update_lstm(O, A, LP, adv, ret, clip, H_stored, C_stored, D)

        self.network.eval()
        self._trained = True
        return episode_points

    # LSTM-aware update  (replaces PPOModel._update)
    def _update_lstm(
        self,
        observations:  torch.Tensor,   # (T, N, C, H, W)
        actions:       torch.Tensor,   # (T, N)
        old_log_probs: torch.Tensor,   # (T, N)
        advantages:    torch.Tensor,   # (T*N,)  already flattened by _gae
        returns:       torch.Tensor,   # (T*N,)
        clip:          float,
        H_stored:      torch.Tensor,   # (T, N, hidden)
        C_stored:      torch.Tensor,   # (T, N, hidden)
        dones:         torch.Tensor,   # (T, N)  needed to reset h,c mid-seq
    ) -> None:
        """
        We treat each of the N actors as an independent sequence of length T.
        For each epoch we iterate over actors in random order.  Within one
        actor's sequence we run the LSTM step-by-step (using the stored h0/c0
        from the beginning of the rollout), re-computing log-probs and values,
        then accumulate the PPO loss and do one gradient step per actor.

        This keeps memory bounded (one sequence at a time) while still
        letting the LSTM see the full T-step context it had during rollout.
        """
        T, N = observations.shape[:2]
        adv_2d = advantages.view(T, N)
        adv_2d = (adv_2d - adv_2d.mean()) / (adv_2d.std() + 1e-8)
        ret_2d = returns.view(T, N).to(self.device)

        for _ in range(self.cfg.epochs):
            actor_order = torch.randperm(N)

            for actor_idx in actor_order:
                actor_idx = actor_idx.item()

                # Starting hidden state for this actor's sequence
                h_seq = H_stored[0, actor_idx].unsqueeze(0).to(self.device)  # (1, hidden)
                c_seq = C_stored[0, actor_idx].unsqueeze(0).to(self.device)

                new_log_probs = []
                new_values    = []

                for t in range(T):
                    obs_t = observations[t, actor_idx].unsqueeze(0).to(self.device)  # (1,C,H,W)
                    logits_t, value_t, h_seq, c_seq = self.network(obs_t, h_seq, c_seq)
                    dist_t = Categorical(logits=logits_t)
                    new_log_probs.append(dist_t.log_prob(actions[t, actor_idx].unsqueeze(0).to(self.device)))
                    new_values.append(value_t)

                    # Reset hidden state if this step ended an episode
                    if dones[t, actor_idx].item():
                        h_seq = torch.zeros_like(h_seq)
                        c_seq = torch.zeros_like(c_seq)

                new_lp = torch.cat(new_log_probs)           # (T,)
                vals   = torch.cat(new_values)              # (T,)

                old_lp_actor = old_log_probs[:, actor_idx].to(self.device)   # (T,)
                adv_actor    = adv_2d[:, actor_idx].to(self.device)          # (T,)
                ret_actor    = ret_2d[:, actor_idx]                          # (T,)

                ratio      = (new_lp - old_lp_actor).exp()
                ratio_clip = ratio.clamp(1.0 - clip, 1.0 + clip)

                p_loss  = -torch.min(ratio * adv_actor, ratio_clip * adv_actor).mean()
                v_loss  = (vals - ret_actor).pow(2).mean()
                entropy = Categorical(logits=logits_t).entropy().mean()  # type: ignore[possibly-undefined]
                loss    = p_loss + self.cfg.val_func_coeff * v_loss - self.cfg.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

    # act()  (overrides PPOModel.act, needs hidden state)
    def act(
        self,
        observation: Any,
        h: torch.Tensor,
        c: torch.Tensor,
        stochastic: bool = True,
    ) -> tuple[int | np.ndarray, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """
        Returns (action, h_new, c_new).
        The caller (PPOAgentLSTM) is responsible for keeping h, c between steps.
        """
        self.network.eval()
        with torch.no_grad():
            logits, _, h_new, c_new = self.network(observation, h, c)
            action = Categorical(logits=logits).sample() if stochastic else torch.argmax(logits, dim=-1)
        result = int(action.item()) if action.numel() == 1 else action.cpu().numpy()
        return result, h_new, c_new
