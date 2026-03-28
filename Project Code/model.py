from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm # for progress bar

import config


class Model:
    """Base model class for project models."""


@dataclass(frozen=True)
class EpisodePoint:
    """Single completed episode logged during PPO training."""

    total_timesteps: int
    game_frames: int
    episode_return: float


class ActorCritic(Model, nn.Module):
    """Paper CNN actor-critic with orthogonal initialization."""

    def __init__(self, action_dim: int) -> None:
        '''
        Defines the model architecture, as in MNI 16 -> https://arxiv.org/pdf/1602.01783 - experimental setup (8.)
        '''
        nn.Module.__init__(self)
        self.cfg = config.ActorCriticConfig()
        self.cnn = nn.Sequential(
            # architecture follows conv of 16 -> Relu -> Flatten -> Linear
            nn.Conv2d(self.cfg.input_channels, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, self.cfg.hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(self.cfg.hidden_size, action_dim) # actor
        self.critic = nn.Linear(self.cfg.hidden_size, 1) # critic

        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, self.cfg.base_init_gain)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, self.cfg.policy_init_gain)
        nn.init.orthogonal_(self.critic.weight, self.cfg.value_init_gain)

    def obs_to_tensor(self, observation: Any) -> torch.Tensor:
        '''Takes raw game observations and transforms them into usable tensors.
        
            Args:
                observation: Game observation of any shape, will be compressed to match expected tensor.
        '''
        x = torch.as_tensor(np.asarray(observation)) # convert 0our observation to a tensor

        if x.ndim == 5 and x.shape[-1] == 1: # compress the shape if it doesn't match
            x = x.squeeze(-1)
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != self.cfg.input_channels and x.shape[-1] == self.cfg.input_channels:
            x = x.permute(0, 3, 1, 2)
        return x.float()

    def forward(self, observation: Any) -> tuple[torch.Tensor, torch.Tensor]:
        '''Normalizes, passes through CNN, returns, gives back policy + value estimates'''
        x = self.obs_to_tensor(observation).to(self.actor.weight.device) / 255.0 # get tensor + normalize
        h = self.cnn(x)
        return self.actor(h), self.critic(h).squeeze(-1)


class PPOModel(Model):
    """Compact PPO implementation used by PPOAgent."""

    def __init__(self, action_dim: int) -> None:
        self.cfg = config.PPOConfig() # use default config from config.py.
        self.env_cfg = config.EnvConfig()
        self.network = ActorCritic(action_dim=action_dim) # uses the action dim from the game it has to train on.
        
        self.optimizer = torch.optim.Adam( # paper uses adam optimizer. config has the correct parameter
            self.network.parameters(),
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_eps,
        )
        self._trained = False # used as check for training

    @property
    def is_trained(self) -> bool:
        '''Check if model is trained'''
        return self._trained


    def _current_value_coeff(self, update_i: int, updates: int) -> float:
        return float(self.cfg.val_func_coeff)

    def train(self, env: gym.vector.VectorEnv, seed: int) -> list[EpisodePoint]:

        N, T = self.cfg.actors, self.cfg.horizon
        updates = self.cfg.total_timesteps // (N * T)
        if updates <= 0:
            raise ValueError(f"PPOConfig.total_timesteps must be more than actors * horizon (here higher than {self.cfg.actors * self.cfg.horizon}).") # added since we gt 0 trainign before


        run_seed = seed

        obs, _ = env.reset(seed=run_seed)
        episode_returns = np.zeros(N, dtype=np.float32)
        total_timesteps = 0
        episode_points: list[EpisodePoint] = []
        self.network.train()
        for update_i in tqdm(range(updates), desc="      Training", unit="update"):
            alpha = 1.0 - (update_i / updates)
            lr = self.cfg.learning_rate * alpha if self.cfg.anneal_learning_rate else self.cfg.learning_rate # if we don't want to anneal we can retest
            clip = self.cfg.clipping * alpha if self.cfg.anneal_clipping else self.cfg.clipping
            value_coeff = self._current_value_coeff(update_i, updates)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            O, A, R, D, LP, V, obs, episode_returns, total_timesteps, new_points = self._rollout(
                env,
                obs,
                episode_returns,
                total_timesteps,
            )
            episode_points.extend(new_points)
            with torch.no_grad():
                _, next_v = self.network(obs)
            adv, ret = self._gae(R, D, V, next_v)
            self._update(O, A, LP, adv, ret, clip, value_coeff)

        self.network.eval()
        self._trained = True
        return episode_points

    def _rollout( self, env: gym.vector.VectorEnv, obs: np.ndarray, episode_returns: np.ndarray, total_timesteps: int,) -> tuple:
        O, A, R, D, LP, V = [], [], [], [], [], []
        episode_points: list[EpisodePoint] = []
        frames_per_env_step = self.env_cfg.frameskip * self.env_cfg.MaxAndSkipObservation
        for _ in range(self.cfg.horizon):
            with torch.no_grad():
                logits, values = self.network(obs)
            dist = Categorical(logits=logits)
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

            O.append(self.network.obs_to_tensor(obs))
            A.append(actions)
            R.append(torch.as_tensor(rewards, dtype=torch.float32))
            D.append(torch.as_tensor(dones, dtype=torch.float32))
            LP.append(dist.log_prob(actions))
            V.append(values)
            obs = next_obs

        return (
            torch.stack(O),
            torch.stack(A),
            torch.stack(R),
            torch.stack(D),
            torch.stack(LP),
            torch.stack(V),
            obs,
            episode_returns,
            total_timesteps,
            episode_points,
        )

    def _gae( self,rewards: torch.Tensor,dones: torch.Tensor,values: torch.Tensor,next_values: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.shape[1], dtype=torch.float32)
        next_v = next_values
        for t in reversed(range(self.cfg.horizon)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.discount * next_v * non_terminal - values[t]
            gae = delta + self.cfg.discount * self.cfg.lambda_bias_var * non_terminal * gae
            advantages[t] = gae
            next_v = values[t]
        returns = advantages + values
        return advantages.flatten(), returns.flatten()

    def _update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        clip: float,
        value_coeff: float,
    ) -> None:
        obs = observations.flatten(0, 1)
        act = actions.flatten()
        old_lp = old_log_probs.flatten()
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ret = returns
        batch_size = obs.shape[0]

        for epo in range(self.cfg.epochs):
            
            perm = torch.randperm(batch_size)
            for start in range(0, batch_size, self.cfg.minibatch_size):
                idx = perm[start : start + self.cfg.minibatch_size]
                logits, values = self.network(obs[idx])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(act[idx])
                ratio = (new_lp - old_lp[idx]).exp()
                ratio_clip = ratio.clamp(1.0 - clip, 1.0 + clip)

                p_loss = -torch.min(ratio * adv[idx], ratio_clip * adv[idx]).mean()
                v_loss = (values - ret[idx]).pow(2).mean()
                e_bonus = dist.entropy().mean()
                loss = p_loss + value_coeff * v_loss - self.cfg.entropy_coeff * e_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

    def act(self, observation: Any, stochastic: bool = True) -> int | np.ndarray:
        self.network.eval()
        with torch.no_grad():
            logits, _ = self.network(observation)
            action = Categorical(logits=logits).sample() if stochastic else torch.argmax(logits, dim=-1)
        return int(action.item()) if action.numel() == 1 else action.cpu().numpy()


class PPOModel_VF_Annealing(PPOModel):
    def _current_value_coeff(self, update_i: int, updates: int) -> float:
        initial = config.PPOConfig_VF_Annealing.initial_val_coeff
        final = config.PPOConfig_VF_Annealing.final_val_coeff
        alpha = 1.0 - (update_i / updates)
        return float(final + (initial - final) * alpha)
