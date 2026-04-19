"""
agent_lstm.py    LSTM PPO agent

Usage in notebook
    from agent_lstm import PPOAgentLSTM
    lstm_agent = PPOAgentLSTM()
    run_benchmark([ppo_agent, lstm_agent], ["Alien"])
"""

import numpy as np
import gymnasium as gym
import torch

import config
from agent import PPOAgent          # re-use the base logic
from game import Game
from model import EpisodePoint
from model_lstm import PPOModelLSTM


class PPOAgentLSTM(PPOAgent):
    """
    Replacement for PPOAgent that uses the LSTM actor-critic.

    Differences:
      - _build_model returns a PPOModelLSTM
      - select_action maintains (h, c) across steps inside an episode,
        reset via reset() at episode boundaries
    """

    def __init__(self) -> None:
        self.ppo_cfg        = config.PPOConfig_LSTM()
        self.stochastic_eval = True
        self.model: PPOModelLSTM | None = None
        self._trained       = False
        self._trained_game: str | None  = None
        # Hidden state for evaluation rollouts (reset each episode)
        self._h: torch.Tensor | None = None
        self._c: torch.Tensor | None = None

    def _build_model(self, action_dim: int) -> PPOModelLSTM:
        return PPOModelLSTM(action_dim=action_dim)

    def reset(self, env: gym.Env) -> None:
        """Called at the start of every evaluation episode. zero out LSTM state."""
        super().reset(env)
        if self.model is not None:
            self._h, self._c = self.model.network.init_hidden(
                batch_size=1,
                device=self.model.device,
            )

    def select_action(self, obs: np.ndarray) -> int:
        if not self._trained or self.model is None:
            raise RuntimeError(
                "ppo agent wasn't trained.. it should be double check run_benchmark?? " #TODO: Remove this after benchmark is correct
            )
        if self._h is None or self._c is None:
            # Safety fallback. shouldn't happen if reset() was called
            self._h, self._c = self.model.network.init_hidden(1, self.model.device)

        action, self._h, self._c = self.model.act(
            obs,
            self._h,
            self._c,
            stochastic=self.stochastic_eval,
        )
        return int(action)
