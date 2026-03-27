from typing import Literal

import gymnasium as gym
import numpy as np

import config
from game import Game
from model import Model, PPOModel
'''
Here is where we can declare each of our agents.
I've set up a random agent for good measure.

The original paper used the following hyperparamaters (copied from the paper)
    Hyperparameter          Value
    Horizon (T)             128
    Adam stepsize           2.5 ×10^{−4} ×α # stupid annealing alpha I missed
    Num. epochs             3
    Minibatch size          32 ×8
    Discount (γ)            0.99
    GAE parameter (λ)       0.95
    Number of actors        8
    Clipping parameter ϵ    0.1 ×α
    VF coeﬀ. c1 (9)         1
    Entropy coeﬀ. c2 (9)    0.01
'''

class Agent:
    '''
    Base agent. Override for LSTM or Replay Buffer or parameter annealing. Does not work as is.
    '''

    def reset(self, env: gym.Env) -> None:
        """Mostly for LSTM or Replay buffer, do here what you need to do per episode (i.e. reset hidden vec)"""
        self.action_space = env.action_space

    def select_action(self, obs: np.ndarray) -> int:
        """select action depend on our policy. as usual actions are ints in gym"""
        return 0
    
class RandomAgent(Agent):
    '''
    Random actions, mostly for testing
    '''

    def select_action(self, obs: np.ndarray) -> int: # observation doesn't matter here, but still take as arg so benchmark runs
        """return random action"""
        return self.action_space.sample()
    

class PolicyAgent(Agent):
    """Abstract policy-backed agent wrapper."""

    def __init__(self, model: Model):
        pass

    def select_action(self, obs: np.ndarray) -> int:
        return 0
    
    def train(self, game: Game, render_mode: Literal["human", "rgb_array", None] = None) -> None:
        pass


class PPOAgent(PolicyAgent):
    """
    Per-game PPO agent. We retrain for each game. Loads the PPO config automatically from config.
    """

    def __init__(self) -> None:
        self.ppo_cfg = config.PPOConfig()
        self.stochastic_eval = True # like in the paper, 
        self.model: PPOModel | None = None
        self._trained = False
        self._trained_game: str | None = None

    @property
    def is_trained(self) -> bool:
        '''Check if the model is trained. Returns True if trained.'''
        return self._trained

    @property
    def trained_game(self) -> str | None:
        '''Check which game we have trained on. Returns the name of the game as a string or None if not trained yet.'''
        return self._trained_game

    def train(self, game: Game) -> None:
        '''
        Trian the PPO agent on a specific game.
        Args:
            game: Game object
        '''
        if self.ppo_cfg.actors <= 0:
            raise ValueError("PPOConfig.actors must be > 0.")

        # Local import avoids circular import because helpers imports Agent.
        from helpers import make_env

        def build_env() -> gym.Env:
            return make_env(game, render_mode=None)

        vector_env = gym.vector.AsyncVectorEnv([build_env for _ in range(self.ppo_cfg.actors)])
        try:
            action_dim = int(vector_env.single_action_space.n)
            self.model = PPOModel(action_dim=action_dim)
            self.model.train(vector_env)
            self._trained = True
            self._trained_game = game.name
            self.action_space = vector_env.single_action_space
        finally:
            vector_env.close()

    def reset(self, env: gym.Env) -> None:
        super().reset(env)

    def select_action(self, obs: np.ndarray) -> int:
        '''Overload select action'''
        if not self._trained or self.model is None:
            raise RuntimeError(
                "ppo agent wasn't trained.. it should be double check run_benchmark?? " #TODO: Remove this after benchmark is correct
            )
        action = self.model.act(obs, stochastic=self.stochastic_eval) # if stochastic, then we select the 
        return int(action)
