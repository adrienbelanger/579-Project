import gymnasium as gym
import numpy as np
from model import Model, PPOModel
'''
Here is where we can declare each of our agents.
I've set up a random agent for good measure.

The original paper used the following hyperparamaters (copied from the paper)
    Hyperparameter          Value
    Horizon (T)             128
    Adam stepsize           2.5 ×10^{−4} ×α
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

    def select_action(self, obs: np.ndarray) -> int:
        """return random action"""
        return self.action_space.sample()
    

class PolicyAgent(Agent):

    def __init__(self, model: Model):
        self.model = model


