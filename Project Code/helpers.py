import gymnasium as gym
from gymnasium.wrappers import (ResizeObservation,GrayscaleObservation,FrameStackObservation,MaxAndSkipObservation,)
import numpy as np
from agent import Agent
from game import Game, VALID_GAMES
import pandas as pd
from dataclasses import dataclass # so we can do game class so it's easier for our plotting
from typing import Literal

def make_env(game: Game,render_mode: Literal["human", "rgb_array", None] = None) -> gym.Env: 
    """
    Make environment for the specified game from the game array.
    These parameters are from PPO, which some werer from DQN or ACER or some paper Machalo et al. that aimed to improve this benchmark. These are the parameters they used

    Args:
        game: name of our atari game, should be from our list from cell 1 (i.e. the standard list)
        render_mode: mode of rendering, btw "human" -> Renders game as it runs, "rgb_array" which when we do env.render() returns the current game frame as a np.array, or None which does no rendering (use for rendering)
    """

    env = gym.make(
        game.env_id,
        render_mode=render_mode,
        frameskip=1,   # we're hadnlign the frame skip in the MaxAndSkipObservation so don't do it in the env as well
        repeat_action_probability=0.25, # this is the part from machalo et al.
        full_action_space=False,                                       
    )
    env = MaxAndSkipObservation(env, skip=4) # DQN
    env = ResizeObservation(env, (84,84))  # DQN
    env = GrayscaleObservation(env, keep_dim=True) # from DQN
    env = FrameStackObservation(env, 4) # DQN

    return env

def run_game(game: Game, render_mode: Literal["human", "rgb_array", None],agent: Agent,n_eps: int,max_steps: int) -> float: #TODO: Might want to pass an env directly? So we can run with whatever render mode? Not sure what's best design-wise # type:ignore
    """
    run n episodes of the game with our agent Agent and return the mean undiscounted return (like in PPO score)."
    
    Args:
        env: gym.Env for the specified game
        agent: agent from our Agent class
        n_eps: number of episodes to run for. PPO ran for 3
        max_steps: max steps to run for if game is not won/loss yet
    """
    env = make_env(game,None)

    return_per_eps = []

    for i in range(n_eps):
        cur_observation = env.reset()[0] # env.reset gives the initial state of game, used by agent ot select action.
        agent.reset(env)
        total_r  = .0
        cur_step = 0

        done = False

        while not done and cur_step < max_steps:
            action = agent.select_action(cur_observation)
            cur_observation, cur_r, done_term, done_trunc, _ = env.step(action) # take ation and get result from game. 

            total_r = total_r + float(cur_r) # have to switch to float bcs cur_reward gives "support float" as a type? Not sure. BUG: Possible bug in the future, double check this if smt breaks

            done = done_term or done_trunc

            cur_step = cur_step + 1

        return_per_eps.append(total_r)

    env.close()

    mean_reward = float(np.mean(return_per_eps))


    return mean_reward #TODO: Might want to return total as well?

@dataclass
class GameResult:
    game: str
    mean_score: float


def run_benchmark(agent: Agent, games: list[str] | None) -> list[GameResult]:
    '''
    Run the list of games with the PPO benchmark parameters. Returns a list of dicts the game names as string and raw scores of all the games. 

    Args:
        agent: agent to be evaluated on the list of games
        games: list of games for the benchmark to be run on. Give None for it to run on all available games
    '''
    if games == None:
        games = VALID_GAMES
    else:
        for game_str in games:
            if game_str not in VALID_GAMES:
                raise ValueError(f"{game_str} not a valid game name. Check VALID_GAMES")
            
    results = []
    for game_str in games:
        game = Game(game_str)
        cur_game_result = run_game(game=game,render_mode=None,agent=agent,n_eps=3,max_steps=108000) # benchmark parameters from PPO
        game_result = GameResult(game_str, cur_game_result)
        results.append(game_result)

    
    return results



def plot_results(game_results: list[GameResult])-> None:
    pass
