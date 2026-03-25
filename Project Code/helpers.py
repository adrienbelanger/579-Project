import gymnasium as gym
from gymnasium.wrappers import (ResizeObservation,GrayscaleObservation,FrameStackObservation,MaxAndSkipObservation,)
import numpy as np
from agent import Agent
from game import Game, VALID_GAMES, make_games
import pandas as pd
from dataclasses import dataclass # so we can do game class so it's easier for our plotting
from typing import Literal, Tuple
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import config

try: # so that when we run benchmark we don't get publicity from ale_py: by printing: A.L.E: Arcade Learning Environment (version 0.11.2+ecc1138) [Powered by Stella]
    from ale_py import ALEInterface, LoggerMode
except ImportError:
    ALEInterface = None
    LoggerMode = None

@dataclass
class frameResult:
    reward: float
    t: int

@dataclass
class seedResult:
    seed: int
    frame_rewards: list[frameResult]
    
@dataclass
class GameResult:
    agent: Agent
    game: Game
    seed_results: list[seedResult]


def make_env(game: Game,render_mode: Literal["human", "rgb_array", None] = None) -> gym.Env: 
    """
    Make environment for the specified game from the game array.
    These parameters are from PPO, which some werer from DQN or ACER or some paper Machalo et al. that aimed to improve this benchmark. These are the parameters they used

    Args:
        game: name of our atari game, should be from our list from cell 1 (i.e. the standard list)
        render_mode: mode of rendering, btw "human" -> Renders game as it runs, "rgb_array" which when we do env.render() returns the current game frame as a np.array, or None which does no rendering (use for rendering)
    """
    if ALEInterface is not None and LoggerMode is not None: # double check to hide message 
        ALEInterface.setLoggerMode(LoggerMode.Warning)
    env_config = config.EnvConfig()

    env = gym.make(
        game.env_id,
        render_mode=render_mode,
        frameskip=env_config.frameskip,
        repeat_action_probability=env_config.repeat_action_probability,
        full_action_space=env_config.full_action_space,                                       
    )
    env = MaxAndSkipObservation(env, skip=env_config.MaxAndSkipObservation) # DQN
    env = ResizeObservation(env, env_config.ResizeObservation)  # DQN
    env = GrayscaleObservation(env, keep_dim=env_config.grayscale) # from DQN
    env = FrameStackObservation(env, env_config.FrameStackObservation) # DQN

    return env

def run_game(game: Game, render_mode: Literal["human", "rgb_array", None],agent: Agent,n_eps: int,max_steps: int) -> list[frameResult]: 
    """
    Run n episodes of the game with our agent and return the average reward at each frame.
    
    Args:
        game: the game to run
        render_mode: rendering mode during execution
        agent: agent from our Agent class
        n_eps: number of episodes to run for. PPO ran for 3
        max_steps: max steps to run for if game is not won/loss yet
        
    Returns:
        list of frameResult objects where reward is the average reward at that frame across all episodes
    """
    env = make_env(game,render_mode)
    
    
    episodes_frames = []
    for i in range(n_eps):
        cur_observation = env.reset()[0]
        agent.reset(env)
        
        episode_frames = []
        done = False
        cur_step = 0

        while not done and cur_step < max_steps:
            action = agent.select_action(cur_observation)
            cur_observation, cur_r, done_term, done_trunc, _ = env.step(action)
            
            cur_r = float(cur_r)
            done = done_term or done_trunc
            
            episode_frames.append(cur_r)
            cur_step += 1

        episodes_frames.append(episode_frames)
    
    env.close()
    
    max_frames = max(len(ep) for ep in episodes_frames) if episodes_frames else 0
    avg_rewards_per_frame = []
    
    for frame_idx in range(max_frames):
        rewards_at_frame = [ep[frame_idx] for ep in episodes_frames if frame_idx < len(ep)]
        avg_reward = np.mean(rewards_at_frame) if rewards_at_frame else 0.0
        avg_rewards_per_frame.append(frameResult(t=frame_idx, reward=float(avg_reward)))
    
    return avg_rewards_per_frame




def run_benchmark(agents: list[Agent], games_list: list[str] | None) -> list[GameResult]:
    '''
    Run the list of games with the PPO benchmark parameters.

    Args:
        agent: agent to be evaluated on the list of games
        games_list: list of games to be run, or None for full list of games
    '''
    if games_list == None:
        games = make_games(VALID_GAMES)
    else:
        games = make_games(games_list)

    bench_config = config.BenchmarkConfig()

    game_results = []
    for a,agent in enumerate(agents):
        print(f"Using agent {agent} - {a}/{len(agents)}")
        for game in games:
            print(f"  Running game {game.name} - ") #TODO: addgame length
            seed_results = []
            for i,seed in enumerate(bench_config.seeds):
                print(f"     Seed {seed} - {i}/{len(bench_config.seeds)}")
                random.seed(seed)
                game_frames = run_game(game,bench_config.render_mode,agent,bench_config.episodes,bench_config.max_steps)

                seed_result = seedResult(seed,game_frames)
                seed_results.append(seed_result)
            
            game_result = GameResult(agent,game,seed_results)
            game_results.append(game_result)

    return game_results



def plot_results(game_results: list[GameResult])-> None:
    """
    Plot like in the PPO paper so we can compare with their results. One color per agent, one line per seed.
    
    Args:
        game_results: List of GameResult objects potentially from multiple agents
    """
    if not game_results:
        return
    
    # first we group results by game, as GameResult is per agent per game
    game_dict = {}
    for result in game_results:
        game_name = result.game.name
        if game_name not in game_dict:
            game_dict[game_name] = []
        game_dict[game_name].append(result)
    
    # find grid size for our subplot so that we know how many we can fit
    n_games = len(game_dict)
    n_cols = 2
    n_rows = (n_games + n_cols - 1) // n_cols
    
    # PLOTTING
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
    axes_all = axes_grid.ravel()
    axes = axes_all[:n_games]  # Only use as many axes as we have games
    
    # we want one color per agent, and  we want those colors to stay the same, so create dict
    color_palette = plt.cm.tab10(np.linspace(0, 1, 10)) #type:ignore -> ignore tab10 error, it works!
    agent_colors = {}
    color_idx = 0
    
    for idx, (game_name, game_results_list) in enumerate(game_dict.items()):
        ax = axes[idx]
        if isinstance(ax, np.ndarray):
            ax = ax.ravel()[0]
        
        for game_result in game_results_list:
            agent_name = game_result.agent.__class__.__name__
            # if we've not seen the agent give it another color
            if agent_name not in agent_colors:
                agent_colors[agent_name] = color_palette[color_idx % len(color_palette)]
                color_idx += 1
            color = agent_colors[agent_name]
            
            for seed_result in game_result.seed_results:
                frame_rewards = seed_result.frame_rewards
                frames = [frame.t for frame in frame_rewards]
                rewards = [frame.reward for frame in frame_rewards]
                
                # Compute cumulative rewards
                cumulative_rewards = np.cumsum(rewards)
                ax.plot(frames, cumulative_rewards, color=color, alpha=0.7, linewidth=1.5)
        
        ax.set_title(game_name, fontsize=12)
        ax.grid()
    
    # Remove extra subplots
    for idx in range(n_games, len(axes_all)):
        fig.delaxes(axes_all[idx])
    
    # Create legend with color per agent (see above)
    legend_elements = [Patch(facecolor=color, label=agent) for agent, color in agent_colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
