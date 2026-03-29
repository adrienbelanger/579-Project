import gymnasium as gym
from gymnasium.wrappers import (ResizeObservation,GrayscaleObservation,FrameStackObservation,MaxAndSkipObservation,)
import numpy as np
from agent import Agent, PolicyAgent
from game import Game, VALID_GAMES, make_games
from model import EpisodePoint
import pandas as pd
from dataclasses import dataclass # so we can do game class so it's easier for our plotting
from typing import Literal, Tuple
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import multiprocessing.pool
import os
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
    episode_points: list[EpisodePoint]
    
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

def run_game(game: Game, render_mode: Literal["human", "rgb_array", None],agent: Agent,n_eps: int,max_episode_steps: int) -> list[frameResult]: 
    """
    Run n episodes of the game with our agent and return the average reward at each frame.
    
    Args:
        game: the game to run
        render_mode: rendering mode during execution
        agent: agent from our Agent class
        n_eps: number of evaluation episodes to run
        max_episode_steps: max steps to run for if game is not won/loss yet
        
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

        while not done and cur_step < max_episode_steps:
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

class NoDaemonProcess(multiprocessing.Process):
    def __init__(self, ctx=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def _train_one(args: tuple) -> seedResult:
    agent_cls, game, seed = args
    random.seed(seed)
    np.random.seed(seed)
    run_agent = agent_cls()
    episode_points = run_agent.train(game, seed=seed)
    return seedResult(seed, episode_points)



def run_benchmark(agents: list[Agent], games_list: list[str] | None) -> None:
    if games_list == None:
        games = make_games(VALID_GAMES)
    else:
        games = make_games(games_list)

    bench_config = config.BenchmarkConfig()

    game_results = []
    for a, agent in enumerate(agents):
        if not isinstance(agent, PolicyAgent):
            raise ValueError("run_benchmark only supports PolicyAgent inputs. ")
        print(f"Using agent {agent.__class__.__name__} - {a+1}/{len(agents)}")
        for ge, game in enumerate(games):
            print(f"  On game {game.name}, {ge+1}/{len(games)}")

            tasks = [(agent.__class__, game, seed) for seed in bench_config.seeds]
            ppo_cfg = config.PPOConfig()
            n_workers = min(len(bench_config.seeds), (os.cpu_count() or 1) // ppo_cfg.actors) # depending onhow many cpu cores, ou might not be able to parallelize.. Colab??
            with NoDaemonPool(processes=max(1, n_workers)) as pool:
                seed_results = pool.map(_train_one, tasks)

            game_result = GameResult(agent, game, seed_results)
            game_results.append(game_result)

    plot_results(game_results)
    plot_final_scores_table(game_results)
def plot_results(game_results: list[GameResult])-> None:
    """
    Plot PPO learning curves. One color per agent, one line per seed.
    
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
                episode_points = seed_result.episode_points
                if not episode_points:
                    continue
                game_frames = [point.game_frames for point in episode_points]
                episode_returns = [point.episode_return for point in episode_points]
                ax.plot(game_frames, episode_returns, color=color, alpha=0.7, linewidth=1.5)
        
        ax.set_title(game_name, fontsize=12)
        ax.set_xlabel("Game Frames")
        ax.set_ylabel("Episode Return")
        ax.grid()
    
    # Remove extra subplots
    for idx in range(n_games, len(axes_all)):
        fig.delaxes(axes_all[idx])
    
    # Create legend with color per agent (see above)
    legend_elements = [Patch(facecolor=color, label=agent) for agent, color in agent_colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_final_scores_table(game_results: list[GameResult]) -> None:
    """
    Plot a table of final scores, like table 6 in the original paper
    """
    rows_by_game: dict[str, dict[str, float]] = {}
    agent_names: list[str] = []

    for result in game_results:
        # get each game name and agent name.
        game_name = result.game.name
        agent_name = result.agent.__class__.__name__
        if agent_name not in agent_names:
            agent_names.append(agent_name)

        per_seed_scores = []
        for seed_result in result.seed_results:
            episode_returns = [point.episode_return for point in seed_result.episode_points]
            if episode_returns:
                per_seed_scores.append(float(np.mean(episode_returns[-100:])))

        mean_final_score = float(np.mean(per_seed_scores)) if per_seed_scores else np.nan
        rows_by_game.setdefault(game_name, {})[agent_name] = mean_final_score

    table_rows = []
    for game_name, agent_scores in rows_by_game.items():
        row = {"Game": game_name}
        for agent_name in agent_names:
            score = agent_scores.get(agent_name, np.nan)
            row[agent_name] = "" if pd.isna(score) else f"{score:.1f}"
        table_rows.append(row)

    table_df = pd.DataFrame(table_rows, columns=["Game", *agent_names])

    fig_height = max(2.5, 0.35 * (len(table_df) + 1))
    fig_width = max(6.0, 2.5 + 1.6 * len(table_df.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title("Mean Final Scores (Last 100 Episodes)", fontsize=12, pad=12)

    table = ax.table(
        cellText=table_df.values, # type:ignore # random typing errors
        colLabels=table_df.columns, #type:ignore
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    plt.tight_layout()
    plt.show()
