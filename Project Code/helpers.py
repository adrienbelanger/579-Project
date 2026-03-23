import gymnasium as gym
from gymnasium.wrappers import (ResizeObservation,GrayscaleObservation,FrameStackObservation,MaxAndSkipObservation,)

def make_env(
    game: str,
    render_mode: Literal["human", "rgb_array", None] = None,
):
    """
    These parameters are from PPO, which some werer from DQN or ACER or some paper Machalo et al. that aimed to improve this benchmark. These are the parameters they used

    Args:
        game: name of our atari game, should be from our list from cell 1 (i.e. the standard list)
        render_mode:
            - None: Keep as None for the trining, no rendering
            - "human": renders game as it runs (fun)
            - "rgb_array": which when we do env.render() returns the current frame as a np array, whcih means we can make a fun video for the presentation
    """

    env = gym.make(
        f"ALE/{game}-v5",
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