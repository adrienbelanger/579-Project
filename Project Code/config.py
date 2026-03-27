from dataclasses import dataclass, field

@dataclass(frozen=True)
class EnvConfig:
    frameskip=1   # we're hadnlign the frame skip in the MaxAndSkipObservation so don't do it in the env as well
    repeat_action_probability=0.25 # this is the part from machalo et al.
    full_action_space=True
    MaxAndSkipObservation=4
    ResizeObservation=(84,84)
    grayscale = True
    FrameStackObservation=4



@dataclass(frozen=True)
class ActorCriticConfig:
    '''
    Taken from the MHNI 16 paper
    '''
    input_channels: int = 4
    input_height: int = 84
    input_width: int = 84
    conv_channels: tuple[int, int] = (16, 32)
    kernel_sizes: tuple[int, int] = (8, 4)
    strides: tuple[int, int] = (4, 2)
    hidden_size: int = 256
    base_init_gain: float = 1.41421356237  # sqrt(2)
    policy_init_gain: float = 0.01
    value_init_gain: float = 1.0



@dataclass(frozen=True)
class BenchmarkConfig:
    seeds= (7, 4002, 451)
    render_mode= None
    episodes= 3
    max_episode_steps = 10_000_000  # cap for evaluation, after this we have enough score


@dataclass(frozen=True)
class PPOConfig:
    # Atari PPO hyperparameters from the original paper. Annealing is done through anneal_learning_rate
    actors = 8
    horizon= 128
    epochs= 3
    minibatch_size = 32 * 8
    discount = 0.99
    lambda_bias_var= 0.95
    learning_rate = 2.5e-4
    adam_eps = 1e-5
    anneal_learning_rate= True # in the PPO paper they anneal by deault
    clipping = 0.1
    anneal_clipping = True
    val_func_coeff = 1.0
    entropy_coeff = 0.01
    total_timesteps = 10_000_000  # supposed to be 10M timesteps with frame skip 4.
    max_grad_norm = 0.5
    seed = None
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
