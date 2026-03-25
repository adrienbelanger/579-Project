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
class PPOConfig:
    pass



@dataclass(frozen=True)
class ModelConfig:
    pass

@dataclass(frozen=True)
class BenchmarkConfig:
    seeds = [7, 4002, 451]
    render_mode = None
    episodes= 3
    max_steps = 108000