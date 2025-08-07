import dataclasses
from typing import Callable, Optional


all_experiments: dict[str, Callable] = {}


@dataclasses.dataclass
class Experiment:
    name: str
    metric: str
    target: float
    actual: float
    dtype: str
    device: str
    arch: str  # GPU name for CUDA or CPU arch for CPU
    is_model: bool = False


def register_experiment(name: Optional[str] = None):
    def decorator(func):
        key = name or func.__name__
        all_experiments[key] = func
        return func

    return decorator
