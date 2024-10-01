import dataclasses
from contextlib import nullcontext
from enum import Enum
from typing import List

import torch

from .metrics import Device, Metrics, profile_range


@dataclasses.dataclass
class BenchmarkConfig:
    device: Device
    dtype: torch.dtype
    phase: str
    max_samples: int
    repeat: int
    metrics: List[Metrics]
    profile: bool
    profile_folder: str
    enable_nvtx: bool


class Phase(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    FULL = "full"


dtype_mapping = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def maybe_record_function(name: str, benchmark_config: BenchmarkConfig, sample_idx: int = None, repeat_idx: int = None):
    if benchmark_config.enable_nvtx:
        if sample_idx is not None:
            return profile_range(name)
        elif repeat_idx is not None and repeat_idx == benchmark_config.repeat - 1:
            # only record the last repeat
            return profile_range(name)
        else:
            return nullcontext()
    elif benchmark_config.profile:
        return torch.profiler.record_function(name)
    else:
        return nullcontext()
