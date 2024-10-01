import dataclasses
from contextlib import nullcontext
from enum import Enum
from typing import List

import torch

from .metrics import Device, Metrics


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


class Phase(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    FULL = "full"


dtype_mapping = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def maybe_record_function(name, profile_enabled):
    return torch.profiler.record_function(name) if profile_enabled else nullcontext()
