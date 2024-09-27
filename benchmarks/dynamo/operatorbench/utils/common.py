import dataclasses
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
