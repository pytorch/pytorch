# Copyright (c) Meta Platforms, Inc. and affiliates
from torch.distributed.tensor.parallel.api import parallelize_module

from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)

__all__ = [
    "ColwiseParallel",
    "ParallelStyle",
    "PrepareModuleInput",
    "PrepareModuleOutput",
    "RowwiseParallel",
    "parallelize_module",
]
