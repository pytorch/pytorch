# Copyright (c) Meta Platforms, Inc. and affiliates
__all__ = [
    "ColwiseParallel",
    "ParallelStyle",
    "PrepareModuleInput",
    "PrepareModuleOutput",
    "RowwiseParallel",
    "SequenceParallel",
    "loss_parallel",
    "parallelize_module",
]

from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.loss import loss_parallel
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
