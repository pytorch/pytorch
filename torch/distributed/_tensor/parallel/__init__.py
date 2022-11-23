# Copyright (c) Meta Platforms, Inc. and affiliates
from torch.distributed._tensor.parallel.multihead_attention_tp import (
    TensorParallelMultiheadAttention,
)

from torch.distributed._tensor.parallel.style import (
    ColwiseParallel,
    PairwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    make_input_replicate_1d,
    make_input_shard_1d,
    make_input_shard_1d_dim_last,
    make_output_replicate_1d,
    make_output_shard_1d,
    make_output_tensor,
)

from torch.distributed._tensor.parallel.api import (
    parallelize_module,
)

__all__ = [
    "ColwiseParallel",
    "PairwiseParallel",
    "ParallelStyle",
    "RowwiseParallel",
    "TensorParallelMultiheadAttention",
    "make_input_replicate_1d",
    "make_input_shard_1d",
    "make_input_shard_1d_dim_last",
    "make_output_replicate_1d",
    "make_output_tensor",
    "make_output_shard_1d",
    "parallelize_module",
]
