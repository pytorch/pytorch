# Copyright (c) Meta Platforms, Inc. and affiliates
from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.multihead_attention_tp import (
    TensorParallelMultiheadAttention,
)

from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    make_input_replicate_1d,
    make_input_reshard_replicate,
    make_input_shard_1d,
    make_input_shard_1d_last_dim,
    make_output_replicate_1d,
    make_output_reshard_tensor,
    make_output_shard_1d,
    make_output_tensor,
    PairwiseParallel,
    PairwiseSequenceParallel,
    ParallelStyle,
    RowwiseParallel,
)

__all__ = [
    "ColwiseParallel",
    "PairwiseParallel",
    "PairwiseSequenceParallel",
    "ParallelStyle",
    "RowwiseParallel",
    "TensorParallelMultiheadAttention",
    "make_input_replicate_1d",
    "make_input_reshard_replicate",
    "make_input_shard_1d",
    "make_input_shard_1d_last_dim",
    "make_output_replicate_1d",
    "make_output_reshard_tensor",
    "make_output_tensor",
    "make_output_shard_1d",
    "parallelize_module",
]
