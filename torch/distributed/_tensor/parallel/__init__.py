# Copyright (c) Meta Platforms, Inc. and affiliates
from torch.distributed._tensor.parallel.multihead_attention_tp import (
    TensorParallelMultiheadAttention,
)

from torch.distributed._tensor.parallel.api import (
    tp_shard_self_attn,
    replicate_input,
    replicate_output,
)

from torch.distributed._tensor.parallel.style import (
    ParallelStyle,
    PairwiseParallel,
    RowwiseParallel,
    ColwiseParallel,
    make_input_shard_1d,
    make_input_replicate_1d,
    make_output_shard_1d,
    make_output_replicate_1d,
    make_output_tensor,
)
