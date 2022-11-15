# Copyright (c) Meta Platforms, Inc. and affiliates
from torch.distributed._tensor.parallel.multihead_attention_tp import (
    TensorParallelMultiheadAttention,
)

from torch.distributed._tensor.parallel.api import (
    tp_shard_self_attn,
    replicate_input,
    replicate_output,
)
