import torch

from ._common import (
    _register_sharded_op_on_local_shards,
)

_register_sharded_op_on_local_shards(torch.nn.functional.gelu)
_register_sharded_op_on_local_shards(torch.nn.functional.relu)
_register_sharded_op_on_local_shards(torch.nn.functional.dropout)
_register_sharded_op_on_local_shards(torch.Tensor.tanh)
