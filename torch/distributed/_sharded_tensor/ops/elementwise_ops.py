import copy

import torch
from torch.distributed._sharded_tensor import (
    sharded_op_impl,
    Shard,
    ShardedTensor,
)


@sharded_op_impl(torch.nn.functional.gelu)
def gelu(types, args=(), kwargs=None, pg=None):
    """
    Handles ``__torch_function__`` dispatch for the elementwise op
    ``torch.nn.functional.gelu``.
    This method computes on either a normal tensor or a sharded tensor.
    """
    input = args[0]
    op = torch.nn.functional.gelu
    # Validate types
    if not isinstance(input, torch.Tensor) and not isinstance(input, ShardedTensor):
        raise TypeError("input needs to be either torch.Tensor or ShardedTensor")
    if isinstance(input, torch.Tensor):
        return op(args[0])
    else:
        local_shards_new = []
        for local_shard in input.local_shards():
            local_shards_new.append(Shard(op(local_shard.tensor), local_shard.metadata))
        new_st = ShardedTensor._init_from_local_shards(
            local_shards_new, input.size(), process_group=pg
        )

        # Manually set sharding_spec
        new_st._sharding_spec = copy.deepcopy(input._sharding_spec)
        return new_st
