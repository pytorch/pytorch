import copy

import torch
from torch.distributed._shard.sharded_tensor import (
    sharded_op_impl,
    Shard,
    ShardedTensor,
)


def register_elementwise_op(op):
    @sharded_op_impl(op)
    def elementwise_op(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the elementwise op such
        as ``torch.nn.functional.gelu`` or ``torch.nn.functional.relu``.
        This method computes on either a normal tensor or a sharded tensor.
        """
        input = args[0]
        # Validate types
        if not isinstance(input, ShardedTensor):
            raise TypeError("input needs to be a ShardedTensor")
        local_shards_new = []
        for local_shard in input.local_shards():
            local_shards_new.append(Shard(op(local_shard.tensor), local_shard.metadata))
        # TODO: After a new API for sharded tensor creation, we need to replace this.
        # https://github.com/pytorch/pytorch/issues/72092
        new_st = ShardedTensor._init_from_local_shards(
            local_shards_new, input.size(), process_group=pg
        )

        # Manually set sharding_spec
        new_st._sharding_spec = copy.deepcopy(input._sharding_spec)
        return new_st


register_elementwise_op(torch.nn.functional.gelu)
register_elementwise_op(torch.nn.functional.relu)
