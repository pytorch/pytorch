import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from ._common import (
    _register_sharded_op_on_local_tensor,
)

@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.softmax)
def sharded_softmax(types, args=(), kwargs=None):
    input = args[0]
    pg = input._process_group
    dim = kwargs['dim']
    sharding_dim = input.sharding_spec().dim
    ndims = input.dim()
    if dim == sharding_dim or dim + ndims == sharding_dim or sharding_dim + ndims == dim:
        exp = torch.exp(input.local_tensor())
        exp_sum = exp.sum(dim=dim).unsqueeze(dim=dim)
        exp_sum = torch.distributed.nn.functional.all_reduce(exp_sum, group=pg)
        smax = torch.div(exp, exp_sum)
    else:
        smax = torch.nn.functional.softmax(input.local_tensor(), dim=dim)
    return ShardedTensor._init_from_local_tensor(smax, input.sharding_spec(), input.size(), process_group=pg)

_register_sharded_op_on_local_tensor(
    torch.nn.functional.softmax,
    customized_func=sharded_softmax,
)
