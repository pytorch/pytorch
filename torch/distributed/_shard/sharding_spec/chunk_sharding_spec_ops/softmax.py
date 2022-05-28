import torch
from ._common import (
    _register_sharded_op_on_local_tensor,
)

def sharded_softmax(args, kwargs, pg):
    input = args[0]
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
    return smax, input.sharding_spec(), input.size()

_register_sharded_op_on_local_tensor(
    torch.nn.functional.softmax,
    customized_func=sharded_softmax,
)
