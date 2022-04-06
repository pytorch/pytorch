import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import (
    sharded_op_impl,
    ShardedTensor,
)

@sharded_op_impl(torch.bmm)
def sharded_bmm(types, args=(), kwargs=None, pg=None):
    """
    Handles ``__torch_function__`` dispatch for the sharded_bmm op.
    """
    input = args[0]
    mat2 = args[1]

    # Validate types
    if not isinstance(input, ShardedTensor) or not isinstance(mat2, ShardedTensor):
        raise TypeError("input or mat2 needs to be a ShardedTensor")
    if input.dim() != 3 or mat2.dim() != 3:
        raise TypeError("input or mat2 needs to be a 3D ShardedTensor")
    local_tensor = torch.bmm(input.local_tensor(), mat2.local_tensor())
    new_st_size = list(input.size())
    new_sharding_dim = input.sharding_spec().dim
    mat2_sharding_dim = mat2.sharding_spec().dim
    new_st_size[-1] = mat2.size()[-1]
    if new_sharding_dim == -1 or new_sharding_dim == (input.dim() - 1):
        if mat2_sharding_dim != -1 and mat2_sharding_dim != (mat2.dim() - 1):
            new_st_size[-1] *= dist.get_world_size(pg)

    return ShardedTensor._init_from_local_tensor(
        local_tensor.contiguous(),
        input.sharding_spec(),
        tuple(new_st_size),
        process_group=pg,
    )

@sharded_op_impl(torch.nn.functional.softmax)
def sharded_softmax(types, args=(), kwargs=None, pg=None):
    """
    Handles ``__torch_function__`` dispatch for the functional.softmax op.
    """
    input = args[0]
    dim = kwargs.get("dim")
    dtype = kwargs.get("dtype")

    # Validate types
    if not isinstance(input, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")
    local_tensor = input.local_tensor()
    local_tensor = torch.nn.functional.softmax(local_tensor, dim=dim, dtype=dtype)
    return ShardedTensor._init_from_local_tensor(
        local_tensor.contiguous(),
        input.sharding_spec(),
        input.size(),
        process_group=pg,
    )

@sharded_op_impl(torch.nn.functional.dropout)
def sharded_dropout(types, args=(), kwargs=None, pg=None):
    """
    Handles ``__torch_function__`` dispatch for the Dropout op.
    """
    input = args[0]
    p = kwargs.get("p")

    # Validate types
    if not isinstance(input, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")
    local_tensor = input.local_tensor()
    local_tensor = torch.nn.functional.dropout(local_tensor, p=p)
    return ShardedTensor._init_from_local_tensor(
        local_tensor.contiguous(),
        input.sharding_spec(),
        input.size(),
        process_group=pg,
    )
