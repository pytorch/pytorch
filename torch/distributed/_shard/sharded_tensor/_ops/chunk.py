import torch
from torch.distributed._shard.sharded_tensor import (
    sharded_op_impl,
    ShardedTensor,
)


@sharded_op_impl(torch.chunk)
def sharded_chunk(types, args=(), kwargs=None, pg=None):
    """
    Handles ``__torch_function__`` dispatch for the chunk op.
    """
    input = args[0]
    chunks = args[1]
    dim = kwargs.get("dim")
    dim = dim if dim else 0

    # Validate types
    if not isinstance(input, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")
    st_size = list(input.size())
    local_tensor = input.local_tensor()
    tensor_chunks = torch.chunk(local_tensor, chunks, dim)
    st_size[dim] //= chunks
    results = []
    for tensor in tensor_chunks:
        results.append(
            ShardedTensor._init_from_local_tensor(
                tensor.contiguous(),
                input.sharding_spec(),
                tuple(st_size),
                process_group=pg,
            )
        )
    return results
