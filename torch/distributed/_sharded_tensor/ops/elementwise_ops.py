import torch


def elementwise_ops(types, args, kwargs):
    """
    Handles ``__torch_function__`` dispatch for elementwise ops like
    ``torch.nn.functional.gelu``.
    This method computes on either a normal tensor or a sharded tensor.
    """
    input = args[0]
    op = kwargs['op']
    from torch.distributed._sharded_tensor import ShardedTensor
    # Validate types
    if not isinstance(input, torch.Tensor) and not isinstance(input, ShardedTensor):
        raise TypeError("input needs to be either torch.Tensor or ShardedTensor")
    if isinstance(input, torch.Tensor):
        return op(args[0])
    else:
        for local_shard in input.local_shards():
            local_shard.tensor = op(local_shard.tensor)
        return input
