import torch


def math_ops(types, args, kwargs):
    """
    Handles ``__torch_function__`` dispatch for  math ops like
    ``torch.nn.functional.gelu``.
    This method computes a sharded linear and has the following limitations:

    1. It is used only for math ops which modifies the tensor in place.
    """
    input = args[0]
    math_op = kwargs['math_op']
    from torch.distributed._sharded_tensor import ShardedTensor
    # Validate types
    if not isinstance(input, torch.Tensor) and not isinstance(input, ShardedTensor):
        raise TypeError("input needs to be either torch.Tensor or ShardedTensor")
    if isinstance(input, torch.Tensor):
        return math_op(args[0])
    else:
        for local_shard in input.local_shards():
            local_shard.tensor = math_op(local_shard.tensor)
        return input
