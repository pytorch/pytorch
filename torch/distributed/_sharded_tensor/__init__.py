from torch.distributed._sharding_spec import ShardingSpec
from .api import (
    ShardedTensor,
)

def empty(
        sharding_spec: ShardingSpec,
        *size,
        dtype=None,
        requires_grad=False,
        pin_memory=False,
        process_group=None,):
    """
    Creates an empty :class:`ShardedTensor`. Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    return ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        process_group=process_group)
