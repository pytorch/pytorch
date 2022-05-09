import copy
import torch
from torch.distributed._shard.sharded_tensor import (
    sharded_op_impl,
    Shard,
    ShardedTensor,
)
from ._common import (
    _register_sharded_op_on_local_shards,
)


def register_default_op(op):
    @sharded_op_impl(op)
    def tensor_default_op(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the default tensor ops that
        behave the same as ``torch.Tensor`` such as ``torch.Tensor.shape`` or
        ``torch.Tensor.dtype``. We simply lower to the real op call with
        DisableTorchFunction context like ``torch.Tensor.__torch_function__``
        to avoid recursions.
        """
        if kwargs is None:
            kwargs = {}

        with torch._C.DisableTorchFunction():
            return op(*args, **kwargs)

# Tensor properties access
register_default_op(torch.Tensor.requires_grad.__get__)  # type: ignore[attr-defined]
register_default_op(torch.Tensor.shape.__get__)  # type: ignore[attr-defined]
register_default_op(torch.Tensor.dtype.__get__)  # type: ignore[attr-defined]
register_default_op(torch.Tensor.layout.__get__)  # type: ignore[attr-defined]
register_default_op(torch.Tensor.size)
register_default_op(torch.Tensor.dim)
register_default_op(torch.Tensor.ndim.__get__)  # type: ignore[attr-defined]
register_default_op(torch.Tensor.is_contiguous)
register_default_op(torch.Tensor.contiguous)

# __reduce_ex__ to dispatch to get_state/set_state
register_default_op(torch.Tensor.__reduce_ex__)

def sharded_type_as_check(*args, **kwargs):
    """
    Perform extra checks for the sharded_type_as op such as the input needs to
    be either a Tensor or ShardedTensor.

    Args: same as ``torch.Tensor.type_as``.

    Return: None
    """
    if len(args) < 2:
        raise ValueError("Needs to give a tensor to cast type as!")
    if not isinstance(args[1], torch.Tensor) and not isinstance(args[1], ShardedTensor):
        raise ValueError("Needs to give a Tensor or ShardedTensor to cast type as!")


def same_dtype(*args, **kwargs):
    """
    When the dtype is the same, return the original ShardedTensor.

    Args: same as ``torch.Tensor.type_as``.

    Return (bool): Whether to return early or not.
    """
    return args[0].dtype == args[1].dtype


def sharded_type_as(args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for the ``torch.Tensor.type_as`` op.

    Args: same as ``torch.Tensor.type_as``.

    Return:
        new_local_shards (List[Shard]): Local shards for the new sharded tensor.
        st_meta (ShardedTensorMetadata): Metadata of the new sharded tensor.
    """
    st = args[0]
    tensor = args[1]
    if isinstance(tensor, ShardedTensor):
        tensor = tensor.local_tensor()
    new_local_shards = []
    for shard in st.local_shards():
        new_local_shards.append(Shard(shard.tensor.type_as(tensor), shard.metadata))
    st_meta = copy.deepcopy(st._metadata)
    st_meta.tensor_properties.dtype = tensor.dtype
    return new_local_shards, st_meta


_register_sharded_op_on_local_shards(
    torch.Tensor.type_as,
    early_stop_func=same_dtype,
    extra_check=sharded_type_as_check,
    customized_func=sharded_type_as,
)
