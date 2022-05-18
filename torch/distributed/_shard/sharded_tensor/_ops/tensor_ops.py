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

def sharded_deepcopy(args, kwargs, pg):
    # NOTE: we directly implement deepcopy magic method
    # instead of using the default tensor.__deepcopy__
    # and implement clone(). This is because the default
    # tensor deepcopy copies every attribute, but the
    # process_group in ShardedTensor cannot be deep copied.
    self_st = args[0]
    new_local_shards = copy.deepcopy(self_st.local_shards())
    new_metadata = copy.deepcopy(self_st.metadata())
    return new_local_shards, new_metadata

_register_sharded_op_on_local_shards(
    torch.Tensor.__deepcopy__,
    customized_func=sharded_deepcopy,
)

def sharded_clone(args, kwargs, pg):
    self_st = args[0]
    desire_memory_format = kwargs.get("memory_format", None)
    if desire_memory_format and desire_memory_format != torch.preserve_format:
        raise RuntimeError("Only support torch.preserve_format for ShardedTensor!")
    cloned_local_shards = [
        Shard(
            local_shard.tensor.clone(memory_format=desire_memory_format),
            metadata=copy.deepcopy(local_shard.metadata),
        )
        for local_shard in self_st.local_shards()
    ]
    new_metadata = copy.deepcopy(self_st.metadata())
    return cloned_local_shards, new_metadata

_register_sharded_op_on_local_shards(
    torch.Tensor.clone,
    customized_func=sharded_clone,
)

def sharded_detach(args, kwargs, pg):
    self_st = args[0]
    detached_local_shards = [
        Shard(
            local_shard.tensor.detach(),
            metadata=copy.deepcopy(local_shard.metadata),
        )
        for local_shard in self_st.local_shards()
    ]
    new_metadata = copy.deepcopy(self_st.metadata())
    new_metadata.tensor_properties.requires_grad = False
    return detached_local_shards, new_metadata

_register_sharded_op_on_local_shards(
    torch.Tensor.detach,
    customized_func=sharded_detach,
)

@sharded_op_impl(torch.Tensor.requires_grad_)
def tensor_requires_grad_set(types, args=(), kwargs=None, pg=None):
    self_st = args[0]
    requires_grad = args[1]
    # Validate types
    if not isinstance(self_st, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")

    if requires_grad == self_st.requires_grad:
        return self_st

    for local_shard in self_st.local_shards():
        local_shard.tensor.requires_grad_(requires_grad)

    # update the metadata in the meanwhile
    self_st._metadata.tensor_properties.requires_grad = requires_grad
    return self_st
