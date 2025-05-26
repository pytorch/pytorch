# mypy: allow-untyped-defs
import copy

import torch
from torch.distributed._shard.common_op_utils import _register_default_op
from torch.distributed._shard.sharded_tensor import (
    _sharded_op_impl,
    Shard,
    ShardedTensor,
)

from ._common import _register_sharded_op_on_local_shards


# Tensor properties access
_register_default_op(torch.Tensor.shape.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.dtype.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.layout.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.size, _sharded_op_impl)
_register_default_op(torch.Tensor.dim, _sharded_op_impl)
_register_default_op(torch.Tensor.ndim.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.is_contiguous, _sharded_op_impl)
_register_default_op(torch.Tensor.contiguous, _sharded_op_impl)
_register_default_op(torch.Tensor.is_floating_point, _sharded_op_impl)

# __reduce_ex__ to dispatch to get_state/set_state
_register_default_op(torch.Tensor.__reduce_ex__, _sharded_op_impl)

# autograd related properties
_register_default_op(torch.Tensor.requires_grad.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
# TODO: set grad with a ShardedTensor that consists of all local grads
_register_default_op(torch.Tensor.grad.__get__, _sharded_op_impl)  # type: ignore[union-attr]
_register_default_op(torch.Tensor.grad_fn.__get__, _sharded_op_impl)  # type: ignore[union-attr]
_register_default_op(torch.Tensor.is_leaf.__get__, _sharded_op_impl)  # type: ignore[attr-defined]


# device property is ambiguous as from a global prospective,
# ShardedTensor.device consists of multiple devices (might even across hosts)
# We choose to return the current device of the local tensor to represent
# the device property on each rank
@_sharded_op_impl(torch.Tensor.device.__get__)
def tensor_device(types, args=(), kwargs=None, pg=None):
    self_st = args[0]
    # Validate types
    if not isinstance(self_st, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")
    dev: torch.device
    if self_st._local_shards:
        dev = self_st._local_shards[0].tensor.device
    elif pg and pg._get_backend_name() == "gloo":
        dev = torch.device("cpu")
    else:
        dev = torch.device(torch.cuda.current_device())
    return dev


@_sharded_op_impl(torch.Tensor.is_meta.__get__)  # type: ignore[attr-defined]
def st_is_meta(types, args=(), kwargs=None, pg=None):
    return args[0].local_tensor().is_meta


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
    new_local_shards = [
        Shard(shard.tensor.type_as(tensor), shard.metadata)
        for shard in st.local_shards()
    ]
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


@_sharded_op_impl(torch.Tensor.copy_)
def sharded_inplace_copy(types, args, kwargs, pg):
    # NOTE: inplace op don't need to rewrap
    kwargs = {} if kwargs is None else kwargs
    self_st = args[0]
    new_st = args[1]
    nonblocking = kwargs.get("non_blocking", False)
    for local_shard, new_shard in zip(self_st.local_shards(), new_st.local_shards()):
        if local_shard.metadata != new_shard.metadata:
            raise RuntimeError(
                "inplace copy can only happen between two ShardedTensor with same metadata!"
            )
    for local_shard, new_shard in zip(self_st.local_shards(), new_st.local_shards()):
        local_shard.tensor.copy_(new_shard.tensor, nonblocking)

    return self_st


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


@_sharded_op_impl(torch.Tensor.requires_grad_)
def tensor_requires_grad_set(types, args=(), kwargs=None, pg=None):
    self_st = args[0]
    # Validate types
    if not isinstance(self_st, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")

    if kwargs is None:
        kwargs = {}

    requires_grad = args[1] if len(args) > 1 else kwargs.get("requires_grad", True)
    if requires_grad == self_st.requires_grad:
        return self_st

    for local_shard in self_st.local_shards():
        local_shard.tensor.requires_grad_(requires_grad)

        # update the wrapper class property
    with torch._C.DisableTorchFunctionSubclass():
        self_st.requires_grad_(requires_grad)
    # update the metadata in the meanwhile
    self_st._metadata.tensor_properties.requires_grad = requires_grad
    return self_st
