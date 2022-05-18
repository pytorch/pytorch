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
from torch.distributed._shard.common_op_utils import _register_default_op

@sharded_op_impl(torch.Tensor.__deepcopy__)
def tensor_deepcopy(types, args=(), kwargs=None, pg=None):
    # NOTE: we directly implement deepcopy magic method
    # instead of using the default tensor.__deepcopy__
    # and implement clone(). This is because the default
    # tensor deepcopy copies every attribute, but the
    # process_group in ShardedTensor cannot be deep copied.
    self_st = args[0]
    # Validate types
    if not isinstance(self_st, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")

    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards=copy.deepcopy(self_st.local_shards()),
        sharded_tensor_metadata=copy.deepcopy(self_st.metadata()),
        process_group=self_st._process_group,
        init_rrefs=self_st._init_rrefs
    )


# Tensor properties access
_register_default_op(torch.Tensor.requires_grad.__get__, sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.shape.__get__, sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.dtype.__get__, sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.layout.__get__, sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.size, sharded_op_impl)
_register_default_op(torch.Tensor.dim, sharded_op_impl)
_register_default_op(torch.Tensor.ndim.__get__, sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.is_contiguous, sharded_op_impl)
_register_default_op(torch.Tensor.contiguous, sharded_op_impl)

# __reduce_ex__ to dispatch to get_state/set_state
_register_default_op(torch.Tensor.__reduce_ex__, sharded_op_impl)

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
