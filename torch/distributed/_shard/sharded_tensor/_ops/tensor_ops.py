import copy

import torch
from torch.distributed._shard.sharded_tensor import (
    sharded_op_impl,
    ShardedTensor
)

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

# __reduce_ex__ to dispatch to get_state/set_state
register_default_op(torch.Tensor.__reduce_ex__)
