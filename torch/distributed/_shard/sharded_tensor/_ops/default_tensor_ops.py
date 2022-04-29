import torch
from torch.distributed._shard.sharded_tensor import (
    sharded_op_impl,
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
