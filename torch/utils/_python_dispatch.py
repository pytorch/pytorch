import torch
import contextlib
from typing import Iterator

# Context manager that causes all pytorch operators to dispatch to the passed-in
# type's __torch_dispatch__ function.
# operation that accepts no tensors but returns a tensor.
#
# enable_python_mode is affected by torch._C._DisableTorchDispatch.
#
# NB: Calling an operator inside __torch_dispatch__ does go through
# __torch_dispatch__ again. Please use _DisableTorchDispatch inside
# __torch_dispatch__ to prevent infinite recursion.
#
# TODO: Limitations and things about enable_python_mode we should fix before exposing it:
# - it currently cannot be nested. This should be simple to implement; we need a
#   stack of TorchDispatchTypeObjects and the next bullet point.
# - We need a better user-facing api for torch._C._DisableTorchDispatch that
#   is able to selectively disable __torch_dispatch__ of a particular class.
# - It doesn't work with the tensor constructors (torch.tensor, torch.Tensor)
# - Better name (see https://github.com/pytorch/pytorch/pull/63496#discussion_r694091694)
@contextlib.contextmanager
def enable_python_mode(cls) -> Iterator[None]:
    if not hasattr(cls, '__torch_dispatch__'):
        raise ValueError('The class passed to enable_python_mode '
                         'must have a __torch_dispatch__ classmethod')
    if not isinstance(cls, type) or not issubclass(cls, (torch.Tensor,)):
        raise ValueError('The argument passed to enable_python_mode '
                         'must be the type of a Tensor subclass')
    torch._C._enter_python_mode(cls)
    try:
        yield
    finally:
        torch._C._exit_python_mode()
