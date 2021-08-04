import torch
import contextlib
from typing import Iterator

# Context manager that causes all factory functions to dispatch to the passed-in
# tensor's __torch_dispatch__ function. A factory function is defined as an
# operation that accepts no tensors but returns a tensor.
#
# enable_factory_dispatch is affected by torch._C._DisableTorchDispatch.
#
# NB: Calling a factory function inside __torch_dispatch__ does go through
# __torch_dispatch__ again. Please use _DisableTorchDispatch inside
# __torch_dispatch__ to prevent infinite recursion. (We are able to prevent
# this from happening by adding a guard into PythonMode.cpp, but I'm not
# sure that is a good idea. For example, we don't automatically prevent the
# user from using the same __torch_dispatch__ object inside __torch_dispatch__).
#
# TODO: Limitations and things about enable_factory_dispatch we should fix:
# - it currently cannot be nested
# - it doesn't work with the tensor constructors (torch.tensor, torch.Tensor)
# - we should test that it works with whatever user-facing API for
#   torch._C._DisableTorchDispatch that we provide.
@contextlib.contextmanager
def enable_factory_dispatch(tensor: torch.Tensor) -> Iterator[None]:
    if not hasattr(tensor, '__torch_dispatch__'):
        raise ValueError('The tensor passed to enable_factory_torch_dispatch '
                         'must have a __torch_dispatch__ classmethod')
    torch._C._python_mode_set_torch_dispatch(tensor)
    try:
        yield
    finally:
        torch._C._python_mode_reset_torch_dispatch()
