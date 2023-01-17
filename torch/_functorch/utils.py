import contextlib
import torch
from torch._C._functorch import (
    set_single_level_autograd_function_allowed,
    get_single_level_autograd_function_allowed,
    unwrap_if_dead,
)

@contextlib.contextmanager
def enable_single_level_autograd_function():
    try:
        prev_state = get_single_level_autograd_function_allowed()
        set_single_level_autograd_function_allowed(True)
        yield
    finally:
        set_single_level_autograd_function_allowed(prev_state)

def unwrap_dead_wrappers(args):
    # NB: doesn't use tree_map_only for performance reasons
    result = tuple(
        unwrap_if_dead(arg) if isinstance(arg, torch.Tensor) else arg
        for arg in args
    )
    return result

# Allows one to expose an API in a private submodule publicly as per the definition
# in PyTorch's public api policy.
#
# It is a temporary solution while we figure out if it should be the long-term solution
# or if we should amend PyTorch's public api policy. The concern is that this approach
# may not be very robust because it's not clear what __module__ is used for.
# However, both numpy and jax overwrite the __module__ attribute of their APIs
# without problem, so it seems fine.
def exposed_in(module):
    def wrapper(fn):
        fn.__module__ = module
        return fn
    return wrapper
