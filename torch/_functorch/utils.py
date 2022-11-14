import contextlib
from torch._C._functorch import (
    set_autograd_function_allowed,
    get_autograd_function_allowed,
)

@contextlib.contextmanager
def enable_autograd_function():
    try:
        prev_state = get_autograd_function_allowed()
        set_autograd_function_allowed(True)
        yield
    finally:
        set_autograd_function_allowed(prev_state)
