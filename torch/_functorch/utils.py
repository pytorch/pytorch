import contextlib
from collections.abc import Generator

from torch._C._functorch import (
    get_single_level_autograd_function_allowed,
    set_single_level_autograd_function_allowed,
    unwrap_dead_wrappers,
)
from torch.utils._exposed_in import exposed_in


__all__ = [
    "exposed_in",
    "argnums_t",
    "enable_single_level_autograd_function",
    "unwrap_dead_wrappers",
]


@contextlib.contextmanager
def enable_single_level_autograd_function() -> Generator[None, None, None]:
    try:
        prev_state = get_single_level_autograd_function_allowed()
        set_single_level_autograd_function_allowed(True)
        yield
    finally:
        set_single_level_autograd_function_allowed(prev_state)


argnums_t = int | tuple[int, ...]
