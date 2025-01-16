# mypy: allow-untyped-defs
import contextlib
from typing import Tuple, Union

import torch
from torch._C._functorch import (
    get_single_level_autograd_function_allowed,
    set_single_level_autograd_function_allowed,
    unwrap_if_dead,
)
from torch.utils._exposed_in import exposed_in


__all__ = [
    "exposed_in",
    "argnums_t",
    "enable_single_level_autograd_function",
    "unwrap_dead_wrappers",
]


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
        unwrap_if_dead(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
    )
    return result


argnums_t = Union[int, Tuple[int, ...]]


def user_fn_call_stub():
    return torch.tensor([])  # pass  # assert False, "Not supported yet"
