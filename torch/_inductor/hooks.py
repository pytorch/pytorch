import contextlib
from collections.abc import Callable
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import torch

# Executed in the order they're registered
INTERMEDIATE_HOOKS: list[Callable[[str, "torch.Tensor"], None]] = []


@contextlib.contextmanager
def intermediate_hook(fn: Any) -> Any:
    INTERMEDIATE_HOOKS.append(fn)
    try:
        yield
    finally:
        INTERMEDIATE_HOOKS.pop()


def run_intermediate_hooks(name: Any, val: Any) -> None:
    global INTERMEDIATE_HOOKS
    hooks = INTERMEDIATE_HOOKS
    INTERMEDIATE_HOOKS = []
    try:
        for hook in hooks:
            hook(name, val)
    finally:
        INTERMEDIATE_HOOKS = hooks
