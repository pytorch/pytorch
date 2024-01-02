# This module contains functions that *will be allowed* by dynamo

import functools

import torch
import torch.utils._pytree as pytree

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


def is_compiling() -> bool:
    return False


def wrap_inline(fn):
    """
    Create an extra frame around fn that is not in skipfiles
    """

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner


def call_hook(hook, *args):
    """
    Used by compiled autograd to handle hook returning None
    """
    result = hook(*args)
    if result is None:
        return args[0]
    return result


def wrap_numpy(f):
    r"""Decorator that turns a function from ``np.ndarray``s to ``np.ndarray``s into a function
    from ``torch.Tensor``s to ``torch.Tensor``s.
    """
    if not np:
        return f

    @functools.wraps(f)
    def wrap(*args, **kwargs):
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, lambda x: x.numpy(), (args, kwargs)
        )
        out = f(*args, **kwargs)
        return pytree.tree_map_only(np.ndarray, lambda x: torch.as_tensor(x), out)

    return wrap
