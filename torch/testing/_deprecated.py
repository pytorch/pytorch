"""This module exists since the `torch.testing` exposed a lot of stuff that shouldn't have been public. Although this
was never documented anywhere, some other internal FB projects as well as downstream OSS projects might use this. Thus,
we don't internalize without warning, but still go through a deprecation cycle.
"""

import functools
import warnings
from typing import Any, Callable

import torch


__all__ = ["rand", "randn"]


def warn_deprecated(instructions: str) -> Callable:
    def outer_wrapper(fn: Callable) -> Callable:
        msg = f"torch.testing.{fn.__name__} is deprecated and will be removed in the future. {instructions.strip()}"

        @functools.wraps(fn)
        def inner_wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, FutureWarning)
            return fn(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


rand = warn_deprecated("Use torch.rand instead.")(torch.rand)
randn = warn_deprecated("Use torch.randn instead.")(torch.randn)
