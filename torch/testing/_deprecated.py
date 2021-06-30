"""This module exists since the `torch.testing` exposed a lot of stuff that shouldn't have been public. Although this
was never documented anywhere, some other internal FB projects as well as downstream OSS projects might use this. Thus,
we don't internalize without warning, but still go through a deprecation cycle.
"""

import functools
import warnings
from typing import Any, Callable, Optional

import torch


__all__ = ["rand", "randn"]


def warn_deprecated(fn: Callable, instructions: str, name: Optional[str] = None) -> Callable:
    if name is None:
        name = fn.__name__

    msg = f"torch.testing.{name} is deprecated and will be removed in the future. {instructions.strip()}"

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(msg, FutureWarning)
        return fn(*args, **kwargs)

    return wrapper


rand = warn_deprecated(torch.rand, "Use torch.rand instead.")
randn = warn_deprecated(torch.randn, "Use torch.randn instead.")
