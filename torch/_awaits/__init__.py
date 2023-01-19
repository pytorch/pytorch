from __future__ import annotations

from typing import cast, Callable, Generic, Type, TypeVar

import torch

__all__ = ['Await']

W = TypeVar("W")

class _PyAwaitMeta(type(torch._C._Await), type(Generic)):  # type: ignore[misc, no-redef]
    pass

class _Await(torch._C._Await, Generic[W], metaclass=_PyAwaitMeta):
    r"""
    Wrapper around a ``torch._C.Await`` which encapsulates delayed execution
    of a callable.
    """
    pass
