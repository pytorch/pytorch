from __future__ import annotations

from typing import cast, Callable, Generic, Type, TypeVar

import torch

__all__ = ['Await']

W = TypeVar("W")

class _PyAwaitMeta(type(torch._C.Await), type(Generic)):
    pass

class Await(torch._C.Await, Generic[W], metaclass=_PyAwaitMeta):
    def __init__(self, f):
        super().__init__()
