from __future__ import annotations

from typing import cast, Callable, Generic, Type, TypeVar

import torch

__all__ = ['Await']

W = TypeVar("W")

class _PyAwaitMeta(type(torch._C.Await), type(Generic)):
    pass

class Await(torch._C.Await, Generic[W], metaclass=_PyAwaitMeta):
    #def __init__(self, func:Callable[[], W]):
    #    super().__init__(func)
    def __init__(self, f):
        super().__init__()

    def wait(self) -> W:
        return super().wait()
