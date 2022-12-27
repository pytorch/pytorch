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

    def wait(self) -> W:
        return super().wait()

    # TODO: Questionable implicit type convertion
    # Will be enough to have in jit implicit conversion Await[W] -> W adding aten::awaitable wait to the graph
    @staticmethod
    def _wait(obj):
        if isinstance(obj, Await):
            return obj.wait()
        return obj

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        new_args = torch.fx.node.map_aggregate(args, Await._wait)
        new_kwargs = torch.fx.node.map_aggregate(kwargs, Await._wait)
        return func(*new_args, **new_kwargs)

    def __getattr__(self, name):
        res = self.wait()
        return getattr(res, name)
