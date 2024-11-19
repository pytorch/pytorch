# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Registry for aten functions."""

from __future__ import annotations


__all__ = ["registry", "onnx_impl"]

from typing import Callable, TypeVar


_T = TypeVar("_T", bound=Callable)


class Registry:
    """Registry for aten functions."""

    def __init__(self):
        self._registry: dict[Callable, list[Callable]] = {}

    def register(self, target: Callable, impl: Callable) -> None:
        """Register a function."""

        self._registry.setdefault(target, []).append(impl)

    def __getitem__(self, name):
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

    def __iter__(self):
        return iter(self._registry)

    def __repr__(self):
        return repr(self._registry)

    def items(self):
        yield from self._registry.items()

    def values(self):
        yield from self._registry.values()


# Default registry
registry = Registry()


def onnx_impl(
    target: Callable,
) -> Callable[[_T], _T]:
    """Register an ONNX implementation of a torch op."""

    def wrapper(
        func: _T,
    ) -> _T:
        registry.register(target, func)
        return func

    return wrapper
