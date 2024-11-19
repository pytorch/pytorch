# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Registry for aten functions."""

from __future__ import annotations

from typing import Callable, TypeVar


_T = TypeVar("_T", bound=Callable)


class Registry:
    """Registry for aten functions."""

    def __init__(self):
        self._registry: dict[Callable, list[Callable]] = {}

    def register(self, target: Callable, impl: Callable) -> None:
        """Register a function."""

        self._registry.setdefault(target, []).append(impl)


# Default registry
default_registry = Registry()


def onnx_impl(
    target: Callable,
) -> Callable[[_T], _T]:
    """Register an ONNX implementation of a torch op."""

    def wrapper(
        func: _T,
    ) -> _T:
        default_registry.register(target, func)
        return func

    return wrapper
