# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Registry for aten functions."""

from __future__ import annotations


__all__ = ["registry", "onnx_impl"]

import collections
from typing import Callable, TypeVar


_T = TypeVar("_T", bound=Callable)


class Registry(collections.UserDict[Callable, list[Callable]]):
    """Registry for aten functions."""

    def register(self, target: Callable, impl: Callable) -> None:
        """Register a function."""

        self.data.setdefault(target, []).append(impl)


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
