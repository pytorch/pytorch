"""
Python polyfills for builtins
"""

import builtins
import functools
import operator
from typing import Iterable, TypeVar

from ..decorators import substitute_in_graph


__all__ = [
    "all",
    "any",
    "sum",
]


_T = TypeVar("_T")


@substitute_in_graph(builtins.all)
def all(iterable: Iterable[object], /) -> bool:
    for elem in iterable:
        if not elem:
            return False
    return True


@substitute_in_graph(builtins.any)
def any(iterable: Iterable[object], /) -> bool:
    for elem in iterable:
        if elem:
            return True
    return False


@substitute_in_graph(builtins.sum)  # type: ignore[arg-type]
def sum(iterable: Iterable[_T], /, start: _T = 0) -> _T:  # type: ignore[assignment]
    return functools.reduce(operator.add, iterable, start)
