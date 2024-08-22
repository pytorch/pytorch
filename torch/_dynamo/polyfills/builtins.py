"""
Python polyfills for builtins
"""

from __future__ import annotations

import builtins
import functools
import operator
from typing import Iterable, TypeVar

from ..decorators import substitute_in_graph


__all__ = [
    "all",
    "any",
    "enumerate___new__",
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


@substitute_in_graph(builtins.enumerate.__new__)  # type: ignore[arg-type]
def enumerate___new__(
    cls: type[builtins.enumerate[_T]],
    iterable: Iterable[_T],
    start: int = 0,
) -> Iterable[tuple[int, _T]]:
    assert cls is builtins.enumerate

    if not isinstance(start, int):
        raise TypeError(
            f"{type(start).__name__!r} object cannot be interpreted as an integer"
        )

    for x in iterable:
        yield start, x
        start += 1


@substitute_in_graph(builtins.sum)  # type: ignore[arg-type]
def sum(iterable: Iterable[_T], /, start: _T = 0) -> _T:  # type: ignore[assignment]
    return functools.reduce(operator.add, iterable, start)
