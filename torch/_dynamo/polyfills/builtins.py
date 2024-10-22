"""
Python polyfills for builtins
"""

from __future__ import annotations

import builtins
import functools
import operator
from typing import Callable, Iterable, TypeVar

from ..decorators import substitute_in_graph


__all__ = [
    "all",
    "any",
    "enumerate",
    # "filter",
    "sum",
]


_T = TypeVar("_T")


@substitute_in_graph(builtins.all, can_constant_fold_through=True)
def all(iterable: Iterable[object], /) -> bool:
    for elem in iterable:
        if not elem:
            return False
    return True


@substitute_in_graph(builtins.any, can_constant_fold_through=True)
def any(iterable: Iterable[object], /) -> bool:
    for elem in iterable:
        if elem:
            return True
    return False


@substitute_in_graph(builtins.enumerate, is_embedded_type=True)  # type: ignore[arg-type]
def enumerate(iterable: Iterable[_T], start: int = 0) -> Iterable[tuple[int, _T]]:
    if not isinstance(start, int):
        raise TypeError(
            f"{type(start).__name__!r} object cannot be interpreted as an integer"
        )

    for x in iterable:
        yield start, x
        start += 1


@substitute_in_graph(builtins.sum, can_constant_fold_through=True)  # type: ignore[arg-type]
def sum(iterable: Iterable[_T], /, start: _T = 0) -> _T:  # type: ignore[assignment]
    return functools.reduce(operator.add, iterable, start)


# TODO: infinite iterators
# Reference: https://docs.python.org/3/library/functions.html#filter
# @substitute_in_graph(builtins.filter, is_embedded_type=True)  # type: ignore[arg-type]
def filter(function: Callable[[_T], bool], iterable: Iterable[_T]) -> _T:  # type: ignore[assignment]
    if function is None:
        function = bool

    for x in iterable:
        if function(x):
            yield x
