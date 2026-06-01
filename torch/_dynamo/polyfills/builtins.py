"""
Python polyfills for builtins
"""

from __future__ import annotations

import builtins
import functools
import operator
import typing
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "all",
    "any",
    "cast",
    "enumerate",
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


# TODO(guilhermeleobas): Implement this iterator as a VariableTracker to see if
# it is faster than tracing through it
class _CallableIterator:
    def __init__(self, fn, sentinel):  # type: ignore[no-untyped-def]
        self.fn = fn
        self.sentinel = sentinel
        self.exhausted = False

    def __iter__(self):  # type: ignore[no-untyped-def]
        return self

    def __next__(self):  # type: ignore[no-untyped-def]
        if self.exhausted:
            raise StopIteration

        # The iterator created in this case will call object with no arguments
        # for each call to its __next__() method;
        r = self.fn()

        # If the value returned is equal to sentinel, StopIteration will be raised
        if r == self.sentinel:
            self.exhausted = True
            raise StopIteration

        # otherwise the value will be returned.
        return r


class _SequenceIterator:
    def __init__(self, iterable) -> None:
        self.iterable = iterable
        self.index = 0
        self.exhausted = False

    def __iter__(self) -> _SequenceIterator:
        return self

    def __next__(self) -> object:
        if self.exhausted:
            raise StopIteration

        try:
            result = self.iterable.__getitem__(self.index)
            self.index += 1
            return result
        except (IndexError, StopIteration):
            self.exhausted = True
            raise StopIteration from None


def sequence_iterator(iterable) -> Iterable[object]:
    if hasattr(iterable, "__getitem__"):
        return _SequenceIterator(iterable)
    raise TypeError(f"'{type(iterable)}' object is not iterable")


def callable_iterator(fn, sentinel, /):
    # If the second argument, sentinel, is given, then object must be a
    # callable object.
    if not isinstance(fn, Callable):  # type: ignore[arg-type]
        raise TypeError("iter(v, w): v must be a callable")

    return _CallableIterator(fn, sentinel)


@substitute_in_graph(typing.cast, can_constant_fold_through=True)
def cast(typ: type, val: _T) -> _T:  # type: ignore[type-var]
    return val
