"""
Python polyfills for builtins
"""

from __future__ import annotations

import builtins
import functools
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "all",
    "any",
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


class _CallableIterator:
    def __init__(self, fn, sentinel):  # type: ignore[no-untyped-def]
        self.fn = fn
        self.sentinel = sentinel

    def __iter__(self):  # type: ignore[no-untyped-def]
        return self

    def __next__(self):  # type: ignore[no-untyped-def]
        # The iterator created in this case will call object with no arguments
        # for each call to its __next__() method;
        r = self.fn()

        # If the value returned is equal to sentinel, StopIteration will be raised
        if r == self.sentinel:
            raise StopIteration

        # otherwise the value will be returned.
        return r


_sentinel_missing = object()


# TODO(guilhermeleobas): use substitute_in_graph for iter()
def iter_(fn_or_iterable, sentinel=_sentinel_missing, /):  # type: ignore[no-untyped-def]
    # Without a second argument, object must be a collection object which supports
    # the iterable (__iter__) or the sequence protocol (__getitem__ with an integer
    # starting at 0)
    if sentinel is _sentinel_missing:
        iterable = fn_or_iterable
        if hasattr(iterable, "__iter__"):
            iterator = iterable.__iter__()
            if hasattr(iterator, "__next__"):
                return iterator
            else:
                raise TypeError(f"'{type(iterator)}' object is not iterable")
        if hasattr(iterable, "__getitem__"):
            # Needs to be a new function to avoid iter becoming a generator
            def sequence_protocol(iterable):  # type: ignore[no-untyped-def]
                i = 0
                while True:
                    try:
                        yield iterable.__getitem__(i)
                        i += 1
                    except IndexError:
                        break

            return sequence_protocol(iterable)
        raise TypeError(f"'{type(iterable)}' object is not iterable")
    else:
        # If the second argument, sentinel, is given, then object must be a
        # callable object.
        fn = fn_or_iterable

        if not isinstance(fn, Callable):  # type: ignore[arg-type]
            raise TypeError("iter(v, w): v must be a callable")

        return _CallableIterator(fn, sentinel)
