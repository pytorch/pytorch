"""
Python polyfills for builtins
"""

from __future__ import annotations

import builtins
import functools
import operator
import sys
from typing import Iterable, TypeVar

from ..decorators import substitute_in_graph


__all__ = [
    "all",
    "any",
    "enumerate___new__",
    "sum",
    "zip___new__",
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


if sys.version_info >= (3, 10):

    @substitute_in_graph(builtins.zip.__new__)  # type: ignore[arg-type]
    def zip___new__(
        cls: type[builtins.zip[tuple[_T, ...]]],
        *iterables: Iterable[_T],
        strict: bool = False,
    ) -> Iterable[tuple[_T, ...]]:
        assert cls is builtins.zip

        if not iterables:
            return
        if len(iterables) == 1:
            for elem in iterables[0]:
                yield (elem,)
            return

        iterators = [iter(it) for it in iterables]
        while True:
            items = []
            for i, it in enumerate(iterators):
                try:
                    items.append(next(it))
                except StopIteration:
                    if strict:
                        if i > 0:
                            raise ValueError(
                                f"zip() argument {i + 1} is longer than "
                                f"argument{'s 1-' if i > 1 else ' '}{i}",
                            ) from None

                        for j in range(1, len(iterators)):
                            try:
                                next(iterators[j])
                            except StopIteration:
                                pass
                            else:
                                raise ValueError(
                                    f"zip() argument {j + 1} is shorter than "
                                    f"argument{'s 1-' if j > 1 else ' '}{j}",
                                ) from None
                    return

            yield tuple(items)

else:

    @substitute_in_graph(builtins.zip.__new__)  # type: ignore[arg-type]
    def zip___new__(
        cls: type[builtins.zip[tuple[_T, ...]]],
        *iterables: Iterable[_T],
    ) -> Iterable[tuple[_T, ...]]:
        assert cls is builtins.zip

        if not iterables:
            return
        if len(iterables) == 1:
            for elem in iterables[0]:
                yield (elem,)
            return

        iterators = [iter(it) for it in iterables]
        while True:
            items = []
            for it in iterators:
                try:
                    items.append(next(it))
                except StopIteration:
                    return

            yield tuple(items)
