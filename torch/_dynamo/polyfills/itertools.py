"""
Python polyfills for itertools
"""

from __future__ import annotations

import itertools
from typing import Iterable, Iterator, TypeVar

from ..decorators import substitute_in_graph
from ..variables.builder import ITERTOOLS_POLYFILLED_CLASSES


__all__ = [
    "chain___new__",
    "chain_from_iterable",
    "count___new__",
    "tee",
]


_T = TypeVar("_T")


# Reference: https://docs.python.org/3/library/itertools.html#itertools.chain
@substitute_in_graph(itertools.chain.__new__)  # type: ignore[arg-type]
def chain___new__(
    cls: type[itertools.chain[_T]],
    *iterables: Iterable[_T],
) -> Iterator[_T]:
    assert cls is itertools.chain

    for iterable in iterables:
        yield from iterable


@substitute_in_graph(itertools.chain.from_iterable)  # type: ignore[arg-type]
def chain_from_iterable(iterable: Iterable[Iterable[_T]], /) -> Iterator[_T]:
    return itertools.chain(*iterable)


ITERTOOLS_POLYFILLED_CLASSES.add(itertools.chain)


# Reference: https://docs.python.org/3/library/itertools.html#itertools.count
@substitute_in_graph(itertools.count.__new__)  # type: ignore[arg-type]
def count___new__(
    cls: type[itertools.count[_T]],  # type: ignore[type-var]
    start: _T = 0,  # type: ignore[assignment]
    step: _T = 1,  # type: ignore[assignment]
) -> Iterator[_T]:
    assert cls is itertools.count

    n = start
    while True:
        yield n
        n += step  # type: ignore[operator]


ITERTOOLS_POLYFILLED_CLASSES.add(itertools.count)


# Reference: https://docs.python.org/3/library/itertools.html#itertools.tee
@substitute_in_graph(itertools.tee)
def tee(iterable: Iterable[_T], n: int = 2, /) -> tuple[Iterator[_T], ...]:
    iterator = iter(iterable)
    shared_link = [None, None]

    def _tee(link) -> Iterator[_T]:  # type: ignore[no-untyped-def]
        try:
            while True:
                if link[1] is None:
                    link[0] = next(iterator)
                    link[1] = [None, None]
                value, link = link
                yield value
        except StopIteration:
            return

    return tuple(_tee(shared_link) for _ in range(n))
