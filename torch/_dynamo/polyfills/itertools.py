"""
Python polyfills for itertools
"""

from __future__ import annotations

import itertools
import operator
from typing import Callable, Optional, overload, TYPE_CHECKING, TypeVar
from typing_extensions import TypeAlias

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


__all__ = [
    "accumulate",
    "chain",
    "chain_from_iterable",
    "compress",
    "cycle",
    "dropwhile",
    "filterfalse",
    "islice",
    "tee",
    "zip_longest",
    "pairwise",
]


_T = TypeVar("_T")
_U = TypeVar("_U")
_Predicate: TypeAlias = Callable[[_T], object]
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


# Reference: https://docs.python.org/3/library/itertools.html#itertools.chain
@substitute_in_graph(itertools.chain, is_embedded_type=True)  # type: ignore[arg-type]
def chain(*iterables: Iterable[_T]) -> Iterator[_T]:
    for iterable in iterables:
        yield from iterable


# Reference: https://docs.python.org/3/library/itertools.html#itertools.accumulate
@substitute_in_graph(itertools.accumulate, is_embedded_type=True)  # type: ignore[arg-type]
def accumulate(
    iterable: Iterable[_T],
    func: Optional[Callable[[_T, _T], _T]] = None,
    *,
    initial: Optional[_T] = None,
) -> Iterator[_T]:
    # call iter outside of the generator to match cypthon behavior
    iterator = iter(iterable)
    if func is None:
        func = operator.add

    def _accumulate(iterator: Iterator[_T]) -> Iterator[_T]:
        total = initial
        if total is None:
            try:
                total = next(iterator)
            except StopIteration:
                return

        yield total
        for element in iterator:
            total = func(total, element)
            yield total

    return _accumulate(iterator)


@substitute_in_graph(itertools.chain.from_iterable)  # type: ignore[arg-type]
def chain_from_iterable(iterable: Iterable[Iterable[_T]], /) -> Iterator[_T]:
    # previous version of this code was:
    #   return itertools.chain(*iterable)
    # If iterable is an infinite generator, this will lead to infinite recursion
    for it in iterable:
        yield from it


chain.from_iterable = chain_from_iterable  # type: ignore[attr-defined]


# Reference: https://docs.python.org/3/library/itertools.html#itertools.compress
@substitute_in_graph(itertools.compress, is_embedded_type=True)  # type: ignore[arg-type]
def compress(data: Iterable[_T], selectors: Iterable[_U], /) -> Iterator[_T]:
    return (datum for datum, selector in zip(data, selectors) if selector)


# Reference: https://docs.python.org/3/library/itertools.html#itertools.cycle
@substitute_in_graph(itertools.cycle, is_embedded_type=True)  # type: ignore[arg-type]
def cycle(iterable: Iterable[_T]) -> Iterator[_T]:
    iterator = iter(iterable)

    def _cycle(iterator: Iterator[_T]) -> Iterator[_T]:
        saved = []
        for element in iterable:
            yield element
            saved.append(element)

        while saved:
            for element in saved:
                yield element

    return _cycle(iterator)


# Reference: https://docs.python.org/3/library/itertools.html#itertools.dropwhile
@substitute_in_graph(itertools.dropwhile, is_embedded_type=True)  # type: ignore[arg-type]
def dropwhile(predicate: _Predicate[_T], iterable: Iterable[_T], /) -> Iterator[_T]:
    # dropwhile(lambda x: x < 5, [1, 4, 6, 3, 8]) -> 6 3 8

    iterator = iter(iterable)
    for x in iterator:
        if not predicate(x):
            yield x
            break

    yield from iterator


@substitute_in_graph(itertools.filterfalse, is_embedded_type=True)  # type: ignore[arg-type]
def filterfalse(function: _Predicate[_T], iterable: Iterable[_T], /) -> Iterator[_T]:
    it = iter(iterable)
    if function is None:
        return filter(operator.not_, it)
    else:
        return filter(lambda x: not function(x), it)


# Reference: https://docs.python.org/3/library/itertools.html#itertools.islice
@substitute_in_graph(itertools.islice, is_embedded_type=True)  # type: ignore[arg-type]
def islice(iterable: Iterable[_T], /, *args: int | None) -> Iterator[_T]:
    s = slice(*args)
    start = 0 if s.start is None else s.start
    stop = s.stop
    step = 1 if s.step is None else s.step
    if start < 0 or (stop is not None and stop < 0) or step <= 0:
        raise ValueError(
            "Indices for islice() must be None or an integer: 0 <= x <= sys.maxsize.",
        )

    if stop is None:
        # TODO: use indices = itertools.count() and merge implementation with the else branch
        #       when we support infinite iterators
        next_i = start
        for i, element in enumerate(iterable):
            if i == next_i:
                yield element
                next_i += step
    else:
        indices = range(max(start, stop))
        next_i = start
        for i, element in zip(indices, iterable):
            if i == next_i:
                yield element
                next_i += step


# Reference: https://docs.python.org/3/library/itertools.html#itertools.pairwise
@substitute_in_graph(itertools.pairwise, is_embedded_type=True)  # type: ignore[arg-type]
def pairwise(iterable: Iterable[_T], /) -> Iterator[tuple[_T, _T]]:
    a = None
    first = True
    for b in iterable:
        if first:
            first = False
        else:
            yield a, b  # type: ignore[misc]
        a = b


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


@overload
# pyrefly: ignore  # inconsistent-overload
def zip_longest(
    iter1: Iterable[_T1],
    /,
    *,
    fillvalue: _U = ...,
) -> Iterator[tuple[_T1]]: ...


@overload
# pyrefly: ignore  # inconsistent-overload
def zip_longest(
    iter1: Iterable[_T1],
    iter2: Iterable[_T2],
    /,
) -> Iterator[tuple[_T1 | None, _T2 | None]]: ...


@overload
# pyrefly: ignore  # inconsistent-overload
def zip_longest(
    iter1: Iterable[_T1],
    iter2: Iterable[_T2],
    /,
    *,
    fillvalue: _U = ...,
) -> Iterator[tuple[_T1 | _U, _T2 | _U]]: ...


@overload
# pyrefly: ignore  # inconsistent-overload
def zip_longest(
    iter1: Iterable[_T],
    iter2: Iterable[_T],
    iter3: Iterable[_T],
    /,
    *iterables: Iterable[_T],
) -> Iterator[tuple[_T | None, ...]]: ...


@overload
# pyrefly: ignore  # inconsistent-overload
def zip_longest(
    iter1: Iterable[_T],
    iter2: Iterable[_T],
    iter3: Iterable[_T],
    /,
    *iterables: Iterable[_T],
    fillvalue: _U = ...,
) -> Iterator[tuple[_T | _U, ...]]: ...


# Reference: https://docs.python.org/3/library/itertools.html#itertools.zip_longest
@substitute_in_graph(itertools.zip_longest, is_embedded_type=True)  # type: ignore[arg-type,misc]
def zip_longest(
    *iterables: Iterable[_T],
    fillvalue: _U = None,  # type: ignore[assignment]
) -> Iterator[tuple[_T | _U, ...]]:
    # zip_longest('ABCD', 'xy', fillvalue='-') -> Ax By C- D-

    iterators = list(map(iter, iterables))
    num_active = len(iterators)
    if not num_active:
        return

    while True:
        values = []
        for i, iterator in enumerate(iterators):
            try:
                value = next(iterator)
            except StopIteration:
                num_active -= 1
                if not num_active:
                    return
                iterators[i] = itertools.repeat(fillvalue)  # type: ignore[arg-type]
                value = fillvalue  # type: ignore[assignment]
            values.append(value)
        yield tuple(values)
