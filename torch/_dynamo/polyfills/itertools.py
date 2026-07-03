"""
Python polyfills for itertools
"""

from __future__ import annotations

import itertools
import operator
import sys
from collections.abc import Callable
from typing import overload, TYPE_CHECKING, TypeAlias, TypeVar

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


__all__ = [
    "accumulate",
    "combinations",
    "combinations_with_replacement",
    "compress",
    "cycle",
    "dropwhile",
    "filterfalse",
    "islice",
    "pairwise",
    "permutations",
    "starmap",
    "takewhile",
    "tee",
]

if sys.version_info >= (3, 12):
    __all__.append("batched")


_T = TypeVar("_T")
_U = TypeVar("_U")
_Predicate: TypeAlias = Callable[[_T], object]
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


# Reference: https://docs.python.org/3/library/itertools.html#itertools.accumulate
@substitute_in_graph(itertools.accumulate, is_embedded_type=True)  # type: ignore[arg-type]
def accumulate(
    iterable: Iterable[_T],
    func: Callable[[_T, _T], _T] | None = None,
    *,
    initial: _T | None = None,
) -> Iterator[_T]:
    # call iter outside of the generator to match cpython behavior
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


# Reference: https://docs.python.org/3/library/itertools.html#itertools.combinations
@substitute_in_graph(itertools.combinations, is_embedded_type=True)  # type: ignore[arg-type]
def combinations(iterable: Iterable[_T], r: int, /) -> Iterator[tuple[_T, ...]]:
    pool = tuple(iterable)
    n = len(pool)

    if r < 0:
        raise ValueError("r must be non-negative")

    def _combinations() -> Iterator[tuple[_T, ...]]:
        if r > n:
            return

        indices = list(range(r))
        yield tuple(pool[i] for i in indices)

        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return

            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1

            yield tuple(pool[i] for i in indices)

    return _combinations()


# Reference: https://docs.python.org/3/library/itertools.html#itertools.compress
@substitute_in_graph(itertools.compress, is_embedded_type=True)  # type: ignore[arg-type]
def compress(data: Iterable[_T], selectors: Iterable[_U], /) -> Iterator[_T]:
    return (datum for datum, selector in zip(data, selectors) if selector)


# Reference: https://docs.python.org/3/library/itertools.html#itertools.cycle
@substitute_in_graph(itertools.cycle, is_embedded_type=True)  # type: ignore[arg-type]
def cycle(iterable: Iterable[_T]) -> Iterator[_T]:
    iterator = iter(iterable)

    def _cycle(iterator: Iterator[_T]) -> Iterator[_T]:
        # pyrefly: ignore [implicit-any]
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
    if not callable(predicate):
        raise TypeError(f"'{type(predicate).__name__}' object is not callable")

    iterator = iter(iterable)
    for x in iterator:
        if not predicate(x):
            yield x
            break

    yield from iterator


# Reference: https://docs.python.org/3/library/itertools.html#itertools.takewhile
@substitute_in_graph(itertools.takewhile, is_embedded_type=True)  # type: ignore[arg-type]
def takewhile(predicate: _Predicate[_T], iterable: Iterable[_T], /) -> Iterator[_T]:
    # takewhile(lambda x: x<5, [1,4,6,3,8]) → 1 4
    if not callable(predicate):
        raise TypeError(f"'{type(predicate).__name__}' object is not callable")

    for x in iterable:
        if not predicate(x):
            break
        yield x


@overload
def starmap(
    function: Callable[[], _U],
    iterable: Iterable[tuple[()]],
    /,
) -> itertools.starmap[_U]: ...


@overload
def starmap(
    function: Callable[[_T], _U],
    iterable: Iterable[tuple[_T]],
    /,
) -> itertools.starmap[_U]: ...


@overload
def starmap(
    function: Callable[[_T, _T1], _U],
    iterable: Iterable[tuple[_T, _T1]],
    /,
) -> itertools.starmap[_U]: ...


@overload
def starmap(
    function: Callable[[_T, _T1, _T2], _U],
    iterable: Iterable[tuple[_T, _T1, _T2]],
    /,
) -> itertools.starmap[_U]: ...


# Reference: https://docs.python.org/3/library/itertools.html#itertools.starmap
@substitute_in_graph(itertools.starmap, is_embedded_type=True)  # type: ignore[arg-type]
# pyrefly: ignore [implicit-any]
def starmap(function: Callable[..., _T], iterable: Iterable, /) -> Iterable[_T]:
    # starmap(pow, [(2,5), (3,2), (10,3)]) → 32 9 1000
    if not callable(function):
        raise TypeError(f"'{type(function).__name__}' object is not callable")

    for args in iterable:
        yield function(*args)


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


# Reference: https://docs.python.org/3/library/itertools.html#itertools.permutations
@substitute_in_graph(itertools.permutations, is_embedded_type=True)  # type: ignore[arg-type]
def permutations(
    iterable: Iterable[_T], r: int | None = None, /
) -> Iterator[tuple[_T, ...]]:
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r

    if r < 0:
        raise ValueError("r must be non-negative")

    def _permutations() -> Iterator[tuple[_T, ...]]:
        if r > n:
            return

        indices = list(range(n))
        cycles = list(range(n, n - r, -1))
        yield tuple(pool[i] for i in indices[:r])

        while n:
            for i in reversed(range(r)):
                cycles[i] -= 1
                if cycles[i] == 0:
                    indices[i:] = indices[i + 1 :] + indices[i : i + 1]
                    cycles[i] = n - i
                else:
                    j = cycles[i]
                    indices[i], indices[-j] = indices[-j], indices[i]
                    yield tuple(pool[i] for i in indices[:r])
                    break
            else:
                return

    return _permutations()


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


# Reference: https://docs.python.org/3/library/itertools.html#itertools.combinations_with_replacement
@substitute_in_graph(itertools.combinations_with_replacement, is_embedded_type=True)  # type: ignore[arg-type]
def combinations_with_replacement(
    iterable: Iterable[_T], r: int, /
) -> Iterator[tuple[_T, ...]]:
    if r < 0:
        raise ValueError("r must be non-negative")

    pool = tuple(iterable)
    n = len(pool)

    def _combinations_with_replacement() -> Iterator[tuple[_T, ...]]:
        if r == 0:
            yield ()
            return
        if n == 0:
            return

        indices = [0] * r
        yield tuple(pool[i] for i in indices)
        while True:
            for i in range(r - 1, -1, -1):
                if indices[i] != n - 1:
                    break
            else:
                return
            indices[i:] = [indices[i] + 1] * (r - i)
            yield tuple(pool[i] for i in indices)

    return _combinations_with_replacement()


if sys.version_info >= (3, 12):
    # Reference: https://docs.python.org/3/library/itertools.html#itertools.batched
    @substitute_in_graph(itertools.batched, is_embedded_type=True)  # type: ignore[arg-type]
    def batched(*args, **kwargs) -> Iterator[tuple[_T, ...]]:  # type: ignore[no-untyped-def]
        if len(args) != 2:
            raise TypeError(
                f"batched takes exactly 2 positional arguments({len(args)} given)"
            )
        if kwargs.keys() - {"strict"}:
            unexpected = next(iter(kwargs.keys() - {"strict"}))
            raise TypeError(
                f"batched() got an unexpected keyword argument '{unexpected}'"
            )

        iterable, n = args
        strict = kwargs.get("strict", False)
        n = operator.index(n)
        if n < 1:
            raise ValueError("n must be at least one")

        iterator = iter(iterable)

        def _batched(iterator: Iterator[_T]) -> Iterator[tuple[_T, ...]]:
            while batch := tuple(islice(iterator, n)):
                if strict and len(batch) != n:
                    raise ValueError("batched(): incomplete batch")
                yield batch

        return _batched(iterator)
