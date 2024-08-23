"""
Python polyfills for functools
"""

from __future__ import annotations

import functools
from typing import Callable, Iterable, TYPE_CHECKING, TypeVar

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from _typeshed import SupportsAllComparisons


__all__ = ["reduce"]


_T = TypeVar("_T")
_U = TypeVar("_U")


class _INITIAL_MISSING:
    pass


# Reference: https://docs.python.org/3/library/functools.html#functools.reduce
@substitute_in_graph(functools.reduce)
def reduce(
    function: Callable[[_U, _T], _U],
    iterable: Iterable[_T],
    initial: _U = _INITIAL_MISSING,  # type: ignore[assignment]
    /,
) -> _U:
    it = iter(iterable)

    value: _U
    if initial is _INITIAL_MISSING:
        try:
            value = next(it)  # type: ignore[assignment]
        except StopIteration:
            raise TypeError(
                "reduce() of empty iterable with no initial value",
            ) from None
    else:
        value = initial

    for element in it:
        value = function(value, element)

    return value


# Copied from functools.py in the standard library
@substitute_in_graph(functools.cmp_to_key)
def cmp_to_key(
    mycmp: Callable[[_T, _T], int],
) -> Callable[[_T], SupportsAllComparisons]:
    class K:
        __slots__ = ("obj",)

        def __init__(self, obj: _T) -> None:
            self.obj = obj

        def __lt__(self, other: K) -> bool:
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other: K) -> bool:
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other: K) -> bool:  # type: ignore[override]
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other: K) -> bool:
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other: K) -> bool:
            return mycmp(self.obj, other.obj) >= 0

        __hash__ = None  # type: ignore[assignment]

    return K
