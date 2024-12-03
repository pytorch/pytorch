"""
Python polyfills for functools
"""

import functools
from typing import Callable, Iterable, TypeVar

from ..decorators import substitute_in_graph


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
