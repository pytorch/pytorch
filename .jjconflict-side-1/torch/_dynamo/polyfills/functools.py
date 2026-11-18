"""
Python polyfills for functools
"""

import functools
from collections.abc import Callable, Iterable
from functools import _initial_missing  # type: ignore[attr-defined]
from typing import Any, TypeVar

from ..decorators import substitute_in_graph


__all__ = ["cmp_to_key", "reduce"]


_T = TypeVar("_T")
_U = TypeVar("_U")


class _KeyWrapper:
    def __init__(self, mycmp: Callable[[Any, Any], int], obj: Any) -> None:
        self.mycmp = mycmp
        self.obj = obj

    def __lt__(self, other: "_KeyWrapper") -> bool:
        return self.mycmp(self.obj, other.obj) < 0

    def __gt__(self, other: "_KeyWrapper") -> bool:
        return self.mycmp(self.obj, other.obj) > 0

    def __eq__(self, other: object) -> bool:
        return self.mycmp(self.obj, other.obj) == 0  # type: ignore[attr-defined]

    def __le__(self, other: "_KeyWrapper") -> bool:
        return self.mycmp(self.obj, other.obj) <= 0

    def __ge__(self, other: "_KeyWrapper") -> bool:
        return self.mycmp(self.obj, other.obj) >= 0

    __hash__ = None  # type: ignore[assignment]


# Reference: https://docs.python.org/3/library/functools.html#functools.cmp_to_key
@substitute_in_graph(functools.cmp_to_key, skip_signature_check=True)
def cmp_to_key(mycmp: Callable[[Any, Any], int]) -> Callable[[Any], _KeyWrapper]:
    return functools.partial(_KeyWrapper, mycmp)


# Reference: https://docs.python.org/3/library/functools.html#functools.reduce
@substitute_in_graph(functools.reduce)
def reduce(
    function: Callable[[_U, _T], _U],
    iterable: Iterable[_T],
    initial: _U = _initial_missing,  # type: ignore[assignment]
    /,
) -> _U:
    it = iter(iterable)

    value: _U
    if initial is _initial_missing:
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
