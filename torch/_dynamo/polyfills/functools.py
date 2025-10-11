"""
Python polyfills for functools
"""

import functools
from collections.abc import Iterable
from typing import Callable, TypeVar

import torch

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


# Reference: https://github.com/python/cpython/blob/56072f9c050b8dd960bb5630eb924eb02a889f9b/Lib/functools.py#L35
# NOTE: DOES NOT support updated != functools.WRAPPER_UPDATES
# NOTE: do not add to __all__ since we do not use substitute_in_graph (dynamo does some additional checks)
def update_wrapper(
    wrapper: Callable[[_U, _T], _U],
    wrapped: Callable[[_U, _T], _U],
    assigned: tuple[str, ...] = functools.WRAPPER_ASSIGNMENTS,
    updated: tuple[str, ...] = functools.WRAPPER_UPDATES,
) -> Callable[[_U, _T], _U]:
    for attr in assigned:
        try:
            value = getattr(wrapped, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)
    if updated != functools.WRAPPER_UPDATES:
        torch._dynamo.graph_break(
            "functools.update_wrapper/wraps does not support `updated` != functools.WRAPPER_UPDATES, i.e. ('__dict__',)"
        )
    for attr in wrapped.__dict__.keys():
        setattr(wrapper, attr, getattr(wrapped, attr))
    return wrapper
