"""
Python polyfills for functools
"""

import functools
from collections.abc import Callable, Iterable
from typing import TypeVar

from ..decorators import substitute_in_graph
from . import import_fresh_module


# Import the pure Python functools module, blocking the C extension
py_functools = import_fresh_module("functools", blocked=["_functools"])


__all__ = ["reduce", "_lru_cache_wrapper", "_make_key"]


_T = TypeVar("_T")
_U = TypeVar("_U")


_initial_missing = object()


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


@substitute_in_graph(functools._lru_cache_wrapper, is_embedded_type=True)
def _lru_cache_wrapper(*args, **kwargs):
    return py_functools._lru_cache_wrapper(*args, **kwargs)


@substitute_in_graph(py_functools._make_key, skip_signature_check=True)
def _make_key(
    args,
    kwds,
    typed,
):
    """Make a cache key from optionally typed positional and keyword arguments

    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.

    """
    # All of code below relies on kwds preserving the order input by the user.
    # Formerly, we sorted() the kwds before looping.  The new way is *much*
    # faster; however, it means that f(x=1, y=2) will now be treated as a
    # distinct call from f(y=2, x=1) which will be cached separately.
    kwd_mark = (object(),)
    fasttypes = {int, str}

    key = args
    if kwds:
        key += kwd_mark
        for item in kwds.items():
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for v in kwds.values())
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return key
