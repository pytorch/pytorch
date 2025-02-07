"""
Python polyfills for functools
"""

import functools
from collections.abc import Iterable
from typing import Callable, TypeVar

from ..decorators import substitute_class, substitute_in_graph


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


@substitute_class(functools.partial, supports_reconstruction=True)
class partial:
    """New function with partial application of the given arguments
    and keywords.
    """

    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(cls, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if isinstance(func, partial):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super(partial, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(*self.args, *args, **keywords)

    @staticmethod
    def convert_to_traceable(original_value):
        assert isinstance(original_value, functools.partial)
        return partial(
            original_value.func, *original_value.args, **original_value.keywords
        )
    
    @staticmethod
    def convert_to_original(value):
        assert isinstance(value, partial)
        return functools.partial(
            value.func, *value.args, **value.keywords
        )
