"""Miscellaneous utilities to aid with typing."""

from collections.abc import Callable
from typing import Any, cast, Concatenate, TypeVar
from typing_extensions import ParamSpec


# Helper to turn Optional[T] into T when we know None either isn't
# possible or should trigger an exception.
T = TypeVar("T")


def not_none(obj: T | None) -> T:
    if obj is None:
        raise TypeError("Invariant encountered: value was None when it should not be")
    return obj


_P = ParamSpec("_P")
_R = TypeVar("_R")
_A1 = TypeVar("_A1")


def copy_func_params(
    source_func: Callable[_P, Any],
) -> Callable[[Callable[..., _R]], Callable[_P, _R]]:
    """Cast the decorated function's call signature to the source_func's.

    Usage:
        def upstream_func(a: int, b: float, *, double: bool = False) -> float: ...
        @copy_func_params(upstream_func)
        def enhanced(a: int, b: float, *args: Any, double: bool = False, **kwargs: Any) -> str: ...
    """

    def return_func(func: Callable[..., _R]) -> Callable[_P, _R]:
        return cast(Callable[_P, _R], func)

    return return_func


def copy_method_params(
    source_method: Callable[Concatenate[Any, _P], Any],
) -> Callable[[Callable[..., _R]], Callable[Concatenate[_A1, _P], _R]]:
    """Cast the decorated *method*'s call signature to the source_method's.
    Keeps the first argument type (e.g., self/cls).
    """

    def return_func(func: Callable[..., _R]) -> Callable[Concatenate[_A1, _P], _R]:
        return cast(Callable[Concatenate[_A1, _P], _R], func)

    return return_func


# stricter variants to preserve the origin callers Return Type too.
# TODO: consider folding both these into the above variants with an optional
# parameter to control whether to copy the return type or not.
def copy_func_sig(
    source_func: Callable[_P, _R],
) -> Callable[[Callable[..., _R]], Callable[_P, _R]]:
    """Cast the decorated function's call signature and return type to the source_func's."""

    def _return(func: Callable[..., _R]) -> Callable[_P, _R]:
        return cast(Callable[_P, _R], func)

    return _return


def copy_method_sig(
    source_method: Callable[Concatenate[_A1, _P], _R],
) -> Callable[[Callable[..., _R]], Callable[Concatenate[_A1, _P], _R]]:
    """Cast the decorated *method*'s call signature to the source_method and return type."""

    def _return(func: Callable[..., _R]) -> Callable[Concatenate[_A1, _P], _R]:
        return cast(Callable[Concatenate[_A1, _P], _R], func)

    return _return
