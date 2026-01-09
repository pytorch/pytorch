"""Miscellaneous utilities to aid with typing."""

from collections.abc import Callable
from typing import Any, cast, Concatenate, Literal, overload, TypeVar
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


# ---------------------------------------------------------------------------
# Copy call signature utilities
#   - validate_return=False: copy params only
#   - validate_return=True:  copy params + enforce return type matches source
# ---------------------------------------------------------------------------


@overload
def copy_func_params(
    source_func: Callable[_P, _R],
    *,
    validate_return: Literal[True],
) -> Callable[[Callable[..., _R]], Callable[_P, _R]]: ...
@overload
def copy_func_params(
    source_func: Callable[_P, Any],
    *,
    validate_return: Literal[False] = False,
) -> Callable[[Callable[..., _R]], Callable[_P, _R]]: ...
def copy_func_params(
    source_func: Callable[_P, Any],
    *,
    validate_return: bool = False,
) -> Callable[[Callable[..., _R]], Callable[_P, _R]]:
    """Cast the decorated function's call signature to the source_func's.

    If validate_return=True, also ties the decorated function's return type to the
    source_func's return type (static type checking only).
    """

    def _return(func: Callable[..., _R]) -> Callable[_P, _R]:
        return cast(Callable[_P, _R], func)

    return _return


@overload
def copy_method_params(
    source_method: Callable[Concatenate[Any, _P], _R],
    *,
    validate_return: Literal[True],
) -> Callable[[Callable[..., _R]], Callable[Concatenate[_A1, _P], _R]]: ...
@overload
def copy_method_params(
    source_method: Callable[Concatenate[Any, _P], Any],
    *,
    validate_return: Literal[False] = False,
) -> Callable[[Callable[..., _R]], Callable[Concatenate[_A1, _P], _R]]: ...
def copy_method_params(
    source_method: Callable[Concatenate[Any, _P], Any],
    *,
    validate_return: bool = False,
) -> Callable[[Callable[..., _R]], Callable[Concatenate[_A1, _P], _R]]:
    """Cast the decorated *method*'s call signature to the source_method's.

    Keeps the first argument type (e.g., self/cls).
    If validate_return=True, also ties the decorated method's return type to the
    source_method's return type (static type checking only).
    """

    def _return(func: Callable[..., _R]) -> Callable[Concatenate[_A1, _P], _R]:
        return cast(Callable[Concatenate[_A1, _P], _R], func)

    return _return
