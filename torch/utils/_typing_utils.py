"""Miscellaneous utilities to aid with typing."""

from typing import TypeVar


# Helper to turn Optional[T] into T when we know None either isn't
# possible or should trigger an exception.
T = TypeVar("T")


def not_none(obj: T | None) -> T:
    if obj is None:
        raise TypeError("Invariant encountered: value was None when it should not be")
    return obj
