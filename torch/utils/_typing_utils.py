"""Miscellaneous utilities to aid with typing."""

from typing import Optional, TYPE_CHECKING, TypeVar


# Helper to turn Optional[T] into T when we know None either isn't
# possible or should trigger an exception.
T = TypeVar("T")


# TorchScript cannot handle the type signature of `not_none` at runtime, because it trips
# over the `Optional[T]`. To allow using `not_none` from inside a TorchScript method/module,
# we split the implementation, and hide the runtime type information from TorchScript.
if TYPE_CHECKING:

    def not_none(obj: Optional[T]) -> T:
        ...

else:

    def not_none(obj):
        if obj is None:
            raise TypeError(
                "Invariant encountered: value was None when it should not be"
            )
        return obj
