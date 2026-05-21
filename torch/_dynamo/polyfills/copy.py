"""
Python polyfills for copy
"""

from __future__ import annotations

from typing import TypeVar

from ..decorators import substitute_in_graph


__all__ = [
    "reduce_ex_user_defined_object",
]

T = TypeVar("T")


@substitute_in_graph(object.__reduce_ex__, skip_signature_check=True)  # type: ignore[arg-type]
def reduce_ex_user_defined_object(obj: T, protocol: int, /) -> tuple:  # type: ignore[type-arg]
    """Traceable polyfill for object.__reduce_ex__ on user-defined objects.

    Returns the same tuple that CPython's _common_reduce produces:
    (copyreg.__newobj__, (cls,), obj.__dict__, None, None).
    copy._reconstruct then calls cls.__new__(cls) and updates __dict__.
    """
    import copyreg

    cls = type(obj)
    return (
        copyreg.__newobj__,  # pyrefly: ignore[missing-attribute]
        (cls,),
        obj.__dict__,
        None,
        None,
    )
