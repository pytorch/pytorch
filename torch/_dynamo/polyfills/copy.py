"""
Python polyfills for copy
"""

from __future__ import annotations

import copy
from typing import Any, TypeVar

from ..decorators import substitute_in_graph


__all__ = [
    "reduce_ex_user_defined_object",
]

T = TypeVar("T")


@substitute_in_graph(object.__reduce_ex__, skip_signature_check=True)  # type: ignore[arg-type]
def reduce_ex_user_defined_object(obj: T, protocol: int, /) -> tuple:  # type: ignore[type-arg]
    """Traceable polyfill for object.__reduce_ex__ (protocol >= 2).

    Mirrors CPython's reduce_newobj (Objects/typeobject.c): builds the __new__
    arguments from __getnewargs_ex__/__getnewargs__, selects
    copyreg.__newobj_ex__ vs __newobj__ based on whether kwargs are present, and
    computes the pickle state (__getstate__ if overridden, else __dict__ if the
    object has one, else None). copy._reconstruct rebuilds the object via
    cls.__new__(cls, *args) and applies the state. List subclasses supply
    iter(obj) as listitems, while dict subclasses supply iter(obj.items()) as
    dictitems; copy._reconstruct consumes them to restore container contents.

    This must not assume the object has a __dict__: tuple/slots objects such as
    namedtuples have __slots__ = () and no __dict__, and reduce to
    (copyreg.__newobj__, (cls, *values), None, None, None).
    """
    import copyreg

    cls = type(obj)

    # Mirror object.__reduce_ex__ (Objects/typeobject.c): if the type overrides
    # __reduce__, defer to it rather than building the copyreg newobj reduction.
    if getattr(cls, "__reduce__", None) is not object.__reduce__:
        return obj.__reduce__()  # type: ignore[attr-defined]

    args: tuple  # type: ignore[type-arg]
    kwargs: dict  # type: ignore[type-arg]
    if hasattr(cls, "__getnewargs_ex__"):
        args, kwargs = obj.__getnewargs_ex__()  # type: ignore[attr-defined]
    elif hasattr(cls, "__getnewargs__"):
        args = obj.__getnewargs__()  # type: ignore[attr-defined]
        kwargs = {}
    else:
        args = ()
        kwargs = {}

    if kwargs:
        func = copyreg.__newobj_ex__  # pyrefly: ignore[missing-attribute]
        newargs = (cls, args, kwargs)
    else:
        func = copyreg.__newobj__  # pyrefly: ignore[missing-attribute]
        newargs = (cls, *args)

    default_getstate = getattr(object, "__getstate__", None)
    if getattr(cls, "__getstate__", None) is not default_getstate:
        state = obj.__getstate__()
    else:
        try:
            state = obj.__dict__
        except AttributeError:
            state = None

    listitems = iter(obj) if isinstance(obj, list) else None
    dictitems = iter(obj.items()) if isinstance(obj, dict) else None

    return (func, newargs, state, listitems, dictitems)


_legacy_deepcopy_atomic = getattr(copy, "_deepcopy_atomic", None)


def is_unmemoized_atomic_key(item: Any, memo: dict[int, Any], owner: Any) -> bool:
    # Legacy deepcopy consults the memo before recognizing atomic keys. A memo
    # containing only the OrderedDict's ID cannot contain its scalar key's ID.
    dispatch = copy._deepcopy_dispatch  # pyrefly: ignore[missing-attribute]
    if type(memo) is not dict or len(memo) != 1 or type(dispatch) is not dict:
        return False
    token = next(iter(memo))
    return (
        type(token) is int
        and token == id(owner)
        and dispatch.get(type(item)) is _legacy_deepcopy_atomic
    )
