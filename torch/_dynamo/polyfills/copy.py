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
    """Traceable polyfill for object.__reduce_ex__ (protocol >= 2).

    Mirrors CPython's reduce_newobj (Objects/typeobject.c): builds the __new__
    arguments from __getnewargs_ex__/__getnewargs__, selects
    copyreg.__newobj_ex__ vs __newobj__ based on whether kwargs are present, and
    computes the pickle state (__getstate__ if overridden, else the instance
    dictionary and slot state). copy._reconstruct rebuilds the object via
    cls.__new__(cls, *args) and applies the state.

    This must not assume the object has a __dict__: tuple/slots objects such as
    namedtuples have __slots__ = () and no __dict__, and reduce to
    (copyreg.__newobj__, (cls, *values), None, None, None).
    """
    import copyreg

    cls = type(obj)

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
        else:
            if not state:
                state = None

        slotstate: dict[str, object] = {}
        for base in cls.__mro__:
            try:
                slots = base.__dict__["__slots__"]
            except KeyError:
                continue
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in ("__dict__", "__weakref__"):
                    continue
                if name.startswith("__") and not name.endswith("__"):
                    stripped = base.__name__.lstrip("_")
                    if stripped:
                        name = f"_{stripped}{name}"
                try:
                    value = getattr(obj, name)
                except AttributeError:
                    pass
                else:
                    slotstate[name] = value
        if slotstate:
            state = (state, slotstate)

    return (func, newargs, state, None, None)
