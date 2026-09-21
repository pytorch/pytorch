from __future__ import annotations

from typing import TYPE_CHECKING

from .core import (  # type: ignore[attr-defined]
    _reify as core_reify,
    _unify as core_unify,
    reify,
    unify,
)
from .dispatch import dispatch


if TYPE_CHECKING:
    from .variable import Var


__all__ = ["unifiable", "reify_object", "unify_object"]


def unifiable(cls: type) -> type:
    """Register standard unify and reify operations on class
    This uses the type and __dict__ or __slots__ attributes to define the
    nature of the term
    See Also:
    >>> # xdoctest: +SKIP
    >>> class A(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    >>> unifiable(A)
    <class 'unification.more.A'>
    >>> x = var("x")
    >>> a = A(1, 2)
    >>> b = A(1, x)
    >>> unify(a, b, {})
    {~x: 2}
    """
    core_unify.add((cls, cls, dict), unify_object)  # type: ignore[attr-defined]
    core_reify.add((cls, dict), reify_object)  # type: ignore[attr-defined]

    return cls


#########
# Reify #
#########


def reify_object(o: object, s: dict[Var, object]) -> object:
    """Reify a Python object with a substitution
    >>> # xdoctest: +SKIP
    >>> class Foo(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    ...
    ...     def __str__(self):
    ...         return "Foo(%s, %s)" % (str(self.a), str(self.b))
    >>> x = var("x")
    >>> f = Foo(1, x)
    >>> print(f)
    Foo(1, ~x)
    >>> print(reify_object(f, {x: 2}))
    Foo(1, 2)
    """
    if hasattr(o, "__slots__"):
        return _reify_object_slots(o, s)
    else:
        return _reify_object_dict(o, s)


def _reify_object_dict(o: object, s: dict[Var, object]) -> object:
    obj = object.__new__(type(o))
    d = reify(o.__dict__, s)  # pyrefly: ignore[missing-attribute]
    if d == o.__dict__:  # pyrefly: ignore[missing-attribute]
        return o
    obj.__dict__.update(d)  # pyrefly: ignore[missing-attribute, no-matching-overload]
    return obj


def _reify_object_slots(o: object, s: dict[Var, object]) -> object:
    attrs = [
        getattr(o, attr)
        for attr in o.__slots__  # pyrefly: ignore[missing-attribute]
    ]
    new_attrs = reify(attrs, s)
    if attrs == new_attrs:
        return o
    else:
        newobj = object.__new__(type(o))
        for slot, attr in zip(
            o.__slots__,  # pyrefly: ignore[missing-attribute]
            new_attrs,  # pyrefly: ignore[bad-argument-type]
        ):
            setattr(newobj, slot, attr)
        return newobj


@dispatch(slice, dict)
def _reify(o: slice, s: dict[Var, object]) -> slice:
    """Reify a Python ``slice`` object"""

    return slice(*reify((o.start, o.stop, o.step), s))  # pyrefly: ignore[not-iterable]


#########
# Unify #
#########


def unify_object(
    u: object, v: object, s: dict[Var, object]
) -> dict[Var, object] | bool:
    """Unify two Python objects
    Unifies their type and ``__dict__`` attributes
    >>> # xdoctest: +SKIP
    >>> class Foo(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    ...
    ...     def __str__(self):
    ...         return "Foo(%s, %s)" % (str(self.a), str(self.b))
    >>> x = var("x")
    >>> f = Foo(1, x)
    >>> g = Foo(1, 2)
    >>> unify_object(f, g, {})
    {~x: 2}
    """
    if type(u) is not type(v):
        return False
    if hasattr(u, "__slots__"):
        return unify(
            [
                getattr(u, slot)
                for slot in u.__slots__  # pyrefly: ignore[missing-attribute]
            ],
            [
                getattr(v, slot)
                for slot in v.__slots__  # pyrefly: ignore[missing-attribute]
            ],
            s,
        )
    else:
        return unify(
            u.__dict__,  # pyrefly: ignore[missing-attribute]
            v.__dict__,  # pyrefly: ignore[missing-attribute]
            s,
        )


@dispatch(slice, slice, dict)
def _unify(u: slice, v: slice, s: dict[Var, object]) -> dict[Var, object] | bool:
    """Unify a Python ``slice`` object"""
    return unify((u.start, u.stop, u.step), (v.start, v.stop, v.step), s)
