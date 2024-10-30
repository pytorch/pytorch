# mypy: allow-untyped-defs
from .core import reify, unify  # type: ignore[attr-defined]
from .dispatch import dispatch


def unifiable(cls):
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
    _unify.add((cls, cls, dict), unify_object)
    _reify.add((cls, dict), reify_object)

    return cls


#########
# Reify #
#########


def reify_object(o, s):
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


def _reify_object_dict(o, s):
    obj = object.__new__(type(o))
    d = reify(o.__dict__, s)
    if d == o.__dict__:
        return o
    obj.__dict__.update(d)
    return obj


def _reify_object_slots(o, s):
    attrs = [getattr(o, attr) for attr in o.__slots__]
    new_attrs = reify(attrs, s)
    if attrs == new_attrs:
        return o
    else:
        newobj = object.__new__(type(o))
        for slot, attr in zip(o.__slots__, new_attrs):
            setattr(newobj, slot, attr)
        return newobj


@dispatch(slice, dict)
def _reify(o, s):
    """Reify a Python ``slice`` object"""
    return slice(*reify((o.start, o.stop, o.step), s))


#########
# Unify #
#########


def unify_object(u, v, s):
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
    if type(u) != type(v):
        return False
    if hasattr(u, "__slots__"):
        return unify(
            [getattr(u, slot) for slot in u.__slots__],
            [getattr(v, slot) for slot in v.__slots__],
            s,
        )
    else:
        return unify(u.__dict__, v.__dict__, s)


@dispatch(slice, slice, dict)
def _unify(u, v, s):
    """Unify a Python ``slice`` object"""
    return unify((u.start, u.stop, u.step), (v.start, v.stop, v.step), s)
