from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import partial
from typing import TYPE_CHECKING
from typing_extensions import TypeVarTuple, Unpack

from .dispatch import dispatch
from .unification_tools import assoc  # type: ignore[import]
from .utils import transitive_get as walk
from .variable import isvar


if TYPE_CHECKING:
    from .variable import Var


_Ts = TypeVarTuple("_Ts")

__all__ = ["reify", "unify"]

###############
# Reification #
###############


@dispatch(Iterator, dict)
def _reify(t: Iterator[object], s: dict[Var, object]) -> Iterator[object]:
    return map(partial(reify, s=s), t)
    # return (reify(arg, s) for arg in t)


_reify


@dispatch(tuple, dict)  # type: ignore[no-redef]
def _reify(t: tuple[Unpack[_Ts]], s: dict[Var, object]) -> tuple[Unpack[_Ts]]:
    return tuple(reify(iter(t), s))  # pyrefly: ignore[bad-argument-type, bad-return]


_reify


@dispatch(list, dict)  # type: ignore[no-redef]
def _reify(t: list[object], s: dict[Var, object]) -> list[object]:
    return list(reify(iter(t), s))  # pyrefly: ignore[bad-argument-type]


_reify


@dispatch(dict, dict)  # type: ignore[no-redef]
def _reify(d: dict[object, object], s: dict[Var, object]) -> dict[object, object]:
    return {k: reify(v, s) for k, v in d.items()}


_reify


@dispatch(object, dict)  # type: ignore[no-redef]
def _reify(o: object, s: dict[Var, object]) -> object:
    return o  # catch all, just return the object


def reify(e: object, s: dict[Var, object]) -> object:
    """Replace variables of expression with substitution
    >>> # xdoctest: +SKIP
    >>> x, y = var(), var()
    >>> e = (1, x, (3, y))
    >>> s = {x: 2, y: 4}
    >>> reify(e, s)
    (1, 2, (3, 4))
    >>> e = {1: x, 3: (y, 5)}
    >>> reify(e, s)
    {1: 2, 3: (4, 5)}
    """
    if isvar(e):
        return reify(s[e], s) if e in s else e
    return _reify(e, s)


###############
# Unification #
###############

seq = tuple, list, Iterator


@dispatch(seq, seq, dict)  # type: ignore[arg-type]
def _unify(
    u: Sequence[object], v: Sequence[object], s: dict[Var, object]
) -> dict[Var, object] | bool:
    if len(u) != len(v):
        return False
    for uu, vv in zip(u, v):  # avoiding recursion
        s = unify(uu, vv, s)  # pyrefly: ignore[bad-assignment]
        if s is False:
            return False
    return s


#
# @dispatch((set, frozenset), (set, frozenset), dict)
# def _unify(u, v, s):
#     i = u & v
#     u = u - i
#     v = v - i
#     return _unify(sorted(u), sorted(v), s)
#
#
# @dispatch(dict, dict, dict)
# def _unify(u, v, s):
#     if len(u) != len(v):
#         return False
#     for key, uval in iteritems(u):
#         if key not in v:
#             return False
#         s = unify(uval, v[key], s)
#         if s is False:
#             return False
#     return s
#
#
# @dispatch(object, object, dict)
# def _unify(u, v, s):
#     return False  # catch all


@dispatch(object, object, dict)
def unify(
    u: object, v: object, s: dict[Var, object]
) -> dict[Var, object] | bool:  # no check at the moment
    """Find substitution so that u == v while satisfying s
    >>> x = var("x")
    >>> unify((1, x), (1, 2), {})
    {~x: 2}
    """
    u = walk(u, s)
    v = walk(v, s)
    if u == v:
        return s
    if isvar(u):
        return assoc(s, u, v)
    if isvar(v):
        return assoc(s, v, u)
    return _unify(u, v, s)


unify


@dispatch(object, object)  # type: ignore[no-redef]
def unify(u: object, v: object) -> dict[Var, object] | bool:
    return unify(u, v, {})
