"""
Python polyfills for operator
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, SupportsIndex, TypeVar, overload

from test.support import import_helper
from typing_extensions import TypeVarTuple, Unpack

from ..decorators import substitute_in_graph

py_operator = import_helper.import_fresh_module("operator", blocked=["_operator"])
assert py_operator is not None


if TYPE_CHECKING:
    from collections.abc import Callable, Container, Iterable, Sequence, MutableSequence


# Most unary and binary operators are handled by BuiltinVariable (e.g., `pos`, `add`)
__all__ = [
    "attrgetter",
    "itemgetter",
    "methodcaller",
    "countOf",
    "add",
    "sub",
    "mul",
    "floordiv",
    "truediv",
    "contains",
    "truth",
    "call",
    "getitem",
    "concat",
    "indexOf",
    "inv",
    "lshift",
    "rshift",
    "mod",
    "neg",
    "or_",
    "and_",
    "pos",
    "pow",
    "setitem",
]


_T = TypeVar("_T")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_Ts = TypeVarTuple("_Ts")
_U = TypeVar("_U")
_U1 = TypeVar("_U1")
_U2 = TypeVar("_U2")
_Us = TypeVarTuple("_Us")


@overload
# pyrefly: ignore [inconsistent-overload]
def attrgetter(attr: str, /) -> Callable[[Any], _U]: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def attrgetter(
    attr1: str, attr2: str, /, *attrs: str
) -> Callable[[Any], tuple[_U1, _U2, Unpack[_Us]]]: ...


# Reference: https://docs.python.org/3/library/operator.html#operator.attrgetter
@substitute_in_graph(operator.attrgetter, is_embedded_type=True)  # type: ignore[arg-type,misc]
def attrgetter(*attrs: str) -> Callable[[Any], Any | tuple[Any, ...]]:
    if len(attrs) == 0:
        raise TypeError("attrgetter expected 1 argument, got 0")

    if any(not isinstance(attr, str) for attr in attrs):
        raise TypeError("attribute name must be a string")

    def resolve_attr(obj: Any, attr: str) -> Any:
        for name in attr.split("."):
            obj = getattr(obj, name)
        return obj

    if len(attrs) == 1:
        attr = attrs[0]

        def getter(obj: Any) -> Any:
            return resolve_attr(obj, attr)

    else:

        def getter(obj: Any) -> tuple[Any, ...]:  # type: ignore[misc]
            return tuple(resolve_attr(obj, attr) for attr in attrs)

    return getter


@overload
# pyrefly: ignore [inconsistent-overload]
def itemgetter(item: _T, /) -> Callable[[Any], _U]: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def itemgetter(
    item1: _T1, item2: _T2, /, *items: Unpack[_Ts]
) -> Callable[[Any], tuple[_U1, _U2, Unpack[_Us]]]: ...


# Reference: https://docs.python.org/3/library/operator.html#operator.itemgetter
@substitute_in_graph(operator.itemgetter, is_embedded_type=True)  # type: ignore[arg-type,misc]
def itemgetter(*items: Any) -> Callable[[Any], Any | tuple[Any, ...]]:
    if len(items) == 0:
        raise TypeError("itemgetter expected 1 argument, got 0")

    if len(items) == 1:
        item = items[0]

        def getter(obj: Any) -> Any:
            return obj[item]

    else:

        def getter(obj: Any) -> tuple[Any, ...]:  # type: ignore[misc]
            return tuple(obj[item] for item in items)

    return getter


# Reference: https://docs.python.org/3/library/operator.html#operator.methodcaller
@substitute_in_graph(operator.methodcaller, is_embedded_type=True)  # type: ignore[arg-type]
def methodcaller(name: str, /, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
    if not isinstance(name, str):
        raise TypeError("method name must be a string")

    def caller(obj: Any) -> Any:
        return getattr(obj, name)(*args, **kwargs)

    return caller


# Reference: https://docs.python.org/3/library/operator.html#operator.countOf
@substitute_in_graph(operator.countOf, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def countOf(a: Iterable[_T], b: _T, /) -> int:
    return sum(it is b or it == b for it in a)


@substitute_in_graph(operator.add, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def add(a: Any, b: Any, /) -> Any:
    return py_operator.add(a, b)


@substitute_in_graph(operator.sub, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def sub(a: Any, b: Any, /) -> Any:
    return py_operator.sub(a, b)


@substitute_in_graph(operator.mul, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def mul(a: Any, b: Any, /) -> Any:
    return py_operator.mul(a, b)


@substitute_in_graph(operator.floordiv, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def floordiv(a: Any, b: Any, /) -> Any:
    return py_operator.floordiv(a, b)


@substitute_in_graph(operator.truediv, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def truediv(a: Any, b: Any, /) -> Any:
    return py_operator.truediv(a, b)


@substitute_in_graph(operator.contains, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def contains(a: Container[object], b: object, /) -> bool:
    return py_operator.contains(a, b)


@substitute_in_graph(operator.truth, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def truth(a: Any, /) -> bool:
    return py_operator.truth(a)


@substitute_in_graph(operator.call, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def call(obj: Callable, /, *args, **kwargs) -> Any:
    return py_operator.call(obj, *args, **kwargs)


@substitute_in_graph(operator.getitem, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def getitem(a, b, /) -> Any:
    return py_operator.getitem(a, b)


@substitute_in_graph(operator.concat, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def concat(a: Sequence[_T], b: Sequence[_T], /) -> Sequence[_T]:
    return py_operator.concat(a, b)


@substitute_in_graph(operator.indexOf, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def indexOf(a: Iterable[_T], b: _T, /) -> int:
    return py_operator.indexOf(a, b)


@substitute_in_graph(operator.inv, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def inv(a: Any, /) -> Any:
    return py_operator.inv(a)


@substitute_in_graph(operator.lshift, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def lshift(a: Any, b: Any, /) -> Any:
    return py_operator.lshift(a, b)


@substitute_in_graph(operator.rshift, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def rshift(a: Any, b: Any, /) -> Any:
    return py_operator.rshift(a, b)


@substitute_in_graph(operator.mod, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def mod(a: Any, b: Any, /) -> Any:
    return py_operator.mod(a, b)

@substitute_in_graph(operator.neg, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def neg(a: Any, /) -> Any:
    return py_operator.neg(a)

@substitute_in_graph(operator.pos, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def pos(a: Any, /) -> Any:
    return py_operator.pos(a)


@substitute_in_graph(operator.pow, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def pow(a: Any, b: Any, /) -> Any:
    return py_operator.pow(a, b)


@substitute_in_graph(operator.or_, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def or_(a: Any, b: Any, /) -> Any:
    return py_operator.or_(a, b)


@substitute_in_graph(operator.and_, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def and_(a: Any, b: Any, /) -> Any:
    return py_operator.and_(a, b)


@substitute_in_graph(operator.setitem, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def setitem(a: MutableSequence[_T], b: SupportsIndex, c: _T, /) -> None:
    return py_operator.setitem(a, b, c)
