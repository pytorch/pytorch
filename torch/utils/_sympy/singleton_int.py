from typing import Any

import sympy
from sympy.multipledispatch import dispatch


__all__ = ["SingletonInt"]


class SingletonInt(sympy.AtomicExpr):
    # This is probably not super important unless we are in multiple dispatch
    # situations with other more exotic Expr types.
    _op_priority = 99999
    _val: int
    _coeff: int

    def __new__(
        cls, *args: Any, coeff: int | None = None, **kwargs: Any
    ) -> "SingletonInt":
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    # The semantics of this class should match that of NestedIntSymNodeImpl in
    # c10/core/NestedIntSymNodeImpl.h
    def __init__(self, val: int, *, coeff: int = 1) -> None:
        self._val = val
        self._coeff = coeff
        super().__init__()

    # See NOTE [ Inequalities with nested int ]
    def _eval_Eq(self, other: sympy.Basic) -> sympy.logic.boolalg.BooleanAtom:
        if (
            isinstance(other, SingletonInt)
            and other._val == self._val
            and self._coeff == other._coeff
        ):
            return sympy.true
        else:
            return sympy.false

    # This is necessary so that calling expr.free_symbols on exprs that contain
    # this Singleton does not error
    @property
    def free_symbols(self) -> set[sympy.Symbol]:
        return set()

    def __mul__(self, other: int) -> "SingletonInt":
        if isinstance(other, SingletonInt):
            raise ValueError(
                "SingletonInt cannot be multiplied by another SingletonInt"
            )
        return SingletonInt(self._val, coeff=self._coeff * other)

    def __rmul__(self, other: int) -> "SingletonInt":
        if isinstance(other, SingletonInt):
            raise ValueError(
                "SingletonInt cannot be multiplied by another SingletonInt"
            )
        return SingletonInt(self._val, coeff=self._coeff * other)

    # Make sure we promptly raise an error instead of falling back to building
    # an expression tree. There are probably more ops, how can we be exhaustive?
    def __add__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("NYI")

    def __sub__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("NYI")

    def __truediv__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("NYI")

    def __floordiv__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("NYI")

    def __mod__(self, other: object) -> "SingletonInt":
        raise NotImplementedError("NYI")


# See NOTE [ Inequalities with nested int ]
@dispatch(sympy.Integer, SingletonInt)
def _eval_is_ge(a: sympy.Integer, b: SingletonInt) -> sympy.logic.boolalg.BooleanAtom:
    if a < 2:
        return sympy.false
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")


@dispatch(SingletonInt, sympy.Integer)  # type: ignore[no-redef]
def _eval_is_ge(a: SingletonInt, b: sympy.Integer) -> sympy.logic.boolalg.BooleanAtom:
    if b <= 2:
        return sympy.true
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")


@dispatch(SingletonInt, SingletonInt)  # type: ignore[no-redef]
def _eval_is_ge(a: SingletonInt, b: SingletonInt) -> sympy.logic.boolalg.BooleanAtom:
    if a._val == b._val:
        if a._coeff >= b._coeff:
            return sympy.true
        else:
            return sympy.false
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")
