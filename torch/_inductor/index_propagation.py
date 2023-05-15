"""
This file implements the IndexPropagation ops handler, which wraps an
underlying handler to add a limited for of constant propagation, as well as
propagation of sympy expressions downstream of `ops.index_expr` calls.

"""
import itertools
from dataclasses import dataclass
from typing import Any, Optional

import sympy

import torch
from torch._prims_common import is_boolean_dtype, is_integer_dtype


@dataclass
class TypedExpr:
    """A SymPy expression with associated type"""

    expr: sympy.Expr
    dtype: torch.dtype


class SymPyOps:
    """An ops handler where all IR values are SymPy expressions

    When a value cannot be represented as a SymPy expression, the method is
    either not defined, or returns NotImplemented

    """

    @staticmethod
    def identity(value):
        return value

    @staticmethod
    def constant(value, dtype):
        if is_boolean_dtype(dtype):
            expr = sympy.Integer(bool(value))
        elif is_integer_dtype(dtype):
            expr = sympy.Integer(int(value))
        else:
            expr = sympy.Float(float(value))
        return TypedExpr(expr, dtype)

    @staticmethod
    def index_expr(value, dtype):
        if isinstance(value, int):
            value = sympy.Integer(value)
        return TypedExpr(value, dtype)

    @staticmethod
    def to_dtype(value, dtype):
        if isinstance(value, (sympy.Integer, sympy.Float)):
            return SymPyOps.constant(value.expr, dtype)
        elif is_integer_dtype(dtype) and is_integer_dtype(value.dtype):
            return SymPyOps.index_expr(value.expr, dtype)
        else:
            # TODO: Inductor doesn't handle floating point in sympy expressions well at the moment
            return NotImplemented

    @staticmethod
    def square(x):
        return TypedExpr(x.expr * x.expr, x.dtype)

    @staticmethod
    def add(x, y):
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr + y.expr, result_type)

    @staticmethod
    def mul(x, y):
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr * y.expr, result_type)

    @staticmethod
    def neg(x):
        return TypedExpr(-x.expr, x.dtype)

    @staticmethod
    def abs(x):
        return TypedExpr(abs(x.expr), x.dtype)


@dataclass
class IndexPropVar:
    rep: Any  # A wrapped IR value, if present
    expr: Optional[TypedExpr] = None


class IndexPropagation:
    """Ops wrapper that tries to propagate constant and index_expr values through the computation.

    This aims to maximize the compile time simplification possible, and convert
    indirect indexing from arange into normal static indexing.

    """

    def __init__(self, inner):
        self._inner = inner
        self.id_iter = itertools.count()

    def new_symbol(self, expr: TypedExpr):
        return IndexPropVar(f"index_tmp{next(self.id_iter)}", expr)

    def materialize_expr(self, expr, dtype):
        # Construct a new constant/index_expr from the SymPy expression
        if isinstance(expr, sympy.Integer):
            return self._inner.constant(int(expr), dtype)
        elif not expr.free_symbols:
            return self._inner.constant(float(expr), dtype)
        return self._inner.index_expr(expr, dtype)

    def unwrap(self, a):
        if not isinstance(a, IndexPropVar):
            return a

        # Prefer the sympy representation if possible
        if a.expr is not None:
            return self.materialize_expr(a.expr.expr, a.expr.dtype)

        return a.rep

    def fallback(self, name, args, kwargs):
        # Fallback to the wrapped handler
        new_args = [self.unwrap(a) for a in args]
        new_kwargs = {k: self.unwrap(v) for k, v in kwargs.items()}
        return IndexPropVar(getattr(self._inner, name)(*new_args, **new_kwargs))

    def propagate_sympy(self, name, args, kwargs):
        # Builld a new SymPy expression from this ops call
        def unwrap(a):
            if not isinstance(a, IndexPropVar):
                return a
            return a.expr

        new_args = [unwrap(a) for a in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)
        if new_expr is NotImplemented:
            return self.fallback(name, args, kwargs)
        return self.new_symbol(new_expr)

    def __getattr__(self, name):
        def inner(*args, **kwargs):
            if not hasattr(SymPyOps, name):
                return self.fallback(name, args, kwargs)

            var_arguments = [
                a
                for a in itertools.chain(args, kwargs.values())
                if isinstance(a, IndexPropVar)
            ]
            if any(v.expr is None for v in var_arguments):
                return self.fallback(name, args, kwargs)

            return self.propagate_sympy(name, args, kwargs)

        return inner

    def indirect_indexing(self, index, size):
        # indirect_indexing returns a sympy value, so no need to wrap in IndexPropVar here
        if isinstance(index, IndexPropVar) and index.expr is not None:
            return index.expr.expr
        return self.fallback("indirect_indexing", (index, size), {}).rep
