"""This file implements the IndexPropagation ops handler, which wraps an
underlying handler to add a limited form of constant propagation, as well as
propagation of sympy expressions downstream of ops.index_expr calls.

For example, say we have the IR:

   tmp0 = ops.index_expr(x, torch.int32)
   tmp1 = ops.constant(2, torch.int32)
   tmp2 = ops.mul(tmp0, tmp1)
   tmp3 = ops.indirect_indexing(tmp2, x_size)
   tmp4 = ops.load("buf0", tmp3)

The underlying handler would just see:

   ops.load("buf0", x * 2)

This is limited by the set of operators handled in the sympy expression
printers. So simple operations like minimum and maximum cannot be translated to
SymPy expressions yet, despite sympy.Min and sympy.Max existing.

"""
import itertools
from dataclasses import dataclass
from typing import Any

import sympy

import torch
from torch._prims_common import is_boolean_dtype, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing


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
        if isinstance(value.expr, (sympy.Integer, sympy.Float)):
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
    def sub(x, y):
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr - y.expr, result_type)

    @staticmethod
    def mul(x, y):
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr * y.expr, result_type)

    @staticmethod
    def neg(x):
        return TypedExpr(-x.expr, x.dtype)

    @staticmethod
    def floordiv(x, y):
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        return TypedExpr(FloorDiv(x.expr, y.expr), result_type)

    @staticmethod
    def remainder(x, y):
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        result_expr = ModularIndexing(x.expr, sympy.Integer(1), y.expr)
        return TypedExpr(result_expr, result_type)

    @staticmethod
    def minimum(x, y):
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(sympy.Min(x.expr, y.expr), result_type)

    @staticmethod
    def maximum(x, y):
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(sympy.Max(x.expr, y.expr), result_type)


@dataclass
class IndexPropVar:
    value: Any  # Either an IR value, or TypedExpr if is_symbolic is true
    is_symbolic: bool = False

    @staticmethod
    def new_symbolic(expr: TypedExpr) -> "IndexPropVar":
        return IndexPropVar(expr, is_symbolic=True)

    def __post_init__(self):
        assert not self.is_symbolic or isinstance(
            self.value, TypedExpr
        ), "Symbolic IndexPropVar must contain a TypedExpr"


class IndexPropagation:
    """Ops wrapper that tries to propagate constant and index_expr values through the computation.

    This aims to maximize the compile time simplification possible, and convert
    indirect indexing from arange into normal static indexing.

    """

    def __init__(self, inner):
        self._inner = inner

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
        if a.is_symbolic:
            return self.materialize_expr(a.value.expr, a.value.dtype)

        return a.value

    def fallback(self, name, args, kwargs):
        # Fallback to the wrapped handler
        new_args = [self.unwrap(a) for a in args]
        new_kwargs = {k: self.unwrap(v) for k, v in kwargs.items()}
        return IndexPropVar(getattr(self._inner, name)(*new_args, **new_kwargs))

    def propagate_sympy(self, name, args, kwargs):
        # Build a new SymPy expression from this ops call
        def unwrap(a):
            if not isinstance(a, IndexPropVar):
                return a
            return a.value

        new_args = [unwrap(a) for a in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)
        if new_expr is NotImplemented:
            return self.fallback(name, args, kwargs)
        return IndexPropVar.new_symbolic(new_expr)

    def __getattr__(self, name):
        def inner(*args, **kwargs):
            if not hasattr(SymPyOps, name):
                return self.fallback(name, args, kwargs)

            var_arguments = [
                a
                for a in itertools.chain(args, kwargs.values())
                if isinstance(a, IndexPropVar)
            ]
            if not all(v.is_symbolic for v in var_arguments):
                return self.fallback(name, args, kwargs)

            return self.propagate_sympy(name, args, kwargs)

        return inner

    def indirect_indexing(self, index, size, check=True):
        # indirect_indexing returns a sympy value, so no need to wrap in IndexPropVar here
        if isinstance(index, IndexPropVar) and index.is_symbolic:
            return index.value.expr
        return self.fallback("indirect_indexing", (index, size, check), {}).value
