import itertools
from dataclasses import dataclass
from typing import Any, Optional

import sympy

import torch
from torch._prims_common import is_boolean_dtype, is_integer_dtype


@dataclass
class TypedExpr:
    expr: sympy.Expr
    dtype: torch.dtype


class SymPyOps:
    @staticmethod
    def identity(value):
        return value

    @staticmethod
    def constant(value, dtype):
        if is_boolean_dtype(dtype) or is_integer_dtype(dtype):
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
    rep: Any
    expr: Optional[TypedExpr] = None


class IndexPropagation:
    def __init__(self, inner):
        self._inner = inner
        self.id_iter = itertools.count()

    def new_symbol(self, expr: TypedExpr):
        return IndexPropVar(f"index_tmp{next(self.id_iter)}", expr)

    def unwrap(self, a):
        if not isinstance(a, IndexPropVar):
            return a
        if a.expr is None:
            return a.rep

        expr, dtype = a.expr.expr, a.expr.dtype
        if isinstance(expr, sympy.Integer):
            return self._inner.constant(int(expr), dtype)
        elif not expr.free_symbols:
            return self._inner.constant(float(expr), dtype)
        return self._inner.index_expr(expr, dtype)

    def fallback(self, name, args, kwargs):
        new_args = [self.unwrap(a) for a in args]
        new_kwargs = {k: self.unwrap(v) for k, v in kwargs.items()}
        return IndexPropVar(getattr(self._inner, name)(*new_args, **new_kwargs))

    def propagate_sympy(self, name, args, kwargs):
        def unwrap(a):
            if not isinstance(a, IndexPropVar):
                return a
            return a.expr

        new_args = [unwrap(a) for a in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)
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
        if isinstance(index, IndexPropVar) and index.expr is not None:
            return index.expr.expr
        return self.fallback("indirect_indexing", (index, size), {}).rep
