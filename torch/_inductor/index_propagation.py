# mypy: allow-untyped-defs
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
from typing import Any, Callable, Dict, Literal, Optional, overload, Tuple, Union
from typing_extensions import TypeAlias

import sympy

import torch
from torch._prims_common import dtype_to_type, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing, Where
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from .sizevars import evaluate_expr
from .utils import generate_assert
from .virtualized import V


_ExprType = Union[sympy.Expr, float, int, bool]


def _is_constant(val: _ExprType):
    if isinstance(val, sympy.Basic):
        return val.is_number
    return isinstance(val, (int, float, bool))


def upper_bound(val: _ExprType):
    return bound_sympy(val).upper if isinstance(val, sympy.Expr) else val


@dataclass
class TypedExpr:
    """A SymPy expression with associated type"""

    expr: _ExprType
    dtype: torch.dtype

    def is_constant(self):
        return _is_constant(self.expr)

    def __post_init__(self):
        if _is_constant(self.expr):
            self.expr = dtype_to_type(self.dtype)(self.expr)


class SymPyOps:
    """An ops handler where all IR values are SymPy expressions

    When a value cannot be represented as a SymPy expression, the method is
    either not defined, or returns NotImplemented

    """

    @staticmethod
    def identity(value: Any) -> Any:
        return value

    @staticmethod
    def constant(value: Union[int, float, bool], dtype: torch.dtype) -> TypedExpr:
        return TypedExpr(value, dtype)

    @staticmethod
    def index_expr(value: Union[sympy.Expr, int], dtype: torch.dtype) -> TypedExpr:
        return TypedExpr(value, dtype)

    @staticmethod
    def to_dtype(
        value: TypedExpr,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types: bool = False,
    ) -> TypedExpr:
        return TypedExpr(value.expr, dtype)

    @staticmethod
    def abs(x: TypedExpr) -> TypedExpr:
        return TypedExpr(abs(x.expr), x.dtype)  # type: ignore[arg-type]

    @staticmethod
    def square(x: TypedExpr) -> TypedExpr:
        return TypedExpr(x.expr * x.expr, x.dtype)

    @staticmethod
    def add(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr + y.expr, result_type)

    @staticmethod
    def sub(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr - y.expr, result_type)

    @staticmethod
    def mul(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr * y.expr, result_type)

    @staticmethod
    def neg(x: TypedExpr) -> TypedExpr:
        return TypedExpr(-x.expr, x.dtype)

    @staticmethod
    def floordiv(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        return TypedExpr(FloorDiv(x.expr, y.expr), result_type)

    @staticmethod
    def mod(x: TypedExpr, y: TypedExpr) -> Optional[TypedExpr]:
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        result_expr = ModularIndexing(x.expr, sympy.Integer(1), y.expr)
        return TypedExpr(result_expr, result_type)

    @staticmethod
    def remainder(x: TypedExpr, y: TypedExpr) -> Optional[TypedExpr]:
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        x_expr = sympy.sympify(x.expr)
        y_expr = sympy.sympify(y.expr)
        # In these cases, remainder in Python == remainder in C++, so this transformation
        # is sound
        if (
            x_expr.is_nonnegative is not None
            and x_expr.is_nonnegative == y_expr.is_positive
        ):
            result_expr = ModularIndexing(x.expr, sympy.Integer(1), y.expr)
            return TypedExpr(result_expr, result_type)
        return NotImplemented

    @staticmethod
    def minimum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(sympy.Min(x.expr, y.expr), result_type)

    @staticmethod
    def maximum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
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


IndexPropResult: TypeAlias = Union[IndexPropVar, Tuple["IndexPropResult", ...]]


class IndexPropagation:
    """Ops wrapper that tries to propagate constant and index_expr values through the computation.

    This aims to maximize the compile time simplification possible, and convert
    indirect indexing from arange into normal static indexing.

    """

    def __init__(
        self,
        inner: Any,
        iter_ranges: Dict[sympy.Symbol, sympy.Expr],
        indirect_var_ranges: Dict[sympy.Symbol, sympy.Expr],
    ) -> None:
        self._inner = inner
        self.shape_env = V.graph.sizevars.shape_env

        var_to_range = {
            k: ValueRanges(0, upper_bound(v) - 1) for k, v in iter_ranges.items()
        }
        self.var_to_range = tuple(
            itertools.chain(self.shape_env.var_to_range.items(), var_to_range.items())
        )
        # NOTE: this is intentionally kept as a reference so the caller can
        # update it in-place
        self.indirect_var_ranges = indirect_var_ranges

        axioms = []
        for x, s in iter_ranges.items():
            axioms.append(0 <= x)
            axioms.append(x < s)
        self.axioms = tuple(axioms) + self.shape_env.get_axioms()

    def materialize_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Any:
        # Construct a new constant/index_expr from the SymPy expression
        if _is_constant(expr):
            val = dtype_to_type(dtype)(expr)
            return self._inner.constant(val, dtype)
        return self._inner.index_expr(expr, dtype)

    def unwrap(self, a: Union[Any, IndexPropVar]) -> Any:
        if isinstance(a, (list, tuple)):
            return tuple(self.unwrap(v) for v in a)

        if not isinstance(a, IndexPropVar):
            return a

        # Prefer the sympy representation if possible
        if a.is_symbolic:
            return self.materialize_expr(a.value.expr, a.value.dtype)

        return a.value

    def wrap(self, a) -> IndexPropResult:
        if isinstance(a, (list, tuple)):
            return tuple(self.wrap(v) for v in a)
        return IndexPropVar(a)

    @overload
    def fallback(
        self,
        name: Literal["indirect_indexing"],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> IndexPropVar:
        ...

    @overload
    def fallback(
        self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> IndexPropResult:
        ...

    def fallback(
        self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> IndexPropResult:
        # Fallback to the wrapped handler
        new_args = [self.unwrap(a) for a in args]
        new_kwargs = {k: self.unwrap(v) for k, v in kwargs.items()}
        return self.wrap(getattr(self._inner, name)(*new_args, **new_kwargs))

    def propagate_sympy(
        self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> IndexPropResult:
        # Build a new SymPy expression from this ops call
        def unwrap(a: Union[Any, IndexPropVar]) -> Any:
            if not isinstance(a, IndexPropVar):
                return a
            return a.value

        new_args = [unwrap(a) for a in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)
        is_valid_expr = new_expr is not NotImplemented and (
            # Inductor doesn't expect floating point in sympy expressions, but
            # allow floating point constants to be propagated
            new_expr.is_constant()
            or new_expr.expr.is_integer
        )
        if not is_valid_expr:
            return self.fallback(name, args, kwargs)
        return IndexPropVar.new_symbolic(new_expr)

    def __getattr__(self, name: str) -> Callable[..., IndexPropResult]:
        def inner(*args: Any, **kwargs: Any) -> IndexPropResult:
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

    def statically_true(self, e):
        """
        Given some iter_ranges, return a function that given an expression, returns whether
        it is true or false using value ranges, guard knowledge and runtime_asserts.

        FIXME I think this may not be entirely right, as we may not be able to use all runtime_asserts
              If this is an issue, just use guards in `self.axioms`.

              The proper way of handling this would be to have a global shape_env that adds
              runtime_asserts as they happen in the code. Then, it shuld be used in SimplifyIndexing
              to perform wrap_expr and in CSEProxy.check_bounds to elide upper / lower bounds also
              for indirect_indexing
        """
        var_to_range = (
            *self.var_to_range,
            *(
                (k, ValueRanges(0, upper_bound(v) - 1))
                for k, v in self.indirect_var_ranges.items()
            ),
        )
        return evaluate_expr(self.shape_env, e, self.axioms, var_to_range)

    def indirect_indexing(
        self,
        index: Union[Any, IndexPropVar],
        size: Any,
        check: bool = True,
        wrap_neg=True,
    ) -> Any:
        if isinstance(index, IndexPropVar) and index.is_symbolic:
            # If we find something we can convert into a direct indexing we do so
            # We still need to (perhaps) wrap the expression and add bound checks
            # We want to do this "constant folding", as we don't allow to fuse
            # kernels into indirect indexing

            expr = sympy.sympify(index.value.expr)

            # TODO Perhaps move this logic to the simplify indexing pass
            def wrap_expr(expr):
                # Positive, negative, mixed
                if self.statically_true(0 <= expr):
                    return expr
                elif self.statically_true(expr < 0):
                    return expr + size
                else:
                    return Where(expr < 0, expr + size, expr)

            # Sometimes it's easier to prove 0 <= expr than the weaker -size <= expr
            can_prove_lower = self.statically_true(0 <= expr) or self.statically_true(
                -size <= expr
            )
            can_prove_upper = self.statically_true(expr < size)
            if wrap_neg:
                expr = wrap_expr(expr)
            if generate_assert(check):
                self.fallback(
                    "check_bounds",
                    (expr, size),
                    dict(lower=not can_prove_lower, upper=not can_prove_upper),
                )
            return expr

        indirect_var = self.fallback(
            "indirect_indexing", (index, size, check, wrap_neg), {}
        ).value
        return indirect_var
