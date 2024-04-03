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

import sympy

from typing_extensions import TypeAlias

import torch
from torch._prims_common import dtype_to_type, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing, Where
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .utils import generate_assert, sympy_subs

from .virtualized import V


_ExprType = Union[sympy.Expr, float, int, bool]


def _is_constant(val: _ExprType):
    if isinstance(val, sympy.Basic):
        return val.is_number
    return isinstance(val, (int, float, bool))


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
        value: TypedExpr, dtype: torch.dtype, src_dtype: Optional[torch.dtype] = None
    ) -> TypedExpr:
        return TypedExpr(value.expr, dtype)

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

    def __init__(self, inner: Any, iter_ranges: Dict[sympy.Symbol, sympy.Expr]):
        self._inner = inner

        def upper_bound(v):
            return bound_sympy(v).upper if isinstance(v, sympy.Expr) else v

        self.iter_ranges = iter_ranges
        self.var_ranges = {
            k: ValueRanges(0, upper_bound(v) - 1) for k, v in iter_ranges.items()
        }

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

    def indirect_indexing(
        self, index: Union[Any, IndexPropVar], size: Any, check: bool = True
    ) -> Any:
        if isinstance(index, IndexPropVar) and index.is_symbolic:
            # FIXME(Lezcano) Here we are trying to fit a square peg in a round hole
            # We need the bounds to implement a fine analysis, but the bounds have
            # not been computed yet. We should either push this transformation down
            # the stack, or move the computation of the bounds before this pass

            # If we are turning a indirect indexing into direct, we need to wrap it
            # and perhaps generate asserts
            expr = sympy.sympify(index.value.expr)
            bounds = bound_sympy(expr, self.var_ranges)

            if bounds.is_singleton():
                expr = bounds.lower

            def wrap_expr(expr):
                # Positive, negative, mixed
                if bounds.lower >= 0:
                    return expr
                elif bounds.upper < 0:
                    return expr + size
                else:
                    return Where(expr < 0, expr + size, expr)

            # To continue through this path, we need to prove that the bounds are correct
            # We do our best effort, otherwise we fallback.
            # We considered adding a `check_bounds` function that lazily adds the asserts
            # while keeping this optimisation on. This has the issue that loads without indirect
            # indexing are lifted to the top of the kernel, so we would need to either lift the asserts
            # or mark these loads not to be lifted

            # Trivial case
            if not generate_assert(check):
                return wrap_expr(expr)

            if isinstance(size, int):
                if bounds.issubset(ValueRanges(-size, size - 1)):
                    return wrap_expr(expr)

                # direct indexing in disguise and it's out of range
                if bounds.is_singleton():
                    raise IndexError(
                        "index {expr} is out of bounds for dimension with size {size}"
                    )
            else:
                # Dynamic shapes case

                # If we note when are bounds tight, we could take this path whenever
                # the bounds are tight, for example, x[torch.arange(4)] even if x has dynamic shapes
                if bounds.is_singleton():
                    # expr \in [-size, size)
                    if expr > 1:
                        V.graph.sizevars.guard_lt(expr, size)
                    elif expr < 0:
                        V.graph.sizevars.guard_lt(-size - 1, expr)
                    return wrap_expr(expr)

                # We try to handle symbolically cases like s0 - x0 - 1 < s0 whenever 0 <= x0 < s0
                # The value range analysis is not good enough to prove these systems of inequalities

                # If they don't have the same symbols, you could have something like 4 < s0 and we
                # would not be able to use the upper bound of s0 just on the LHS.
                expr_upper = sympy_subs(expr, self.iter_ranges)
                if (
                    isinstance(expr_upper, sympy.Expr)
                    and expr_upper.free_symbols == size.free_symbols
                ):
                    # nb(lezcano) I think this transformation is not entirely sound
                    bound_size = bound_sympy(size).upper
                    if bounds.issubset(ValueRanges(-bound_size, bound_size - 1)):
                        return wrap_expr(expr)
        return self.fallback("indirect_indexing", (index, size, check), {}).value
