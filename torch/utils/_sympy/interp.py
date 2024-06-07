"""
This is a simple interpreter for Sympy expressions that dispatches to
classes following the torch._inductor.virtualized calling convention.
For directness, the interpreter takes the handler directly rather than
consulting the TLS.  It does not use most of the methods on the full
handler; only those with corresponding Sympy expressions.  To see an example
of a full handler, see torch.utils._sympy.value_ranges.ValueRangeAnalysis.
"""

import functools
from typing import Any, Dict, Union

import sympy
from sympy.logic.boolalg import Boolean as SympyBoolean, BooleanAtom

import torch
from .functions import (
    CeilToInt,
    CleanDiv,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    FloorToInt,
    IntTrueDiv,
    IsNonOverlappingAndDenseIndicator,
    Mod,
    ModularIndexing,
    PowByNatural,
    PythonMod,
    RoundDecimal,
    RoundToInt,
    ToFloat,
    TruncToFloat,
    TruncToInt,
    Where,
)


# TODO: Dedupe this with SYMPY_INTERP


@functools.lru_cache(None)
def handlers():
    # TODO add CeilDiv (it doesn't appear in the index_expr)

    # TODO default to some decompositions if the interpreter doesn't have them
    # like decomposing ModularIndexing or implementing Le(a,b) as Ge(b, a)

    HANDLERS = {
        sympy.Or: "or_",
        sympy.And: "and_",
        sympy.Eq: "eq",
        sympy.Ne: "ne",
        sympy.Lt: "lt",
        sympy.Gt: "gt",
        sympy.Le: "le",
        sympy.Ge: "ge",
        sympy.Not: "not_",
        IntTrueDiv: "int_truediv",
        FloatTrueDiv: "truediv",
        FloorDiv: "floordiv",
        CleanDiv: "floordiv",  # TODO: hmm?
        TruncToFloat: "trunc",
        Where: "where",
        sympy.Add: "add",
        sympy.Mul: "mul",
        FloatPow: "pow",
        PowByNatural: "pow_by_natural",
        # sympy simplifies x * x into Pow(x, 2), so we need to handle this.
        # Do NOT use builtin Pow for floats
        # TODO: There is a hazard here, if we have float * float it will
        # also get turned into Pow(float, 2) but we don't want this because
        # pow_by_natural is assumed to only be integers.  Probably the fix is
        # to add a FloatMul to impede this optimization
        sympy.Pow: "pow_by_natural",
        Mod: "mod",
        PythonMod: "mod",  # TODO: this is wrong
        # TODO: Inductor can generate these, but it's ill-specified which
        # semantics were intended here.  Needs to be cleaned up along with
        # FloorDiv in a bigger cleanup
        sympy.Mod: "mod",
        sympy.Abs: "abs",
        sympy.log: "log",
        sympy.exp: "exp",
        sympy.Min: "minimum",
        sympy.Max: "maximum",
        ModularIndexing: "modular_indexing",
        sympy.functions.elementary.piecewise.ExprCondPair: "expr_cond_pair",
        sympy.Piecewise: "piecewise",
        IsNonOverlappingAndDenseIndicator: "is_non_overlapping_and_dense_indicator",
        RoundDecimal: "round_decimal",
    }
    for name in ["cos", "sin", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan"]:
        HANDLERS[getattr(sympy, name)] = name

    return HANDLERS


ASSOCIATIVE_OPS = {"minimum", "maximum", "mul", "add", "and_", "or_"}


def sympy_interp(
    analysis,
    env: Dict[sympy.Symbol, Any],
    expr: Union[sympy.Expr, SympyBoolean],
    *,
    index_dtype=torch.int64,
):
    # Handle base cases
    dtype = None
    if isinstance(expr, BooleanAtom):
        dtype = torch.bool
    elif isinstance(expr, sympy.Integer):
        dtype = torch.int64
    elif isinstance(expr, sympy.Number):
        dtype = torch.double

    if dtype is not None:
        return analysis.constant(expr, dtype)
    elif isinstance(expr, sympy.Symbol):
        return env[expr]

    # Special cases
    if isinstance(expr, sympy.Pow) and isinstance(
        expr.args[1], sympy.core.numbers.Half
    ):
        return analysis.sqrt(sympy_interp(analysis, env, expr.args[0]))
    if isinstance(expr, ToFloat):
        return analysis.to_dtype(
            sympy_interp(analysis, env, expr.args[0]), torch.float64
        )

    # Recursive case
    args = [sympy_interp(analysis, env, arg) for arg in expr.args]  # type: ignore[arg-type]

    # These handlers are special because they take an extra dtype argument
    # specifying what they should convert to, and we need to appropriately set
    # this up when we convert from Sympy.  A reasonable default when you
    # are translating is to conservatively do int64, and then narrow these
    # arguments later when you discover you can narrow the index range.  But
    # if you already know that 32-bit indexing is OK, you can directly do the
    # sympy translation with index_dtype=torch.int32
    INDEX_DTYPE_HANDLERS = {
        TruncToInt: "trunc_to_int",
        sympy.floor: "floor_to_int",
        sympy.ceiling: "ceil_to_int",
        FloorToInt: "floor_to_int",
        CeilToInt: "ceil_to_int",
        RoundToInt: "round_to_int",
    }
    if (handler_name := INDEX_DTYPE_HANDLERS.get(expr.func)) is not None:
        return getattr(analysis, handler_name)(*args, index_dtype)

    if hasattr(expr.func, "_torch_handler_name"):
        handler_name = expr.func._torch_handler_name
    else:
        handler_name = handlers()[expr.func]
    handler = getattr(analysis, handler_name)
    if handler_name in ASSOCIATIVE_OPS:
        assert len(args) > 1
        acc = handler(args[0], args[1])
        for i in range(2, len(args)):
            acc = handler(acc, args[i])
        return acc
    else:
        return handler(*args)
