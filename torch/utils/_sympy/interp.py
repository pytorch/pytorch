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
from .functions import CleanDiv, FloorDiv, Mod, ModularIndexing


# TODO: Dedupe this with SYMPY_INTERP


@functools.lru_cache(None)
def handlers():
    from torch.fx.experimental.symbolic_shapes import Pow, TrueDiv

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
        TrueDiv: "truediv",
        FloorDiv: "floordiv",
        CleanDiv: "div",
        sympy.Add: "add",
        sympy.Mul: "mul",
        Pow: "pow",
        sympy.Pow: "pow",
        Mod: "mod",
        sympy.Mod: "mod",
        sympy.Abs: "abs",
        sympy.log: "log",
        sympy.exp: "exp",
        sympy.floor: "floor",
        sympy.ceiling: "ceil",
        sympy.Min: "minimum",
        sympy.Max: "maximum",
        ModularIndexing: "modular_indexing",
    }
    return HANDLERS


ASSOCIATIVE_OPS = {"minimum", "maximum", "mul", "add", "and_", "or_"}


def sympy_interp(
    analysis, env: Dict[sympy.Symbol, Any], expr: Union[sympy.Expr, SympyBoolean]
):
    # Handle base cases
    # TODO: not really sure if I'm passing the right dtype here
    # TODO: wouldn't it be better to pass the sympy expression through
    # sometimes?
    if isinstance(expr, sympy.Integer):
        return analysis.constant(int(expr), torch.int64)
    elif isinstance(expr, sympy.Number):
        return analysis.constant(float(expr), torch.double)
    elif isinstance(expr, BooleanAtom):
        return analysis.constant(bool(expr), torch.bool)
    elif isinstance(expr, sympy.Symbol):
        return env[expr]

    # Special cases
    if isinstance(expr, sympy.Pow) and isinstance(
        expr.args[1], sympy.core.numbers.Half
    ):
        return analysis.sqrt(sympy_interp(analysis, env, expr.args[0]))

    # Recursive case
    args = [sympy_interp(analysis, env, arg) for arg in expr.args]  # type: ignore[arg-type]
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
