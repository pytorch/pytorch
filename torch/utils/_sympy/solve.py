import sympy

from typing import Dict, Optional, Tuple, Type

from torch.utils._sympy.functions import FloorDiv


def try_solve(expr: sympy.Basic, thing: sympy.Basic, trials: int =5) -> Optional[Tuple[sympy.Rel, sympy.Basic]]:
    MIRROR: Dict[Type[sympy.Basic], Type[sympy.Rel]] = {
        sympy.Eq: sympy.Eq,
        sympy.Ne: sympy.Ne,
        sympy.Ge: sympy.Le,
        sympy.Gt: sympy.Lt,
        sympy.Le: sympy.Ge,
        sympy.Lt: sympy.Gt,
    }

    if not (isinstance(expr, sympy.Rel) and type(expr) in MIRROR):
        return

    # Here, we try considering both LHS and RHS by mirroring the
    # original expression: a < b ==> b > a
    for e in (expr, MIRROR[type(expr)](expr.rhs, expr.lhs)):
        if e is None:
            continue

        assert isinstance(e, sympy.Rel)

        if not e.lhs.has(thing):
            continue

        for _ in range(trials):
            e = _try_isolate_lhs(e, thing)

        if isinstance(e, sympy.Rel) and e.lhs == thing:
            return e, e.rhs

# The transformations below only work if b is positive.
# Note: we only have this information for constants.
def _is_floordiv_with_positive_denominator(e) -> bool:
    if not isinstance(e, FloorDiv):
        return False
    number = e.args[1]
    return isinstance(number, sympy.Integer) and bool(number > 0)


def _try_isolate_lhs(expr: sympy.Basic, thing: sympy.Basic) -> sympy.Basic:
    if not isinstance(expr, sympy.Rel) or not expr.lhs.has(thing):
        return expr

    lhs, rhs = expr.args

    # Move any constants in the left-hand side to the right-hand side.
    lhs_const = sum([a for a in lhs.args if isinstance(a, sympy.Integer)]) if isinstance(lhs, sympy.Add) else 0
    lhs = lhs - lhs_const  # type: ignore[arg-type]
    rhs = rhs - lhs_const  # type: ignore[arg-type]

    ################################################################################
    # left-hand side is FloorDiv
    ################################################################################
    # a // b == expr
    # => a >= (b * expr) and a < ((b + 1) * expr)
    if isinstance(expr, sympy.Eq) and isinstance(lhs, FloorDiv):
        numerator, denominator = lhs.args
        return sympy.And(
            sympy.Ge(numerator, (rhs * denominator)),  # type: ignore[arg-type]
            sympy.Lt(numerator, ((rhs + 1) * denominator))  # type: ignore[arg-type]
        )
    # a // b != expr
    # => a < (b * expr) or a >= ((b + 1) * expr)
    if isinstance(expr, sympy.Ne) and isinstance(lhs, FloorDiv):
        numerator, denominator = lhs.args
        return sympy.Or(
            sympy.Lt(numerator, (rhs * denominator)),  # type: ignore[arg-type]
            sympy.Ge(numerator, ((rhs + 1) * denominator))  # type: ignore[arg-type]
        )
    # a // b > expr  => a >= (b + 1) * expr
    # a // b >= expr => a >= b * expr
    if isinstance(expr, (sympy.Gt, sympy.Ge)) and _is_floordiv_with_positive_denominator(lhs):
        quotient = rhs if isinstance(expr, sympy.Ge) else (rhs + 1)  # type: ignore[arg-type]
        return sympy.Ge(lhs.args[0], (quotient * lhs.args[1]))  # type: ignore[arg-type]
    # a // b < expr  => a < b * expr
    # a // b <= expr => a < (b + 1) * expr
    if isinstance(expr, (sympy.Lt, sympy.Le)) and _is_floordiv_with_positive_denominator(lhs):
        quotient = rhs if isinstance(expr, sympy.Lt) else (rhs + 1)  # type: ignore[arg-type]
        return sympy.Lt(lhs.args[0], (quotient * lhs.args[1]))  # type: ignore[arg-type]

    return type(expr)(lhs, rhs)

