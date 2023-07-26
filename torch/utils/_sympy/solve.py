from typing import Dict, Optional, Tuple, Type

import sympy

from torch.utils._sympy.functions import FloorDiv


# Tries to simplify 'expr', so as to leave only 'thing' in the left-hand side.
# Returns a tuple of:
#   1. The simplified expression
#   2. The expression on the right-hand side
# Returns 'None' if it can't reach a state where the only thing in the left
# hand side is 'thing'.
def try_solve(
    expr: sympy.Basic, thing: sympy.Basic, trials: int = 5
) -> Optional[Tuple[sympy.Rel, sympy.Basic]]:
    MIRROR: Dict[Type[sympy.Basic], Type[sympy.Rel]] = {
        sympy.Eq: sympy.Eq,
        sympy.Ne: sympy.Ne,
        sympy.Ge: sympy.Le,
        sympy.Gt: sympy.Lt,
        sympy.Le: sympy.Ge,
        sympy.Lt: sympy.Gt,
    }

    if not (isinstance(expr, sympy.Rel) and type(expr) in MIRROR):
        return None

    # Here, we try considering both LHS and RHS by mirroring the
    # original expression: a < b ==> b > a
    for e in (expr, MIRROR[type(expr)](expr.rhs, expr.lhs)):
        if e is None:
            continue

        assert isinstance(e, sympy.Rel)

        if not e.lhs.has(thing):
            continue

        for _ in range(trials):
            e = _try_isolate_lhs(e, thing)  # type: ignore[assignment]

        if isinstance(e, sympy.Rel) and e.lhs == thing:
            return e, e.rhs

    return None


def _try_isolate_lhs(expr: sympy.Basic, thing: sympy.Basic) -> sympy.Basic:
    if not isinstance(expr, sympy.Rel) or not expr.lhs.has(thing):
        return expr

    lhs, rhs = expr.args

    # Move any constants in the left-hand side to the right-hand side.
    lhs_const = (
        sum([a for a in lhs.args if isinstance(a, sympy.Integer)])
        if isinstance(lhs, sympy.Add)
        else 0
    )
    lhs = lhs - lhs_const  # type: ignore[arg-type]
    rhs = rhs - lhs_const  # type: ignore[arg-type]

    # Divide both sides by the factors that don't contain thing.
    if isinstance(lhs, sympy.Mul):
        other = sympy.Mul(*[a for a in lhs.args if not a.has(thing)])
        # TODO: mirror the operation if 'other' is negative.
        if isinstance(expr, (sympy.Eq, sympy.Ne)) or other.is_positive:
            lhs = lhs / other
            rhs = rhs / other

    ################################################################################
    # left-hand side is FloorDiv
    ################################################################################
    #
    # Given the expression: a // b op c
    # where 'op' is a relational operation, these rules only work if:
    #   - b > 0
    #   - c is an integer
    if isinstance(lhs, FloorDiv) and lhs.divisor.is_positive and rhs.is_integer:
        # a // b == expr
        # => a >= (b * expr) and a < (b * (expr + 1))
        if isinstance(expr, sympy.Eq):
            numerator, denominator = lhs.args
            return sympy.And(
                sympy.Ge(numerator, (rhs * denominator)),  # type: ignore[arg-type]
                sympy.Lt(numerator, ((rhs + 1) * denominator)),  # type: ignore[arg-type]
            )
        # a // b != expr
        # => a < (b * expr) or a >= (b * (expr + 1))
        if isinstance(expr, sympy.Ne):
            numerator, denominator = lhs.args
            return sympy.Or(
                sympy.Lt(numerator, (rhs * denominator)),  # type: ignore[arg-type]
                sympy.Ge(numerator, ((rhs + 1) * denominator)),  # type: ignore[arg-type]
            )
        # The transformations below only work if b is positive.
        # Note: we only have this information for constants.
        # a // b > expr  => a >= b * (expr + 1)
        # a // b >= expr => a >= b * expr
        if isinstance(expr, (sympy.Gt, sympy.Ge)):
            quotient = rhs if isinstance(expr, sympy.Ge) else (rhs + 1)  # type: ignore[arg-type]
            return sympy.Ge(lhs.args[0], (quotient * lhs.args[1]))  # type: ignore[arg-type]
        # a // b < expr  => a < b * expr
        # a // b <= expr => a < b * (expr + 1)
        if isinstance(expr, (sympy.Lt, sympy.Le)):
            quotient = rhs if isinstance(expr, sympy.Lt) else (rhs + 1)  # type: ignore[arg-type]
            return sympy.Lt(lhs.args[0], (quotient * lhs.args[1]))  # type: ignore[arg-type]

    return type(expr)(lhs, rhs)
