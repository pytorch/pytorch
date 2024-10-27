"""
This module contains query handlers responsible for calculus queries:
infinitesimal, finite, etc.
"""

from sympy.assumptions import Q, ask
from sympy.core import Add, Mul, Pow, Symbol
from sympy.core.numbers import (NegativeInfinity, GoldenRatio,
    Infinity, Exp1, ComplexInfinity, ImaginaryUnit, NaN, Number, Pi, E,
    TribonacciConstant)
from sympy.functions import cos, exp, log, sign, sin
from sympy.logic.boolalg import conjuncts

from ..predicates.calculus import (FinitePredicate, InfinitePredicate,
    PositiveInfinitePredicate, NegativeInfinitePredicate)


# FinitePredicate


@FinitePredicate.register(Symbol)
def _(expr, assumptions):
    """
    Handles Symbol.
    """
    if expr.is_finite is not None:
        return expr.is_finite
    if Q.finite(expr) in conjuncts(assumptions):
        return True
    return None

@FinitePredicate.register(Add)
def _(expr, assumptions):
    """
    Return True if expr is bounded, False if not and None if unknown.

    Truth Table:

    +-------+-----+-----------+-----------+
    |       |     |           |           |
    |       |  B  |     U     |     ?     |
    |       |     |           |           |
    +-------+-----+---+---+---+---+---+---+
    |       |     |   |   |   |   |   |   |
    |       |     |'+'|'-'|'x'|'+'|'-'|'x'|
    |       |     |   |   |   |   |   |   |
    +-------+-----+---+---+---+---+---+---+
    |       |     |           |           |
    |   B   |  B  |     U     |     ?     |
    |       |     |           |           |
    +---+---+-----+---+---+---+---+---+---+
    |   |   |     |   |   |   |   |   |   |
    |   |'+'|     | U | ? | ? | U | ? | ? |
    |   |   |     |   |   |   |   |   |   |
    |   +---+-----+---+---+---+---+---+---+
    |   |   |     |   |   |   |   |   |   |
    | U |'-'|     | ? | U | ? | ? | U | ? |
    |   |   |     |   |   |   |   |   |   |
    |   +---+-----+---+---+---+---+---+---+
    |   |   |     |           |           |
    |   |'x'|     |     ?     |     ?     |
    |   |   |     |           |           |
    +---+---+-----+---+---+---+---+---+---+
    |       |     |           |           |
    |   ?   |     |           |     ?     |
    |       |     |           |           |
    +-------+-----+-----------+---+---+---+

        * 'B' = Bounded

        * 'U' = Unbounded

        * '?' = unknown boundedness

        * '+' = positive sign

        * '-' = negative sign

        * 'x' = sign unknown

        * All Bounded -> True

        * 1 Unbounded and the rest Bounded -> False

        * >1 Unbounded, all with same known sign -> False

        * Any Unknown and unknown sign -> None

        * Else -> None

    When the signs are not the same you can have an undefined
    result as in oo - oo, hence 'bounded' is also undefined.
    """
    sign = -1  # sign of unknown or infinite
    result = True
    for arg in expr.args:
        _bounded = ask(Q.finite(arg), assumptions)
        if _bounded:
            continue
        s = ask(Q.extended_positive(arg), assumptions)
        # if there has been more than one sign or if the sign of this arg
        # is None and Bounded is None or there was already
        # an unknown sign, return None
        if sign != -1 and s != sign or \
                s is None and None in (_bounded, sign):
            return None
        else:
            sign = s
        # once False, do not change
        if result is not False:
            result = _bounded
    return result

@FinitePredicate.register(Mul)
def _(expr, assumptions):
    """
    Return True if expr is bounded, False if not and None if unknown.

    Truth Table:

    +---+---+---+--------+
    |   |   |   |        |
    |   | B | U |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+
    |   |   |   |   |    |
    |   |   |   | s | /s |
    |   |   |   |   |    |
    +---+---+---+---+----+
    |   |   |   |        |
    | B | B | U |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+
    |   |   |   |   |    |
    | U |   | U | U | ?  |
    |   |   |   |   |    |
    +---+---+---+---+----+
    |   |   |   |        |
    | ? |   |   |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+

        * B = Bounded

        * U = Unbounded

        * ? = unknown boundedness

        * s = signed (hence nonzero)

        * /s = not signed
    """
    result = True
    for arg in expr.args:
        _bounded = ask(Q.finite(arg), assumptions)
        if _bounded:
            continue
        elif _bounded is None:
            if result is None:
                return None
            if ask(Q.extended_nonzero(arg), assumptions) is None:
                return None
            if result is not False:
                result = None
        else:
            result = False
    return result

@FinitePredicate.register(Pow)
def _(expr, assumptions):
    """
    * Unbounded ** NonZero -> Unbounded

    * Bounded ** Bounded -> Bounded

    * Abs()<=1 ** Positive -> Bounded

    * Abs()>=1 ** Negative -> Bounded

    * Otherwise unknown
    """
    if expr.base == E:
        return ask(Q.finite(expr.exp), assumptions)

    base_bounded = ask(Q.finite(expr.base), assumptions)
    exp_bounded = ask(Q.finite(expr.exp), assumptions)
    if base_bounded is None and exp_bounded is None:  # Common Case
        return None
    if base_bounded is False and ask(Q.extended_nonzero(expr.exp), assumptions):
        return False
    if base_bounded and exp_bounded:
        return True
    if (abs(expr.base) <= 1) == True and ask(Q.extended_positive(expr.exp), assumptions):
        return True
    if (abs(expr.base) >= 1) == True and ask(Q.extended_negative(expr.exp), assumptions):
        return True
    if (abs(expr.base) >= 1) == True and exp_bounded is False:
        return False
    return None

@FinitePredicate.register(exp)
def _(expr, assumptions):
    return ask(Q.finite(expr.exp), assumptions)

@FinitePredicate.register(log)
def _(expr, assumptions):
    # After complex -> finite fact is registered to new assumption system,
    # querying Q.infinite may be removed.
    if ask(Q.infinite(expr.args[0]), assumptions):
        return False
    return ask(~Q.zero(expr.args[0]), assumptions)

@FinitePredicate.register_many(cos, sin, Number, Pi, Exp1, GoldenRatio,
    TribonacciConstant, ImaginaryUnit, sign)
def _(expr, assumptions):
    return True

@FinitePredicate.register_many(ComplexInfinity, Infinity, NegativeInfinity)
def _(expr, assumptions):
    return False

@FinitePredicate.register(NaN)
def _(expr, assumptions):
    return None


# InfinitePredicate


@InfinitePredicate.register_many(ComplexInfinity, Infinity, NegativeInfinity)
def _(expr, assumptions):
    return True


# PositiveInfinitePredicate


@PositiveInfinitePredicate.register(Infinity)
def _(expr, assumptions):
    return True


@PositiveInfinitePredicate.register_many(NegativeInfinity, ComplexInfinity)
def _(expr, assumptions):
    return False


# NegativeInfinitePredicate


@NegativeInfinitePredicate.register(NegativeInfinity)
def _(expr, assumptions):
    return True


@NegativeInfinitePredicate.register_many(Infinity, ComplexInfinity)
def _(expr, assumptions):
    return False
