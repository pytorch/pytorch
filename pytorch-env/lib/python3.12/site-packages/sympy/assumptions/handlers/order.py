"""
Handlers related to order relations: positive, negative, etc.
"""

from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Mul, Pow
from sympy.core.logic import fuzzy_not, fuzzy_and, fuzzy_or
from sympy.core.numbers import E, ImaginaryUnit, NaN, I, pi
from sympy.functions import Abs, acos, acot, asin, atan, exp, factorial, log
from sympy.matrices import Determinant, Trace
from sympy.matrices.expressions.matexpr import MatrixElement

from sympy.multipledispatch import MDNotImplementedError

from ..predicates.order import (NegativePredicate, NonNegativePredicate,
    NonZeroPredicate, ZeroPredicate, NonPositivePredicate, PositivePredicate,
    ExtendedNegativePredicate, ExtendedNonNegativePredicate,
    ExtendedNonPositivePredicate, ExtendedNonZeroPredicate,
    ExtendedPositivePredicate,)


# NegativePredicate

def _NegativePredicate_number(expr, assumptions):
    r, i = expr.as_real_imag()
    # If the imaginary part can symbolically be shown to be zero then
    # we just evaluate the real part; otherwise we evaluate the imaginary
    # part to see if it actually evaluates to zero and if it does then
    # we make the comparison between the real part and zero.
    if not i:
        r = r.evalf(2)
        if r._prec != 1:
            return r < 0
    else:
        i = i.evalf(2)
        if i._prec != 1:
            if i != 0:
                return False
            r = r.evalf(2)
            if r._prec != 1:
                return r < 0

@NegativePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _NegativePredicate_number(expr, assumptions)

@NegativePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_negative
    if ret is None:
        raise MDNotImplementedError
    return ret

@NegativePredicate.register(Add)
def _(expr, assumptions):
    """
    Positive + Positive -> Positive,
    Negative + Negative -> Negative
    """
    if expr.is_number:
        return _NegativePredicate_number(expr, assumptions)

    r = ask(Q.real(expr), assumptions)
    if r is not True:
        return r

    nonpos = 0
    for arg in expr.args:
        if ask(Q.negative(arg), assumptions) is not True:
            if ask(Q.positive(arg), assumptions) is False:
                nonpos += 1
            else:
                break
    else:
        if nonpos < len(expr.args):
            return True

@NegativePredicate.register(Mul)
def _(expr, assumptions):
    if expr.is_number:
        return _NegativePredicate_number(expr, assumptions)
    result = None
    for arg in expr.args:
        if result is None:
            result = False
        if ask(Q.negative(arg), assumptions):
            result = not result
        elif ask(Q.positive(arg), assumptions):
            pass
        else:
            return
    return result

@NegativePredicate.register(Pow)
def _(expr, assumptions):
    """
    Real ** Even -> NonNegative
    Real ** Odd  -> same_as_base
    NonNegative ** Positive -> NonNegative
    """
    if expr.base == E:
        # Exponential is always positive:
        if ask(Q.real(expr.exp), assumptions):
            return False
        return

    if expr.is_number:
        return _NegativePredicate_number(expr, assumptions)
    if ask(Q.real(expr.base), assumptions):
        if ask(Q.positive(expr.base), assumptions):
            if ask(Q.real(expr.exp), assumptions):
                return False
        if ask(Q.even(expr.exp), assumptions):
            return False
        if ask(Q.odd(expr.exp), assumptions):
            return ask(Q.negative(expr.base), assumptions)

@NegativePredicate.register_many(Abs, ImaginaryUnit)
def _(expr, assumptions):
    return False

@NegativePredicate.register(exp)
def _(expr, assumptions):
    if ask(Q.real(expr.exp), assumptions):
        return False
    raise MDNotImplementedError


# NonNegativePredicate

@NonNegativePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        notnegative = fuzzy_not(_NegativePredicate_number(expr, assumptions))
        if notnegative:
            return ask(Q.real(expr), assumptions)
        else:
            return notnegative

@NonNegativePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_nonnegative
    if ret is None:
        raise MDNotImplementedError
    return ret


# NonZeroPredicate

@NonZeroPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_nonzero
    if ret is None:
        raise MDNotImplementedError
    return ret

@NonZeroPredicate.register(Basic)
def _(expr, assumptions):
    if ask(Q.real(expr)) is False:
        return False
    if expr.is_number:
        # if there are no symbols just evalf
        i = expr.evalf(2)
        def nonz(i):
            if i._prec != 1:
                return i != 0
        return fuzzy_or(nonz(i) for i in i.as_real_imag())

@NonZeroPredicate.register(Add)
def _(expr, assumptions):
    if all(ask(Q.positive(x), assumptions) for x in expr.args) \
            or all(ask(Q.negative(x), assumptions) for x in expr.args):
        return True

@NonZeroPredicate.register(Mul)
def _(expr, assumptions):
    for arg in expr.args:
        result = ask(Q.nonzero(arg), assumptions)
        if result:
            continue
        return result
    return True

@NonZeroPredicate.register(Pow)
def _(expr, assumptions):
    return ask(Q.nonzero(expr.base), assumptions)

@NonZeroPredicate.register(Abs)
def _(expr, assumptions):
    return ask(Q.nonzero(expr.args[0]), assumptions)

@NonZeroPredicate.register(NaN)
def _(expr, assumptions):
    return None


# ZeroPredicate

@ZeroPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_zero
    if ret is None:
        raise MDNotImplementedError
    return ret

@ZeroPredicate.register(Basic)
def _(expr, assumptions):
    return fuzzy_and([fuzzy_not(ask(Q.nonzero(expr), assumptions)),
        ask(Q.real(expr), assumptions)])

@ZeroPredicate.register(Mul)
def _(expr, assumptions):
    # TODO: This should be deducible from the nonzero handler
    return fuzzy_or(ask(Q.zero(arg), assumptions) for arg in expr.args)


# NonPositivePredicate

@NonPositivePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_nonpositive
    if ret is None:
        raise MDNotImplementedError
    return ret

@NonPositivePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        notpositive = fuzzy_not(_PositivePredicate_number(expr, assumptions))
        if notpositive:
            return ask(Q.real(expr), assumptions)
        else:
            return notpositive


# PositivePredicate

def _PositivePredicate_number(expr, assumptions):
    r, i = expr.as_real_imag()
    # If the imaginary part can symbolically be shown to be zero then
    # we just evaluate the real part; otherwise we evaluate the imaginary
    # part to see if it actually evaluates to zero and if it does then
    # we make the comparison between the real part and zero.
    if not i:
        r = r.evalf(2)
        if r._prec != 1:
            return r > 0
    else:
        i = i.evalf(2)
        if i._prec != 1:
            if i != 0:
                return False
            r = r.evalf(2)
            if r._prec != 1:
                return r > 0

@PositivePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_positive
    if ret is None:
        raise MDNotImplementedError
    return ret

@PositivePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)

@PositivePredicate.register(Mul)
def _(expr, assumptions):
    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)
    result = True
    for arg in expr.args:
        if ask(Q.positive(arg), assumptions):
            continue
        elif ask(Q.negative(arg), assumptions):
            result = result ^ True
        else:
            return
    return result

@PositivePredicate.register(Add)
def _(expr, assumptions):
    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)

    r = ask(Q.real(expr), assumptions)
    if r is not True:
        return r

    nonneg = 0
    for arg in expr.args:
        if ask(Q.positive(arg), assumptions) is not True:
            if ask(Q.negative(arg), assumptions) is False:
                nonneg += 1
            else:
                break
    else:
        if nonneg < len(expr.args):
            return True

@PositivePredicate.register(Pow)
def _(expr, assumptions):
    if expr.base == E:
        if ask(Q.real(expr.exp), assumptions):
            return True
        if ask(Q.imaginary(expr.exp), assumptions):
            return ask(Q.even(expr.exp/(I*pi)), assumptions)
        return

    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)
    if ask(Q.positive(expr.base), assumptions):
        if ask(Q.real(expr.exp), assumptions):
            return True
    if ask(Q.negative(expr.base), assumptions):
        if ask(Q.even(expr.exp), assumptions):
            return True
        if ask(Q.odd(expr.exp), assumptions):
            return False

@PositivePredicate.register(exp)
def _(expr, assumptions):
    if ask(Q.real(expr.exp), assumptions):
        return True
    if ask(Q.imaginary(expr.exp), assumptions):
        return ask(Q.even(expr.exp/(I*pi)), assumptions)

@PositivePredicate.register(log)
def _(expr, assumptions):
    r = ask(Q.real(expr.args[0]), assumptions)
    if r is not True:
        return r
    if ask(Q.positive(expr.args[0] - 1), assumptions):
        return True
    if ask(Q.negative(expr.args[0] - 1), assumptions):
        return False

@PositivePredicate.register(factorial)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.integer(x) & Q.positive(x), assumptions):
            return True

@PositivePredicate.register(ImaginaryUnit)
def _(expr, assumptions):
    return False

@PositivePredicate.register(Abs)
def _(expr, assumptions):
    return ask(Q.nonzero(expr), assumptions)

@PositivePredicate.register(Trace)
def _(expr, assumptions):
    if ask(Q.positive_definite(expr.arg), assumptions):
        return True

@PositivePredicate.register(Determinant)
def _(expr, assumptions):
    if ask(Q.positive_definite(expr.arg), assumptions):
        return True

@PositivePredicate.register(MatrixElement)
def _(expr, assumptions):
    if (expr.i == expr.j
            and ask(Q.positive_definite(expr.parent), assumptions)):
        return True

@PositivePredicate.register(atan)
def _(expr, assumptions):
    return ask(Q.positive(expr.args[0]), assumptions)

@PositivePredicate.register(asin)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.positive(x) & Q.nonpositive(x - 1), assumptions):
        return True
    if ask(Q.negative(x) & Q.nonnegative(x + 1), assumptions):
        return False

@PositivePredicate.register(acos)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.nonpositive(x - 1) & Q.nonnegative(x + 1), assumptions):
        return True

@PositivePredicate.register(acot)
def _(expr, assumptions):
    return ask(Q.real(expr.args[0]), assumptions)

@PositivePredicate.register(NaN)
def _(expr, assumptions):
    return None


# ExtendedNegativePredicate

@ExtendedNegativePredicate.register(object)
def _(expr, assumptions):
    return ask(Q.negative(expr) | Q.negative_infinite(expr), assumptions)


# ExtendedPositivePredicate

@ExtendedPositivePredicate.register(object)
def _(expr, assumptions):
    return ask(Q.positive(expr) | Q.positive_infinite(expr), assumptions)


# ExtendedNonZeroPredicate

@ExtendedNonZeroPredicate.register(object)
def _(expr, assumptions):
    return ask(
        Q.negative_infinite(expr) | Q.negative(expr) | Q.positive(expr) | Q.positive_infinite(expr),
        assumptions)


# ExtendedNonPositivePredicate

@ExtendedNonPositivePredicate.register(object)
def _(expr, assumptions):
    return ask(
        Q.negative_infinite(expr) | Q.negative(expr) | Q.zero(expr),
        assumptions)


# ExtendedNonNegativePredicate

@ExtendedNonNegativePredicate.register(object)
def _(expr, assumptions):
    return ask(
        Q.zero(expr) | Q.positive(expr) | Q.positive_infinite(expr),
        assumptions)
