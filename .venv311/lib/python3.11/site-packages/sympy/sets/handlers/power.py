from sympy.core import Basic, Expr
from sympy.core.function import Lambda
from sympy.core.numbers import oo, Infinity, NegativeInfinity, Zero, Integer
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import (Max, Min)
from sympy.sets.fancysets import ImageSet
from sympy.sets.setexpr import set_div
from sympy.sets.sets import Set, Interval, FiniteSet, Union
from sympy.multipledispatch import Dispatcher


_x, _y = symbols("x y")


_set_pow = Dispatcher('_set_pow')


@_set_pow.register(Basic, Basic)
def _(x, y):
    return None

@_set_pow.register(Set, Set)
def _(x, y):
    return ImageSet(Lambda((_x, _y), (_x ** _y)), x, y)

@_set_pow.register(Expr, Expr)
def _(x, y):
    return x**y

@_set_pow.register(Interval, Zero)
def _(x, z):
    return FiniteSet(S.One)

@_set_pow.register(Interval, Integer)
def _(x, exponent):
    """
    Powers in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    s1 = x.start**exponent
    s2 = x.end**exponent
    if ((s2 > s1) if exponent > 0 else (x.end > -x.start)) == True:
        left_open = x.left_open
        right_open = x.right_open
        # TODO: handle unevaluated condition.
        sleft = s2
    else:
        # TODO: `s2 > s1` could be unevaluated.
        left_open = x.right_open
        right_open = x.left_open
        sleft = s1

    if x.start.is_positive:
        return Interval(
            Min(s1, s2),
            Max(s1, s2), left_open, right_open)
    elif x.end.is_negative:
        return Interval(
            Min(s1, s2),
            Max(s1, s2), left_open, right_open)

    # Case where x.start < 0 and x.end > 0:
    if exponent.is_odd:
        if exponent.is_negative:
            if x.start.is_zero:
                return Interval(s2, oo, x.right_open)
            if x.end.is_zero:
                return Interval(-oo, s1, True, x.left_open)
            return Union(Interval(-oo, s1, True, x.left_open), Interval(s2, oo, x.right_open))
        else:
            return Interval(s1, s2, x.left_open, x.right_open)
    elif exponent.is_even:
        if exponent.is_negative:
            if x.start.is_zero:
                return Interval(s2, oo, x.right_open)
            if x.end.is_zero:
                return Interval(s1, oo, x.left_open)
            return Interval(0, oo)
        else:
            return Interval(S.Zero, sleft, S.Zero not in x, left_open)

@_set_pow.register(Interval, Infinity)
def _(b, e):
    # TODO: add logic for open intervals?
    if b.start.is_nonnegative:
        if b.end < 1:
            return FiniteSet(S.Zero)
        if b.start > 1:
            return FiniteSet(S.Infinity)
        return Interval(0, oo)
    elif b.end.is_negative:
        if b.start > -1:
            return FiniteSet(S.Zero)
        if b.end < -1:
            return FiniteSet(-oo, oo)
        return Interval(-oo, oo)
    else:
        if b.start > -1:
            if b.end < 1:
                return FiniteSet(S.Zero)
            return Interval(0, oo)
        return Interval(-oo, oo)

@_set_pow.register(Interval, NegativeInfinity)
def _(b, e):
    return _set_pow(set_div(S.One, b), oo)
