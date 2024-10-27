from sympy.core.numbers import oo, Infinity, NegativeInfinity
from sympy.core.singleton import S
from sympy.core import Basic, Expr
from sympy.multipledispatch import Dispatcher
from sympy.sets import Interval, FiniteSet



# XXX: The functions in this module are clearly not tested and are broken in a
# number of ways.

_set_add = Dispatcher('_set_add')
_set_sub = Dispatcher('_set_sub')


@_set_add.register(Basic, Basic)
def _(x, y):
    return None


@_set_add.register(Expr, Expr)
def _(x, y):
    return x+y


@_set_add.register(Interval, Interval)
def _(x, y):
    """
    Additions in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    return Interval(x.start + y.start, x.end + y.end,
                    x.left_open or y.left_open, x.right_open or y.right_open)


@_set_add.register(Interval, Infinity)
def _(x, y):
    if x.start is S.NegativeInfinity:
        return Interval(-oo, oo)
    return FiniteSet({S.Infinity})

@_set_add.register(Interval, NegativeInfinity)
def _(x, y):
    if x.end is S.Infinity:
        return Interval(-oo, oo)
    return FiniteSet({S.NegativeInfinity})


@_set_sub.register(Basic, Basic)
def _(x, y):
    return None


@_set_sub.register(Expr, Expr)
def _(x, y):
    return x-y


@_set_sub.register(Interval, Interval)
def _(x, y):
    """
    Subtractions in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    return Interval(x.start - y.end, x.end - y.start,
                    x.left_open or y.right_open, x.right_open or y.left_open)


@_set_sub.register(Interval, Infinity)
def _(x, y):
    if x.start is S.NegativeInfinity:
        return Interval(-oo, oo)
    return FiniteSet(-oo)

@_set_sub.register(Interval, NegativeInfinity)
def _(x, y):
    if x.start is S.NegativeInfinity:
        return Interval(-oo, oo)
    return FiniteSet(-oo)
