from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, tan
from sympy.polys.rootoftools import RootOf, CRootOf
from sympy import Integers
from sympy.sets.sets import Interval
from sympy.sets.fancysets import ImageSet
from sympy.core.function import Lambda

from sympy.core.expr import unchanged
from sympy.testing.pytest import XFAIL, raises

x = Symbol('x')
i = Symbol('i', imaginary=True)
y = Symbol('y', real=True)
k, n = symbols('k,n', integer=True)


def test_floor():

    assert floor(nan) is nan

    assert floor(oo) is oo
    assert floor(-oo) is -oo
    assert floor(zoo) is zoo

    assert floor(0) == 0

    assert floor(1) == 1
    assert floor(-1) == -1

    assert floor(E) == 2
    assert floor(-E) == -3

    assert floor(2*E) == 5
    assert floor(-2*E) == -6

    assert floor(pi) == 3
    assert floor(-pi) == -4

    assert floor(S.Half) == 0
    assert floor(Rational(-1, 2)) == -1

    assert floor(Rational(7, 3)) == 2
    assert floor(Rational(-7, 3)) == -3
    assert floor(-Rational(7, 3)) == -3

    assert floor(Float(17.0)) == 17
    assert floor(-Float(17.0)) == -17

    assert floor(Float(7.69)) == 7
    assert floor(-Float(7.69)) == -8

    assert floor(I) == I
    assert floor(-I) == -I
    e = floor(i)
    assert e.func is floor and e.args[0] == i

    assert floor(oo*I) == oo*I
    assert floor(-oo*I) == -oo*I
    assert floor(exp(I*pi/4)*oo) == exp(I*pi/4)*oo

    assert floor(2*I) == 2*I
    assert floor(-2*I) == -2*I

    assert floor(I/2) == 0
    assert floor(-I/2) == -I

    assert floor(E + 17) == 19
    assert floor(pi + 2) == 5

    assert floor(E + pi) == 5
    assert floor(I + pi) == 3 + I

    assert floor(floor(pi)) == 3
    assert floor(floor(y)) == floor(y)
    assert floor(floor(x)) == floor(x)

    assert unchanged(floor, x)
    assert unchanged(floor, 2*x)
    assert unchanged(floor, k*x)

    assert floor(k) == k
    assert floor(2*k) == 2*k
    assert floor(k*n) == k*n

    assert unchanged(floor, k/2)

    assert unchanged(floor, x + y)

    assert floor(x + 3) == floor(x) + 3
    assert floor(x + k) == floor(x) + k

    assert floor(y + 3) == floor(y) + 3
    assert floor(y + k) == floor(y) + k

    assert floor(3 + I*y + pi) == 6 + floor(y)*I

    assert floor(k + n) == k + n

    assert unchanged(floor, x*I)
    assert floor(k*I) == k*I

    assert floor(Rational(23, 10) - E*I) == 2 - 3*I

    assert floor(sin(1)) == 0
    assert floor(sin(-1)) == -1

    assert floor(exp(2)) == 7

    assert floor(log(8)/log(2)) != 2
    assert int(floor(log(8)/log(2)).evalf(chop=True)) == 3

    assert floor(factorial(50)/exp(1)) == \
        11188719610782480504630258070757734324011354208865721592720336800

    assert (floor(y) < y) == False
    assert (floor(y) <= y) == True
    assert (floor(y) > y) == False
    assert (floor(y) >= y) == False
    assert (floor(x) <= x).is_Relational  # x could be non-real
    assert (floor(x) > x).is_Relational
    assert (floor(x) <= y).is_Relational  # arg is not same as rhs
    assert (floor(x) > y).is_Relational
    assert (floor(y) <= oo) == True
    assert (floor(y) < oo) == True
    assert (floor(y) >= -oo) == True
    assert (floor(y) > -oo) == True

    assert floor(y).rewrite(frac) == y - frac(y)
    assert floor(y).rewrite(ceiling) == -ceiling(-y)
    assert floor(y).rewrite(frac).subs(y, -pi) == floor(-pi)
    assert floor(y).rewrite(frac).subs(y, E) == floor(E)
    assert floor(y).rewrite(ceiling).subs(y, E) == -ceiling(-E)
    assert floor(y).rewrite(ceiling).subs(y, -pi) == -ceiling(pi)

    assert Eq(floor(y), y - frac(y))
    assert Eq(floor(y), -ceiling(-y))

    neg = Symbol('neg', negative=True)
    nn = Symbol('nn', nonnegative=True)
    pos = Symbol('pos', positive=True)
    np = Symbol('np', nonpositive=True)

    assert (floor(neg) < 0) == True
    assert (floor(neg) <= 0) == True
    assert (floor(neg) > 0) == False
    assert (floor(neg) >= 0) == False
    assert (floor(neg) <= -1) == True
    assert (floor(neg) >= -3) == (neg >= -3)
    assert (floor(neg) < 5) == (neg < 5)

    assert (floor(nn) < 0) == False
    assert (floor(nn) >= 0) == True

    assert (floor(pos) < 0) == False
    assert (floor(pos) <= 0) == (pos < 1)
    assert (floor(pos) > 0) == (pos >= 1)
    assert (floor(pos) >= 0) == True
    assert (floor(pos) >= 3) == (pos >= 3)

    assert (floor(np) <= 0) == True
    assert (floor(np) > 0) == False

    assert floor(neg).is_negative == True
    assert floor(neg).is_nonnegative == False
    assert floor(nn).is_negative == False
    assert floor(nn).is_nonnegative == True
    assert floor(pos).is_negative == False
    assert floor(pos).is_nonnegative == True
    assert floor(np).is_negative is None
    assert floor(np).is_nonnegative is None

    assert (floor(7, evaluate=False) >= 7) == True
    assert (floor(7, evaluate=False) > 7) == False
    assert (floor(7, evaluate=False) <= 7) == True
    assert (floor(7, evaluate=False) < 7) == False

    assert (floor(7, evaluate=False) >= 6) == True
    assert (floor(7, evaluate=False) > 6) == True
    assert (floor(7, evaluate=False) <= 6) == False
    assert (floor(7, evaluate=False) < 6) == False

    assert (floor(7, evaluate=False) >= 8) == False
    assert (floor(7, evaluate=False) > 8) == False
    assert (floor(7, evaluate=False) <= 8) == True
    assert (floor(7, evaluate=False) < 8) == True

    assert (floor(x) <= 5.5) == Le(floor(x), 5.5, evaluate=False)
    assert (floor(x) >= -3.2) == Ge(floor(x), -3.2, evaluate=False)
    assert (floor(x) < 2.9) == Lt(floor(x), 2.9, evaluate=False)
    assert (floor(x) > -1.7) == Gt(floor(x), -1.7, evaluate=False)

    assert (floor(y) <= 5.5) == (y < 6)
    assert (floor(y) >= -3.2) == (y >= -3)
    assert (floor(y) < 2.9) == (y < 3)
    assert (floor(y) > -1.7) == (y >= -1)

    assert (floor(y) <= n) == (y < n + 1)
    assert (floor(y) >= n) == (y >= n)
    assert (floor(y) < n) == (y < n)
    assert (floor(y) > n) == (y >= n + 1)

    assert floor(RootOf(x**3 - 27*x, 2)) == 5


def test_ceiling():

    assert ceiling(nan) is nan

    assert ceiling(oo) is oo
    assert ceiling(-oo) is -oo
    assert ceiling(zoo) is zoo

    assert ceiling(0) == 0

    assert ceiling(1) == 1
    assert ceiling(-1) == -1

    assert ceiling(E) == 3
    assert ceiling(-E) == -2

    assert ceiling(2*E) == 6
    assert ceiling(-2*E) == -5

    assert ceiling(pi) == 4
    assert ceiling(-pi) == -3

    assert ceiling(S.Half) == 1
    assert ceiling(Rational(-1, 2)) == 0

    assert ceiling(Rational(7, 3)) == 3
    assert ceiling(-Rational(7, 3)) == -2

    assert ceiling(Float(17.0)) == 17
    assert ceiling(-Float(17.0)) == -17

    assert ceiling(Float(7.69)) == 8
    assert ceiling(-Float(7.69)) == -7

    assert ceiling(I) == I
    assert ceiling(-I) == -I
    e = ceiling(i)
    assert e.func is ceiling and e.args[0] == i

    assert ceiling(oo*I) == oo*I
    assert ceiling(-oo*I) == -oo*I
    assert ceiling(exp(I*pi/4)*oo) == exp(I*pi/4)*oo

    assert ceiling(2*I) == 2*I
    assert ceiling(-2*I) == -2*I

    assert ceiling(I/2) == I
    assert ceiling(-I/2) == 0

    assert ceiling(E + 17) == 20
    assert ceiling(pi + 2) == 6

    assert ceiling(E + pi) == 6
    assert ceiling(I + pi) == I + 4

    assert ceiling(ceiling(pi)) == 4
    assert ceiling(ceiling(y)) == ceiling(y)
    assert ceiling(ceiling(x)) == ceiling(x)

    assert unchanged(ceiling, x)
    assert unchanged(ceiling, 2*x)
    assert unchanged(ceiling, k*x)

    assert ceiling(k) == k
    assert ceiling(2*k) == 2*k
    assert ceiling(k*n) == k*n

    assert unchanged(ceiling, k/2)

    assert unchanged(ceiling, x + y)

    assert ceiling(x + 3) == ceiling(x) + 3
    assert ceiling(x + 3.0) == ceiling(x) + 3
    assert ceiling(x + 3.0*I) == ceiling(x) + 3*I
    assert ceiling(x + k) == ceiling(x) + k

    assert ceiling(y + 3) == ceiling(y) + 3
    assert ceiling(y + k) == ceiling(y) + k

    assert ceiling(3 + pi + y*I) == 7 + ceiling(y)*I

    assert ceiling(k + n) == k + n

    assert unchanged(ceiling, x*I)
    assert ceiling(k*I) == k*I

    assert ceiling(Rational(23, 10) - E*I) == 3 - 2*I

    assert ceiling(sin(1)) == 1
    assert ceiling(sin(-1)) == 0

    assert ceiling(exp(2)) == 8

    assert ceiling(-log(8)/log(2)) != -2
    assert int(ceiling(-log(8)/log(2)).evalf(chop=True)) == -3

    assert ceiling(factorial(50)/exp(1)) == \
        11188719610782480504630258070757734324011354208865721592720336801

    assert (ceiling(y) >= y) == True
    assert (ceiling(y) > y) == False
    assert (ceiling(y) < y) == False
    assert (ceiling(y) <= y) == False
    assert (ceiling(x) >= x).is_Relational  # x could be non-real
    assert (ceiling(x) < x).is_Relational
    assert (ceiling(x) >= y).is_Relational  # arg is not same as rhs
    assert (ceiling(x) < y).is_Relational
    assert (ceiling(y) >= -oo) == True
    assert (ceiling(y) > -oo) == True
    assert (ceiling(y) <= oo) == True
    assert (ceiling(y) < oo) == True

    assert ceiling(y).rewrite(floor) == -floor(-y)
    assert ceiling(y).rewrite(frac) == y + frac(-y)
    assert ceiling(y).rewrite(floor).subs(y, -pi) == -floor(pi)
    assert ceiling(y).rewrite(floor).subs(y, E) == -floor(-E)
    assert ceiling(y).rewrite(frac).subs(y, pi) == ceiling(pi)
    assert ceiling(y).rewrite(frac).subs(y, -E) == ceiling(-E)

    assert Eq(ceiling(y), y + frac(-y))
    assert Eq(ceiling(y), -floor(-y))

    neg = Symbol('neg', negative=True)
    nn = Symbol('nn', nonnegative=True)
    pos = Symbol('pos', positive=True)
    np = Symbol('np', nonpositive=True)

    assert (ceiling(neg) <= 0) == True
    assert (ceiling(neg) < 0) == (neg <= -1)
    assert (ceiling(neg) > 0) == False
    assert (ceiling(neg) >= 0) == (neg > -1)
    assert (ceiling(neg) > -3) == (neg > -3)
    assert (ceiling(neg) <= 10) == (neg <= 10)

    assert (ceiling(nn) < 0) == False
    assert (ceiling(nn) >= 0) == True

    assert (ceiling(pos) < 0) == False
    assert (ceiling(pos) <= 0) == False
    assert (ceiling(pos) > 0) == True
    assert (ceiling(pos) >= 0) == True
    assert (ceiling(pos) >= 1) == True
    assert (ceiling(pos) > 5) == (pos > 5)

    assert (ceiling(np) <= 0) == True
    assert (ceiling(np) > 0) == False

    assert ceiling(neg).is_positive == False
    assert ceiling(neg).is_nonpositive == True
    assert ceiling(nn).is_positive is None
    assert ceiling(nn).is_nonpositive is None
    assert ceiling(pos).is_positive == True
    assert ceiling(pos).is_nonpositive == False
    assert ceiling(np).is_positive == False
    assert ceiling(np).is_nonpositive == True

    assert (ceiling(7, evaluate=False) >= 7) == True
    assert (ceiling(7, evaluate=False) > 7) == False
    assert (ceiling(7, evaluate=False) <= 7) == True
    assert (ceiling(7, evaluate=False) < 7) == False

    assert (ceiling(7, evaluate=False) >= 6) == True
    assert (ceiling(7, evaluate=False) > 6) == True
    assert (ceiling(7, evaluate=False) <= 6) == False
    assert (ceiling(7, evaluate=False) < 6) == False

    assert (ceiling(7, evaluate=False) >= 8) == False
    assert (ceiling(7, evaluate=False) > 8) == False
    assert (ceiling(7, evaluate=False) <= 8) == True
    assert (ceiling(7, evaluate=False) < 8) == True

    assert (ceiling(x) <= 5.5) == Le(ceiling(x), 5.5, evaluate=False)
    assert (ceiling(x) >= -3.2) == Ge(ceiling(x), -3.2, evaluate=False)
    assert (ceiling(x) < 2.9) == Lt(ceiling(x), 2.9, evaluate=False)
    assert (ceiling(x) > -1.7) == Gt(ceiling(x), -1.7, evaluate=False)

    assert (ceiling(y) <= 5.5) == (y <= 5)
    assert (ceiling(y) >= -3.2) == (y > -4)
    assert (ceiling(y) < 2.9) == (y <= 2)
    assert (ceiling(y) > -1.7) == (y > -2)

    assert (ceiling(y) <= n) == (y <= n)
    assert (ceiling(y) >= n) == (y > n - 1)
    assert (ceiling(y) < n) == (y <= n - 1)
    assert (ceiling(y) > n) == (y > n)

    assert ceiling(RootOf(x**3 - 27*x, 2)) == 6
    s = ImageSet(Lambda(n, n + (CRootOf(x**5 - x**2 + 1, 0))), Integers)
    f = CRootOf(x**5 - x**2 + 1, 0)
    s = ImageSet(Lambda(n, n + f), Integers)
    assert s.intersect(Interval(-10, 10)) == {i + f for i in range(-9, 11)}


def test_frac():
    assert isinstance(frac(x), frac)
    assert frac(oo) == AccumBounds(0, 1)
    assert frac(-oo) == AccumBounds(0, 1)
    assert frac(zoo) is nan

    assert frac(n) == 0
    assert frac(nan) is nan
    assert frac(Rational(4, 3)) == Rational(1, 3)
    assert frac(-Rational(4, 3)) == Rational(2, 3)
    assert frac(Rational(-4, 3)) == Rational(2, 3)

    r = Symbol('r', real=True)
    assert frac(I*r) == I*frac(r)
    assert frac(1 + I*r) == I*frac(r)
    assert frac(0.5 + I*r) == 0.5 + I*frac(r)
    assert frac(n + I*r) == I*frac(r)
    assert frac(n + I*k) == 0
    assert unchanged(frac, x + I*x)
    assert frac(x + I*n) == frac(x)

    assert frac(x).rewrite(floor) == x - floor(x)
    assert frac(x).rewrite(ceiling) == x + ceiling(-x)
    assert frac(y).rewrite(floor).subs(y, pi) == frac(pi)
    assert frac(y).rewrite(floor).subs(y, -E) == frac(-E)
    assert frac(y).rewrite(ceiling).subs(y, -pi) == frac(-pi)
    assert frac(y).rewrite(ceiling).subs(y, E) == frac(E)

    assert Eq(frac(y), y - floor(y))
    assert Eq(frac(y), y + ceiling(-y))

    r = Symbol('r', real=True)
    p_i = Symbol('p_i', integer=True, positive=True)
    n_i = Symbol('p_i', integer=True, negative=True)
    np_i = Symbol('np_i', integer=True, nonpositive=True)
    nn_i = Symbol('nn_i', integer=True, nonnegative=True)
    p_r = Symbol('p_r', positive=True)
    n_r = Symbol('n_r', negative=True)
    np_r = Symbol('np_r', real=True, nonpositive=True)
    nn_r = Symbol('nn_r', real=True, nonnegative=True)

    # Real frac argument, integer rhs
    assert frac(r) <= p_i
    assert not frac(r) <= n_i
    assert (frac(r) <= np_i).has(Le)
    assert (frac(r) <= nn_i).has(Le)
    assert frac(r) < p_i
    assert not frac(r) < n_i
    assert not frac(r) < np_i
    assert (frac(r) < nn_i).has(Lt)
    assert not frac(r) >= p_i
    assert frac(r) >= n_i
    assert frac(r) >= np_i
    assert (frac(r) >= nn_i).has(Ge)
    assert not frac(r) > p_i
    assert frac(r) > n_i
    assert (frac(r) > np_i).has(Gt)
    assert (frac(r) > nn_i).has(Gt)

    assert not Eq(frac(r), p_i)
    assert not Eq(frac(r), n_i)
    assert Eq(frac(r), np_i).has(Eq)
    assert Eq(frac(r), nn_i).has(Eq)

    assert Ne(frac(r), p_i)
    assert Ne(frac(r), n_i)
    assert Ne(frac(r), np_i).has(Ne)
    assert Ne(frac(r), nn_i).has(Ne)


    # Real frac argument, real rhs
    assert (frac(r) <= p_r).has(Le)
    assert not frac(r) <= n_r
    assert (frac(r) <= np_r).has(Le)
    assert (frac(r) <= nn_r).has(Le)
    assert (frac(r) < p_r).has(Lt)
    assert not frac(r) < n_r
    assert not frac(r) < np_r
    assert (frac(r) < nn_r).has(Lt)
    assert (frac(r) >= p_r).has(Ge)
    assert frac(r) >= n_r
    assert frac(r) >= np_r
    assert (frac(r) >= nn_r).has(Ge)
    assert (frac(r) > p_r).has(Gt)
    assert frac(r) > n_r
    assert (frac(r) > np_r).has(Gt)
    assert (frac(r) > nn_r).has(Gt)

    assert not Eq(frac(r), n_r)
    assert Eq(frac(r), p_r).has(Eq)
    assert Eq(frac(r), np_r).has(Eq)
    assert Eq(frac(r), nn_r).has(Eq)

    assert Ne(frac(r), p_r).has(Ne)
    assert Ne(frac(r), n_r)
    assert Ne(frac(r), np_r).has(Ne)
    assert Ne(frac(r), nn_r).has(Ne)

    # Real frac argument, +/- oo rhs
    assert frac(r) < oo
    assert frac(r) <= oo
    assert not frac(r) > oo
    assert not frac(r) >= oo

    assert not frac(r) < -oo
    assert not frac(r) <= -oo
    assert frac(r) > -oo
    assert frac(r) >= -oo

    assert frac(r) < 1
    assert frac(r) <= 1
    assert not frac(r) > 1
    assert not frac(r) >= 1

    assert not frac(r) < 0
    assert (frac(r) <= 0).has(Le)
    assert (frac(r) > 0).has(Gt)
    assert frac(r) >= 0

    # Some test for numbers
    assert frac(r) <= sqrt(2)
    assert (frac(r) <= sqrt(3) - sqrt(2)).has(Le)
    assert not frac(r) <= sqrt(2) - sqrt(3)
    assert not frac(r) >= sqrt(2)
    assert (frac(r) >= sqrt(3) - sqrt(2)).has(Ge)
    assert frac(r) >= sqrt(2) - sqrt(3)

    assert not Eq(frac(r), sqrt(2))
    assert Eq(frac(r), sqrt(3) - sqrt(2)).has(Eq)
    assert not Eq(frac(r), sqrt(2) - sqrt(3))
    assert Ne(frac(r), sqrt(2))
    assert Ne(frac(r), sqrt(3) - sqrt(2)).has(Ne)
    assert Ne(frac(r), sqrt(2) - sqrt(3))

    assert frac(p_i, evaluate=False).is_zero
    assert frac(p_i, evaluate=False).is_finite
    assert frac(p_i, evaluate=False).is_integer
    assert frac(p_i, evaluate=False).is_real
    assert frac(r).is_finite
    assert frac(r).is_real
    assert frac(r).is_zero is None
    assert frac(r).is_integer is None

    assert frac(oo).is_finite
    assert frac(oo).is_real


def test_series():
    x, y = symbols('x,y')
    assert floor(x).nseries(x, y, 100) == floor(y)
    assert ceiling(x).nseries(x, y, 100) == ceiling(y)
    assert floor(x).nseries(x, pi, 100) == 3
    assert ceiling(x).nseries(x, pi, 100) == 4
    assert floor(x).nseries(x, 0, 100) == 0
    assert ceiling(x).nseries(x, 0, 100) == 1
    assert floor(-x).nseries(x, 0, 100) == -1
    assert ceiling(-x).nseries(x, 0, 100) == 0


def test_issue_14355():
    # This test checks the leading term and series for the floor and ceil
    # function when arg0 evaluates to S.NaN.
    assert floor((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir = 1) == -2
    assert floor((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir = -1) == -1
    assert floor((cos(x) - 1)/x).as_leading_term(x, cdir = 1) == -1
    assert floor((cos(x) - 1)/x).as_leading_term(x, cdir = -1) == 0
    assert floor(sin(x)/x).as_leading_term(x, cdir = 1) == 0
    assert floor(sin(x)/x).as_leading_term(x, cdir = -1) == 0
    assert floor(-tan(x)/x).as_leading_term(x, cdir = 1) == -2
    assert floor(-tan(x)/x).as_leading_term(x, cdir = -1) == -2
    assert floor(sin(x)/x/3).as_leading_term(x, cdir = 1) == 0
    assert floor(sin(x)/x/3).as_leading_term(x, cdir = -1) == 0
    assert ceiling((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir = 1) == -1
    assert ceiling((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir = -1) == 0
    assert ceiling((cos(x) - 1)/x).as_leading_term(x, cdir = 1) == 0
    assert ceiling((cos(x) - 1)/x).as_leading_term(x, cdir = -1) == 1
    assert ceiling(sin(x)/x).as_leading_term(x, cdir = 1) == 1
    assert ceiling(sin(x)/x).as_leading_term(x, cdir = -1) == 1
    assert ceiling(-tan(x)/x).as_leading_term(x, cdir = 1) == -1
    assert ceiling(-tan(x)/x).as_leading_term(x, cdir = 1) == -1
    assert ceiling(sin(x)/x/3).as_leading_term(x, cdir = 1) == 1
    assert ceiling(sin(x)/x/3).as_leading_term(x, cdir = -1) == 1
    # test for series
    assert floor(sin(x)/x).series(x, 0, 100, cdir = 1) == 0
    assert floor(sin(x)/x).series(x, 0, 100, cdir = 1) == 0
    assert floor((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir = 1) == -2
    assert floor((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir = -1) == -1
    assert ceiling(sin(x)/x).series(x, 0, 100, cdir = 1) == 1
    assert ceiling(sin(x)/x).series(x, 0, 100, cdir = -1) == 1
    assert ceiling((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir = 1) == -1
    assert ceiling((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir = -1) == 0


def test_frac_leading_term():
    assert frac(x).as_leading_term(x) == x
    assert frac(x).as_leading_term(x, cdir = 1) == x
    assert frac(x).as_leading_term(x, cdir = -1) == 1
    assert frac(x + S.Half).as_leading_term(x, cdir = 1) == S.Half
    assert frac(x + S.Half).as_leading_term(x, cdir = -1) == S.Half
    assert frac(-2*x + 1).as_leading_term(x, cdir = 1) == S.One
    assert frac(-2*x + 1).as_leading_term(x, cdir = -1) == -2*x
    assert frac(sin(x) + 5).as_leading_term(x, cdir = 1) == x
    assert frac(sin(x) + 5).as_leading_term(x, cdir = -1) == S.One
    assert frac(sin(x**2) + 5).as_leading_term(x, cdir = 1) == x**2
    assert frac(sin(x**2) + 5).as_leading_term(x, cdir = -1) == x**2


@XFAIL
def test_issue_4149():
    assert floor(3 + pi*I + y*I) == 3 + floor(pi + y)*I
    assert floor(3*I + pi*I + y*I) == floor(3 + pi + y)*I
    assert floor(3 + E + pi*I + y*I) == 5 + floor(pi + y)*I


def test_issue_21651():
    k = Symbol('k', positive=True, integer=True)
    exp = 2*2**(-k)
    assert isinstance(floor(exp), floor)


def test_issue_11207():
    assert floor(floor(x)) == floor(x)
    assert floor(ceiling(x)) == ceiling(x)
    assert ceiling(floor(x)) == floor(x)
    assert ceiling(ceiling(x)) == ceiling(x)


def test_nested_floor_ceiling():
    assert floor(-floor(ceiling(x**3)/y)) == -floor(ceiling(x**3)/y)
    assert ceiling(-floor(ceiling(x**3)/y)) == -floor(ceiling(x**3)/y)
    assert floor(ceiling(-floor(x**Rational(7, 2)/y))) == -floor(x**Rational(7, 2)/y)
    assert -ceiling(-ceiling(floor(x)/y)) == ceiling(floor(x)/y)

def test_issue_18689():
    assert floor(floor(floor(x)) + 3) == floor(x) + 3
    assert ceiling(ceiling(ceiling(x)) + 1) == ceiling(x) + 1
    assert ceiling(ceiling(floor(x)) + 3) == floor(x) + 3

def test_issue_18421():
    assert floor(float(0)) is S.Zero
    assert ceiling(float(0)) is S.Zero

def test_issue_25230():
    a = Symbol('a', real = True)
    b = Symbol('b', positive = True)
    c = Symbol('c', negative = True)
    raises(NotImplementedError, lambda: floor(x/a).as_leading_term(x, cdir = 1))
    raises(NotImplementedError, lambda: ceiling(x/a).as_leading_term(x, cdir = 1))
    assert floor(x/b).as_leading_term(x, cdir = 1) == 0
    assert floor(x/b).as_leading_term(x, cdir = -1) == -1
    assert floor(x/c).as_leading_term(x, cdir = 1) == -1
    assert floor(x/c).as_leading_term(x, cdir = -1) == 0
    assert ceiling(x/b).as_leading_term(x, cdir = 1) == 1
    assert ceiling(x/b).as_leading_term(x, cdir = -1) == 0
    assert ceiling(x/c).as_leading_term(x, cdir = 1) == 0
    assert ceiling(x/c).as_leading_term(x, cdir = -1) == 1
