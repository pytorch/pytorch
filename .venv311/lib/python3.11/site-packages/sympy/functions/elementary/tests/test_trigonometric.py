from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.function import (Lambda, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, atan2,
                                                      cos, cot, csc, sec, sin, sinc, tan)
from sympy.functions.special.bessel import (besselj, jn)
from sympy.functions.special.delta_functions import Heaviside
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (cancel, gcd)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.series.series import series
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError, PoleError
from sympy.core.relational import Ne, Eq
from sympy.functions.elementary.piecewise import Piecewise
from sympy.sets.setexpr import SetExpr
from sympy.testing.pytest import XFAIL, slow, raises


x, y, z = symbols('x y z')
r = Symbol('r', real=True)
k, m = symbols('k m', integer=True)
p = Symbol('p', positive=True)
n = Symbol('n', negative=True)
np = Symbol('p', nonpositive=True)
nn = Symbol('n', nonnegative=True)
nz = Symbol('nz', nonzero=True)
ep = Symbol('ep', extended_positive=True)
en = Symbol('en', extended_negative=True)
enp = Symbol('ep', extended_nonpositive=True)
enn = Symbol('en', extended_nonnegative=True)
enz = Symbol('enz', extended_nonzero=True)
a = Symbol('a', algebraic=True)
na = Symbol('na', nonzero=True, algebraic=True)


def test_sin():
    x, y = symbols('x y')
    z = symbols('z', imaginary=True)

    assert sin.nargs == FiniteSet(1)
    assert sin(nan) is nan
    assert sin(zoo) is nan

    assert sin(oo) == AccumBounds(-1, 1)
    assert sin(oo) - sin(oo) == AccumBounds(-2, 2)
    assert sin(oo*I) == oo*I
    assert sin(-oo*I) == -oo*I
    assert 0*sin(oo) is S.Zero
    assert 0/sin(oo) is S.Zero
    assert 0 + sin(oo) == AccumBounds(-1, 1)
    assert 5 + sin(oo) == AccumBounds(4, 6)

    assert sin(0) == 0

    assert sin(z*I) == I*sinh(z)
    assert sin(asin(x)) == x
    assert sin(atan(x)) == x / sqrt(1 + x**2)
    assert sin(acos(x)) == sqrt(1 - x**2)
    assert sin(acot(x)) == 1 / (sqrt(1 + 1 / x**2) * x)
    assert sin(acsc(x)) == 1 / x
    assert sin(asec(x)) == sqrt(1 - 1 / x**2)
    assert sin(atan2(y, x)) == y / sqrt(x**2 + y**2)

    assert sin(pi*I) == sinh(pi)*I
    assert sin(-pi*I) == -sinh(pi)*I
    assert sin(-2*I) == -sinh(2)*I

    assert sin(pi) == 0
    assert sin(-pi) == 0
    assert sin(2*pi) == 0
    assert sin(-2*pi) == 0
    assert sin(-3*10**73*pi) == 0
    assert sin(7*10**103*pi) == 0

    assert sin(pi/2) == 1
    assert sin(-pi/2) == -1
    assert sin(pi*Rational(5, 2)) == 1
    assert sin(pi*Rational(7, 2)) == -1

    ne = symbols('ne', integer=True, even=False)
    e = symbols('e', even=True)
    assert sin(pi*ne/2) == (-1)**(ne/2 - S.Half)
    assert sin(pi*k/2).func == sin
    assert sin(pi*e/2) == 0
    assert sin(pi*k) == 0
    assert sin(pi*k).subs(k, 3) == sin(pi*k/2).subs(k, 6)  # issue 8298

    assert sin(pi/3) == S.Half*sqrt(3)
    assert sin(pi*Rational(-2, 3)) == Rational(-1, 2)*sqrt(3)

    assert sin(pi/4) == S.Half*sqrt(2)
    assert sin(-pi/4) == Rational(-1, 2)*sqrt(2)
    assert sin(pi*Rational(17, 4)) == S.Half*sqrt(2)
    assert sin(pi*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)

    assert sin(pi/6) == S.Half
    assert sin(-pi/6) == Rational(-1, 2)
    assert sin(pi*Rational(7, 6)) == Rational(-1, 2)
    assert sin(pi*Rational(-5, 6)) == Rational(-1, 2)

    assert sin(pi*Rational(1, 5)) == sqrt((5 - sqrt(5)) / 8)
    assert sin(pi*Rational(2, 5)) == sqrt((5 + sqrt(5)) / 8)
    assert sin(pi*Rational(3, 5)) == sin(pi*Rational(2, 5))
    assert sin(pi*Rational(4, 5)) == sin(pi*Rational(1, 5))
    assert sin(pi*Rational(6, 5)) == -sin(pi*Rational(1, 5))
    assert sin(pi*Rational(8, 5)) == -sin(pi*Rational(2, 5))

    assert sin(pi*Rational(-1273, 5)) == -sin(pi*Rational(2, 5))

    assert sin(pi/8) == sqrt((2 - sqrt(2))/4)

    assert sin(pi/10) == Rational(-1, 4) + sqrt(5)/4

    assert sin(pi/12) == -sqrt(2)/4 + sqrt(6)/4
    assert sin(pi*Rational(5, 12)) == sqrt(2)/4 + sqrt(6)/4
    assert sin(pi*Rational(-7, 12)) == -sqrt(2)/4 - sqrt(6)/4
    assert sin(pi*Rational(-11, 12)) == sqrt(2)/4 - sqrt(6)/4

    assert sin(pi*Rational(104, 105)) == sin(pi/105)
    assert sin(pi*Rational(106, 105)) == -sin(pi/105)

    assert sin(pi*Rational(-104, 105)) == -sin(pi/105)
    assert sin(pi*Rational(-106, 105)) == sin(pi/105)

    assert sin(x*I) == sinh(x)*I

    assert sin(k*pi) == 0
    assert sin(17*k*pi) == 0
    assert sin(2*k*pi + 4) == sin(4)
    assert sin(2*k*pi + m*pi + 1) == (-1)**(m + 2*k)*sin(1)

    assert sin(k*pi*I) == sinh(k*pi)*I

    assert sin(r).is_real is True

    assert sin(0, evaluate=False).is_algebraic
    assert sin(a).is_algebraic is None
    assert sin(na).is_algebraic is False
    q = Symbol('q', rational=True)
    assert sin(pi*q).is_algebraic
    qn = Symbol('qn', rational=True, nonzero=True)
    assert sin(qn).is_rational is False
    assert sin(q).is_rational is None  # issue 8653

    assert isinstance(sin( re(x) - im(y)), sin) is True
    assert isinstance(sin(-re(x) + im(y)), sin) is False

    assert sin(SetExpr(Interval(0, 1))) == SetExpr(ImageSet(Lambda(x, sin(x)),
                       Interval(0, 1)))

    for d in list(range(1, 22)) + [60, 85]:
        for n in range(d*2 + 1):
            x = n*pi/d
            e = abs( float(sin(x)) - sin(float(x)) )
            assert e < 1e-12

    assert sin(0, evaluate=False).is_zero is True
    assert sin(k*pi, evaluate=False).is_zero is True

    assert sin(Add(1, -1, evaluate=False), evaluate=False).is_zero is True


def test_sin_cos():
    for d in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 24, 30, 40, 60, 120]:  # list is not exhaustive...
        for n in range(-2*d, d*2):
            x = n*pi/d
            assert sin(x + pi/2) == cos(x), "fails for %d*pi/%d" % (n, d)
            assert sin(x - pi/2) == -cos(x), "fails for %d*pi/%d" % (n, d)
            assert sin(x) == cos(x - pi/2), "fails for %d*pi/%d" % (n, d)
            assert -sin(x) == cos(x + pi/2), "fails for %d*pi/%d" % (n, d)


def test_sin_series():
    assert sin(x).series(x, 0, 9) == \
        x - x**3/6 + x**5/120 - x**7/5040 + O(x**9)


def test_sin_rewrite():
    assert sin(x).rewrite(exp) == -I*(exp(I*x) - exp(-I*x))/2
    assert sin(x).rewrite(tan) == 2*tan(x/2)/(1 + tan(x/2)**2)
    assert sin(x).rewrite(cot) == \
        Piecewise((0, Eq(im(x), 0) & Eq(Mod(x, pi), 0)),
                  (2*cot(x/2)/(cot(x/2)**2 + 1), True))
    assert sin(sinh(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, sinh(3)).n()
    assert sin(cosh(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cosh(3)).n()
    assert sin(tanh(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, tanh(3)).n()
    assert sin(coth(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, coth(3)).n()
    assert sin(sin(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, sin(3)).n()
    assert sin(cos(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cos(3)).n()
    assert sin(tan(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, tan(3)).n()
    assert sin(cot(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cot(3)).n()
    assert sin(log(x)).rewrite(Pow) == I*x**-I / 2 - I*x**I /2
    assert sin(x).rewrite(csc) == 1/csc(x)
    assert sin(x).rewrite(cos) == cos(x - pi / 2, evaluate=False)
    assert sin(x).rewrite(sec) == 1 / sec(x - pi / 2, evaluate=False)
    assert sin(cos(x)).rewrite(Pow) == sin(cos(x))
    assert sin(x).rewrite(besselj) == sqrt(pi*x/2)*besselj(S.Half, x)
    assert sin(x).rewrite(besselj).subs(x, 0) == sin(0)


def _test_extrig(f, i, e):
    from sympy.core.function import expand_trig
    assert unchanged(f, i)
    assert expand_trig(f(i)) == f(i)
    # testing directly instead of with .expand(trig=True)
    # because the other expansions undo the unevaluated Mul
    assert expand_trig(f(Mul(i, 1, evaluate=False))) == e
    assert abs(f(i) - e).n() < 1e-10


def test_sin_expansion():
    # Note: these formulas are not unique.  The ones here come from the
    # Chebyshev formulas.
    assert sin(x + y).expand(trig=True) == sin(x)*cos(y) + cos(x)*sin(y)
    assert sin(x - y).expand(trig=True) == sin(x)*cos(y) - cos(x)*sin(y)
    assert sin(y - x).expand(trig=True) == cos(x)*sin(y) - sin(x)*cos(y)
    assert sin(2*x).expand(trig=True) == 2*sin(x)*cos(x)
    assert sin(3*x).expand(trig=True) == -4*sin(x)**3 + 3*sin(x)
    assert sin(4*x).expand(trig=True) == -8*sin(x)**3*cos(x) + 4*sin(x)*cos(x)
    assert sin(2*pi/17).expand(trig=True) == sin(2*pi/17, evaluate=False)
    assert sin(x+pi/17).expand(trig=True) == sin(pi/17)*cos(x) + cos(pi/17)*sin(x)
    _test_extrig(sin, 2, 2*sin(1)*cos(1))
    _test_extrig(sin, 3, -4*sin(1)**3 + 3*sin(1))


def test_sin_AccumBounds():
    assert sin(AccumBounds(-oo, oo)) == AccumBounds(-1, 1)
    assert sin(AccumBounds(0, oo)) == AccumBounds(-1, 1)
    assert sin(AccumBounds(-oo, 0)) == AccumBounds(-1, 1)
    assert sin(AccumBounds(0, 2*S.Pi)) == AccumBounds(-1, 1)
    assert sin(AccumBounds(0, S.Pi*Rational(3, 4))) == AccumBounds(0, 1)
    assert sin(AccumBounds(S.Pi*Rational(3, 4), S.Pi*Rational(7, 4))) == AccumBounds(-1, sin(S.Pi*Rational(3, 4)))
    assert sin(AccumBounds(S.Pi/4, S.Pi/3)) == AccumBounds(sin(S.Pi/4), sin(S.Pi/3))
    assert sin(AccumBounds(S.Pi*Rational(3, 4), S.Pi*Rational(5, 6))) == AccumBounds(sin(S.Pi*Rational(5, 6)), sin(S.Pi*Rational(3, 4)))


def test_sin_fdiff():
    assert sin(x).fdiff() == cos(x)
    raises(ArgumentIndexError, lambda: sin(x).fdiff(2))


def test_trig_symmetry():
    assert sin(-x) == -sin(x)
    assert cos(-x) == cos(x)
    assert tan(-x) == -tan(x)
    assert cot(-x) == -cot(x)
    assert sin(x + pi) == -sin(x)
    assert sin(x + 2*pi) == sin(x)
    assert sin(x + 3*pi) == -sin(x)
    assert sin(x + 4*pi) == sin(x)
    assert sin(x - 5*pi) == -sin(x)
    assert cos(x + pi) == -cos(x)
    assert cos(x + 2*pi) == cos(x)
    assert cos(x + 3*pi) == -cos(x)
    assert cos(x + 4*pi) == cos(x)
    assert cos(x - 5*pi) == -cos(x)
    assert tan(x + pi) == tan(x)
    assert tan(x - 3*pi) == tan(x)
    assert cot(x + pi) == cot(x)
    assert cot(x - 3*pi) == cot(x)
    assert sin(pi/2 - x) == cos(x)
    assert sin(pi*Rational(3, 2) - x) == -cos(x)
    assert sin(pi*Rational(5, 2) - x) == cos(x)
    assert cos(pi/2 - x) == sin(x)
    assert cos(pi*Rational(3, 2) - x) == -sin(x)
    assert cos(pi*Rational(5, 2) - x) == sin(x)
    assert tan(pi/2 - x) == cot(x)
    assert tan(pi*Rational(3, 2) - x) == cot(x)
    assert tan(pi*Rational(5, 2) - x) == cot(x)
    assert cot(pi/2 - x) == tan(x)
    assert cot(pi*Rational(3, 2) - x) == tan(x)
    assert cot(pi*Rational(5, 2) - x) == tan(x)
    assert sin(pi/2 + x) == cos(x)
    assert cos(pi/2 + x) == -sin(x)
    assert tan(pi/2 + x) == -cot(x)
    assert cot(pi/2 + x) == -tan(x)


def test_cos():
    x, y = symbols('x y')

    assert cos.nargs == FiniteSet(1)
    assert cos(nan) is nan

    assert cos(oo) == AccumBounds(-1, 1)
    assert cos(oo) - cos(oo) == AccumBounds(-2, 2)
    assert cos(oo*I) is oo
    assert cos(-oo*I) is oo
    assert cos(zoo) is nan

    assert cos(0) == 1

    assert cos(acos(x)) == x
    assert cos(atan(x)) == 1 / sqrt(1 + x**2)
    assert cos(asin(x)) == sqrt(1 - x**2)
    assert cos(acot(x)) == 1 / sqrt(1 + 1 / x**2)
    assert cos(acsc(x)) == sqrt(1 - 1 / x**2)
    assert cos(asec(x)) == 1 / x
    assert cos(atan2(y, x)) == x / sqrt(x**2 + y**2)

    assert cos(pi*I) == cosh(pi)
    assert cos(-pi*I) == cosh(pi)
    assert cos(-2*I) == cosh(2)

    assert cos(pi/2) == 0
    assert cos(-pi/2) == 0
    assert cos(pi/2) == 0
    assert cos(-pi/2) == 0
    assert cos((-3*10**73 + 1)*pi/2) == 0
    assert cos((7*10**103 + 1)*pi/2) == 0

    n = symbols('n', integer=True, even=False)
    e = symbols('e', even=True)
    assert cos(pi*n/2) == 0
    assert cos(pi*e/2) == (-1)**(e/2)

    assert cos(pi) == -1
    assert cos(-pi) == -1
    assert cos(2*pi) == 1
    assert cos(5*pi) == -1
    assert cos(8*pi) == 1

    assert cos(pi/3) == S.Half
    assert cos(pi*Rational(-2, 3)) == Rational(-1, 2)

    assert cos(pi/4) == S.Half*sqrt(2)
    assert cos(-pi/4) == S.Half*sqrt(2)
    assert cos(pi*Rational(11, 4)) == Rational(-1, 2)*sqrt(2)
    assert cos(pi*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)

    assert cos(pi/6) == S.Half*sqrt(3)
    assert cos(-pi/6) == S.Half*sqrt(3)
    assert cos(pi*Rational(7, 6)) == Rational(-1, 2)*sqrt(3)
    assert cos(pi*Rational(-5, 6)) == Rational(-1, 2)*sqrt(3)

    assert cos(pi*Rational(1, 5)) == (sqrt(5) + 1)/4
    assert cos(pi*Rational(2, 5)) == (sqrt(5) - 1)/4
    assert cos(pi*Rational(3, 5)) == -cos(pi*Rational(2, 5))
    assert cos(pi*Rational(4, 5)) == -cos(pi*Rational(1, 5))
    assert cos(pi*Rational(6, 5)) == -cos(pi*Rational(1, 5))
    assert cos(pi*Rational(8, 5)) == cos(pi*Rational(2, 5))

    assert cos(pi*Rational(-1273, 5)) == -cos(pi*Rational(2, 5))

    assert cos(pi/8) == sqrt((2 + sqrt(2))/4)

    assert cos(pi/12) == sqrt(2)/4 + sqrt(6)/4
    assert cos(pi*Rational(5, 12)) == -sqrt(2)/4 + sqrt(6)/4
    assert cos(pi*Rational(7, 12)) == sqrt(2)/4 - sqrt(6)/4
    assert cos(pi*Rational(11, 12)) == -sqrt(2)/4 - sqrt(6)/4

    assert cos(pi*Rational(104, 105)) == -cos(pi/105)
    assert cos(pi*Rational(106, 105)) == -cos(pi/105)

    assert cos(pi*Rational(-104, 105)) == -cos(pi/105)
    assert cos(pi*Rational(-106, 105)) == -cos(pi/105)

    assert cos(x*I) == cosh(x)
    assert cos(k*pi*I) == cosh(k*pi)

    assert cos(r).is_real is True

    assert cos(0, evaluate=False).is_algebraic
    assert cos(a).is_algebraic is None
    assert cos(na).is_algebraic is False
    q = Symbol('q', rational=True)
    assert cos(pi*q).is_algebraic
    assert cos(pi*Rational(2, 7)).is_algebraic

    assert cos(k*pi) == (-1)**k
    assert cos(2*k*pi) == 1
    assert cos(0, evaluate=False).is_zero is False
    assert cos(Rational(1, 2)).is_zero is False
    # The following test will return None as the result, but really it should
    # be True even if it is not always possible to resolve an assumptions query.
    assert cos(asin(-1, evaluate=False), evaluate=False).is_zero is None
    for d in list(range(1, 22)) + [60, 85]:
        for n in range(2*d + 1):
            x = n*pi/d
            e = abs( float(cos(x)) - cos(float(x)) )
            assert e < 1e-12


def test_issue_6190():
    c = Float('123456789012345678901234567890.25', '')
    for cls in [sin, cos, tan, cot]:
        assert cls(c*pi) == cls(pi/4)
        assert cls(4.125*pi) == cls(pi/8)
        assert cls(4.7*pi) == cls((4.7 % 2)*pi)


def test_cos_series():
    assert cos(x).series(x, 0, 9) == \
        1 - x**2/2 + x**4/24 - x**6/720 + x**8/40320 + O(x**9)


def test_cos_rewrite():
    assert cos(x).rewrite(exp) == exp(I*x)/2 + exp(-I*x)/2
    assert cos(x).rewrite(tan) == (1 - tan(x/2)**2)/(1 + tan(x/2)**2)
    assert cos(x).rewrite(cot) == \
        Piecewise((1, Eq(im(x), 0) & Eq(Mod(x, 2*pi), 0)),
                  ((cot(x/2)**2 - 1)/(cot(x/2)**2 + 1), True))
    assert cos(sinh(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, sinh(3)).n()
    assert cos(cosh(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, cosh(3)).n()
    assert cos(tanh(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, tanh(3)).n()
    assert cos(coth(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, coth(3)).n()
    assert cos(sin(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, sin(3)).n()
    assert cos(cos(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, cos(3)).n()
    assert cos(tan(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, tan(3)).n()
    assert cos(cot(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, cot(3)).n()
    assert cos(log(x)).rewrite(Pow) == x**I/2 + x**-I/2
    assert cos(x).rewrite(sec) == 1/sec(x)
    assert cos(x).rewrite(sin) == sin(x + pi/2, evaluate=False)
    assert cos(x).rewrite(csc) == 1/csc(-x + pi/2, evaluate=False)
    assert cos(sin(x)).rewrite(Pow) == cos(sin(x))
    assert cos(x).rewrite(besselj) == Piecewise(
                (sqrt(pi*x/2)*besselj(-S.Half, x), Ne(x, 0)),
                (1, True)
            )
    assert cos(x).rewrite(besselj).subs(x, 0) == cos(0)


def test_cos_expansion():
    assert cos(x + y).expand(trig=True) == cos(x)*cos(y) - sin(x)*sin(y)
    assert cos(x - y).expand(trig=True) == cos(x)*cos(y) + sin(x)*sin(y)
    assert cos(y - x).expand(trig=True) == cos(x)*cos(y) + sin(x)*sin(y)
    assert cos(2*x).expand(trig=True) == 2*cos(x)**2 - 1
    assert cos(3*x).expand(trig=True) == 4*cos(x)**3 - 3*cos(x)
    assert cos(4*x).expand(trig=True) == 8*cos(x)**4 - 8*cos(x)**2 + 1
    assert cos(2*pi/17).expand(trig=True) == cos(2*pi/17, evaluate=False)
    assert cos(x+pi/17).expand(trig=True) == cos(pi/17)*cos(x) - sin(pi/17)*sin(x)
    _test_extrig(cos, 2, 2*cos(1)**2 - 1)
    _test_extrig(cos, 3, 4*cos(1)**3 - 3*cos(1))


def test_cos_AccumBounds():
    assert cos(AccumBounds(-oo, oo)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(0, oo)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(-oo, 0)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(0, 2*S.Pi)) == AccumBounds(-1, 1)
    assert cos(AccumBounds(-S.Pi/3, S.Pi/4)) == AccumBounds(cos(-S.Pi/3), 1)
    assert cos(AccumBounds(S.Pi*Rational(3, 4), S.Pi*Rational(5, 4))) == AccumBounds(-1, cos(S.Pi*Rational(3, 4)))
    assert cos(AccumBounds(S.Pi*Rational(5, 4), S.Pi*Rational(4, 3))) == AccumBounds(cos(S.Pi*Rational(5, 4)), cos(S.Pi*Rational(4, 3)))
    assert cos(AccumBounds(S.Pi/4, S.Pi/3)) == AccumBounds(cos(S.Pi/3), cos(S.Pi/4))


def test_cos_fdiff():
    assert cos(x).fdiff() == -sin(x)
    raises(ArgumentIndexError, lambda: cos(x).fdiff(2))


def test_tan():
    assert tan(nan) is nan

    assert tan(zoo) is nan
    assert tan(oo) == AccumBounds(-oo, oo)
    assert tan(oo) - tan(oo) == AccumBounds(-oo, oo)
    assert tan.nargs == FiniteSet(1)
    assert tan(oo*I) == I
    assert tan(-oo*I) == -I

    assert tan(0) == 0

    assert tan(atan(x)) == x
    assert tan(asin(x)) == x / sqrt(1 - x**2)
    assert tan(acos(x)) == sqrt(1 - x**2) / x
    assert tan(acot(x)) == 1 / x
    assert tan(acsc(x)) == 1 / (sqrt(1 - 1 / x**2) * x)
    assert tan(asec(x)) == sqrt(1 - 1 / x**2) * x
    assert tan(atan2(y, x)) == y/x

    assert tan(pi*I) == tanh(pi)*I
    assert tan(-pi*I) == -tanh(pi)*I
    assert tan(-2*I) == -tanh(2)*I

    assert tan(pi) == 0
    assert tan(-pi) == 0
    assert tan(2*pi) == 0
    assert tan(-2*pi) == 0
    assert tan(-3*10**73*pi) == 0

    assert tan(pi/2) is zoo
    assert tan(pi*Rational(3, 2)) is zoo

    assert tan(pi/3) == sqrt(3)
    assert tan(pi*Rational(-2, 3)) == sqrt(3)

    assert tan(pi/4) is S.One
    assert tan(-pi/4) is S.NegativeOne
    assert tan(pi*Rational(17, 4)) is S.One
    assert tan(pi*Rational(-3, 4)) is S.One

    assert tan(pi/5) == sqrt(5 - 2*sqrt(5))
    assert tan(pi*Rational(2, 5)) == sqrt(5 + 2*sqrt(5))
    assert tan(pi*Rational(18, 5)) == -sqrt(5 + 2*sqrt(5))
    assert tan(pi*Rational(-16, 5)) == -sqrt(5 - 2*sqrt(5))

    assert tan(pi/6) == 1/sqrt(3)
    assert tan(-pi/6) == -1/sqrt(3)
    assert tan(pi*Rational(7, 6)) == 1/sqrt(3)
    assert tan(pi*Rational(-5, 6)) == 1/sqrt(3)

    assert tan(pi/8) == -1 + sqrt(2)
    assert tan(pi*Rational(3, 8)) == 1 + sqrt(2)  # issue 15959
    assert tan(pi*Rational(5, 8)) == -1 - sqrt(2)
    assert tan(pi*Rational(7, 8)) == 1 - sqrt(2)

    assert tan(pi/10) == sqrt(1 - 2*sqrt(5)/5)
    assert tan(pi*Rational(3, 10)) == sqrt(1 + 2*sqrt(5)/5)
    assert tan(pi*Rational(17, 10)) == -sqrt(1 + 2*sqrt(5)/5)
    assert tan(pi*Rational(-31, 10)) == -sqrt(1 - 2*sqrt(5)/5)

    assert tan(pi/12) == -sqrt(3) + 2
    assert tan(pi*Rational(5, 12)) == sqrt(3) + 2
    assert tan(pi*Rational(7, 12)) == -sqrt(3) - 2
    assert tan(pi*Rational(11, 12)) == sqrt(3) - 2

    assert tan(pi/24).radsimp() == -2 - sqrt(3) + sqrt(2) + sqrt(6)
    assert tan(pi*Rational(5, 24)).radsimp() == -2 + sqrt(3) - sqrt(2) + sqrt(6)
    assert tan(pi*Rational(7, 24)).radsimp() == 2 - sqrt(3) - sqrt(2) + sqrt(6)
    assert tan(pi*Rational(11, 24)).radsimp() == 2 + sqrt(3) + sqrt(2) + sqrt(6)
    assert tan(pi*Rational(13, 24)).radsimp() == -2 - sqrt(3) - sqrt(2) - sqrt(6)
    assert tan(pi*Rational(17, 24)).radsimp() == -2 + sqrt(3) + sqrt(2) - sqrt(6)
    assert tan(pi*Rational(19, 24)).radsimp() == 2 - sqrt(3) + sqrt(2) - sqrt(6)
    assert tan(pi*Rational(23, 24)).radsimp() == 2 + sqrt(3) - sqrt(2) - sqrt(6)

    assert tan(x*I) == tanh(x)*I

    assert tan(k*pi) == 0
    assert tan(17*k*pi) == 0

    assert tan(k*pi*I) == tanh(k*pi)*I

    assert tan(r).is_real is None
    assert tan(r).is_extended_real is True

    assert tan(0, evaluate=False).is_algebraic
    assert tan(a).is_algebraic is None
    assert tan(na).is_algebraic is False

    assert tan(pi*Rational(10, 7)) == tan(pi*Rational(3, 7))
    assert tan(pi*Rational(11, 7)) == -tan(pi*Rational(3, 7))
    assert tan(pi*Rational(-11, 7)) == tan(pi*Rational(3, 7))

    assert tan(pi*Rational(15, 14)) == tan(pi/14)
    assert tan(pi*Rational(-15, 14)) == -tan(pi/14)

    assert tan(r).is_finite is None
    assert tan(I*r).is_finite is True

    # https://github.com/sympy/sympy/issues/21177
    f = tan(pi*(x + S(3)/2))/(3*x)
    assert f.as_leading_term(x) == -1/(3*pi*x**2)


def test_tan_series():
    assert tan(x).series(x, 0, 9) == \
        x + x**3/3 + 2*x**5/15 + 17*x**7/315 + O(x**9)


def test_tan_rewrite():
    neg_exp, pos_exp = exp(-x*I), exp(x*I)
    assert tan(x).rewrite(exp) == I*(neg_exp - pos_exp)/(neg_exp + pos_exp)
    assert tan(x).rewrite(sin) == 2*sin(x)**2/sin(2*x)
    assert tan(x).rewrite(cos) == cos(x - S.Pi/2, evaluate=False)/cos(x)
    assert tan(x).rewrite(cot) == 1/cot(x)
    assert tan(sinh(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, sinh(3)).n()
    assert tan(cosh(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, cosh(3)).n()
    assert tan(tanh(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, tanh(3)).n()
    assert tan(coth(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, coth(3)).n()
    assert tan(sin(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, sin(3)).n()
    assert tan(cos(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, cos(3)).n()
    assert tan(tan(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, tan(3)).n()
    assert tan(cot(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, cot(3)).n()
    assert tan(log(x)).rewrite(Pow) == I*(x**-I - x**I)/(x**-I + x**I)
    assert tan(x).rewrite(sec) == sec(x)/sec(x - pi/2, evaluate=False)
    assert tan(x).rewrite(csc) == csc(-x + pi/2, evaluate=False)/csc(x)
    assert tan(sin(x)).rewrite(Pow) == tan(sin(x))
    assert tan(pi*Rational(2, 5), evaluate=False).rewrite(sqrt) == sqrt(sqrt(5)/8 +
               Rational(5, 8))/(Rational(-1, 4) + sqrt(5)/4)
    assert tan(x).rewrite(besselj) == besselj(S.Half, x)/besselj(-S.Half, x)
    assert tan(x).rewrite(besselj).subs(x, 0) == tan(0)


@slow
def test_tan_rewrite_slow():
    assert 0 == (cos(pi/34)*tan(pi/34) - sin(pi/34)).rewrite(pow)
    assert 0 == (cos(pi/17)*tan(pi/17) - sin(pi/17)).rewrite(pow)
    assert tan(pi/19).rewrite(pow) == tan(pi/19)
    assert tan(pi*Rational(8, 19)).rewrite(sqrt) == tan(pi*Rational(8, 19))
    assert tan(pi*Rational(2, 5), evaluate=False).rewrite(sqrt) == sqrt(sqrt(5)/8 +
               Rational(5, 8))/(Rational(-1, 4) + sqrt(5)/4)


def test_tan_subs():
    assert tan(x).subs(tan(x), y) == y
    assert tan(x).subs(x, y) == tan(y)
    assert tan(x).subs(x, S.Pi/2) is zoo
    assert tan(x).subs(x, S.Pi*Rational(3, 2)) is zoo


def test_tan_expansion():
    assert tan(x + y).expand(trig=True) == ((tan(x) + tan(y))/(1 - tan(x)*tan(y))).expand()
    assert tan(x - y).expand(trig=True) == ((tan(x) - tan(y))/(1 + tan(x)*tan(y))).expand()
    assert tan(x + y + z).expand(trig=True) == (
        (tan(x) + tan(y) + tan(z) - tan(x)*tan(y)*tan(z))/
        (1 - tan(x)*tan(y) - tan(x)*tan(z) - tan(y)*tan(z))).expand()
    assert 0 == tan(2*x).expand(trig=True).rewrite(tan).subs([(tan(x), Rational(1, 7))])*24 - 7
    assert 0 == tan(3*x).expand(trig=True).rewrite(tan).subs([(tan(x), Rational(1, 5))])*55 - 37
    assert 0 == tan(4*x - pi/4).expand(trig=True).rewrite(tan).subs([(tan(x), Rational(1, 5))])*239 - 1
    _test_extrig(tan, 2, 2*tan(1)/(1 - tan(1)**2))
    _test_extrig(tan, 3, (-tan(1)**3 + 3*tan(1))/(1 - 3*tan(1)**2))


def test_tan_AccumBounds():
    assert tan(AccumBounds(-oo, oo)) == AccumBounds(-oo, oo)
    assert tan(AccumBounds(S.Pi/3, S.Pi*Rational(2, 3))) == AccumBounds(-oo, oo)
    assert tan(AccumBounds(S.Pi/6, S.Pi/3)) == AccumBounds(tan(S.Pi/6), tan(S.Pi/3))


def test_tan_fdiff():
    assert tan(x).fdiff() == tan(x)**2 + 1
    raises(ArgumentIndexError, lambda: tan(x).fdiff(2))


def test_cot():
    assert cot(nan) is nan

    assert cot.nargs == FiniteSet(1)
    assert cot(oo*I) == -I
    assert cot(-oo*I) == I
    assert cot(zoo) is nan

    assert cot(0) is zoo
    assert cot(2*pi) is zoo

    assert cot(acot(x)) == x
    assert cot(atan(x)) == 1 / x
    assert cot(asin(x)) == sqrt(1 - x**2) / x
    assert cot(acos(x)) == x / sqrt(1 - x**2)
    assert cot(acsc(x)) == sqrt(1 - 1 / x**2) * x
    assert cot(asec(x)) == 1 / (sqrt(1 - 1 / x**2) * x)
    assert cot(atan2(y, x)) == x/y

    assert cot(pi*I) == -coth(pi)*I
    assert cot(-pi*I) == coth(pi)*I
    assert cot(-2*I) == coth(2)*I

    assert cot(pi) == cot(2*pi) == cot(3*pi)
    assert cot(-pi) == cot(-2*pi) == cot(-3*pi)

    assert cot(pi/2) == 0
    assert cot(-pi/2) == 0
    assert cot(pi*Rational(5, 2)) == 0
    assert cot(pi*Rational(7, 2)) == 0

    assert cot(pi/3) == 1/sqrt(3)
    assert cot(pi*Rational(-2, 3)) == 1/sqrt(3)

    assert cot(pi/4) is S.One
    assert cot(-pi/4) is S.NegativeOne
    assert cot(pi*Rational(17, 4)) is S.One
    assert cot(pi*Rational(-3, 4)) is S.One

    assert cot(pi/6) == sqrt(3)
    assert cot(-pi/6) == -sqrt(3)
    assert cot(pi*Rational(7, 6)) == sqrt(3)
    assert cot(pi*Rational(-5, 6)) == sqrt(3)

    assert cot(pi/8) == 1 + sqrt(2)
    assert cot(pi*Rational(3, 8)) == -1 + sqrt(2)
    assert cot(pi*Rational(5, 8)) == 1 - sqrt(2)
    assert cot(pi*Rational(7, 8)) == -1 - sqrt(2)

    assert cot(pi/12) == sqrt(3) + 2
    assert cot(pi*Rational(5, 12)) == -sqrt(3) + 2
    assert cot(pi*Rational(7, 12)) == sqrt(3) - 2
    assert cot(pi*Rational(11, 12)) == -sqrt(3) - 2

    assert cot(pi/24).radsimp() == sqrt(2) + sqrt(3) + 2 + sqrt(6)
    assert cot(pi*Rational(5, 24)).radsimp() == -sqrt(2) - sqrt(3) + 2 + sqrt(6)
    assert cot(pi*Rational(7, 24)).radsimp() == -sqrt(2) + sqrt(3) - 2 + sqrt(6)
    assert cot(pi*Rational(11, 24)).radsimp() == sqrt(2) - sqrt(3) - 2 + sqrt(6)
    assert cot(pi*Rational(13, 24)).radsimp() == -sqrt(2) + sqrt(3) + 2 - sqrt(6)
    assert cot(pi*Rational(17, 24)).radsimp() == sqrt(2) - sqrt(3) + 2 - sqrt(6)
    assert cot(pi*Rational(19, 24)).radsimp() == sqrt(2) + sqrt(3) - 2 - sqrt(6)
    assert cot(pi*Rational(23, 24)).radsimp() == -sqrt(2) - sqrt(3) - 2 - sqrt(6)

    assert cot(x*I) == -coth(x)*I
    assert cot(k*pi*I) == -coth(k*pi)*I

    assert cot(r).is_real is None
    assert cot(r).is_extended_real is True

    assert cot(a).is_algebraic is None
    assert cot(na).is_algebraic is False

    assert cot(pi*Rational(10, 7)) == cot(pi*Rational(3, 7))
    assert cot(pi*Rational(11, 7)) == -cot(pi*Rational(3, 7))
    assert cot(pi*Rational(-11, 7)) == cot(pi*Rational(3, 7))

    assert cot(pi*Rational(39, 34)) == cot(pi*Rational(5, 34))
    assert cot(pi*Rational(-41, 34)) == -cot(pi*Rational(7, 34))

    assert cot(x).is_finite is None
    assert cot(r).is_finite is None
    i = Symbol('i', imaginary=True)
    assert cot(i).is_finite is True

    assert cot(x).subs(x, 3*pi) is zoo

    # https://github.com/sympy/sympy/issues/21177
    f = cot(pi*(x + 4))/(3*x)
    assert f.as_leading_term(x) == 1/(3*pi*x**2)


def test_tan_cot_sin_cos_evalf():
    assert abs((tan(pi*Rational(8, 15))*cos(pi*Rational(8, 15))/sin(pi*Rational(8, 15)) - 1).evalf()) < 1e-14
    assert abs((cot(pi*Rational(4, 15))*sin(pi*Rational(4, 15))/cos(pi*Rational(4, 15)) - 1).evalf()) < 1e-14

@XFAIL
def test_tan_cot_sin_cos_ratsimp():
    assert 1 == (tan(pi*Rational(8, 15))*cos(pi*Rational(8, 15))/sin(pi*Rational(8, 15))).ratsimp()
    assert 1 == (cot(pi*Rational(4, 15))*sin(pi*Rational(4, 15))/cos(pi*Rational(4, 15))).ratsimp()


def test_cot_series():
    assert cot(x).series(x, 0, 9) == \
        1/x - x/3 - x**3/45 - 2*x**5/945 - x**7/4725 + O(x**9)
    # issue 6210
    assert cot(x**4 + x**5).series(x, 0, 1) == \
        x**(-4) - 1/x**3 + x**(-2) - 1/x + 1 + O(x)
    assert cot(pi*(1-x)).series(x, 0, 3) == -1/(pi*x) + pi*x/3 + O(x**3)
    assert cot(x).taylor_term(0, x) == 1/x
    assert cot(x).taylor_term(2, x) is S.Zero
    assert cot(x).taylor_term(3, x) == -x**3/45


def test_cot_rewrite():
    neg_exp, pos_exp = exp(-x*I), exp(x*I)
    assert cot(x).rewrite(exp) == I*(pos_exp + neg_exp)/(pos_exp - neg_exp)
    assert cot(x).rewrite(sin) == sin(2*x)/(2*(sin(x)**2))
    assert cot(x).rewrite(cos) == cos(x)/cos(x - pi/2, evaluate=False)
    assert cot(x).rewrite(tan) == 1/tan(x)
    def check(func):
        z = cot(func(x)).rewrite(exp) - cot(x).rewrite(exp).subs(x, func(x))
        assert z.rewrite(exp).expand() == 0
    check(sinh)
    check(cosh)
    check(tanh)
    check(coth)
    check(sin)
    check(cos)
    check(tan)
    assert cot(log(x)).rewrite(Pow) == -I*(x**-I + x**I)/(x**-I - x**I)
    assert cot(x).rewrite(sec) == sec(x - pi / 2, evaluate=False) / sec(x)
    assert cot(x).rewrite(csc) == csc(x) / csc(- x + pi / 2, evaluate=False)
    assert cot(sin(x)).rewrite(Pow) == cot(sin(x))
    assert cot(pi*Rational(2, 5), evaluate=False).rewrite(sqrt) == (Rational(-1, 4) + sqrt(5)/4)/\
                                                        sqrt(sqrt(5)/8 + Rational(5, 8))
    assert cot(x).rewrite(besselj) == besselj(-S.Half, x)/besselj(S.Half, x)
    assert cot(x).rewrite(besselj).subs(x, 0) == cot(0)


@slow
def test_cot_rewrite_slow():
    assert cot(pi*Rational(4, 34)).rewrite(pow).ratsimp() == \
        (cos(pi*Rational(4, 34))/sin(pi*Rational(4, 34))).rewrite(pow).ratsimp()
    assert cot(pi*Rational(4, 17)).rewrite(pow) == \
        (cos(pi*Rational(4, 17))/sin(pi*Rational(4, 17))).rewrite(pow)
    assert cot(pi/19).rewrite(pow) == cot(pi/19)
    assert cot(pi/19).rewrite(sqrt) == cot(pi/19)
    assert cot(pi*Rational(2, 5), evaluate=False).rewrite(sqrt) == \
        (Rational(-1, 4) + sqrt(5)/4) / sqrt(sqrt(5)/8 + Rational(5, 8))


def test_cot_subs():
    assert cot(x).subs(cot(x), y) == y
    assert cot(x).subs(x, y) == cot(y)
    assert cot(x).subs(x, 0) is zoo
    assert cot(x).subs(x, S.Pi) is zoo


def test_cot_expansion():
    assert cot(x + y).expand(trig=True).together() == (
        (cot(x)*cot(y) - 1)/(cot(x) + cot(y)))
    assert cot(x - y).expand(trig=True).together() == (
        cot(x)*cot(-y) - 1)/(cot(x) + cot(-y))
    assert cot(x + y + z).expand(trig=True).together() == (
        (cot(x)*cot(y)*cot(z) - cot(x) - cot(y) - cot(z))/
        (-1 + cot(x)*cot(y) + cot(x)*cot(z) + cot(y)*cot(z)))
    assert cot(3*x).expand(trig=True).together() == (
        (cot(x)**2 - 3)*cot(x)/(3*cot(x)**2 - 1))
    assert cot(2*x).expand(trig=True) == cot(x)/2 - 1/(2*cot(x))
    assert cot(3*x).expand(trig=True).together() == (
        cot(x)**2 - 3)*cot(x)/(3*cot(x)**2 - 1)
    assert cot(4*x - pi/4).expand(trig=True).cancel() == (
        -tan(x)**4 + 4*tan(x)**3 + 6*tan(x)**2 - 4*tan(x) - 1
        )/(tan(x)**4 + 4*tan(x)**3 - 6*tan(x)**2 - 4*tan(x) + 1)
    _test_extrig(cot, 2, (-1 + cot(1)**2)/(2*cot(1)))
    _test_extrig(cot, 3, (-3*cot(1) + cot(1)**3)/(-1 + 3*cot(1)**2))


def test_cot_AccumBounds():
    assert cot(AccumBounds(-oo, oo)) == AccumBounds(-oo, oo)
    assert cot(AccumBounds(-S.Pi/3, S.Pi/3)) == AccumBounds(-oo, oo)
    assert cot(AccumBounds(S.Pi/6, S.Pi/3)) == AccumBounds(cot(S.Pi/3), cot(S.Pi/6))


def test_cot_fdiff():
    assert cot(x).fdiff() == -cot(x)**2 - 1
    raises(ArgumentIndexError, lambda: cot(x).fdiff(2))


def test_sinc():
    assert isinstance(sinc(x), sinc)

    s = Symbol('s', zero=True)
    assert sinc(s) is S.One
    assert sinc(S.Infinity) is S.Zero
    assert sinc(S.NegativeInfinity) is S.Zero
    assert sinc(S.NaN) is S.NaN
    assert sinc(S.ComplexInfinity) is S.NaN

    n = Symbol('n', integer=True, nonzero=True)
    assert sinc(n*pi) is S.Zero
    assert sinc(-n*pi) is S.Zero
    assert sinc(pi/2) == 2 / pi
    assert sinc(-pi/2) == 2 / pi
    assert sinc(pi*Rational(5, 2)) == 2 / (5*pi)
    assert sinc(pi*Rational(7, 2)) == -2 / (7*pi)

    assert sinc(-x) == sinc(x)

    assert sinc(x).diff(x) == cos(x)/x - sin(x)/x**2
    assert sinc(x).diff(x) == (sin(x)/x).diff(x)
    assert sinc(x).diff(x, x) == (-sin(x) - 2*cos(x)/x + 2*sin(x)/x**2)/x
    assert sinc(x).diff(x, x) == (sin(x)/x).diff(x, x)
    assert limit(sinc(x).diff(x), x, 0) == 0
    assert limit(sinc(x).diff(x, x), x, 0) == -S(1)/3

    # https://github.com/sympy/sympy/issues/11402
    #
    # assert sinc(x).diff(x) == Piecewise(((x*cos(x) - sin(x)) / x**2, Ne(x, 0)), (0, True))
    #
    # assert sinc(x).diff(x).equals(sinc(x).rewrite(sin).diff(x))
    #
    # assert sinc(x).diff(x).subs(x, 0) is S.Zero

    assert sinc(x).series() == 1 - x**2/6 + x**4/120 + O(x**6)

    assert sinc(x).rewrite(jn) == jn(0, x)
    assert sinc(x).rewrite(sin) == Piecewise((sin(x)/x, Ne(x, 0)), (1, True))
    assert sinc(pi, evaluate=False).is_zero is True
    assert sinc(0, evaluate=False).is_zero is False
    assert sinc(n*pi, evaluate=False).is_zero is True
    assert sinc(x).is_zero is None
    xr = Symbol('xr', real=True, nonzero=True)
    assert sinc(x).is_real is None
    assert sinc(xr).is_real is True
    assert sinc(I*xr).is_real is True
    assert sinc(I*100).is_real is True
    assert sinc(x).is_finite is None
    assert sinc(xr).is_finite is True


def test_asin():
    assert asin(nan) is nan

    assert asin.nargs == FiniteSet(1)
    assert asin(oo) == -I*oo
    assert asin(-oo) == I*oo
    assert asin(zoo) is zoo

    # Note: asin(-x) = - asin(x)
    assert asin(0) == 0
    assert asin(1) == pi/2
    assert asin(-1) == -pi/2
    assert asin(sqrt(3)/2) == pi/3
    assert asin(-sqrt(3)/2) == -pi/3
    assert asin(sqrt(2)/2) == pi/4
    assert asin(-sqrt(2)/2) == -pi/4
    assert asin(sqrt((5 - sqrt(5))/8)) == pi/5
    assert asin(-sqrt((5 - sqrt(5))/8)) == -pi/5
    assert asin(S.Half) == pi/6
    assert asin(Rational(-1, 2)) == -pi/6
    assert asin((sqrt(2 - sqrt(2)))/2) == pi/8
    assert asin(-(sqrt(2 - sqrt(2)))/2) == -pi/8
    assert asin((sqrt(5) - 1)/4) == pi/10
    assert asin(-(sqrt(5) - 1)/4) == -pi/10
    assert asin((sqrt(3) - 1)/sqrt(2**3)) == pi/12
    assert asin(-(sqrt(3) - 1)/sqrt(2**3)) == -pi/12

    # check round-trip for exact values:
    for d in [5, 6, 8, 10, 12]:
        for n in range(-(d//2), d//2 + 1):
            if gcd(n, d) == 1:
                assert asin(sin(n*pi/d)) == n*pi/d

    assert asin(x).diff(x) == 1/sqrt(1 - x**2)

    assert asin(0.2, evaluate=False).is_real is True
    assert asin(-2).is_real is False
    assert asin(r).is_real is None

    assert asin(-2*I) == -I*asinh(2)

    assert asin(Rational(1, 7), evaluate=False).is_positive is True
    assert asin(Rational(-1, 7), evaluate=False).is_positive is False
    assert asin(p).is_positive is None
    assert asin(sin(Rational(7, 2))) == Rational(-7, 2) + pi
    assert asin(sin(Rational(-7, 4))) == Rational(7, 4) - pi
    assert unchanged(asin, cos(x))


def test_asin_series():
    assert asin(x).series(x, 0, 9) == \
        x + x**3/6 + 3*x**5/40 + 5*x**7/112 + O(x**9)
    t5 = asin(x).taylor_term(5, x)
    assert t5 == 3*x**5/40
    assert asin(x).taylor_term(7, x, t5, 0) == 5*x**7/112


def test_asin_leading_term():
    assert asin(x).as_leading_term(x) == x
    # Tests concerning branch points
    assert asin(x + 1).as_leading_term(x) == pi/2
    assert asin(x - 1).as_leading_term(x) == -pi/2
    assert asin(1/x).as_leading_term(x, cdir=1) == I*log(x) + pi/2 - I*log(2)
    assert asin(1/x).as_leading_term(x, cdir=-1) == -I*log(x) - 3*pi/2 + I*log(2)
    # Tests concerning points lying on branch cuts
    assert asin(I*x + 2).as_leading_term(x, cdir=1) == pi - asin(2)
    assert asin(-I*x + 2).as_leading_term(x, cdir=1) == asin(2)
    assert asin(I*x - 2).as_leading_term(x, cdir=1) == -asin(2)
    assert asin(-I*x - 2).as_leading_term(x, cdir=1) == -pi + asin(2)
    # Tests concerning im(ndir) == 0
    assert asin(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == -pi/2 + I*log(2 - sqrt(3))
    assert asin(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == -pi/2 + I*log(2 - sqrt(3))


def test_asin_rewrite():
    assert asin(x).rewrite(log) == -I*log(I*x + sqrt(1 - x**2))
    assert asin(x).rewrite(atan) == 2*atan(x/(1 + sqrt(1 - x**2)))
    assert asin(x).rewrite(acos) == S.Pi/2 - acos(x)
    assert asin(x).rewrite(acot) == 2*acot((sqrt(-x**2 + 1) + 1)/x)
    assert asin(x).rewrite(asec) == -asec(1/x) + pi/2
    assert asin(x).rewrite(acsc) == acsc(1/x)


def test_asin_fdiff():
    assert asin(x).fdiff() == 1/sqrt(1 - x**2)
    raises(ArgumentIndexError, lambda: asin(x).fdiff(2))


def test_acos():
    assert acos(nan) is nan
    assert acos(zoo) is zoo

    assert acos.nargs == FiniteSet(1)
    assert acos(oo) == I*oo
    assert acos(-oo) == -I*oo

    # Note: acos(-x) = pi - acos(x)
    assert acos(0) == pi/2
    assert acos(S.Half) == pi/3
    assert acos(Rational(-1, 2)) == pi*Rational(2, 3)
    assert acos(1) == 0
    assert acos(-1) == pi
    assert acos(sqrt(2)/2) == pi/4
    assert acos(-sqrt(2)/2) == pi*Rational(3, 4)

    # check round-trip for exact values:
    for d in range(5, 13):
        for num in range(d):
            if gcd(num, d) == 1:
                assert acos(cos(num*pi/d)) == num*pi/d
                assert acos(-cos(num*pi/d)) == pi - num*pi/d
                assert acos(sin(num*pi/d)) == pi/2 - asin(sin(num*pi/d))
                assert acos(-sin(num*pi/d)) == pi/2 - asin(-sin(num*pi/d))

    assert acos(2*I) == pi/2 - asin(2*I)

    assert acos(x).diff(x) == -1/sqrt(1 - x**2)

    assert acos(0.2).is_real is True
    assert acos(-2).is_real is False
    assert acos(r).is_real is None

    assert acos(Rational(1, 7), evaluate=False).is_positive is True
    assert acos(Rational(-1, 7), evaluate=False).is_positive is True
    assert acos(Rational(3, 2), evaluate=False).is_positive is False
    assert acos(p).is_positive is None

    assert acos(2 + p).conjugate() != acos(10 + p)
    assert acos(-3 + n).conjugate() != acos(-3 + n)
    assert acos(Rational(1, 3)).conjugate() == acos(Rational(1, 3))
    assert acos(Rational(-1, 3)).conjugate() == acos(Rational(-1, 3))
    assert acos(p + n*I).conjugate() == acos(p - n*I)
    assert acos(z).conjugate() != acos(conjugate(z))


def test_acos_leading_term():
    assert acos(x).as_leading_term(x) == pi/2
    # Tests concerning branch points
    assert acos(x + 1).as_leading_term(x) == sqrt(2)*sqrt(-x)
    assert acos(x - 1).as_leading_term(x) == pi
    assert acos(1/x).as_leading_term(x, cdir=1) == -I*log(x) + I*log(2)
    assert acos(1/x).as_leading_term(x, cdir=-1) == I*log(x) + 2*pi - I*log(2)
    # Tests concerning points lying on branch cuts
    assert acos(I*x + 2).as_leading_term(x, cdir=1) == -acos(2)
    assert acos(-I*x + 2).as_leading_term(x, cdir=1) == acos(2)
    assert acos(I*x - 2).as_leading_term(x, cdir=1) == acos(-2)
    assert acos(-I*x - 2).as_leading_term(x, cdir=1) == 2*pi - acos(-2)
    # Tests concerning im(ndir) == 0
    assert acos(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == pi + I*log(sqrt(3) + 2)
    assert acos(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == pi + I*log(sqrt(3) + 2)


def test_acos_series():
    assert acos(x).series(x, 0, 8) == \
        pi/2 - x - x**3/6 - 3*x**5/40 - 5*x**7/112 + O(x**8)
    assert acos(x).series(x, 0, 8) == pi/2 - asin(x).series(x, 0, 8)
    t5 = acos(x).taylor_term(5, x)
    assert t5 == -3*x**5/40
    assert acos(x).taylor_term(7, x, t5, 0) == -5*x**7/112
    assert acos(x).taylor_term(0, x) == pi/2
    assert acos(x).taylor_term(2, x) is S.Zero


def test_acos_rewrite():
    assert acos(x).rewrite(log) == pi/2 + I*log(I*x + sqrt(1 - x**2))
    assert acos(x).rewrite(atan) == pi*(-x*sqrt(x**(-2)) + 1)/2 + atan(sqrt(1 - x**2)/x)
    assert acos(0).rewrite(atan) == S.Pi/2
    assert acos(0.5).rewrite(atan) == acos(0.5).rewrite(log)
    assert acos(x).rewrite(asin) == S.Pi/2 - asin(x)
    assert acos(x).rewrite(acot) == -2*acot((sqrt(-x**2 + 1) + 1)/x) + pi/2
    assert acos(x).rewrite(asec) == asec(1/x)
    assert acos(x).rewrite(acsc) == -acsc(1/x) + pi/2


def test_acos_fdiff():
    assert acos(x).fdiff() == -1/sqrt(1 - x**2)
    raises(ArgumentIndexError, lambda: acos(x).fdiff(2))


def test_atan():
    assert atan(nan) is nan

    assert atan.nargs == FiniteSet(1)
    assert atan(oo) == pi/2
    assert atan(-oo) == -pi/2
    assert atan(zoo) == AccumBounds(-pi/2, pi/2)

    assert atan(0) == 0
    assert atan(1) == pi/4
    assert atan(sqrt(3)) == pi/3
    assert atan(-(1 + sqrt(2))) == pi*Rational(-3, 8)
    assert atan(sqrt(5 - 2 * sqrt(5))) == pi/5
    assert atan(-sqrt(1 - 2 * sqrt(5)/ 5)) == -pi/10
    assert atan(sqrt(1 + 2 * sqrt(5) / 5)) == pi*Rational(3, 10)
    assert atan(-2 + sqrt(3)) == -pi/12
    assert atan(2 + sqrt(3)) == pi*Rational(5, 12)
    assert atan(-2 - sqrt(3)) == pi*Rational(-5, 12)

    # check round-trip for exact values:
    for d in [5, 6, 8, 10, 12]:
        for num in range(-(d//2), d//2 + 1):
            if gcd(num, d) == 1:
                assert atan(tan(num*pi/d)) == num*pi/d

    assert atan(oo) == pi/2
    assert atan(x).diff(x) == 1/(1 + x**2)

    assert atan(r).is_real is True

    assert atan(-2*I) == -I*atanh(2)
    assert unchanged(atan, cot(x))
    assert atan(cot(Rational(1, 4))) == Rational(-1, 4) + pi/2
    assert acot(Rational(1, 4)).is_rational is False

    for s in (x, p, n, np, nn, nz, ep, en, enp, enn, enz):
        if s.is_real or s.is_extended_real is None:
            assert s.is_nonzero is atan(s).is_nonzero
            assert s.is_positive is atan(s).is_positive
            assert s.is_negative is atan(s).is_negative
            assert s.is_nonpositive is atan(s).is_nonpositive
            assert s.is_nonnegative is atan(s).is_nonnegative
        else:
            assert s.is_extended_nonzero is atan(s).is_nonzero
            assert s.is_extended_positive is atan(s).is_positive
            assert s.is_extended_negative is atan(s).is_negative
            assert s.is_extended_nonpositive is atan(s).is_nonpositive
            assert s.is_extended_nonnegative is atan(s).is_nonnegative
        assert s.is_extended_nonzero is atan(s).is_extended_nonzero
        assert s.is_extended_positive is atan(s).is_extended_positive
        assert s.is_extended_negative is atan(s).is_extended_negative
        assert s.is_extended_nonpositive is atan(s).is_extended_nonpositive
        assert s.is_extended_nonnegative is atan(s).is_extended_nonnegative


def test_atan_rewrite():
    assert atan(x).rewrite(log) == I*(log(1 - I*x)-log(1 + I*x))/2
    assert atan(x).rewrite(asin) == (-asin(1/sqrt(x**2 + 1)) + pi/2)*sqrt(x**2)/x
    assert atan(x).rewrite(acos) == sqrt(x**2)*acos(1/sqrt(x**2 + 1))/x
    assert atan(x).rewrite(acot) == acot(1/x)
    assert atan(x).rewrite(asec) == sqrt(x**2)*asec(sqrt(x**2 + 1))/x
    assert atan(x).rewrite(acsc) == (-acsc(sqrt(x**2 + 1)) + pi/2)*sqrt(x**2)/x

    assert atan(-5*I).evalf() == atan(x).rewrite(log).evalf(subs={x:-5*I})
    assert atan(5*I).evalf() == atan(x).rewrite(log).evalf(subs={x:5*I})


def test_atan_fdiff():
    assert atan(x).fdiff() == 1/(x**2 + 1)
    raises(ArgumentIndexError, lambda: atan(x).fdiff(2))


def test_atan_leading_term():
    assert atan(x).as_leading_term(x) == x
    assert atan(1/x).as_leading_term(x, cdir=1) == pi/2
    assert atan(1/x).as_leading_term(x, cdir=-1) == -pi/2
    # Tests concerning branch points
    assert atan(x + I).as_leading_term(x, cdir=1) == -I*log(x)/2 + pi/4 + I*log(2)/2
    assert atan(x + I).as_leading_term(x, cdir=-1) == -I*log(x)/2 - 3*pi/4 + I*log(2)/2
    assert atan(x - I).as_leading_term(x, cdir=1) == I*log(x)/2 + pi/4 - I*log(2)/2
    assert atan(x - I).as_leading_term(x, cdir=-1) == I*log(x)/2 + pi/4 - I*log(2)/2
    # Tests concerning points lying on branch cuts
    assert atan(x + 2*I).as_leading_term(x, cdir=1) == I*atanh(2)
    assert atan(x + 2*I).as_leading_term(x, cdir=-1) == -pi + I*atanh(2)
    assert atan(x - 2*I).as_leading_term(x, cdir=1) == pi - I*atanh(2)
    assert atan(x - 2*I).as_leading_term(x, cdir=-1) == -I*atanh(2)
    # Tests concerning re(ndir) == 0
    assert atan(2*I - I*x - x**2).as_leading_term(x, cdir=1) == -pi/2 + I*log(3)/2
    assert atan(2*I - I*x - x**2).as_leading_term(x, cdir=-1) == -pi/2 + I*log(3)/2


def test_atan2():
    assert atan2.nargs == FiniteSet(2)
    assert atan2(0, 0) is S.NaN
    assert atan2(0, 1) == 0
    assert atan2(1, 1) == pi/4
    assert atan2(1, 0) == pi/2
    assert atan2(1, -1) == pi*Rational(3, 4)
    assert atan2(0, -1) == pi
    assert atan2(-1, -1) == pi*Rational(-3, 4)
    assert atan2(-1, 0) == -pi/2
    assert atan2(-1, 1) == -pi/4
    i = symbols('i', imaginary=True)
    r = symbols('r', real=True)
    eq = atan2(r, i)
    ans = -I*log((i + I*r)/sqrt(i**2 + r**2))
    reps = ((r, 2), (i, I))
    assert eq.subs(reps) == ans.subs(reps)

    x = Symbol('x', negative=True)
    y = Symbol('y', negative=True)
    assert atan2(y, x) == atan(y/x) - pi
    y = Symbol('y', nonnegative=True)
    assert atan2(y, x) == atan(y/x) + pi
    y = Symbol('y')
    assert atan2(y, x) == atan2(y, x, evaluate=False)

    u = Symbol("u", positive=True)
    assert atan2(0, u) == 0
    u = Symbol("u", negative=True)
    assert atan2(0, u) == pi

    assert atan2(y, oo) ==  0
    assert atan2(y, -oo)==  2*pi*Heaviside(re(y), S.Half) - pi

    assert atan2(y, x).rewrite(log) == -I*log((x + I*y)/sqrt(x**2 + y**2))
    assert atan2(0, 0) is S.NaN

    ex = atan2(y, x) - arg(x + I*y)
    assert ex.subs({x:2, y:3}).rewrite(arg) == 0
    assert ex.subs({x:2, y:3*I}).rewrite(arg) == -pi - I*log(sqrt(5)*I/5)
    assert ex.subs({x:2*I, y:3}).rewrite(arg) == -pi/2 - I*log(sqrt(5)*I)
    assert ex.subs({x:2*I, y:3*I}).rewrite(arg) == -pi + atan(Rational(2, 3)) + atan(Rational(3, 2))
    i = symbols('i', imaginary=True)
    r = symbols('r', real=True)
    e = atan2(i, r)
    rewrite = e.rewrite(arg)
    reps = {i: I, r: -2}
    assert rewrite == -I*log(abs(I*i + r)/sqrt(abs(i**2 + r**2))) + arg((I*i + r)/sqrt(i**2 + r**2))
    assert (e - rewrite).subs(reps).equals(0)

    assert atan2(0, x).rewrite(atan) == Piecewise((pi, re(x) < 0),
                                            (0, Ne(x, 0)),
                                            (nan, True))
    assert atan2(0, r).rewrite(atan) == Piecewise((pi, r < 0), (0, Ne(r, 0)), (S.NaN, True))
    assert atan2(0, i),rewrite(atan) == 0
    assert atan2(0, r + i).rewrite(atan) == Piecewise((pi, r < 0), (0, True))

    assert atan2(y, x).rewrite(atan) == Piecewise(
            (2*atan(y/(x + sqrt(x**2 + y**2))), Ne(y, 0)),
            (pi, re(x) < 0),
            (0, (re(x) > 0) | Ne(im(x), 0)),
            (nan, True))
    assert conjugate(atan2(x, y)) == atan2(conjugate(x), conjugate(y))

    assert diff(atan2(y, x), x) == -y/(x**2 + y**2)
    assert diff(atan2(y, x), y) == x/(x**2 + y**2)

    assert simplify(diff(atan2(y, x).rewrite(log), x)) == -y/(x**2 + y**2)
    assert simplify(diff(atan2(y, x).rewrite(log), y)) ==  x/(x**2 + y**2)

    assert str(atan2(1, 2).evalf(5)) == '0.46365'
    raises(ArgumentIndexError, lambda: atan2(x, y).fdiff(3))

def test_issue_17461():
    class A(Symbol):
        is_extended_real = True

        def _eval_evalf(self, prec):
            return Float(5.0)

    x = A('X')
    y = A('Y')
    assert abs(atan2(x, y).evalf() - 0.785398163397448) <= 1e-10

def test_acot():
    assert acot(nan) is nan

    assert acot.nargs == FiniteSet(1)
    assert acot(-oo) == 0
    assert acot(oo) == 0
    assert acot(zoo) == 0
    assert acot(1) == pi/4
    assert acot(0) == pi/2
    assert acot(sqrt(3)/3) == pi/3
    assert acot(1/sqrt(3)) == pi/3
    assert acot(-1/sqrt(3)) == -pi/3
    assert acot(x).diff(x) == -1/(1 + x**2)

    assert acot(r).is_extended_real is True

    assert acot(I*pi) == -I*acoth(pi)
    assert acot(-2*I) == I*acoth(2)
    assert acot(x).is_positive is None
    assert acot(n).is_positive is False
    assert acot(p).is_positive is True
    assert acot(I).is_positive is False
    assert acot(Rational(1, 4)).is_rational is False
    assert unchanged(acot, cot(x))
    assert unchanged(acot, tan(x))
    assert acot(cot(Rational(1, 4))) == Rational(1, 4)
    assert acot(tan(Rational(-1, 4))) == Rational(1, 4) - pi/2


def test_acot_rewrite():
    assert acot(x).rewrite(log) == I*(log(1 - I/x)-log(1 + I/x))/2
    assert acot(x).rewrite(asin) == x*(-asin(sqrt(-x**2)/sqrt(-x**2 - 1)) + pi/2)*sqrt(x**(-2))
    assert acot(x).rewrite(acos) == x*sqrt(x**(-2))*acos(sqrt(-x**2)/sqrt(-x**2 - 1))
    assert acot(x).rewrite(atan) == atan(1/x)
    assert acot(x).rewrite(asec) == x*sqrt(x**(-2))*asec(sqrt((x**2 + 1)/x**2))
    assert acot(x).rewrite(acsc) == x*(-acsc(sqrt((x**2 + 1)/x**2)) + pi/2)*sqrt(x**(-2))

    assert acot(-I/5).evalf() == acot(x).rewrite(log).evalf(subs={x:-I/5})
    assert acot(I/5).evalf() == acot(x).rewrite(log).evalf(subs={x:I/5})


def test_acot_fdiff():
    assert acot(x).fdiff() == -1/(x**2 + 1)
    raises(ArgumentIndexError, lambda: acot(x).fdiff(2))

def test_acot_leading_term():
    assert acot(1/x).as_leading_term(x) == x
    # Tests concerning branch points
    assert acot(x + I).as_leading_term(x, cdir=1) == I*log(x)/2 + pi/4 - I*log(2)/2
    assert acot(x + I).as_leading_term(x, cdir=-1) == I*log(x)/2 + pi/4 - I*log(2)/2
    assert acot(x - I).as_leading_term(x, cdir=1) == -I*log(x)/2 + pi/4 + I*log(2)/2
    assert acot(x - I).as_leading_term(x, cdir=-1) == -I*log(x)/2 - 3*pi/4 + I*log(2)/2
    # Tests concerning points lying on branch cuts
    assert acot(x).as_leading_term(x, cdir=1) == pi/2
    assert acot(x).as_leading_term(x, cdir=-1) == -pi/2
    assert acot(x + I/2).as_leading_term(x, cdir=1) == pi - I*acoth(S(1)/2)
    assert acot(x + I/2).as_leading_term(x, cdir=-1) == -I*acoth(S(1)/2)
    assert acot(x - I/2).as_leading_term(x, cdir=1) == I*acoth(S(1)/2)
    assert acot(x - I/2).as_leading_term(x, cdir=-1) == -pi + I*acoth(S(1)/2)
    # Tests concerning re(ndir) == 0
    assert acot(I/2 - I*x - x**2).as_leading_term(x, cdir=1) == -pi/2 - I*log(3)/2
    assert acot(I/2 - I*x - x**2).as_leading_term(x, cdir=-1) == -pi/2 - I*log(3)/2


def test_attributes():
    assert sin(x).args == (x,)


def test_sincos_rewrite():
    assert sin(pi/2 - x) == cos(x)
    assert sin(pi - x) == sin(x)
    assert cos(pi/2 - x) == sin(x)
    assert cos(pi - x) == -cos(x)


def _check_even_rewrite(func, arg):
    """Checks that the expr has been rewritten using f(-x) -> f(x)
    arg : -x
    """
    return func(arg).args[0] == -arg


def _check_odd_rewrite(func, arg):
    """Checks that the expr has been rewritten using f(-x) -> -f(x)
    arg : -x
    """
    return func(arg).func.is_Mul


def _check_no_rewrite(func, arg):
    """Checks that the expr is not rewritten"""
    return func(arg).args[0] == arg


def test_evenodd_rewrite():
    a = cos(2)  # negative
    b = sin(1)  # positive
    even = [cos]
    odd = [sin, tan, cot, asin, atan, acot]
    with_minus = [-1, -2**1024 * E, -pi/105, -x*y, -x - y]
    for func in even:
        for expr in with_minus:
            assert _check_even_rewrite(func, expr)
        assert _check_no_rewrite(func, a*b)
        assert func(
            x - y) == func(y - x)  # it doesn't matter which form is canonical
    for func in odd:
        for expr in with_minus:
            assert _check_odd_rewrite(func, expr)
        assert _check_no_rewrite(func, a*b)
        assert func(
            x - y) == -func(y - x)  # it doesn't matter which form is canonical


def test_as_leading_term_issue_5272():
    assert sin(x).as_leading_term(x) == x
    assert cos(x).as_leading_term(x) == 1
    assert tan(x).as_leading_term(x) == x
    assert cot(x).as_leading_term(x) == 1/x


def test_leading_terms():
    assert sin(1/x).as_leading_term(x) == AccumBounds(-1, 1)
    assert sin(S.Half).as_leading_term(x) == sin(S.Half)
    assert cos(1/x).as_leading_term(x) == AccumBounds(-1, 1)
    assert cos(S.Half).as_leading_term(x) == cos(S.Half)
    assert sec(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert csc(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert tan(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert cot(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)

    # https://github.com/sympy/sympy/issues/21038
    f = sin(pi*(x + 4))/(3*x)
    assert f.as_leading_term(x) == pi/3


def test_atan2_expansion():
    assert cancel(atan2(x**2, x + 1).diff(x) - atan(x**2/(x + 1)).diff(x)) == 0
    assert cancel(atan(y/x).series(y, 0, 5) - atan2(y, x).series(y, 0, 5)
                  + atan2(0, x) - atan(0)) == O(y**5)
    assert cancel(atan(y/x).series(x, 1, 4) - atan2(y, x).series(x, 1, 4)
                  + atan2(y, 1) - atan(y)) == O((x - 1)**4, (x, 1))
    assert cancel(atan((y + x)/x).series(x, 1, 3) - atan2(y + x, x).series(x, 1, 3)
                  + atan2(1 + y, 1) - atan(1 + y)) == O((x - 1)**3, (x, 1))
    assert Matrix([atan2(y, x)]).jacobian([y, x]) == \
        Matrix([[x/(y**2 + x**2), -y/(y**2 + x**2)]])


def test_aseries():
    def t(n, v, d, e):
        assert abs(
            n(1/v).evalf() - n(1/x).series(x, dir=d).removeO().subs(x, v)) < e
    t(atan, 0.1, '+', 1e-5)
    t(atan, -0.1, '-', 1e-5)
    t(acot, 0.1, '+', 1e-5)
    t(acot, -0.1, '-', 1e-5)


def test_issue_4420():
    i = Symbol('i', integer=True)
    e = Symbol('e', even=True)
    o = Symbol('o', odd=True)

    # unknown parity for variable
    assert cos(4*i*pi) == 1
    assert sin(4*i*pi) == 0
    assert tan(4*i*pi) == 0
    assert cot(4*i*pi) is zoo

    assert cos(3*i*pi) == cos(pi*i)  # +/-1
    assert sin(3*i*pi) == 0
    assert tan(3*i*pi) == 0
    assert cot(3*i*pi) is zoo

    assert cos(4.0*i*pi) == 1
    assert sin(4.0*i*pi) == 0
    assert tan(4.0*i*pi) == 0
    assert cot(4.0*i*pi) is zoo

    assert cos(3.0*i*pi) == cos(pi*i)  # +/-1
    assert sin(3.0*i*pi) == 0
    assert tan(3.0*i*pi) == 0
    assert cot(3.0*i*pi) is zoo

    assert cos(4.5*i*pi) == cos(0.5*pi*i)
    assert sin(4.5*i*pi) == sin(0.5*pi*i)
    assert tan(4.5*i*pi) == tan(0.5*pi*i)
    assert cot(4.5*i*pi) == cot(0.5*pi*i)

    # parity of variable is known
    assert cos(4*e*pi) == 1
    assert sin(4*e*pi) == 0
    assert tan(4*e*pi) == 0
    assert cot(4*e*pi) is zoo

    assert cos(3*e*pi) == 1
    assert sin(3*e*pi) == 0
    assert tan(3*e*pi) == 0
    assert cot(3*e*pi) is zoo

    assert cos(4.0*e*pi) == 1
    assert sin(4.0*e*pi) == 0
    assert tan(4.0*e*pi) == 0
    assert cot(4.0*e*pi) is zoo

    assert cos(3.0*e*pi) == 1
    assert sin(3.0*e*pi) == 0
    assert tan(3.0*e*pi) == 0
    assert cot(3.0*e*pi) is zoo

    assert cos(4.5*e*pi) == cos(0.5*pi*e)
    assert sin(4.5*e*pi) == sin(0.5*pi*e)
    assert tan(4.5*e*pi) == tan(0.5*pi*e)
    assert cot(4.5*e*pi) == cot(0.5*pi*e)

    assert cos(4*o*pi) == 1
    assert sin(4*o*pi) == 0
    assert tan(4*o*pi) == 0
    assert cot(4*o*pi) is zoo

    assert cos(3*o*pi) == -1
    assert sin(3*o*pi) == 0
    assert tan(3*o*pi) == 0
    assert cot(3*o*pi) is zoo

    assert cos(4.0*o*pi) == 1
    assert sin(4.0*o*pi) == 0
    assert tan(4.0*o*pi) == 0
    assert cot(4.0*o*pi) is zoo

    assert cos(3.0*o*pi) == -1
    assert sin(3.0*o*pi) == 0
    assert tan(3.0*o*pi) == 0
    assert cot(3.0*o*pi) is zoo

    assert cos(4.5*o*pi) == cos(0.5*pi*o)
    assert sin(4.5*o*pi) == sin(0.5*pi*o)
    assert tan(4.5*o*pi) == tan(0.5*pi*o)
    assert cot(4.5*o*pi) == cot(0.5*pi*o)

    # x could be imaginary
    assert cos(4*x*pi) == cos(4*pi*x)
    assert sin(4*x*pi) == sin(4*pi*x)
    assert tan(4*x*pi) == tan(4*pi*x)
    assert cot(4*x*pi) == cot(4*pi*x)

    assert cos(3*x*pi) == cos(3*pi*x)
    assert sin(3*x*pi) == sin(3*pi*x)
    assert tan(3*x*pi) == tan(3*pi*x)
    assert cot(3*x*pi) == cot(3*pi*x)

    assert cos(4.0*x*pi) == cos(4.0*pi*x)
    assert sin(4.0*x*pi) == sin(4.0*pi*x)
    assert tan(4.0*x*pi) == tan(4.0*pi*x)
    assert cot(4.0*x*pi) == cot(4.0*pi*x)

    assert cos(3.0*x*pi) == cos(3.0*pi*x)
    assert sin(3.0*x*pi) == sin(3.0*pi*x)
    assert tan(3.0*x*pi) == tan(3.0*pi*x)
    assert cot(3.0*x*pi) == cot(3.0*pi*x)

    assert cos(4.5*x*pi) == cos(4.5*pi*x)
    assert sin(4.5*x*pi) == sin(4.5*pi*x)
    assert tan(4.5*x*pi) == tan(4.5*pi*x)
    assert cot(4.5*x*pi) == cot(4.5*pi*x)


def test_inverses():
    raises(AttributeError, lambda: sin(x).inverse())
    raises(AttributeError, lambda: cos(x).inverse())
    assert tan(x).inverse() == atan
    assert cot(x).inverse() == acot
    raises(AttributeError, lambda: csc(x).inverse())
    raises(AttributeError, lambda: sec(x).inverse())
    assert asin(x).inverse() == sin
    assert acos(x).inverse() == cos
    assert atan(x).inverse() == tan
    assert acot(x).inverse() == cot


def test_real_imag():
    a, b = symbols('a b', real=True)
    z = a + b*I
    for deep in [True, False]:
        assert sin(
            z).as_real_imag(deep=deep) == (sin(a)*cosh(b), cos(a)*sinh(b))
        assert cos(
            z).as_real_imag(deep=deep) == (cos(a)*cosh(b), -sin(a)*sinh(b))
        assert tan(z).as_real_imag(deep=deep) == (sin(2*a)/(cos(2*a) +
            cosh(2*b)), sinh(2*b)/(cos(2*a) + cosh(2*b)))
        assert cot(z).as_real_imag(deep=deep) == (-sin(2*a)/(cos(2*a) -
            cosh(2*b)), sinh(2*b)/(cos(2*a) - cosh(2*b)))
        assert sin(a).as_real_imag(deep=deep) == (sin(a), 0)
        assert cos(a).as_real_imag(deep=deep) == (cos(a), 0)
        assert tan(a).as_real_imag(deep=deep) == (tan(a), 0)
        assert cot(a).as_real_imag(deep=deep) == (cot(a), 0)


@slow
def test_sincos_rewrite_sqrt():
    # equivalent to testing rewrite(pow)
    for p in [1, 3, 5, 17]:
        for t in [1, 8]:
            n = t*p
            # The vertices `exp(i*pi/n)` of a regular `n`-gon can
            # be expressed by means of nested square roots if and
            # only if `n` is a product of Fermat primes, `p`, and
            # powers of 2, `t'. The code aims to check all vertices
            # not belonging to an `m`-gon for `m < n`(`gcd(i, n) == 1`).
            # For large `n` this makes the test too slow, therefore
            # the vertices are limited to those of index `i < 10`.
            for i in range(1, min((n + 1)//2 + 1, 10)):
                if 1 == gcd(i, n):
                    x = i*pi/n
                    s1 = sin(x).rewrite(sqrt)
                    c1 = cos(x).rewrite(sqrt)
                    assert not s1.has(cos, sin), "fails for %d*pi/%d" % (i, n)
                    assert not c1.has(cos, sin), "fails for %d*pi/%d" % (i, n)
                    assert 1e-3 > abs(sin(x.evalf(5)) - s1.evalf(2)), "fails for %d*pi/%d" % (i, n)
                    assert 1e-3 > abs(cos(x.evalf(5)) - c1.evalf(2)), "fails for %d*pi/%d" % (i, n)
    assert cos(pi/14).rewrite(sqrt) == sqrt(cos(pi/7)/2 + S.Half)
    assert cos(pi*Rational(-15, 2)/11, evaluate=False).rewrite(
        sqrt) == -sqrt(-cos(pi*Rational(4, 11))/2 + S.Half)
    assert cos(Mul(2, pi, S.Half, evaluate=False), evaluate=False).rewrite(
        sqrt) == -1
    e = cos(pi/3/17)  # don't use pi/15 since that is caught at instantiation
    a = (
        -3*sqrt(-sqrt(17) + 17)*sqrt(sqrt(17) + 17)/64 -
        3*sqrt(34)*sqrt(sqrt(17) + 17)/128 - sqrt(sqrt(17) +
        17)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) + 17)
        + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/64 - sqrt(-sqrt(17)
        + 17)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/128 - Rational(1, 32) +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/64 +
        3*sqrt(2)*sqrt(sqrt(17) + 17)/128 + sqrt(34)*sqrt(-sqrt(17) + 17)/128
        + 13*sqrt(2)*sqrt(-sqrt(17) + 17)/128 + sqrt(17)*sqrt(-sqrt(17) +
        17)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) + 17)
        + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/128 + 5*sqrt(17)/32
        + sqrt(3)*sqrt(-sqrt(2)*sqrt(sqrt(17) + 17)*sqrt(sqrt(17)/32 +
        sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 + Rational(15, 32))/8 -
        5*sqrt(2)*sqrt(sqrt(17)/32 + sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 +
        Rational(15, 32))*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/64 -
        3*sqrt(2)*sqrt(-sqrt(17) + 17)*sqrt(sqrt(17)/32 +
        sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 + Rational(15, 32))/32
        + sqrt(34)*sqrt(sqrt(17)/32 + sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 +
        Rational(15, 32))*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/64 +
        sqrt(sqrt(17)/32 + sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 + Rational(15, 32))/2 +
        S.Half + sqrt(-sqrt(17) + 17)*sqrt(sqrt(17)/32 + sqrt(2)*sqrt(-sqrt(17) +
        17)/32 + sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) -
        sqrt(2)*sqrt(-sqrt(17) + 17) + sqrt(34)*sqrt(-sqrt(17) + 17) +
        6*sqrt(17) + 34)/32 + Rational(15, 32))*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) -
        sqrt(2)*sqrt(-sqrt(17) + 17) + sqrt(34)*sqrt(-sqrt(17) + 17) +
        6*sqrt(17) + 34)/32 + sqrt(34)*sqrt(-sqrt(17) + 17)*sqrt(sqrt(17)/32 +
        sqrt(2)*sqrt(-sqrt(17) + 17)/32 +
        sqrt(2)*sqrt(-8*sqrt(2)*sqrt(sqrt(17) + 17) - sqrt(2)*sqrt(-sqrt(17) +
        17) + sqrt(34)*sqrt(-sqrt(17) + 17) + 6*sqrt(17) + 34)/32 +
        Rational(15, 32))/32)/2)
    assert e.rewrite(sqrt) == a
    assert e.n() == a.n()
    # coverage of fermatCoords: multiplicity > 1; the following could be
    # different but that portion of the code should be tested in some way
    assert cos(pi/9/17).rewrite(sqrt) == \
        sin(pi/9)*sin(pi*Rational(2, 17)) + cos(pi/9)*cos(pi*Rational(2, 17))


@slow
def test_sincos_rewrite_sqrt_257():
    assert cos(pi/257).rewrite(sqrt).evalf(64) == cos(pi/257).evalf(64)


@slow
def test_tancot_rewrite_sqrt():
    # equivalent to testing rewrite(pow)
    for p in [1, 3, 5, 17]:
        for t in [1, 8]:
            n = t*p
            for i in range(1, min((n + 1)//2 + 1, 10)):
                if 1 == gcd(i, n):
                    x = i*pi/n
                    if  2*i != n and 3*i != 2*n:
                        t1 = tan(x).rewrite(sqrt)
                        assert not t1.has(cot, tan), "fails for %d*pi/%d" % (i, n)
                        assert 1e-3 > abs( tan(x.evalf(7)) - t1.evalf(4) ), "fails for %d*pi/%d" % (i, n)
                    if  i != 0 and i != n:
                        c1 = cot(x).rewrite(sqrt)
                        assert not c1.has(cot, tan), "fails for %d*pi/%d" % (i, n)
                        assert 1e-3 > abs( cot(x.evalf(7)) - c1.evalf(4) ), "fails for %d*pi/%d" % (i, n)


def test_sec():
    x = symbols('x', real=True)
    z = symbols('z')

    assert sec.nargs == FiniteSet(1)

    assert sec(zoo) is nan
    assert sec(0) == 1
    assert sec(pi) == -1
    assert sec(pi/2) is zoo
    assert sec(-pi/2) is zoo
    assert sec(pi/6) == 2*sqrt(3)/3
    assert sec(pi/3) == 2
    assert sec(pi*Rational(5, 2)) is zoo
    assert sec(pi*Rational(9, 7)) == -sec(pi*Rational(2, 7))
    assert sec(pi*Rational(3, 4)) == -sqrt(2)  # issue 8421
    assert sec(I) == 1/cosh(1)
    assert sec(x*I) == 1/cosh(x)
    assert sec(-x) == sec(x)

    assert sec(asec(x)) == x

    assert sec(z).conjugate() == sec(conjugate(z))

    assert (sec(z).as_real_imag() ==
    (cos(re(z))*cosh(im(z))/(sin(re(z))**2*sinh(im(z))**2 +
                             cos(re(z))**2*cosh(im(z))**2),
     sin(re(z))*sinh(im(z))/(sin(re(z))**2*sinh(im(z))**2 +
                             cos(re(z))**2*cosh(im(z))**2)))

    assert sec(x).expand(trig=True) == 1/cos(x)
    assert sec(2*x).expand(trig=True) == 1/(2*cos(x)**2 - 1)

    assert sec(x).is_extended_real == True
    assert sec(z).is_real == None

    assert sec(a).is_algebraic is None
    assert sec(na).is_algebraic is False

    assert sec(x).as_leading_term() == sec(x)

    assert sec(0, evaluate=False).is_finite == True
    assert sec(x).is_finite == None
    assert sec(pi/2, evaluate=False).is_finite == False

    assert series(sec(x), x, x0=0, n=6) == 1 + x**2/2 + 5*x**4/24 + O(x**6)

    # https://github.com/sympy/sympy/issues/7166
    assert series(sqrt(sec(x))) == 1 + x**2/4 + 7*x**4/96 + O(x**6)

    # https://github.com/sympy/sympy/issues/7167
    assert (series(sqrt(sec(x)), x, x0=pi*3/2, n=4) ==
            1/sqrt(x - pi*Rational(3, 2)) + (x - pi*Rational(3, 2))**Rational(3, 2)/12 +
            (x - pi*Rational(3, 2))**Rational(7, 2)/160 + O((x - pi*Rational(3, 2))**4, (x, pi*Rational(3, 2))))

    assert sec(x).diff(x) == tan(x)*sec(x)

    # Taylor Term checks
    assert sec(z).taylor_term(4, z) == 5*z**4/24
    assert sec(z).taylor_term(6, z) == 61*z**6/720
    assert sec(z).taylor_term(5, z) == 0


def test_sec_rewrite():
    assert sec(x).rewrite(exp) == 1/(exp(I*x)/2 + exp(-I*x)/2)
    assert sec(x).rewrite(cos) == 1/cos(x)
    assert sec(x).rewrite(tan) == (tan(x/2)**2 + 1)/(-tan(x/2)**2 + 1)
    assert sec(x).rewrite(pow) == sec(x)
    assert sec(x).rewrite(sqrt) == sec(x)
    assert sec(z).rewrite(cot) == (cot(z/2)**2 + 1)/(cot(z/2)**2 - 1)
    assert sec(x).rewrite(sin) == 1 / sin(x + pi / 2, evaluate=False)
    assert sec(x).rewrite(tan) == (tan(x / 2)**2 + 1) / (-tan(x / 2)**2 + 1)
    assert sec(x).rewrite(csc) == csc(-x + pi/2, evaluate=False)
    assert sec(x).rewrite(besselj) == Piecewise(
                (sqrt(2)/(sqrt(pi*x)*besselj(-S.Half, x)), Ne(x, 0)),
                (1, True)
            )
    assert sec(x).rewrite(besselj).subs(x, 0) == sec(0)


def test_sec_fdiff():
    assert sec(x).fdiff() == tan(x)*sec(x)
    raises(ArgumentIndexError, lambda: sec(x).fdiff(2))


def test_csc():
    x = symbols('x', real=True)
    z = symbols('z')

    # https://github.com/sympy/sympy/issues/6707
    cosecant = csc('x')
    alternate = 1/sin('x')
    assert cosecant.equals(alternate) == True
    assert alternate.equals(cosecant) == True

    assert csc.nargs == FiniteSet(1)

    assert csc(0) is zoo
    assert csc(pi) is zoo
    assert csc(zoo) is nan

    assert csc(pi/2) == 1
    assert csc(-pi/2) == -1
    assert csc(pi/6) == 2
    assert csc(pi/3) == 2*sqrt(3)/3
    assert csc(pi*Rational(5, 2)) == 1
    assert csc(pi*Rational(9, 7)) == -csc(pi*Rational(2, 7))
    assert csc(pi*Rational(3, 4)) == sqrt(2)  # issue 8421
    assert csc(I) == -I/sinh(1)
    assert csc(x*I) == -I/sinh(x)
    assert csc(-x) == -csc(x)

    assert csc(acsc(x)) == x

    assert csc(z).conjugate() == csc(conjugate(z))

    assert (csc(z).as_real_imag() ==
            (sin(re(z))*cosh(im(z))/(sin(re(z))**2*cosh(im(z))**2 +
                                     cos(re(z))**2*sinh(im(z))**2),
             -cos(re(z))*sinh(im(z))/(sin(re(z))**2*cosh(im(z))**2 +
                          cos(re(z))**2*sinh(im(z))**2)))

    assert csc(x).expand(trig=True) == 1/sin(x)
    assert csc(2*x).expand(trig=True) == 1/(2*sin(x)*cos(x))

    assert csc(x).is_extended_real == True
    assert csc(z).is_real == None

    assert csc(a).is_algebraic is None
    assert csc(na).is_algebraic is False

    assert csc(x).as_leading_term() == csc(x)

    assert csc(0, evaluate=False).is_finite == False
    assert csc(x).is_finite == None
    assert csc(pi/2, evaluate=False).is_finite == True

    assert series(csc(x), x, x0=pi/2, n=6) == \
        1 + (x - pi/2)**2/2 + 5*(x - pi/2)**4/24 + O((x - pi/2)**6, (x, pi/2))
    assert series(csc(x), x, x0=0, n=6) == \
            1/x + x/6 + 7*x**3/360 + 31*x**5/15120 + O(x**6)

    assert csc(x).diff(x) == -cot(x)*csc(x)

    assert csc(x).taylor_term(2, x) == 0
    assert csc(x).taylor_term(3, x) == 7*x**3/360
    assert csc(x).taylor_term(5, x) == 31*x**5/15120
    raises(ArgumentIndexError, lambda: csc(x).fdiff(2))


def test_asec():
    z = Symbol('z', zero=True)
    assert asec(z) is zoo
    assert asec(nan) is nan
    assert asec(1) == 0
    assert asec(-1) == pi
    assert asec(oo) == pi/2
    assert asec(-oo) == pi/2
    assert asec(zoo) == pi/2

    assert asec(sec(pi*Rational(13, 4))) == pi*Rational(3, 4)
    assert asec(1 + sqrt(5)) == pi*Rational(2, 5)
    assert asec(2/sqrt(3)) == pi/6
    assert asec(sqrt(4 - 2*sqrt(2))) == pi/8
    assert asec(-sqrt(4 + 2*sqrt(2))) == pi*Rational(5, 8)
    assert asec(sqrt(2 + 2*sqrt(5)/5)) == pi*Rational(3, 10)
    assert asec(-sqrt(2 + 2*sqrt(5)/5)) == pi*Rational(7, 10)
    assert asec(sqrt(2) - sqrt(6)) == pi*Rational(11, 12)

    for d in [3, 4, 6]:
        for num in range(d):
            if gcd(num, d) == 1:
                assert asec(sec(num*pi/d)) == num*pi/d
                assert asec(-sec(num*pi/d)) == pi - num*pi/d
                assert asec(csc(num*pi/d)) == pi/2 - acsc(csc(num*pi/d))
                assert asec(-csc(num*pi/d)) == pi/2 - acsc(-csc(num*pi/d))

    assert asec(x).diff(x) == 1/(x**2*sqrt(1 - 1/x**2))

    assert asec(x).rewrite(log) == I*log(sqrt(1 - 1/x**2) + I/x) + pi/2
    assert asec(x).rewrite(asin) == -asin(1/x) + pi/2
    assert asec(x).rewrite(acos) == acos(1/x)
    assert asec(x).rewrite(atan) == \
        pi*(1 - sqrt(x**2)/x)/2 + sqrt(x**2)*atan(sqrt(x**2 - 1))/x
    assert asec(x).rewrite(acot) == \
        pi*(1 - sqrt(x**2)/x)/2 + sqrt(x**2)*acot(1/sqrt(x**2 - 1))/x
    assert asec(x).rewrite(acsc) == -acsc(x) + pi/2
    raises(ArgumentIndexError, lambda: asec(x).fdiff(2))


def test_asec_is_real():
    assert asec(S.Half).is_real is False
    n = Symbol('n', positive=True, integer=True)
    assert asec(n).is_extended_real is True
    assert asec(x).is_real is None
    assert asec(r).is_real is None
    t = Symbol('t', real=False, finite=True)
    assert asec(t).is_real is False


def test_asec_leading_term():
    assert asec(1/x).as_leading_term(x) == pi/2
    # Tests concerning branch points
    assert asec(x + 1).as_leading_term(x) == sqrt(2)*sqrt(x)
    assert asec(x - 1).as_leading_term(x) == pi
    # Tests concerning points lying on branch cuts
    assert asec(x).as_leading_term(x, cdir=1) == -I*log(x) + I*log(2)
    assert asec(x).as_leading_term(x, cdir=-1) == I*log(x) + 2*pi - I*log(2)
    assert asec(I*x + 1/2).as_leading_term(x, cdir=1) == asec(1/2)
    assert asec(-I*x + 1/2).as_leading_term(x, cdir=1) == -asec(1/2)
    assert asec(I*x - 1/2).as_leading_term(x, cdir=1) == 2*pi - asec(-1/2)
    assert asec(-I*x - 1/2).as_leading_term(x, cdir=1) == asec(-1/2)
    # Tests concerning im(ndir) == 0
    assert asec(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=1) == pi + I*log(2 - sqrt(3))
    assert asec(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=-1) == pi + I*log(2 - sqrt(3))


def test_asec_series():
    assert asec(x).series(x, 0, 9) == \
        I*log(2) - I*log(x) - I*x**2/4 - 3*I*x**4/32 \
        - 5*I*x**6/96 - 35*I*x**8/1024 + O(x**9)
    t4 = asec(x).taylor_term(4, x)
    assert t4 == -3*I*x**4/32
    assert asec(x).taylor_term(6, x, t4, 0) == -5*I*x**6/96


def test_acsc():
    assert acsc(nan) is nan
    assert acsc(1) == pi/2
    assert acsc(-1) == -pi/2
    assert acsc(oo) == 0
    assert acsc(-oo) == 0
    assert acsc(zoo) == 0
    assert acsc(0) is zoo

    assert acsc(csc(3)) == -3 + pi
    assert acsc(csc(4)) == -4 + pi
    assert acsc(csc(6)) == 6 - 2*pi
    assert unchanged(acsc, csc(x))
    assert unchanged(acsc, sec(x))

    assert acsc(2/sqrt(3)) == pi/3
    assert acsc(csc(pi*Rational(13, 4))) == -pi/4
    assert acsc(sqrt(2 + 2*sqrt(5)/5)) == pi/5
    assert acsc(-sqrt(2 + 2*sqrt(5)/5)) == -pi/5
    assert acsc(-2) == -pi/6
    assert acsc(-sqrt(4 + 2*sqrt(2))) == -pi/8
    assert acsc(sqrt(4 - 2*sqrt(2))) == pi*Rational(3, 8)
    assert acsc(1 + sqrt(5)) == pi/10
    assert acsc(sqrt(2) - sqrt(6)) == pi*Rational(-5, 12)

    assert acsc(x).diff(x) == -1/(x**2*sqrt(1 - 1/x**2))

    assert acsc(x).rewrite(log) == -I*log(sqrt(1 - 1/x**2) + I/x)
    assert acsc(x).rewrite(asin) == asin(1/x)
    assert acsc(x).rewrite(acos) == -acos(1/x) + pi/2
    assert acsc(x).rewrite(atan) == \
        (-atan(sqrt(x**2 - 1)) + pi/2)*sqrt(x**2)/x
    assert acsc(x).rewrite(acot) == (-acot(1/sqrt(x**2 - 1)) + pi/2)*sqrt(x**2)/x
    assert acsc(x).rewrite(asec) == -asec(x) + pi/2
    raises(ArgumentIndexError, lambda: acsc(x).fdiff(2))


def test_csc_rewrite():
    assert csc(x).rewrite(pow) == csc(x)
    assert csc(x).rewrite(sqrt) == csc(x)

    assert csc(x).rewrite(exp) == 2*I/(exp(I*x) - exp(-I*x))
    assert csc(x).rewrite(sin) == 1/sin(x)
    assert csc(x).rewrite(tan) == (tan(x/2)**2 + 1)/(2*tan(x/2))
    assert csc(x).rewrite(cot) == (cot(x/2)**2 + 1)/(2*cot(x/2))
    assert csc(x).rewrite(cos) == 1/cos(x - pi/2, evaluate=False)
    assert csc(x).rewrite(sec) == sec(-x + pi/2, evaluate=False)

    # issue 17349
    assert csc(1 - exp(-besselj(I, I))).rewrite(cos) == \
           -1/cos(-pi/2 - 1 + cos(I*besselj(I, I)) +
                  I*cos(-pi/2 + I*besselj(I, I), evaluate=False), evaluate=False)
    assert csc(x).rewrite(besselj) == sqrt(2)/(sqrt(pi*x)*besselj(S.Half, x))
    assert csc(x).rewrite(besselj).subs(x, 0) == csc(0)


def test_acsc_leading_term():
    assert acsc(1/x).as_leading_term(x) == x
    # Tests concerning branch points
    assert acsc(x + 1).as_leading_term(x) == pi/2
    assert acsc(x - 1).as_leading_term(x) == -pi/2
    # Tests concerning points lying on branch cuts
    assert acsc(x).as_leading_term(x, cdir=1) == I*log(x) + pi/2 - I*log(2)
    assert acsc(x).as_leading_term(x, cdir=-1) == -I*log(x) - 3*pi/2 + I*log(2)
    assert acsc(I*x + 1/2).as_leading_term(x, cdir=1) == acsc(1/2)
    assert acsc(-I*x + 1/2).as_leading_term(x, cdir=1) == pi - acsc(1/2)
    assert acsc(I*x - 1/2).as_leading_term(x, cdir=1) == -pi - acsc(-1/2)
    assert acsc(-I*x - 1/2).as_leading_term(x, cdir=1) == -acsc(1/2)
    # Tests concerning im(ndir) == 0
    assert acsc(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=1) == -pi/2 + I*log(sqrt(3) + 2)
    assert acsc(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=-1) == -pi/2 + I*log(sqrt(3) + 2)


def test_acsc_series():
    assert acsc(x).series(x, 0, 9) == \
        -I*log(2) + pi/2 + I*log(x) + I*x**2/4 \
        + 3*I*x**4/32 + 5*I*x**6/96 + 35*I*x**8/1024 + O(x**9)
    t6 = acsc(x).taylor_term(6, x)
    assert t6 == 5*I*x**6/96
    assert acsc(x).taylor_term(8, x, t6, 0) == 35*I*x**8/1024


def test_asin_nseries():
    assert asin(x + 2)._eval_nseries(x, 4, None, I) == -asin(2) + pi + \
    sqrt(3)*I*x/3 - sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert asin(x + 2)._eval_nseries(x, 4, None, -I) == asin(2) - \
    sqrt(3)*I*x/3 + sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asin(x - 2)._eval_nseries(x, 4, None, I) == -asin(2) - \
    sqrt(3)*I*x/3 - sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asin(x - 2)._eval_nseries(x, 4, None, -I) == asin(2) - pi + \
    sqrt(3)*I*x/3 + sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    # testing nseries for asin at branch points
    assert asin(1 + x)._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(-x) - \
    sqrt(2)*(-x)**(S(3)/2)/12 - 3*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)
    assert asin(-1 + x)._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(x) + \
    sqrt(2)*x**(S(3)/2)/12 + 3*sqrt(2)*x**(S(5)/2)/160 + O(x**3)
    assert asin(exp(x))._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(-x) + \
    sqrt(2)*(-x)**(S(3)/2)/6 - sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)
    assert asin(-exp(x))._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(-x) - \
    sqrt(2)*(-x)**(S(3)/2)/6 + sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)


def test_acos_nseries():
    assert acos(x + 2)._eval_nseries(x, 4, None, I) == -acos(2) - sqrt(3)*I*x/3 + \
    sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert acos(x + 2)._eval_nseries(x, 4, None, -I) == acos(2) + sqrt(3)*I*x/3 - \
    sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert acos(x - 2)._eval_nseries(x, 4, None, I) == acos(-2) + sqrt(3)*I*x/3 + \
    sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert acos(x - 2)._eval_nseries(x, 4, None, -I) == -acos(-2) + 2*pi - \
    sqrt(3)*I*x/3 - sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    # testing nseries for acos at branch points
    assert acos(1 + x)._eval_nseries(x, 3, None) == sqrt(2)*sqrt(-x) + \
    sqrt(2)*(-x)**(S(3)/2)/12 + 3*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)
    assert acos(-1 + x)._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(x) - \
    sqrt(2)*x**(S(3)/2)/12 - 3*sqrt(2)*x**(S(5)/2)/160 + O(x**3)
    assert acos(exp(x))._eval_nseries(x, 3, None) == sqrt(2)*sqrt(-x) - \
    sqrt(2)*(-x)**(S(3)/2)/6 + sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)
    assert acos(-exp(x))._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(-x) + \
    sqrt(2)*(-x)**(S(3)/2)/6 - sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)


def test_atan_nseries():
    assert atan(x + 2*I)._eval_nseries(x, 4, None, 1) == I*atanh(2) - x/3 - \
    2*I*x**2/9 + 13*x**3/81 + O(x**4)
    assert atan(x + 2*I)._eval_nseries(x, 4, None, -1) == I*atanh(2) - pi - \
    x/3 - 2*I*x**2/9 + 13*x**3/81 + O(x**4)
    assert atan(x - 2*I)._eval_nseries(x, 4, None, 1) == -I*atanh(2) + pi - \
    x/3 + 2*I*x**2/9 + 13*x**3/81 + O(x**4)
    assert atan(x - 2*I)._eval_nseries(x, 4, None, -1) == -I*atanh(2) - x/3 + \
    2*I*x**2/9 + 13*x**3/81 + O(x**4)
    assert atan(1/x)._eval_nseries(x, 2, None, 1) == pi/2 - x + O(x**2)
    assert atan(1/x)._eval_nseries(x, 2, None, -1) == -pi/2 - x + O(x**2)
    # testing nseries for atan at branch points
    assert atan(x + I)._eval_nseries(x, 4, None) == I*log(2)/2 + pi/4 - \
    I*log(x)/2 + x/4 + I*x**2/16 - x**3/48 + O(x**4)
    assert atan(x - I)._eval_nseries(x, 4, None) == -I*log(2)/2 + pi/4 + \
    I*log(x)/2 + x/4 - I*x**2/16 - x**3/48 + O(x**4)


def test_acot_nseries():
    assert acot(x + S(1)/2*I)._eval_nseries(x, 4, None, 1) == -I*acoth(S(1)/2) + \
    pi - 4*x/3 + 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    assert acot(x + S(1)/2*I)._eval_nseries(x, 4, None, -1) == -I*acoth(S(1)/2) - \
    4*x/3 + 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    assert acot(x - S(1)/2*I)._eval_nseries(x, 4, None, 1) == I*acoth(S(1)/2) - \
    4*x/3 - 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    assert acot(x - S(1)/2*I)._eval_nseries(x, 4, None, -1) == I*acoth(S(1)/2) - \
    pi - 4*x/3 - 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    assert acot(x)._eval_nseries(x, 2, None, 1) == pi/2 - x + O(x**2)
    assert acot(x)._eval_nseries(x, 2, None, -1) == -pi/2 - x + O(x**2)
    # testing nseries for acot at branch points
    assert acot(x + I)._eval_nseries(x, 4, None) == -I*log(2)/2 + pi/4 + \
    I*log(x)/2 - x/4 - I*x**2/16 + x**3/48 + O(x**4)
    assert acot(x - I)._eval_nseries(x, 4, None) == I*log(2)/2 + pi/4 - \
    I*log(x)/2 - x/4 + I*x**2/16 + x**3/48 + O(x**4)


def test_asec_nseries():
    assert asec(x + S(1)/2)._eval_nseries(x, 4, None, I) == asec(S(1)/2) - \
    4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert asec(x + S(1)/2)._eval_nseries(x, 4, None, -I) == -asec(S(1)/2) + \
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert asec(x - S(1)/2)._eval_nseries(x, 4, None, I) == -asec(-S(1)/2) + \
    2*pi + 4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert asec(x - S(1)/2)._eval_nseries(x, 4, None, -I) == asec(-S(1)/2) - \
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    # testing nseries for asec at branch points
    assert asec(1 + x)._eval_nseries(x, 3, None) == sqrt(2)*sqrt(x) - \
    5*sqrt(2)*x**(S(3)/2)/12 + 43*sqrt(2)*x**(S(5)/2)/160 + O(x**3)
    assert asec(-1 + x)._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(-x) + \
    5*sqrt(2)*(-x)**(S(3)/2)/12 - 43*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)
    assert asec(exp(x))._eval_nseries(x, 3, None) == sqrt(2)*sqrt(x) - \
    sqrt(2)*x**(S(3)/2)/6 + sqrt(2)*x**(S(5)/2)/120 + O(x**3)
    assert asec(-exp(x))._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(x) + \
    sqrt(2)*x**(S(3)/2)/6 - sqrt(2)*x**(S(5)/2)/120 + O(x**3)


def test_acsc_nseries():
    assert acsc(x + S(1)/2)._eval_nseries(x, 4, None, I) == acsc(S(1)/2) + \
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsc(x + S(1)/2)._eval_nseries(x, 4, None, -I) == -acsc(S(1)/2) + \
    pi - 4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsc(x - S(1)/2)._eval_nseries(x, 4, None, I) == acsc(S(1)/2) - pi -\
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsc(x - S(1)/2)._eval_nseries(x, 4, None, -I) == -acsc(S(1)/2) + \
    4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    # testing nseries for acsc at branch points
    assert acsc(1 + x)._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(x) + \
    5*sqrt(2)*x**(S(3)/2)/12 - 43*sqrt(2)*x**(S(5)/2)/160 + O(x**3)
    assert acsc(-1 + x)._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(-x) - \
    5*sqrt(2)*(-x)**(S(3)/2)/12 + 43*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)
    assert acsc(exp(x))._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(x) + \
    sqrt(2)*x**(S(3)/2)/6 - sqrt(2)*x**(S(5)/2)/120 + O(x**3)
    assert acsc(-exp(x))._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(x) - \
    sqrt(2)*x**(S(3)/2)/6 + sqrt(2)*x**(S(5)/2)/120 + O(x**3)


def test_issue_8653():
    n = Symbol('n', integer=True)
    assert sin(n).is_irrational is None
    assert cos(n).is_irrational is None
    assert tan(n).is_irrational is None


def test_issue_9157():
    n = Symbol('n', integer=True, positive=True)
    assert atan(n - 1).is_nonnegative is True


def test_trig_period():
    x, y = symbols('x, y')

    assert sin(x).period() == 2*pi
    assert cos(x).period() == 2*pi
    assert tan(x).period() == pi
    assert cot(x).period() == pi
    assert sec(x).period() == 2*pi
    assert csc(x).period() == 2*pi
    assert sin(2*x).period() == pi
    assert cot(4*x - 6).period() == pi/4
    assert cos((-3)*x).period() == pi*Rational(2, 3)
    assert cos(x*y).period(x) == 2*pi/abs(y)
    assert sin(3*x*y + 2*pi).period(y) == 2*pi/abs(3*x)
    assert tan(3*x).period(y) is S.Zero
    raises(NotImplementedError, lambda: sin(x**2).period(x))


def test_issue_7171():
    assert sin(x).rewrite(sqrt) == sin(x)
    assert sin(x).rewrite(pow) == sin(x)


def test_issue_11864():
    w, k = symbols('w, k', real=True)
    F = Piecewise((1, Eq(2*pi*k, 0)), (sin(pi*k)/(pi*k), True))
    soln = Piecewise((1, Eq(2*pi*k, 0)), (sinc(pi*k), True))
    assert F.rewrite(sinc) == soln

def test_real_assumptions():
    z = Symbol('z', real=False, finite=True)
    assert sin(z).is_real is None
    assert cos(z).is_real is None
    assert tan(z).is_real is False
    assert sec(z).is_real is None
    assert csc(z).is_real is None
    assert cot(z).is_real is False
    assert asin(p).is_real is None
    assert asin(n).is_real is None
    assert asec(p).is_real is None
    assert asec(n).is_real is None
    assert acos(p).is_real is None
    assert acos(n).is_real is None
    assert acsc(p).is_real is None
    assert acsc(n).is_real is None
    assert atan(p).is_positive is True
    assert atan(n).is_negative is True
    assert acot(p).is_positive is True
    assert acot(n).is_negative is True

def test_issue_14320():
    assert asin(sin(2)) == -2 + pi and (-pi/2 <= -2 + pi <= pi/2) and sin(2) == sin(-2 + pi)
    assert asin(cos(2)) == -2 + pi/2 and (-pi/2 <= -2 + pi/2 <= pi/2) and cos(2) == sin(-2 + pi/2)
    assert acos(sin(2)) == -pi/2 + 2 and (0 <= -pi/2 + 2 <= pi) and sin(2) == cos(-pi/2 + 2)
    assert acos(cos(20)) == -6*pi + 20 and (0 <= -6*pi + 20 <= pi) and cos(20) == cos(-6*pi + 20)
    assert acos(cos(30)) == -30 + 10*pi and (0 <= -30 + 10*pi <= pi) and cos(30) == cos(-30 + 10*pi)

    assert atan(tan(17)) == -5*pi + 17 and (-pi/2 < -5*pi + 17 < pi/2) and tan(17) == tan(-5*pi + 17)
    assert atan(tan(15)) == -5*pi + 15 and (-pi/2 < -5*pi + 15 < pi/2) and tan(15) == tan(-5*pi + 15)
    assert atan(cot(12)) == -12 + pi*Rational(7, 2) and (-pi/2 < -12 + pi*Rational(7, 2) < pi/2) and cot(12) == tan(-12 + pi*Rational(7, 2))
    assert acot(cot(15)) == -5*pi + 15 and (-pi/2 < -5*pi + 15 <= pi/2) and cot(15) == cot(-5*pi + 15)
    assert acot(tan(19)) == -19 + pi*Rational(13, 2) and (-pi/2 < -19 + pi*Rational(13, 2) <= pi/2) and tan(19) == cot(-19 + pi*Rational(13, 2))

    assert asec(sec(11)) == -11 + 4*pi and (0 <= -11 + 4*pi <= pi) and cos(11) == cos(-11 + 4*pi)
    assert asec(csc(13)) == -13 + pi*Rational(9, 2) and (0 <= -13 + pi*Rational(9, 2) <= pi) and sin(13) == cos(-13 + pi*Rational(9, 2))
    assert acsc(csc(14)) == -4*pi + 14 and (-pi/2 <= -4*pi + 14 <= pi/2) and sin(14) == sin(-4*pi + 14)
    assert acsc(sec(10)) == pi*Rational(-7, 2) + 10 and (-pi/2 <= pi*Rational(-7, 2) + 10 <= pi/2) and cos(10) == sin(pi*Rational(-7, 2) + 10)

def test_issue_14543():
    assert sec(2*pi + 11) == sec(11)
    assert sec(2*pi - 11) == sec(11)
    assert sec(pi + 11) == -sec(11)
    assert sec(pi - 11) == -sec(11)

    assert csc(2*pi + 17) == csc(17)
    assert csc(2*pi - 17) == -csc(17)
    assert csc(pi + 17) == -csc(17)
    assert csc(pi - 17) == csc(17)

    x = Symbol('x')
    assert csc(pi/2 + x) == sec(x)
    assert csc(pi/2 - x) == sec(x)
    assert csc(pi*Rational(3, 2) + x) == -sec(x)
    assert csc(pi*Rational(3, 2) - x) == -sec(x)

    assert sec(pi/2 - x) == csc(x)
    assert sec(pi/2 + x) == -csc(x)
    assert sec(pi*Rational(3, 2) + x) == csc(x)
    assert sec(pi*Rational(3, 2) - x) == -csc(x)


def test_as_real_imag():
    # This is for https://github.com/sympy/sympy/issues/17142
    # If it start failing again in irrelevant builds or in the master
    # please open up the issue again.
    expr = atan(I/(I + I*tan(1)))
    assert expr.as_real_imag() == (expr, 0)


def test_issue_18746():
    e3 = cos(S.Pi*(x/4 + 1/4))
    assert e3.period() == 8


def test_issue_25833():
    assert limit(atan(x**2), x, oo) == pi/2
    assert limit(atan(x**2 - 1), x, oo) == pi/2
    assert limit(atan(log(2**x)/log(2*x)), x, oo) == pi/2


def test_issue_25847():
    #atan
    assert atan(sin(x)/x).as_leading_term(x) == pi/4
    raises(PoleError, lambda: atan(exp(1/x)).as_leading_term(x))

    #asin
    assert asin(sin(x)/x).as_leading_term(x) == pi/2
    raises(PoleError, lambda: asin(exp(1/x)).as_leading_term(x))

    #acos
    assert acos(sin(x)/x).as_leading_term(x) == 0
    raises(PoleError, lambda: acos(exp(1/x)).as_leading_term(x))

    #acot
    assert acot(sin(x)/x).as_leading_term(x) == pi/4
    raises(PoleError, lambda: acot(exp(1/x)).as_leading_term(x))

    #asec
    assert asec(sin(x)/x).as_leading_term(x) == 0
    raises(PoleError, lambda: asec(exp(1/x)).as_leading_term(x))

    #acsc
    assert acsc(sin(x)/x).as_leading_term(x) == pi/2
    raises(PoleError, lambda: acsc(exp(1/x)).as_leading_term(x))

def test_issue_23843():
    #atan
    assert atan(x + I).series(x, oo) == -16/(5*x**5) - 2*I/x**4 + 4/(3*x**3) + I/x**2 - 1/x + pi/2 + O(x**(-6), (x, oo))
    assert atan(x + I).series(x, -oo) == -16/(5*x**5) - 2*I/x**4 + 4/(3*x**3) + I/x**2 - 1/x - pi/2 + O(x**(-6), (x, -oo))
    assert atan(x - I).series(x, oo) == -16/(5*x**5) + 2*I/x**4 + 4/(3*x**3) - I/x**2 - 1/x + pi/2 + O(x**(-6), (x, oo))
    assert atan(x - I).series(x, -oo) == -16/(5*x**5) + 2*I/x**4 + 4/(3*x**3) - I/x**2 - 1/x - pi/2 + O(x**(-6), (x, -oo))

    #acot
    assert acot(x + I).series(x, oo) == 16/(5*x**5) + 2*I/x**4 - 4/(3*x**3) - I/x**2 + 1/x + O(x**(-6), (x, oo))
    assert acot(x + I).series(x, -oo) == 16/(5*x**5) + 2*I/x**4 - 4/(3*x**3) - I/x**2 + 1/x + O(x**(-6), (x, -oo))
    assert acot(x - I).series(x, oo) == 16/(5*x**5) - 2*I/x**4 - 4/(3*x**3) + I/x**2 + 1/x + O(x**(-6), (x, oo))
    assert acot(x - I).series(x, -oo) == 16/(5*x**5) - 2*I/x**4 - 4/(3*x**3) + I/x**2 + 1/x + O(x**(-6), (x, -oo))
