from itertools import product

from sympy.concrete.summations import Sum
from sympy.core.function import (Function, diff)
from sympy.core import EulerGamma, GoldenRatio
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.combinatorial.factorials import (binomial, factorial, subfactorial)
from sympy.functions.elementary.complexes import (Abs, re, sign)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (atanh, asinh, acosh, acoth, acsch, asech, tanh, sinh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, real_root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin,
                                                      atan, cos, cot, csc, sec, sin, tan)
from sympy.functions.special.bessel import (besseli, bessely, besselj, besselk)
from sympy.functions.special.error_functions import (Ei, erf, erfc, erfi, fresnelc, fresnels)
from sympy.functions.special.gamma_functions import (digamma, gamma, uppergamma)
from sympy.functions.special.hyper import meijerg
from sympy.integrals.integrals import (Integral, integrate)
from sympy.series.limits import (Limit, limit)
from sympy.simplify.simplify import (logcombine, simplify)
from sympy.simplify.hyperexpand import hyperexpand

from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.mul import Mul
from sympy.series.limits import heuristics
from sympy.series.order import Order
from sympy.testing.pytest import XFAIL, raises

from sympy import elliptic_e, elliptic_k

from sympy.abc import x, y, z, k
n = Symbol('n', integer=True, positive=True)


def test_basic1():
    assert limit(x, x, oo) is oo
    assert limit(x, x, -oo) is -oo
    assert limit(-x, x, oo) is -oo
    assert limit(x**2, x, -oo) is oo
    assert limit(-x**2, x, oo) is -oo
    assert limit(x*log(x), x, 0, dir="+") == 0
    assert limit(1/x, x, oo) == 0
    assert limit(exp(x), x, oo) is oo
    assert limit(-exp(x), x, oo) is -oo
    assert limit(exp(x)/x, x, oo) is oo
    assert limit(1/x - exp(-x), x, oo) == 0
    assert limit(x + 1/x, x, oo) is oo
    assert limit(x - x**2, x, oo) is -oo
    assert limit((1 + x)**(1 + sqrt(2)), x, 0) == 1
    assert limit((1 + x)**oo, x, 0) == Limit((x + 1)**oo, x, 0)
    assert limit((1 + x)**oo, x, 0, dir='-') == Limit((x + 1)**oo, x, 0, dir='-')
    assert limit((1 + x + y)**oo, x, 0, dir='-') == Limit((1 + x + y)**oo, x, 0, dir='-')
    assert limit(y/x/log(x), x, 0) == -oo*sign(y)
    assert limit(cos(x + y)/x, x, 0) == sign(cos(y))*oo
    assert limit(gamma(1/x + 3), x, oo) == 2
    assert limit(S.NaN, x, -oo) is S.NaN
    assert limit(Order(2)*x, x, S.NaN) is S.NaN
    assert limit(1/(x - 1), x, 1, dir="+") is oo
    assert limit(1/(x - 1), x, 1, dir="-") is -oo
    assert limit(1/(5 - x)**3, x, 5, dir="+") is -oo
    assert limit(1/(5 - x)**3, x, 5, dir="-") is oo
    assert limit(1/sin(x), x, pi, dir="+") is -oo
    assert limit(1/sin(x), x, pi, dir="-") is oo
    assert limit(1/cos(x), x, pi/2, dir="+") is -oo
    assert limit(1/cos(x), x, pi/2, dir="-") is oo
    assert limit(1/tan(x**3), x, (2*pi)**Rational(1, 3), dir="+") is oo
    assert limit(1/tan(x**3), x, (2*pi)**Rational(1, 3), dir="-") is -oo
    assert limit(1/cot(x)**3, x, (pi*Rational(3, 2)), dir="+") is -oo
    assert limit(1/cot(x)**3, x, (pi*Rational(3, 2)), dir="-") is oo
    assert limit(tan(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(cot(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(sec(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(csc(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)

    # test bi-directional limits
    assert limit(sin(x)/x, x, 0, dir="+-") == 1
    assert limit(x**2, x, 0, dir="+-") == 0
    assert limit(1/x**2, x, 0, dir="+-") is oo

    # test failing bi-directional limits
    assert limit(1/x, x, 0, dir="+-") is zoo
    # approaching 0
    # from dir="+"
    assert limit(1 + 1/x, x, 0) is oo
    # from dir='-'
    # Add
    assert limit(1 + 1/x, x, 0, dir='-') is -oo
    # Pow
    assert limit(x**(-2), x, 0, dir='-') is oo
    assert limit(x**(-3), x, 0, dir='-') is -oo
    assert limit(1/sqrt(x), x, 0, dir='-') == (-oo)*I
    assert limit(x**2, x, 0, dir='-') == 0
    assert limit(sqrt(x), x, 0, dir='-') == 0
    assert limit(x**-pi, x, 0, dir='-') == -oo*(-1)**(1 - pi)
    assert limit((1 + cos(x))**oo, x, 0) == Limit((cos(x) + 1)**oo, x, 0)

    # test pull request 22491
    assert limit(1/asin(x), x, 0, dir = '+') == oo
    assert limit(1/asin(x), x, 0, dir = '-') == -oo
    assert limit(1/sinh(x), x, 0, dir = '+') == oo
    assert limit(1/sinh(x), x, 0, dir = '-') == -oo
    assert limit(log(1/x) + 1/sin(x), x, 0, dir = '+') == oo
    assert limit(log(1/x) + 1/x, x, 0, dir = '+') == oo


def test_basic2():
    assert limit(x**x, x, 0, dir="+") == 1
    assert limit((exp(x) - 1)/x, x, 0) == 1
    assert limit(1 + 1/x, x, oo) == 1
    assert limit(-exp(1/x), x, oo) == -1
    assert limit(x + exp(-x), x, oo) is oo
    assert limit(x + exp(-x**2), x, oo) is oo
    assert limit(x + exp(-exp(x)), x, oo) is oo
    assert limit(13 + 1/x - exp(-x), x, oo) == 13


def test_basic3():
    assert limit(1/x, x, 0, dir="+") is oo
    assert limit(1/x, x, 0, dir="-") is -oo


def test_basic4():
    assert limit(2*x + y*x, x, 0) == 0
    assert limit(2*x + y*x, x, 1) == 2 + y
    assert limit(2*x**8 + y*x**(-3), x, -2) == 512 - y/8
    assert limit(sqrt(x + 1) - sqrt(x), x, oo) == 0
    assert integrate(1/(x**3 + 1), (x, 0, oo)) == 2*pi*sqrt(3)/9


def test_log():
    # https://github.com/sympy/sympy/issues/21598
    a, b, c = symbols('a b c', positive=True)
    A = log(a/b) - (log(a) - log(b))
    assert A.limit(a, oo) == 0
    assert (A * c).limit(a, oo) == 0

    tau, x = symbols('tau x', positive=True)
    # The value of manualintegrate in the issue
    expr = tau**2*((tau - 1)*(tau + 1)*log(x + 1)/(tau**2 + 1)**2 + 1/((tau**2\
            + 1)*(x + 1)) - (-2*tau*atan(x/tau) + (tau**2/2 - 1/2)*log(tau**2\
            + x**2))/(tau**2 + 1)**2)
    assert limit(expr, x, oo) == pi*tau**3/(tau**2 + 1)**2


def test_piecewise():
    # https://github.com/sympy/sympy/issues/18363
    assert limit((real_root(x - 6, 3) + 2)/(x + 2), x, -2, '+') == Rational(1, 12)


def test_piecewise2():
    func1 = 2*sqrt(x)*Piecewise(((4*x - 2)/Abs(sqrt(4 - 4*(2*x - 1)**2)), 4*x - 2\
            >= 0), ((2 - 4*x)/Abs(sqrt(4 - 4*(2*x - 1)**2)), True))
    func2 = Piecewise((x**2/2, x <= 0.5), (x/2 - 0.125, True))
    func3 = Piecewise(((x - 9) / 5, x < -1), ((x - 9) / 5, x > 4), (sqrt(Abs(x - 3)), True))
    assert limit(func1, x, 0) == 1
    assert limit(func2, x, 0) == 0
    assert limit(func3, x, -1) == 2


def test_basic5():
    class my(Function):
        @classmethod
        def eval(cls, arg):
            if arg is S.Infinity:
                return S.NaN
    assert limit(my(x), x, oo) == Limit(my(x), x, oo)


def test_issue_3885():
    assert limit(x*y + x*z, z, 2) == x*y + 2*x


def test_Limit():
    assert Limit(sin(x)/x, x, 0) != 1
    assert Limit(sin(x)/x, x, 0).doit() == 1
    assert Limit(x, x, 0, dir='+-').args == (x, x, 0, Symbol('+-'))


def test_floor():
    assert limit(floor(x), x, -2, "+") == -2
    assert limit(floor(x), x, -2, "-") == -3
    assert limit(floor(x), x, -1, "+") == -1
    assert limit(floor(x), x, -1, "-") == -2
    assert limit(floor(x), x, 0, "+") == 0
    assert limit(floor(x), x, 0, "-") == -1
    assert limit(floor(x), x, 1, "+") == 1
    assert limit(floor(x), x, 1, "-") == 0
    assert limit(floor(x), x, 2, "+") == 2
    assert limit(floor(x), x, 2, "-") == 1
    assert limit(floor(x), x, 248, "+") == 248
    assert limit(floor(x), x, 248, "-") == 247

    # https://github.com/sympy/sympy/issues/14478
    assert limit(x*floor(3/x)/2, x, 0, '+') == Rational(3, 2)
    assert limit(floor(x + 1/2) - floor(x), x, oo) == AccumBounds(-S.Half, S(3)/2)

    # test issue 9158
    assert limit(floor(atan(x)), x, oo) == 1
    assert limit(floor(atan(x)), x, -oo) == -2
    assert limit(ceiling(atan(x)), x, oo) == 2
    assert limit(ceiling(atan(x)), x, -oo) == -1


def test_floor_requires_robust_assumptions():
    assert limit(floor(sin(x)), x, 0, "+") == 0
    assert limit(floor(sin(x)), x, 0, "-") == -1
    assert limit(floor(cos(x)), x, 0, "+") == 0
    assert limit(floor(cos(x)), x, 0, "-") == 0
    assert limit(floor(5 + sin(x)), x, 0, "+") == 5
    assert limit(floor(5 + sin(x)), x, 0, "-") == 4
    assert limit(floor(5 + cos(x)), x, 0, "+") == 5
    assert limit(floor(5 + cos(x)), x, 0, "-") == 5


def test_ceiling():
    assert limit(ceiling(x), x, -2, "+") == -1
    assert limit(ceiling(x), x, -2, "-") == -2
    assert limit(ceiling(x), x, -1, "+") == 0
    assert limit(ceiling(x), x, -1, "-") == -1
    assert limit(ceiling(x), x, 0, "+") == 1
    assert limit(ceiling(x), x, 0, "-") == 0
    assert limit(ceiling(x), x, 1, "+") == 2
    assert limit(ceiling(x), x, 1, "-") == 1
    assert limit(ceiling(x), x, 2, "+") == 3
    assert limit(ceiling(x), x, 2, "-") == 2
    assert limit(ceiling(x), x, 248, "+") == 249
    assert limit(ceiling(x), x, 248, "-") == 248

    # https://github.com/sympy/sympy/issues/14478
    assert limit(x*ceiling(3/x)/2, x, 0, '+') == Rational(3, 2)
    assert limit(ceiling(x + 1/2) - ceiling(x), x, oo) == AccumBounds(-S.Half, S(3)/2)


def test_ceiling_requires_robust_assumptions():
    assert limit(ceiling(sin(x)), x, 0, "+") == 1
    assert limit(ceiling(sin(x)), x, 0, "-") == 0
    assert limit(ceiling(cos(x)), x, 0, "+") == 1
    assert limit(ceiling(cos(x)), x, 0, "-") == 1
    assert limit(ceiling(5 + sin(x)), x, 0, "+") == 6
    assert limit(ceiling(5 + sin(x)), x, 0, "-") == 5
    assert limit(ceiling(5 + cos(x)), x, 0, "+") == 6
    assert limit(ceiling(5 + cos(x)), x, 0, "-") == 6


def test_frac():
    assert limit(frac(x), x, oo) == AccumBounds(0, 1)
    assert limit(frac(x)**(1/x), x, oo) == AccumBounds(0, 1)
    assert limit(frac(x)**(1/x), x, -oo) == AccumBounds(1, oo)
    assert limit(frac(x)**x, x, oo) == AccumBounds(0, oo)  # wolfram gives (0, 1)
    assert limit(frac(sin(x)), x, 0, "+") == 0
    assert limit(frac(sin(x)), x, 0, "-") == 1
    assert limit(frac(cos(x)), x, 0, "+-") == 1
    assert limit(frac(x**2), x, 0, "+-") == 0
    raises(ValueError, lambda: limit(frac(x), x, 0, '+-'))
    assert limit(frac(-2*x + 1), x, 0, "+") == 1
    assert limit(frac(-2*x + 1), x, 0, "-") == 0
    assert limit(frac(x + S.Half), x, 0, "+-") == S(1)/2
    assert limit(frac(1/x), x, 0) == AccumBounds(0, 1)


def test_issue_14355():
    assert limit(floor(sin(x)/x), x, 0, '+') == 0
    assert limit(floor(sin(x)/x), x, 0, '-') == 0
    # test comment https://github.com/sympy/sympy/issues/14355#issuecomment-372121314
    assert limit(floor(-tan(x)/x), x, 0, '+') == -2
    assert limit(floor(-tan(x)/x), x, 0, '-') == -2


def test_atan():
    x = Symbol("x", real=True)
    assert limit(atan(x)*sin(1/x), x, 0) == 0
    assert limit(atan(x) + sqrt(x + 1) - sqrt(x), x, oo) == pi/2


def test_set_signs():
    assert limit(abs(x), x, 0) == 0
    assert limit(abs(sin(x)), x, 0) == 0
    assert limit(abs(cos(x)), x, 0) == 1
    assert limit(abs(sin(x + 1)), x, 0) == sin(1)

    # https://github.com/sympy/sympy/issues/9449
    assert limit((Abs(x + y) - Abs(x - y))/(2*x), x, 0) == sign(y)

    # https://github.com/sympy/sympy/issues/12398
    assert limit(Abs(log(x)/x**3), x, oo) == 0
    assert limit(x*(Abs(log(x)/x**3)/Abs(log(x + 1)/(x + 1)**3) - 1), x, oo) == 3

    # https://github.com/sympy/sympy/issues/18501
    assert limit(Abs(log(x - 1)**3 - 1), x, 1, '+') == oo

    # https://github.com/sympy/sympy/issues/18997
    assert limit(Abs(log(x)), x, 0) == oo
    assert limit(Abs(log(Abs(x))), x, 0) == oo

    # https://github.com/sympy/sympy/issues/19026
    z = Symbol('z', positive=True)
    assert limit(Abs(log(z) + 1)/log(z), z, oo) == 1

    # https://github.com/sympy/sympy/issues/20704
    assert limit(z*(Abs(1/z + y) - Abs(y - 1/z))/2, z, 0) == 0

    # https://github.com/sympy/sympy/issues/21606
    assert limit(cos(z)/sign(z), z, pi, '-') == -1


def test_heuristic():
    x = Symbol("x", real=True)
    assert heuristics(sin(1/x) + atan(x), x, 0, '+') == AccumBounds(-1, 1)
    assert limit(log(2 + sqrt(atan(x))*sqrt(sin(1/x))), x, 0) == log(2)


def test_issue_3871():
    z = Symbol("z", positive=True)
    f = -1/z*exp(-z*x)
    assert limit(f, x, oo) == 0
    assert f.limit(x, oo) == 0


def test_exponential():
    n = Symbol('n')
    x = Symbol('x', real=True)
    assert limit((1 + x/n)**n, n, oo) == exp(x)
    assert limit((1 + x/(2*n))**n, n, oo) == exp(x/2)
    assert limit((1 + x/(2*n + 1))**n, n, oo) == exp(x/2)
    assert limit(((x - 1)/(x + 1))**x, x, oo) == exp(-2)
    assert limit(1 + (1 + 1/x)**x, x, oo) == 1 + S.Exp1
    assert limit((2 + 6*x)**x/(6*x)**x, x, oo) == exp(S('1/3'))


def test_exponential2():
    n = Symbol('n')
    assert limit((1 + x/(n + sin(n)))**n, n, oo) == exp(x)


def test_doit():
    f = Integral(2 * x, x)
    l = Limit(f, x, oo)
    assert l.doit() is oo


def test_series_AccumBounds():
    assert limit(sin(k) - sin(k + 1), k, oo) == AccumBounds(-2, 2)
    assert limit(cos(k) - cos(k + 1) + 1, k, oo) == AccumBounds(-1, 3)

    # not the exact bound
    assert limit(sin(k) - sin(k)*cos(k), k, oo) == AccumBounds(-2, 2)

    # test for issue #9934
    lo = (-3 + cos(1))/2
    hi = (1 + cos(1))/2
    t1 = Mul(AccumBounds(lo, hi), 1/(-1 + cos(1)), evaluate=False)
    assert limit(simplify(Sum(cos(n).rewrite(exp), (n, 0, k)).doit().rewrite(sin)), k, oo) == t1

    t2 = Mul(AccumBounds(-1 + sin(1)/2, sin(1)/2 + 1), 1/(1 - cos(1)))
    assert limit(simplify(Sum(sin(n).rewrite(exp), (n, 0, k)).doit().rewrite(sin)), k, oo) == t2

    assert limit(((sin(x) + 1)/2)**x, x, oo) == AccumBounds(0, oo)  # wolfram says 0

    # https://github.com/sympy/sympy/issues/12312
    e = 2**(-x)*(sin(x) + 1)**x
    assert limit(e, x, oo) == AccumBounds(0, oo)


def test_bessel_functions_at_infinity():
    # Pull Request 23844 implements limits for all bessel and modified bessel
    # functions approaching infinity along any direction i.e. abs(z0) tends to oo

    assert limit(besselj(1, x), x, oo) == 0
    assert limit(besselj(1, x), x, -oo) == 0
    assert limit(besselj(1, x), x, I*oo) == oo*I
    assert limit(besselj(1, x), x, -I*oo) == -oo*I
    assert limit(bessely(1, x), x, oo) == 0
    assert limit(bessely(1, x), x, -oo) == 0
    assert limit(bessely(1, x), x, I*oo) == -oo
    assert limit(bessely(1, x), x, -I*oo) == -oo
    assert limit(besseli(1, x), x, oo) == oo
    assert limit(besseli(1, x), x, -oo) == -oo
    assert limit(besseli(1, x), x, I*oo) == 0
    assert limit(besseli(1, x), x, -I*oo) == 0
    assert limit(besselk(1, x), x, oo) == 0
    assert limit(besselk(1, x), x, -oo) == -oo*I
    assert limit(besselk(1, x), x, I*oo) == 0
    assert limit(besselk(1, x), x, -I*oo) == 0

    # test issue 14874
    assert limit(besselk(0, x), x, oo) == 0


@XFAIL
def test_doit2():
    f = Integral(2 * x, x)
    l = Limit(f, x, oo)
    # limit() breaks on the contained Integral.
    assert l.doit(deep=False) == l


def test_issue_2929():
    assert limit((x * exp(x))/(exp(x) - 1), x, -oo) == 0


def test_issue_3792():
    assert limit((1 - cos(x))/x**2, x, S.Half) == 4 - 4*cos(S.Half)
    assert limit(sin(sin(x + 1) + 1), x, 0) == sin(1 + sin(1))
    assert limit(abs(sin(x + 1) + 1), x, 0) == 1 + sin(1)


def test_issue_4090():
    assert limit(1/(x + 3), x, 2) == Rational(1, 5)
    assert limit(1/(x + pi), x, 2) == S.One/(2 + pi)
    assert limit(log(x)/(x**2 + 3), x, 2) == log(2)/7
    assert limit(log(x)/(x**2 + pi), x, 2) == log(2)/(4 + pi)


def test_issue_4547():
    assert limit(cot(x), x, 0, dir='+') is oo
    assert limit(cot(x), x, pi/2, dir='+') == 0


def test_issue_5164():
    assert limit(x**0.5, x, oo) == oo**0.5 is oo
    assert limit(x**0.5, x, 16) == 4 # Should this be a float?
    assert limit(x**0.5, x, 0) == 0
    assert limit(x**(-0.5), x, oo) == 0
    assert limit(x**(-0.5), x, 4) == S.Half # Should this be a float?


def test_issue_5383():
    func = (1.0 * 1 + 1.0 * x)**(1.0 * 1 / x)
    assert limit(func, x, 0) == E


def test_issue_14793():
    expr = ((x + S(1)/2) * log(x) - x + log(2*pi)/2 - \
        log(factorial(x)) + S(1)/(12*x))*x**3
    assert limit(expr, x, oo) == S(1)/360


def test_issue_5183():
    # using list(...) so py.test can recalculate values
    tests = list(product([x, -x],
                         [-1, 1],
                         [2, 3, S.Half, Rational(2, 3)],
                         ['-', '+']))
    results = (oo, oo, -oo, oo, -oo*I, oo, -oo*(-1)**Rational(1, 3), oo,
               0, 0, 0, 0, 0, 0, 0, 0,
               oo, oo, oo, -oo, oo, -oo*I, oo, -oo*(-1)**Rational(1, 3),
               0, 0, 0, 0, 0, 0, 0, 0)
    assert len(tests) == len(results)
    for i, (args, res) in enumerate(zip(tests, results)):
        y, s, e, d = args
        eq = y**(s*e)
        try:
            assert limit(eq, x, 0, dir=d) == res
        except AssertionError:
            if 0:  # change to 1 if you want to see the failing tests
                print()
                print(i, res, eq, d, limit(eq, x, 0, dir=d))
            else:
                assert None


def test_issue_5184():
    assert limit(sin(x)/x, x, oo) == 0
    assert limit(atan(x), x, oo) == pi/2
    assert limit(gamma(x), x, oo) is oo
    assert limit(cos(x)/x, x, oo) == 0
    assert limit(gamma(x), x, S.Half) == sqrt(pi)

    r = Symbol('r', real=True)
    assert limit(r*sin(1/r), r, 0) == 0


def test_issue_5229():
    assert limit((1 + y)**(1/y) - S.Exp1, y, 0) == 0


def test_issue_4546():
    # using list(...) so py.test can recalculate values
    tests = list(product([cot, tan],
                         [-pi/2, 0, pi/2, pi, pi*Rational(3, 2)],
                         ['-', '+']))
    results = (0, 0, -oo, oo, 0, 0, -oo, oo, 0, 0,
               oo, -oo, 0, 0, oo, -oo, 0, 0, oo, -oo)
    assert len(tests) == len(results)
    for i, (args, res) in enumerate(zip(tests, results)):
        f, l, d = args
        eq = f(x)
        try:
            assert limit(eq, x, l, dir=d) == res
        except AssertionError:
            if 0:  # change to 1 if you want to see the failing tests
                print()
                print(i, res, eq, l, d, limit(eq, x, l, dir=d))
            else:
                assert None


def test_issue_3934():
    assert limit((1 + x**log(3))**(1/x), x, 0) == 1
    assert limit((5**(1/x) + 3**(1/x))**x, x, 0) == 5


def test_calculate_series():
    # NOTE
    # The calculate_series method is being deprecated and is no longer responsible
    # for result being returned. The mrv_leadterm function now uses simple leadterm
    # calls rather than calculate_series.

    # needs gruntz calculate_series to go to n = 32
    assert limit(x**Rational(77, 3)/(1 + x**Rational(77, 3)), x, oo) == 1
    # needs gruntz calculate_series to go to n = 128
    assert limit(x**101.1/(1 + x**101.1), x, oo) == 1


def test_issue_5955():
    assert limit((x**16)/(1 + x**16), x, oo) == 1
    assert limit((x**100)/(1 + x**100), x, oo) == 1
    assert limit((x**1885)/(1 + x**1885), x, oo) == 1
    assert limit((x**1000/((x + 1)**1000 + exp(-x))), x, oo) == 1


def test_newissue():
    assert limit(exp(1/sin(x))/exp(cot(x)), x, 0) == 1


def test_extended_real_line():
    assert limit(x - oo, x, oo) == Limit(x - oo, x, oo)
    assert limit(1/(x + sin(x)) - oo, x, 0) == Limit(1/(x + sin(x)) - oo, x, 0)
    assert limit(oo/x, x, oo) == Limit(oo/x, x, oo)
    assert limit(x - oo + 1/x, x, oo) == Limit(x - oo + 1/x, x, oo)


@XFAIL
def test_order_oo():
    x = Symbol('x', positive=True)
    assert Order(x)*oo != Order(1, x)
    assert limit(oo/(x**2 - 4), x, oo) is oo


def test_issue_5436():
    raises(NotImplementedError, lambda: limit(exp(x*y), x, oo))
    raises(NotImplementedError, lambda: limit(exp(-x*y), x, oo))


def test_Limit_dir():
    raises(TypeError, lambda: Limit(x, x, 0, dir=0))
    raises(ValueError, lambda: Limit(x, x, 0, dir='0'))


def test_polynomial():
    assert limit((x + 1)**1000/((x + 1)**1000 + 1), x, oo) == 1
    assert limit((x + 1)**1000/((x + 1)**1000 + 1), x, -oo) == 1


def test_rational():
    assert limit(1/y - (1/(y + x) + x/(y + x)/y)/z, x, oo) == (z - 1)/(y*z)
    assert limit(1/y - (1/(y + x) + x/(y + x)/y)/z, x, -oo) == (z - 1)/(y*z)


def test_issue_5740():
    assert limit(log(x)*z - log(2*x)*y, x, 0) == oo*sign(y - z)


def test_issue_6366():
    n = Symbol('n', integer=True, positive=True)
    r = (n + 1)*x**(n + 1)/(x**(n + 1) - 1) - x/(x - 1)
    assert limit(r, x, 1).cancel() == n/2


def test_factorial():
    f = factorial(x)
    assert limit(f, x, oo) is oo
    assert limit(x/f, x, oo) == 0
    # see Stirling's approximation:
    # https://en.wikipedia.org/wiki/Stirling's_approximation
    assert limit(f/(sqrt(2*pi*x)*(x/E)**x), x, oo) == 1
    assert limit(f, x, -oo) == gamma(-oo)


def test_issue_6560():
    e = (5*x**3/4 - x*Rational(3, 4) + (y*(3*x**2/2 - S.Half) +
                             35*x**4/8 - 15*x**2/4 + Rational(3, 8))/(2*(y + 1)))
    assert limit(e, y, oo) == 5*x**3/4 + 3*x**2/4 - 3*x/4 - Rational(1, 4)

@XFAIL
def test_issue_5172():
    n = Symbol('n')
    r = Symbol('r', positive=True)
    c = Symbol('c')
    p = Symbol('p', positive=True)
    m = Symbol('m', negative=True)
    expr = ((2*n*(n - r + 1)/(n + r*(n - r + 1)))**c +
            (r - 1)*(n*(n - r + 2)/(n + r*(n - r + 1)))**c - n)/(n**c - n)
    expr = expr.subs(c, c + 1)
    raises(NotImplementedError, lambda: limit(expr, n, oo))
    assert limit(expr.subs(c, m), n, oo) == 1
    assert limit(expr.subs(c, p), n, oo).simplify() == \
        (2**(p + 1) + r - 1)/(r + 1)**(p + 1)


def test_issue_7088():
    a = Symbol('a')
    assert limit(sqrt(x/(x + a)), x, oo) == 1


def test_branch_cuts():
    assert limit(asin(I*x + 2), x, 0) == pi - asin(2)
    assert limit(asin(I*x + 2), x, 0, '-') == asin(2)
    assert limit(asin(I*x - 2), x, 0) == -asin(2)
    assert limit(asin(I*x - 2), x, 0, '-') == -pi + asin(2)
    assert limit(acos(I*x + 2), x, 0) == -acos(2)
    assert limit(acos(I*x + 2), x, 0, '-') == acos(2)
    assert limit(acos(I*x - 2), x, 0) == acos(-2)
    assert limit(acos(I*x - 2), x, 0, '-') == 2*pi - acos(-2)
    assert limit(atan(x + 2*I), x, 0) == I*atanh(2)
    assert limit(atan(x + 2*I), x, 0, '-') == -pi + I*atanh(2)
    assert limit(atan(x - 2*I), x, 0) == pi - I*atanh(2)
    assert limit(atan(x - 2*I), x, 0, '-') == -I*atanh(2)
    assert limit(atan(1/x), x, 0) == pi/2
    assert limit(atan(1/x), x, 0, '-') == -pi/2
    assert limit(atan(x), x, oo) == pi/2
    assert limit(atan(x), x, -oo) == -pi/2
    assert limit(acot(x + S(1)/2*I), x, 0) == pi - I*acoth(S(1)/2)
    assert limit(acot(x + S(1)/2*I), x, 0, '-') == -I*acoth(S(1)/2)
    assert limit(acot(x - S(1)/2*I), x, 0) == I*acoth(S(1)/2)
    assert limit(acot(x - S(1)/2*I), x, 0, '-') == -pi + I*acoth(S(1)/2)
    assert limit(acot(x), x, 0) == pi/2
    assert limit(acot(x), x, 0, '-') == -pi/2
    assert limit(asec(I*x + S(1)/2), x, 0) == asec(S(1)/2)
    assert limit(asec(I*x + S(1)/2), x, 0, '-') == -asec(S(1)/2)
    assert limit(asec(I*x - S(1)/2), x, 0) == 2*pi - asec(-S(1)/2)
    assert limit(asec(I*x - S(1)/2), x, 0, '-') == asec(-S(1)/2)
    assert limit(acsc(I*x + S(1)/2), x, 0) == acsc(S(1)/2)
    assert limit(acsc(I*x + S(1)/2), x, 0, '-') == pi - acsc(S(1)/2)
    assert limit(acsc(I*x - S(1)/2), x, 0) == -pi + acsc(S(1)/2)
    assert limit(acsc(I*x - S(1)/2), x, 0, '-') == -acsc(S(1)/2)

    assert limit(log(I*x - 1), x, 0) == I*pi
    assert limit(log(I*x - 1), x, 0, '-') == -I*pi
    assert limit(log(-I*x - 1), x, 0) == -I*pi
    assert limit(log(-I*x - 1), x, 0, '-') == I*pi

    assert limit(sqrt(I*x - 1), x, 0) == I
    assert limit(sqrt(I*x - 1), x, 0, '-') == -I
    assert limit(sqrt(-I*x - 1), x, 0) == -I
    assert limit(sqrt(-I*x - 1), x, 0, '-') == I

    assert limit(cbrt(I*x - 1), x, 0) == (-1)**(S(1)/3)
    assert limit(cbrt(I*x - 1), x, 0, '-') == -(-1)**(S(2)/3)
    assert limit(cbrt(-I*x - 1), x, 0) == -(-1)**(S(2)/3)
    assert limit(cbrt(-I*x - 1), x, 0, '-') == (-1)**(S(1)/3)


def test_issue_6364():
    a = Symbol('a')
    e = z/(1 - sqrt(1 + z)*sin(a)**2 - sqrt(1 - z)*cos(a)**2)
    assert limit(e, z, 0) == 1/(cos(a)**2 - S.Half)


def test_issue_6682():
    assert limit(exp(2*Ei(-x))/x**2, x, 0) == exp(2*EulerGamma)


def test_issue_4099():
    a = Symbol('a')
    assert limit(a/x, x, 0) == oo*sign(a)
    assert limit(-a/x, x, 0) == -oo*sign(a)
    assert limit(-a*x, x, oo) == -oo*sign(a)
    assert limit(a*x, x, oo) == oo*sign(a)


def test_issue_4503():
    dx = Symbol('dx')
    assert limit((sqrt(1 + exp(x + dx)) - sqrt(1 + exp(x)))/dx, dx, 0) == \
        exp(x)/(2*sqrt(exp(x) + 1))


def test_issue_6052():
    G = meijerg((), (), (1,), (0,), -x)
    g = hyperexpand(G)
    assert limit(g, x, 0, '+-') == 0
    assert limit(g, x, oo) == -oo


def test_issue_7224():
    expr = sqrt(x)*besseli(1,sqrt(8*x))
    assert limit(x*diff(expr, x, x)/expr, x, 0) == 2
    assert limit(x*diff(expr, x, x)/expr, x, 1).evalf() == 2.0


def test_issue_8208():
    assert limit(n**(Rational(1, 1e9) - 1), n, oo) == 0


def test_issue_8229():
    assert limit((x**Rational(1, 4) - 2)/(sqrt(x) - 4)**Rational(2, 3), x, 16) == 0


def test_issue_8433():
    d, t = symbols('d t', positive=True)
    assert limit(erf(1 - t/d), t, oo) == -1


def test_issue_8481():
    k = Symbol('k', integer=True, nonnegative=True)
    lamda = Symbol('lamda', positive=True)
    assert limit(lamda**k * exp(-lamda) / factorial(k), k, oo) == 0


def test_issue_8462():
    assert limit(binomial(n, n/2), n, oo) == oo
    assert limit(binomial(n, n/2) * 3 ** (-n), n, oo) == 0


def test_issue_8634():
    n = Symbol('n', integer=True, positive=True)
    x = Symbol('x')
    assert limit(x**n, x, -oo) == oo*sign((-1)**n)


def test_issue_8635_18176():
    x = Symbol('x', real=True)
    k = Symbol('k', positive=True)
    assert limit(x**n - x**(n - 0), x, oo) == 0
    assert limit(x**n - x**(n - 5), x, oo) == oo
    assert limit(x**n - x**(n - 2.5), x, oo) == oo
    assert limit(x**n - x**(n - k - 1), x, oo) == oo
    x = Symbol('x', positive=True)
    assert limit(x**n - x**(n - 1), x, oo) == oo
    assert limit(x**n - x**(n + 2), x, oo) == -oo


def test_issue_8730():
    assert limit(subfactorial(x), x, oo) is oo


def test_issue_9252():
    n = Symbol('n', integer=True)
    c = Symbol('c', positive=True)
    assert limit((log(n))**(n/log(n)) / (1 + c)**n, n, oo) == 0
    # limit should depend on the value of c
    raises(NotImplementedError, lambda: limit((log(n))**(n/log(n)) / c**n, n, oo))


def test_issue_9558():
    assert limit(sin(x)**15, x, 0, '-') == 0


def test_issue_10801():
    # make sure limits work with binomial
    assert limit(16**k / (k * binomial(2*k, k)**2), k, oo) == pi


def test_issue_10976():
    s, x = symbols('s x', real=True)
    assert limit(erf(s*x)/erf(s), s, 0) == x


def test_issue_9041():
    assert limit(factorial(n) / ((n/exp(1))**n * sqrt(2*pi*n)), n, oo) == 1


def test_issue_9205():
    x, y, a = symbols('x, y, a')
    assert Limit(x, x, a).free_symbols == {a}
    assert Limit(x, x, a, '-').free_symbols == {a}
    assert Limit(x + y, x + y, a).free_symbols == {a}
    assert Limit(-x**2 + y, x**2, a).free_symbols == {y, a}


def test_issue_9471():
    assert limit(((27**(log(n,3)))/n**3),n,oo) == 1
    assert limit(((27**(log(n,3)+1))/n**3),n,oo) == 27


def test_issue_10382():
    assert limit(fibonacci(n + 1)/fibonacci(n), n, oo) == GoldenRatio


def test_issue_11496():
    assert limit(erfc(log(1/x)), x, oo) == 2


def test_issue_11879():
    assert simplify(limit(((x+y)**n-x**n)/y, y, 0)) == n*x**(n-1)


def test_limit_with_Float():
    k = symbols("k")
    assert limit(1.0 ** k, k, oo) == 1
    assert limit(0.3*1.0**k, k, oo) == Rational(3, 10)


def test_issue_10610():
    assert limit(3**x*3**(-x - 1)*(x + 1)**2/x**2, x, oo) == Rational(1, 3)


def test_issue_10868():
    assert limit(log(x) + asech(x), x, 0, '+') == log(2)
    assert limit(log(x) + asech(x), x, 0, '-') == log(2) + 2*I*pi
    raises(ValueError, lambda: limit(log(x) + asech(x), x, 0, '+-'))
    assert limit(log(x) + asech(x), x, oo) == oo
    assert limit(log(x) + acsch(x), x, 0, '+') == log(2)
    assert limit(log(x) + acsch(x), x, 0, '-') == -oo
    raises(ValueError, lambda: limit(log(x) + acsch(x), x, 0, '+-'))
    assert limit(log(x) + acsch(x), x, oo) == oo


def test_issue_6599():
    assert limit((n + cos(n))/n, n, oo) == 1


def test_issue_12555():
    assert limit((3**x + 2* x**10) / (x**10 + exp(x)), x, -oo) == 2
    assert limit((3**x + 2* x**10) / (x**10 + exp(x)), x, oo) is oo


def test_issue_12769():
    r, z, x = symbols('r z x', real=True)
    a, b, s0, K, F0, s, T = symbols('a b s0 K F0 s T', positive=True, real=True)
    fx = (F0**b*K**b*r*s0 - sqrt((F0**2*K**(2*b)*a**2*(b - 1) + \
        F0**(2*b)*K**2*a**2*(b - 1) + F0**(2*b)*K**(2*b)*s0**2*(b - 1)*(b**2 - 2*b + 1) - \
        2*F0**(2*b)*K**(b + 1)*a*r*s0*(b**2 - 2*b +  1) + \
        2*F0**(b + 1)*K**(2*b)*a*r*s0*(b**2 - 2*b + 1) - \
        2*F0**(b + 1)*K**(b + 1)*a**2*(b - 1))/((b - 1)*(b**2 - 2*b + 1))))*(b*r -  b - r + 1)

    assert fx.subs(K, F0).factor(deep=True) == limit(fx, K, F0).factor(deep=True)


def test_issue_13332():
    assert limit(sqrt(30)*5**(-5*x - 1)*(46656*x)**x*(5*x + 2)**(5*x + 5*S.Half) *
                (6*x + 2)**(-6*x - 5*S.Half), x, oo) == Rational(25, 36)


def test_issue_12564():
    assert limit(x**2 + x*sin(x) + cos(x), x, -oo) is oo
    assert limit(x**2 + x*sin(x) + cos(x), x, oo) is oo
    assert limit(((x + cos(x))**2).expand(), x, oo) is oo
    assert limit(((x + sin(x))**2).expand(), x, oo) is oo
    assert limit(((x + cos(x))**2).expand(), x, -oo) is oo
    assert limit(((x + sin(x))**2).expand(), x, -oo) is oo


def test_issue_14456():
    raises(NotImplementedError, lambda: Limit(exp(x), x, zoo).doit())
    raises(NotImplementedError, lambda: Limit(x**2/(x+1), x, zoo).doit())


def test_issue_14411():
    assert limit(3*sec(4*pi*x - x/3), x, 3*pi/(24*pi - 2)) is -oo


def test_issue_13382():
    assert limit(x*(((x + 1)**2 + 1)/(x**2 + 1) - 1), x, oo) == 2


def test_issue_13403():
    assert limit(x*(-1 + (x + log(x + 1) + 1)/(x + log(x))), x, oo) == 1


def test_issue_13416():
    assert limit((-x**3*log(x)**3 + (x - 1)*(x + 1)**2*log(x + 1)**3)/(x**2*log(x)**3), x, oo) == 1


def test_issue_13462():
    assert limit(n**2*(2*n*(-(1 - 1/(2*n))**x + 1) - x - (-x**2/4 + x/4)/n), n, oo) == x**3/24 - x**2/8 + x/12


def test_issue_13750():
    a = Symbol('a')
    assert limit(erf(a - x), x, oo) == -1
    assert limit(erf(sqrt(x) - x), x, oo) == -1


def test_issue_14276():
    assert isinstance(limit(sin(x)**log(x), x, oo), Limit)
    assert isinstance(limit(sin(x)**cos(x), x, oo), Limit)
    assert isinstance(limit(sin(log(cos(x))), x, oo), Limit)
    assert limit((1 + 1/(x**2 + cos(x)))**(x**2 + x), x, oo) == E


def test_issue_14514():
    assert limit((1/(log(x)**log(x)))**(1/x), x, oo) == 1


def test_issues_14525():
    assert limit(sin(x)**2 - cos(x) + tan(x)*csc(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(sin(x)**2 - cos(x) + sin(x)*cot(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(cot(x) - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(cos(x) - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.One)
    assert limit(sin(x) - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.One)
    assert limit(cos(x)**2 - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.One)
    assert limit(tan(x)**2 + sin(x)**2 - cos(x), x, oo) == AccumBounds(-S.One, S.Infinity)


def test_issue_14574():
    assert limit(sqrt(x)*cos(x - x**2) / (x + 1), x, oo) == 0


def test_issue_10102():
    assert limit(fresnels(x), x, oo) == S.Half
    assert limit(3 + fresnels(x), x, oo) == 3 + S.Half
    assert limit(5*fresnels(x), x, oo) == Rational(5, 2)
    assert limit(fresnelc(x), x, oo) == S.Half
    assert limit(fresnels(x), x, -oo) == Rational(-1, 2)
    assert limit(4*fresnelc(x), x, -oo) == -2


def test_issue_14377():
    raises(NotImplementedError, lambda: limit(exp(I*x)*sin(pi*x), x, oo))


def test_issue_15146():
    e = (x/2) * (-2*x**3 - 2*(x**3 - 1) * x**2 * digamma(x**3 + 1) + \
        2*(x**3 - 1) * x**2 * digamma(x**3 + x + 1) + x + 3)
    assert limit(e, x, oo) == S(1)/3


def test_issue_15202():
    e = (2**x*(2 + 2**(-x)*(-2*2**x + x + 2))/(x + 1))**(x + 1)
    assert limit(e, x, oo) == exp(1)

    e = (log(x, 2)**7 + 10*x*factorial(x) + 5**x) / (factorial(x + 1) + 3*factorial(x) + 10**x)
    assert limit(e, x, oo) == 10


def test_issue_15282():
    assert limit((x**2000 - (x + 1)**2000) / x**1999, x, oo) == -2000


def test_issue_15984():
    assert limit((-x + log(exp(x) + 1))/x, x, oo, dir='-') == 0


def test_issue_13571():
    assert limit(uppergamma(x, 1) / gamma(x), x, oo) == 1


def test_issue_13575():
    assert limit(acos(erfi(x)), x, 1) == acos(erfi(S.One))


def test_issue_17325():
    assert Limit(sin(x)/x, x, 0, dir="+-").doit() == 1
    assert Limit(x**2, x, 0, dir="+-").doit() == 0
    assert Limit(1/x**2, x, 0, dir="+-").doit() is oo
    assert Limit(1/x, x, 0, dir="+-").doit() is zoo


def test_issue_10978():
    assert LambertW(x).limit(x, 0) == 0


def test_issue_14313_comment():
    assert limit(floor(n/2), n, oo) is oo


@XFAIL
def test_issue_15323():
    d = ((1 - 1/x)**x).diff(x)
    assert limit(d, x, 1, dir='+') == 1


def test_issue_12571():
    assert limit(-LambertW(-log(x))/log(x), x, 1) == 1


def test_issue_14590():
    assert limit((x**3*((x + 1)/x)**x)/((x + 1)*(x + 2)*(x + 3)), x, oo) == exp(1)


def test_issue_14393():
    a, b = symbols('a b')
    assert limit((x**b - y**b)/(x**a - y**a), x, y) == b*y**(-a + b)/a


def test_issue_14556():
    assert limit(factorial(n + 1)**(1/(n + 1)) - factorial(n)**(1/n), n, oo) == exp(-1)


def test_issue_14811():
    assert limit(((1 + ((S(2)/3)**(x + 1)))**(2**x))/(2**((S(4)/3)**(x - 1))), x, oo) == oo


def test_issue_16222():
    assert limit(exp(x), x, 1000000000) == exp(1000000000)


def test_issue_16714():
    assert limit(((x**(x + 1) + (x + 1)**x) / x**(x + 1))**x, x, oo) == exp(exp(1))


def test_issue_16722():
    z = symbols('z', positive=True)
    assert limit(binomial(n + z, n)*n**-z, n, oo) == 1/gamma(z + 1)
    z = symbols('z', positive=True, integer=True)
    assert limit(binomial(n + z, n)*n**-z, n, oo) == 1/gamma(z + 1)


def test_issue_17431():
    assert limit(((n + 1) + 1) / (((n + 1) + 2) * factorial(n + 1)) *
                 (n + 2) * factorial(n) / (n + 1), n, oo) == 0
    assert limit((n + 2)**2*factorial(n)/((n + 1)*(n + 3)*factorial(n + 1))
                 , n, oo) == 0
    assert limit((n + 1) * factorial(n) / (n * factorial(n + 1)), n, oo) == 0


def test_issue_17671():
    assert limit(Ei(-log(x)) - log(log(x))/x, x, 1) == EulerGamma


def test_issue_17751():
    a, b, c, x = symbols('a b c x', positive=True)
    assert limit((a + 1)*x - sqrt((a + 1)**2*x**2 + b*x + c), x, oo) == -b/(2*a + 2)


def test_issue_17792():
    assert limit(factorial(n)/sqrt(n)*(exp(1)/n)**n, n, oo) == sqrt(2)*sqrt(pi)


def test_issue_18118():
    assert limit(sign(sin(x)), x, 0, "-") == -1
    assert limit(sign(sin(x)), x, 0, "+") == 1


def test_issue_18306():
    assert limit(sin(sqrt(x))/sqrt(sin(x)), x, 0, '+') == 1


def test_issue_18378():
    assert limit(log(exp(3*x) + x)/log(exp(x) + x**100), x, oo) == 3


def test_issue_18399():
    assert limit((1 - S(1)/2*x)**(3*x), x, oo) is zoo
    assert limit((-x)**x, x, oo) is zoo


def test_issue_18442():
    assert limit(tan(x)**(2**(sqrt(pi))), x, oo, dir='-') == Limit(tan(x)**(2**(sqrt(pi))), x, oo, dir='-')


def test_issue_18452():
    assert limit(abs(log(x))**x, x, 0) == 1
    assert limit(abs(log(x))**x, x, 0, "-") == 1


def test_issue_18473():
    assert limit(sin(x)**(1/x), x, oo) == Limit(sin(x)**(1/x), x, oo, dir='-')
    assert limit(cos(x)**(1/x), x, oo) == Limit(cos(x)**(1/x), x, oo, dir='-')
    assert limit(tan(x)**(1/x), x, oo) == Limit(tan(x)**(1/x), x, oo, dir='-')
    assert limit((cos(x) + 2)**(1/x), x, oo) == 1
    assert limit((sin(x) + 10)**(1/x), x, oo) == 1
    assert limit((cos(x) - 2)**(1/x), x, oo) == Limit((cos(x) - 2)**(1/x), x, oo, dir='-')
    assert limit((cos(x) + 1)**(1/x), x, oo) == AccumBounds(0, 1)
    assert limit((tan(x)**2)**(2/x) , x, oo) == AccumBounds(0, oo)
    assert limit((sin(x)**2)**(1/x), x, oo) == AccumBounds(0, 1)
    # Tests for issue #23751
    assert limit((cos(x) + 1)**(1/x), x, -oo) == AccumBounds(1, oo)
    assert limit((sin(x)**2)**(1/x), x, -oo) == AccumBounds(1, oo)
    assert limit((tan(x)**2)**(2/x) , x, -oo) == AccumBounds(0, oo)


def test_issue_18482():
    assert limit((2*exp(3*x)/(exp(2*x) + 1))**(1/x), x, oo) == exp(1)


def test_issue_18508():
    assert limit(sin(x)/sqrt(1-cos(x)), x, 0) == sqrt(2)
    assert limit(sin(x)/sqrt(1-cos(x)), x, 0, dir='+') == sqrt(2)
    assert limit(sin(x)/sqrt(1-cos(x)), x, 0, dir='-') == -sqrt(2)


def test_issue_18521():
    raises(NotImplementedError, lambda: limit(exp((2 - n) * x), x, oo))


def test_issue_18969():
    a, b = symbols('a b', positive=True)
    assert limit(LambertW(a), a, b) == LambertW(b)
    assert limit(exp(LambertW(a)), a, b) == exp(LambertW(b))


def test_issue_18992():
    assert limit(n/(factorial(n)**(1/n)), n, oo) == exp(1)


def test_issue_19067():
    x = Symbol('x')
    assert limit(gamma(x)/(gamma(x - 1)*gamma(x + 2)), x, 0) == -1


def test_issue_19586():
    assert limit(x**(2**x*3**(-x)), x, oo) == 1


def test_issue_13715():
    n = Symbol('n')
    p = Symbol('p', zero=True)
    assert limit(n + p, n, 0) == 0


def test_issue_15055():
    assert limit(n**3*((-n - 1)*sin(1/n) + (n + 2)*sin(1/(n + 1)))/(-n + 1), n, oo) == 1


def test_issue_16708():
    m, vi = symbols('m vi', positive=True)
    B, ti, d = symbols('B ti d')
    assert limit((B*ti*vi - sqrt(m)*sqrt(-2*B*d*vi + m*(vi)**2) + m*vi)/(B*vi), B, 0) == (d + ti*vi)/vi


def test_issue_19154():
    assert limit(besseli(1, 3 *x)/(x *besseli(1, x)**3), x , oo) == 2*sqrt(3)*pi/3
    assert limit(besseli(1, 3 *x)/(x *besseli(1, x)**3), x , -oo) == -2*sqrt(3)*pi/3


def test_issue_19453():
    beta = Symbol("beta", positive=True)
    h = Symbol("h", positive=True)
    m = Symbol("m", positive=True)
    w = Symbol("omega", positive=True)
    g = Symbol("g", positive=True)

    e = exp(1)
    q = 3*h**2*beta*g*e**(0.5*h*beta*w)
    p = m**2*w**2
    s = e**(h*beta*w) - 1
    Z = -q/(4*p*s) - q/(2*p*s**2) - q*(e**(h*beta*w) + 1)/(2*p*s**3)\
            + e**(0.5*h*beta*w)/s
    E = -diff(log(Z), beta)

    assert limit(E - 0.5*h*w, beta, oo) == 0
    assert limit(E.simplify() - 0.5*h*w, beta, oo) == 0


def test_issue_19739():
    assert limit((-S(1)/4)**x, x, oo) == 0


def test_issue_19766():
    assert limit(2**(-x)*sqrt(4**(x + 1) + 1), x, oo) == 2


def test_issue_19770():
    m = Symbol('m')
    # the result is not 0 for non-real m
    assert limit(cos(m*x)/x, x, oo) == Limit(cos(m*x)/x, x, oo, dir='-')
    m = Symbol('m', real=True)
    # can be improved to give the correct result 0
    assert limit(cos(m*x)/x, x, oo) == Limit(cos(m*x)/x, x, oo, dir='-')
    m = Symbol('m', nonzero=True)
    assert limit(cos(m*x), x, oo) == AccumBounds(-1, 1)
    assert limit(cos(m*x)/x, x, oo) == 0


def test_issue_7535():
    assert limit(tan(x)/sin(tan(x)), x, pi/2) == Limit(tan(x)/sin(tan(x)), x, pi/2, dir='+')
    assert limit(tan(x)/sin(tan(x)), x, pi/2, dir='-') == Limit(tan(x)/sin(tan(x)), x, pi/2, dir='-')
    assert limit(tan(x)/sin(tan(x)), x, pi/2, dir='+-') == Limit(tan(x)/sin(tan(x)), x, pi/2, dir='+-')
    assert limit(sin(tan(x)),x,pi/2) == AccumBounds(-1, 1)
    assert -oo*(1/sin(-oo)) == AccumBounds(-oo, oo)
    assert oo*(1/sin(oo)) == AccumBounds(-oo, oo)
    assert oo*(1/sin(-oo)) == AccumBounds(-oo, oo)
    assert -oo*(1/sin(oo)) == AccumBounds(-oo, oo)


def test_issue_20365():
    assert limit(((x + 1)**(1/x) - E)/x, x, 0) == -E/2


def test_issue_21031():
    assert limit(((1 + x)**(1/x) - (1 + 2*x)**(1/(2*x)))/asin(x), x, 0) == E/2


def test_issue_21038():
    assert limit(sin(pi*x)/(3*x - 12), x, 4) == pi/3


def test_issue_20578():
    expr = abs(x) * sin(1/x)
    assert limit(expr,x,0,'+') == 0
    assert limit(expr,x,0,'-') == 0
    assert limit(expr,x,0,'+-') == 0


def test_issue_21227():
    f = log(x)

    assert f.nseries(x, logx=y) == y
    assert f.nseries(x, logx=-x) == -x

    f = log(-log(x))

    assert f.nseries(x, logx=y) == log(-y)
    assert f.nseries(x, logx=-x) == log(x)

    f = log(log(x))

    assert f.nseries(x, logx=y) == log(y)
    assert f.nseries(x, logx=-x) == log(-x)
    assert f.nseries(x, logx=x) == log(x)

    f = log(log(log(1/x)))

    assert f.nseries(x, logx=y) == log(log(-y))
    assert f.nseries(x, logx=-y) == log(log(y))
    assert f.nseries(x, logx=x) == log(log(-x))
    assert f.nseries(x, logx=-x) == log(log(x))


def test_issue_21415():
    exp = (x-1)*cos(1/(x-1))
    assert exp.limit(x,1) == 0
    assert exp.expand().limit(x,1) == 0


def test_issue_21530():
    assert limit(sinh(n + 1)/sinh(n), n, oo) == E


def test_issue_21550():
    r = (sqrt(5) - 1)/2
    assert limit((x - r)/(x**2 + x - 1), x, r) == sqrt(5)/5


def test_issue_21661():
    out = limit((x**(x + 1) * (log(x) + 1) + 1) / x, x, 11)
    assert out == S(3138428376722)/11 + 285311670611*log(11)


def test_issue_21701():
    assert limit((besselj(z, x)/x**z).subs(z, 7), x, 0) == S(1)/645120


def test_issue_21721():
    a = Symbol('a', real=True)
    I = integrate(1/(pi*(1 + (x - a)**2)), x)
    assert I.limit(x, oo) == S.Half


def test_issue_21756():
    term = (1 - exp(-2*I*pi*z))/(1 - exp(-2*I*pi*z/5))
    assert term.limit(z, 0) == 5
    assert re(term).limit(z, 0) == 5


def test_issue_21785():
    a = Symbol('a')
    assert sqrt((-a**2 + x**2)/(1 - x**2)).limit(a, 1, '-') == I


def test_issue_22181():
    assert limit((-1)**x * 2**(-x), x, oo) == 0


def test_issue_22220():
    e1 = sqrt(30)*atan(sqrt(30)*tan(x/2)/6)/30
    e2 = sqrt(30)*I*(-log(sqrt(2)*tan(x/2) - 2*sqrt(15)*I/5) +
                     +log(sqrt(2)*tan(x/2) + 2*sqrt(15)*I/5))/60

    assert limit(e1, x, -pi) == -sqrt(30)*pi/60
    assert limit(e2, x, -pi) == -sqrt(30)*pi/30

    assert limit(e1, x, -pi, '-') == sqrt(30)*pi/60
    assert limit(e2, x, -pi, '-') == 0

    # test https://github.com/sympy/sympy/issues/22220#issuecomment-972727694
    expr = log(x - I) - log(-x - I)
    expr2 = logcombine(expr, force=True)
    assert limit(expr, x, oo) == limit(expr2, x, oo) == I*pi

    # test https://github.com/sympy/sympy/issues/22220#issuecomment-1077618340
    expr = expr = (-log(tan(x/2) - I) +log(tan(x/2) + I))
    assert limit(expr, x, pi, '+') == 2*I*pi
    assert limit(expr, x, pi, '-') == 0


def test_issue_22334():
    k, n  = symbols('k, n', positive=True)
    assert limit((n+1)**k/((n+1)**(k+1) - (n)**(k+1)), n, oo) == 1/(k + 1)
    assert limit((n+1)**k/((n+1)**(k+1) - (n)**(k+1)).expand(), n, oo) == 1/(k + 1)
    assert limit((n+1)**k/(n*(-n**k + (n + 1)**k) + (n + 1)**k), n, oo) == 1/(k + 1)


def test_sympyissue_22986():
    assert limit(acosh(1 + 1/x)*sqrt(x), x, oo) == sqrt(2)


def test_issue_23231():
    f = (2**x - 2**(-x))/(2**x + 2**(-x))
    assert limit(f, x, -oo) == -1


def test_issue_23596():
    assert integrate(((1 + x)/x**2)*exp(-1/x), (x, 0, oo)) == oo


def test_issue_23752():
    expr1 = sqrt(-I*x**2 + x - 3)
    expr2 = sqrt(-I*x**2 + I*x - 3)
    assert limit(expr1, x, 0, '+') == -sqrt(3)*I
    assert limit(expr1, x, 0, '-') == -sqrt(3)*I
    assert limit(expr2, x, 0, '+') == sqrt(3)*I
    assert limit(expr2, x, 0, '-') == -sqrt(3)*I


def test_issue_24276():
    fx = log(tan(pi/2*tanh(x))).diff(x)
    assert fx.limit(x, oo) == 2
    assert fx.simplify().limit(x, oo) == 2
    assert fx.rewrite(sin).limit(x, oo) == 2
    assert fx.rewrite(sin).simplify().limit(x, oo) == 2

def test_issue_25230():
    a = Symbol('a', real = True)
    b = Symbol('b', positive = True)
    c = Symbol('c', negative = True)
    n = Symbol('n', integer = True)
    raises(NotImplementedError, lambda: limit(Mod(x, a), x, a))
    assert limit(Mod(x, b), x, n*b, '+') == 0
    assert limit(Mod(x, b), x, n*b, '-') == b
    assert limit(Mod(x, c), x, n*c, '+') == c
    assert limit(Mod(x, c), x, n*c, '-') == 0


def test_issue_25582():

    assert limit(asin(exp(x)), x, oo, '-') == -oo*I
    assert limit(acos(exp(x)), x, oo, '-') == oo*I
    assert limit(atan(exp(x)), x, oo, '-') == pi/2
    assert limit(acot(exp(x)), x, oo, '-') == 0
    assert limit(asec(exp(x)), x, oo, '-') == pi/2
    assert limit(acsc(exp(x)), x, oo, '-') == 0


def test_issue_25847():
    #atan
    assert limit(atan(sin(x)/x), x, 0, '+-') == pi/4
    assert limit(atan(exp(1/x)), x, 0, '+') == pi/2
    assert limit(atan(exp(1/x)), x, 0, '-') == 0

    #asin
    assert limit(asin(sin(x)/x), x, 0, '+-') == pi/2
    assert limit(asin(exp(1/x)), x, 0, '+') == -oo*I
    assert limit(asin(exp(1/x)), x, 0, '-') == 0

    #acos
    assert limit(acos(sin(x)/x), x, 0, '+-') == 0
    assert limit(acos(exp(1/x)), x, 0, '+') == oo*I
    assert limit(acos(exp(1/x)), x, 0, '-') == pi/2

    #acot
    assert limit(acot(sin(x)/x), x, 0, '+-') == pi/4
    assert limit(acot(exp(1/x)), x, 0, '+') == 0
    assert limit(acot(exp(1/x)), x, 0, '-') == pi/2

    #asec
    assert limit(asec(sin(x)/x), x, 0, '+-') == 0
    assert limit(asec(exp(1/x)), x, 0, '+') == pi/2
    assert limit(asec(exp(1/x)), x, 0, '-') == oo*I

    #acsc
    assert limit(acsc(sin(x)/x), x, 0, '+-') == pi/2
    assert limit(acsc(exp(1/x)), x, 0, '+') == 0
    assert limit(acsc(exp(1/x)), x, 0, '-') == -oo*I

    #atanh
    assert limit(atanh(sin(x)/x), x, 0, '+-') == oo
    assert limit(atanh(exp(1/x)), x, 0, '+') == -I*pi/2
    assert limit(atanh(exp(1/x)), x, 0, '-') == 0

    #asinh
    assert limit(asinh(sin(x)/x), x, 0, '+-') == log(1 + sqrt(2))
    assert limit(asinh(exp(1/x)), x, 0, '+') == oo
    assert limit(asinh(exp(1/x)), x, 0, '-') == 0

    #acosh
    assert limit(acosh(sin(x)/x), x, 0, '+-') == 0
    assert limit(acosh(exp(1/x)), x, 0, '+') == oo
    assert limit(acosh(exp(1/x)), x, 0, '-') == I*pi/2

    #acoth
    assert limit(acoth(sin(x)/x), x, 0, '+-') == oo
    assert limit(acoth(exp(1/x)), x, 0, '+') == 0
    assert limit(acoth(exp(1/x)), x, 0, '-') == -I*pi/2

    #asech
    assert limit(asech(sin(x)/x), x, 0, '+-') == 0
    assert limit(asech(exp(1/x)), x, 0, '+') == I*pi/2
    assert limit(asech(exp(1/x)), x, 0, '-') == oo

    #acsch
    assert limit(acsch(sin(x)/x), x, 0, '+-') == log(1 + sqrt(2))
    assert limit(acsch(exp(1/x)), x, 0, '+') == 0
    assert limit(acsch(exp(1/x)), x, 0, '-') == oo


def test_issue_26040():
    assert limit(besseli(0, x + 1)/besseli(0, x), x, oo) == S.Exp1


def test_issue_26250():
    e = elliptic_e(4*x/(x**2 + 2*x + 1))
    k = elliptic_k(4*x/(x**2 + 2*x + 1))
    e1 = ((1-3*x**2)*e**2/2 - (x**2-2*x+1)*e*k/2)
    e2 = pi**2*(x**8 - 2*x**7 - x**6 + 4*x**5 - x**4 - 2*x**3 + x**2)
    assert limit(e1/e2, x, 0) == -S(1)/8
