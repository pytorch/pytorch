# A collection of failing integrals from the issues.

from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (sech, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, atan, cos, sin, tan)
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import (Integral, integrate)
from sympy.simplify.fu import fu


from sympy.testing.pytest import XFAIL, slow, tooslow

from sympy.abc import x, k, c, y, b, h, a, m, z, n, t


@tooslow
@XFAIL
def test_issue_3880():
    # integrate_hyperexponential(Poly(t*2*(1 - t0**2)*t0*(x**3 + x**2), t), Poly((1 + t0**2)**2*2*(x**2 + x + 1), t), [Poly(1, x), Poly(1 + t0**2, t0), Poly(t, t)], [x, t0, t], [exp, tan])
    assert not integrate(exp(x)*cos(2*x)*sin(2*x) * (x**3 + x**2)/(2*(x**2 + x + 1)), x).has(Integral)


def test_issue_4212_real():
    xr = symbols('xr', real=True)
    negabsx = Piecewise((-xr, xr < 0), (xr, True))
    assert integrate(sign(xr), xr) == negabsx


@XFAIL
def test_issue_4212():
    # XXX: Maybe this should be expected to fail without real assumptions on x.
    # As a complex function sign(x) is not analytic and so there is no complex
    # function whose complex derivative is sign(x). With real assumptions this
    # works (see test_issue_4212_real above).
    assert not integrate(sign(x), x).has(Integral)


def test_issue_4511():
    # This works, but gives a slightly over-complicated answer.
    f = integrate(cos(x)**2 / (1 - sin(x)), x)
    assert fu(f) == x - cos(x) - 1
    assert f == ((x*tan(x/2)**2 + x - 2)/(tan(x/2)**2 + 1)).expand()


def test_integrate_DiracDelta_no_meijerg():
    assert integrate(integrate(integrate(
        DiracDelta(x - y - z), (z, 0, oo)), (y, 0, 1), meijerg=False), (x, 0, 1)) == S.Half


@XFAIL
def test_integrate_DiracDelta_fails():
    # issue 6427
    # works without meijerg. See test_integrate_DiracDelta_no_meijerg above.
    assert integrate(integrate(integrate(
        DiracDelta(x - y - z), (z, 0, oo)), (y, 0, 1)), (x, 0, 1)) == S.Half


@XFAIL
@slow
def test_issue_4525():
    # Warning: takes a long time
    assert not integrate((x**m * (1 - x)**n * (a + b*x + c*x**2))/(1 + x**2), (x, 0, 1)).has(Integral)


@XFAIL
@tooslow
def test_issue_4540():
    # Note, this integral is probably nonelementary
    assert not integrate(
        (sin(1/x) - x*exp(x)) /
        ((-sin(1/x) + x*exp(x))*x + x*sin(1/x)), x).has(Integral)


@XFAIL
@slow
def test_issue_4891():
    # Requires the hypergeometric function.
    assert not integrate(cos(x)**y, x).has(Integral)


@XFAIL
@slow
def test_issue_1796a():
    assert not integrate(exp(2*b*x)*exp(-a*x**2), x).has(Integral)


@XFAIL
def test_issue_4895b():
    assert not integrate(exp(2*b*x)*exp(-a*x**2), (x, -oo, 0)).has(Integral)


@XFAIL
def test_issue_4895c():
    assert not integrate(exp(2*b*x)*exp(-a*x**2), (x, -oo, oo)).has(Integral)


@XFAIL
def test_issue_4895d():
    assert not integrate(exp(2*b*x)*exp(-a*x**2), (x, 0, oo)).has(Integral)


@XFAIL
@slow
def test_issue_4941():
    assert not integrate(sqrt(1 + sinh(x/20)**2), (x, -25, 25)).has(Integral)


@XFAIL
def test_issue_4992():
    # Nonelementary integral.  Requires hypergeometric/Meijer-G handling.
    assert not integrate(log(x) * x**(k - 1) * exp(-x) / gamma(k), (x, 0, oo)).has(Integral)


@XFAIL
def test_issue_16396a():
    i = integrate(1/(1+sqrt(tan(x))), (x, pi/3, pi/6))
    assert not i.has(Integral)


@XFAIL
def test_issue_16396b():
    i = integrate(x*sin(x)/(1+cos(x)**2), (x, 0, pi))
    assert not i.has(Integral)


@XFAIL
def test_issue_16046():
    assert integrate(exp(exp(I*x)), [x, 0, 2*pi]) == 2*pi


@XFAIL
def test_issue_15925a():
    assert not integrate(sqrt((1+sin(x))**2+(cos(x))**2), (x, -pi/2, pi/2)).has(Integral)


def test_issue_15925b():
    f = sqrt((-12*cos(x)**2*sin(x))**2+(12*cos(x)*sin(x)**2)**2)
    assert integrate(f, (x, 0, pi/6)) == Rational(3, 2)


@XFAIL
def test_issue_15925b_manual():
    assert not integrate(sqrt((-12*cos(x)**2*sin(x))**2+(12*cos(x)*sin(x)**2)**2),
                         (x, 0, pi/6), manual=True).has(Integral)


@XFAIL
@tooslow
def test_issue_15227():
    i = integrate(log(1-x)*log((1+x)**2)/x, (x, 0, 1))
    assert not i.has(Integral)
    # assert i == -5*zeta(3)/4


@XFAIL
@slow
def test_issue_14716():
    i = integrate(log(x + 5)*cos(pi*x),(x, S.Half, 1))
    assert not i.has(Integral)
    # Mathematica can not solve it either, but
    # integrate(log(x + 5)*cos(pi*x),(x, S.Half, 1)).transform(x, y - 5).doit()
    # works
    # assert i == -log(Rational(11, 2))/pi - Si(pi*Rational(11, 2))/pi + Si(6*pi)/pi


@XFAIL
def test_issue_14709a():
    i = integrate(x*acos(1 - 2*x/h), (x, 0, h))
    assert not i.has(Integral)
    # assert i == 5*h**2*pi/16


@slow
@XFAIL
def test_issue_14398():
    assert not integrate(exp(x**2)*cos(x), x).has(Integral)


@XFAIL
def test_issue_14074():
    i = integrate(log(sin(x)), (x, 0, pi/2))
    assert not i.has(Integral)
    # assert i == -pi*log(2)/2


@XFAIL
@slow
def test_issue_14078b():
    i = integrate((atan(4*x)-atan(2*x))/x, (x, 0, oo))
    assert not i.has(Integral)
    # assert i == pi*log(2)/2


@XFAIL
def test_issue_13792():
    i =  integrate(log(1/x) / (1 - x), (x, 0, 1))
    assert not i.has(Integral)
    # assert i in [polylog(2, -exp_polar(I*pi)), pi**2/6]


@XFAIL
def test_issue_11845a():
    assert not integrate(exp(y - x**3), (x, 0, 1)).has(Integral)


@XFAIL
def test_issue_11845b():
    assert not integrate(exp(-y - x**3), (x, 0, 1)).has(Integral)


@XFAIL
def test_issue_11813():
    assert not integrate((a - x)**Rational(-1, 2)*x, (x, 0, a)).has(Integral)


@XFAIL
def test_issue_11254c():
    assert not integrate(sech(x)**2, (x, 0, 1)).has(Integral)


@XFAIL
def test_issue_10584():
    assert not integrate(sqrt(x**2 + 1/x**2), x).has(Integral)


@XFAIL
def test_issue_9101():
    assert not integrate(log(x + sqrt(x**2 + y**2 + z**2)), z).has(Integral)


@XFAIL
def test_issue_7147():
    assert not integrate(x/sqrt(a*x**2 + b*x + c)**3, x).has(Integral)


@XFAIL
def test_issue_7109():
    assert not integrate(sqrt(a**2/(a**2 - x**2)), x).has(Integral)


@XFAIL
def test_integrate_Piecewise_rational_over_reals():
    f = Piecewise(
        (0,                                              t - 478.515625*pi <  0),
        (13.2075145209219*pi/(0.000871222*t + 0.995)**2, t - 478.515625*pi >= 0))

    assert abs((integrate(f, (t, 0, oo)) - 15235.9375*pi).evalf()) <= 1e-7


@XFAIL
def test_issue_4311_slow():
    # Not slow when bypassing heurish
    assert not integrate(x*abs(9-x**2), x).has(Integral)

@XFAIL
def test_issue_20370():
    a = symbols('a', positive=True)
    assert integrate((1 + a * cos(x))**-1, (x, 0, 2 * pi)) == (2 * pi / sqrt(1 - a**2))


@XFAIL
def test_polylog():
    # log(1/x)*log(x+1)-polylog(2, -x)
    assert not integrate(log(1/x)/(x + 1), x).has(Integral)


@XFAIL
def test_polylog_manual():
    # Make sure _parts_rule does not go into an infinite loop here
    assert not integrate(log(1/x)/(x + 1), x, manual=True).has(Integral)
