from sympy.core import EulerGamma
from sympy.core.numbers import (E, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acot, atan, cos, sin)
from sympy.functions.elementary.complexes import sign as _sign
from sympy.functions.special.error_functions import (Ei, erf)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.polys.polytools import cancel
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh
from sympy.series.gruntz import compare, mrv, rewrite, mrv_leadterm, gruntz, \
    sign
from sympy.testing.pytest import XFAIL, skip, slow

"""
This test suite is testing the limit algorithm using the bottom up approach.
See the documentation in limits2.py. The algorithm itself is highly recursive
by nature, so "compare" is logically the lowest part of the algorithm, yet in
some sense it's the most complex part, because it needs to calculate a limit
to return the result.

Nevertheless, the rest of the algorithm depends on compare working correctly.
"""

x = Symbol('x', real=True)
m = Symbol('m', real=True)


runslow = False


def _sskip():
    if not runslow:
        skip("slow")


@slow
def test_gruntz_evaluation():
    # Gruntz' thesis pp. 122 to 123
    # 8.1
    assert gruntz(exp(x)*(exp(1/x - exp(-x)) - exp(1/x)), x, oo) == -1
    # 8.2
    assert gruntz(exp(x)*(exp(1/x + exp(-x) + exp(-x**2))
                  - exp(1/x - exp(-exp(x)))), x, oo) == 1
    # 8.3
    assert gruntz(exp(exp(x - exp(-x))/(1 - 1/x)) - exp(exp(x)), x, oo) is oo
    # 8.5
    assert gruntz(exp(exp(exp(x + exp(-x)))) / exp(exp(exp(x))), x, oo) is oo
    # 8.6
    assert gruntz(exp(exp(exp(x))) / exp(exp(exp(x - exp(-exp(x))))),
                  x, oo) is oo
    # 8.7
    assert gruntz(exp(exp(exp(x))) / exp(exp(exp(x - exp(-exp(exp(x)))))),
                  x, oo) == 1
    # 8.8
    assert gruntz(exp(exp(x)) / exp(exp(x - exp(-exp(exp(x))))), x, oo) == 1
    # 8.9
    assert gruntz(log(x)**2 * exp(sqrt(log(x))*(log(log(x)))**2
                  * exp(sqrt(log(log(x))) * (log(log(log(x))))**3)) / sqrt(x),
                  x, oo) == 0
    # 8.10
    assert gruntz((x*log(x)*(log(x*exp(x) - x**2))**2)
                  / (log(log(x**2 + 2*exp(exp(3*x**3*log(x)))))), x, oo) == Rational(1, 3)
    # 8.11
    assert gruntz((exp(x*exp(-x)/(exp(-x) + exp(-2*x**2/(x + 1)))) - exp(x))/x,
                  x, oo) == -exp(2)
    # 8.12
    assert gruntz((3**x + 5**x)**(1/x), x, oo) == 5
    # 8.13
    assert gruntz(x/log(x**(log(x**(log(2)/log(x))))), x, oo) is oo
    # 8.14
    assert gruntz(exp(exp(2*log(x**5 + x)*log(log(x))))
                  / exp(exp(10*log(x)*log(log(x)))), x, oo) is oo
    # 8.15
    assert gruntz(exp(exp(Rational(5, 2)*x**Rational(-5, 7) + Rational(21, 8)*x**Rational(6, 11)
                          + 2*x**(-8) + Rational(54, 17)*x**Rational(49, 45)))**8
                  / log(log(-log(Rational(4, 3)*x**Rational(-5, 14))))**Rational(7, 6), x, oo) is oo
    # 8.16
    assert gruntz((exp(4*x*exp(-x)/(1/exp(x) + 1/exp(2*x**2/(x + 1)))) - exp(x))
                  / exp(x)**4, x, oo) == 1
    # 8.17
    assert gruntz(exp(x*exp(-x)/(exp(-x) + exp(-2*x**2/(x + 1))))/exp(x), x, oo) \
        == 1
    # 8.19
    assert gruntz(log(x)*(log(log(x) + log(log(x))) - log(log(x)))
                  / (log(log(x) + log(log(log(x))))), x, oo) == 1
    # 8.20
    assert gruntz(exp((log(log(x + exp(log(x)*log(log(x))))))
                  / (log(log(log(exp(x) + x + log(x)))))), x, oo) == E
    # Another
    assert gruntz(exp(exp(exp(x + exp(-x)))) / exp(exp(x)), x, oo) is oo


def test_gruntz_evaluation_slow():
    _sskip()
    # 8.4
    assert gruntz(exp(exp(exp(x)/(1 - 1/x)))
                  - exp(exp(exp(x)/(1 - 1/x - log(x)**(-log(x))))), x, oo) is -oo
    # 8.18
    assert gruntz((exp(exp(-x/(1 + exp(-x))))*exp(-x/(1 + exp(-x/(1 + exp(-x)))))
                   *exp(exp(-x + exp(-x/(1 + exp(-x))))))
                  / (exp(-x/(1 + exp(-x))))**2 - exp(x) + x, x, oo) == 2


@slow
def test_gruntz_eval_special():
    # Gruntz, p. 126
    assert gruntz(exp(x)*(sin(1/x + exp(-x)) - sin(1/x + exp(-x**2))), x, oo) == 1
    assert gruntz((erf(x - exp(-exp(x))) - erf(x)) * exp(exp(x)) * exp(x**2),
                  x, oo) == -2/sqrt(pi)
    assert gruntz(exp(exp(x)) * (exp(sin(1/x + exp(-exp(x)))) - exp(sin(1/x))),
                  x, oo) == 1
    assert gruntz(exp(x)*(gamma(x + exp(-x)) - gamma(x)), x, oo) is oo
    assert gruntz(exp(exp(digamma(digamma(x))))/x, x, oo) == exp(Rational(-1, 2))
    assert gruntz(exp(exp(digamma(log(x))))/x, x, oo) == exp(Rational(-1, 2))
    assert gruntz(digamma(digamma(digamma(x))), x, oo) is oo
    assert gruntz(loggamma(loggamma(x)), x, oo) is oo
    assert gruntz(((gamma(x + 1/gamma(x)) - gamma(x))/log(x) - cos(1/x))
                  * x*log(x), x, oo) == Rational(-1, 2)
    assert gruntz(x * (gamma(x - 1/gamma(x)) - gamma(x) + log(x)), x, oo) \
        == S.Half
    assert gruntz((gamma(x + 1/gamma(x)) - gamma(x)) / log(x), x, oo) == 1


def test_gruntz_eval_special_slow():
    _sskip()
    assert gruntz(gamma(x + 1)/sqrt(2*pi)
                  - exp(-x)*(x**(x + S.Half) + x**(x - S.Half)/12), x, oo) is oo
    assert gruntz(exp(exp(exp(digamma(digamma(digamma(x))))))/x, x, oo) == 0


@XFAIL
def test_grunts_eval_special_slow_sometimes_fail():
    _sskip()
    # XXX This sometimes fails!!!
    assert gruntz(exp(gamma(x - exp(-x))*exp(1/x)) - exp(gamma(x)), x, oo) is oo


def test_gruntz_Ei():
    assert gruntz((Ei(x - exp(-exp(x))) - Ei(x)) *exp(-x)*exp(exp(x))*x, x, oo) == -1


@XFAIL
def test_gruntz_eval_special_fail():
    # TODO zeta function series
    assert gruntz(
        exp((log(2) + 1)*x) * (zeta(x + exp(-x)) - zeta(x)), x, oo) == -log(2)

    # TODO 8.35 - 8.37 (bessel, max-min)


def test_gruntz_hyperbolic():
    assert gruntz(cosh(x), x, oo) is oo
    assert gruntz(cosh(x), x, -oo) is oo
    assert gruntz(sinh(x), x, oo) is oo
    assert gruntz(sinh(x), x, -oo) is -oo
    assert gruntz(2*cosh(x)*exp(x), x, oo) is oo
    assert gruntz(2*cosh(x)*exp(x), x, -oo) == 1
    assert gruntz(2*sinh(x)*exp(x), x, oo) is oo
    assert gruntz(2*sinh(x)*exp(x), x, -oo) == -1
    assert gruntz(tanh(x), x, oo) == 1
    assert gruntz(tanh(x), x, -oo) == -1
    assert gruntz(coth(x), x, oo) == 1
    assert gruntz(coth(x), x, -oo) == -1


def test_compare1():
    assert compare(2, x, x) == "<"
    assert compare(x, exp(x), x) == "<"
    assert compare(exp(x), exp(x**2), x) == "<"
    assert compare(exp(x**2), exp(exp(x)), x) == "<"
    assert compare(1, exp(exp(x)), x) == "<"

    assert compare(x, 2, x) == ">"
    assert compare(exp(x), x, x) == ">"
    assert compare(exp(x**2), exp(x), x) == ">"
    assert compare(exp(exp(x)), exp(x**2), x) == ">"
    assert compare(exp(exp(x)), 1, x) == ">"

    assert compare(2, 3, x) == "="
    assert compare(3, -5, x) == "="
    assert compare(2, -5, x) == "="

    assert compare(x, x**2, x) == "="
    assert compare(x**2, x**3, x) == "="
    assert compare(x**3, 1/x, x) == "="
    assert compare(1/x, x**m, x) == "="
    assert compare(x**m, -x, x) == "="

    assert compare(exp(x), exp(-x), x) == "="
    assert compare(exp(-x), exp(2*x), x) == "="
    assert compare(exp(2*x), exp(x)**2, x) == "="
    assert compare(exp(x)**2, exp(x + exp(-x)), x) == "="
    assert compare(exp(x), exp(x + exp(-x)), x) == "="

    assert compare(exp(x**2), 1/exp(x**2), x) == "="


def test_compare2():
    assert compare(exp(x), x**5, x) == ">"
    assert compare(exp(x**2), exp(x)**2, x) == ">"
    assert compare(exp(x), exp(x + exp(-x)), x) == "="
    assert compare(exp(x + exp(-x)), exp(x), x) == "="
    assert compare(exp(x + exp(-x)), exp(-x), x) == "="
    assert compare(exp(-x), x, x) == ">"
    assert compare(x, exp(-x), x) == "<"
    assert compare(exp(x + 1/x), x, x) == ">"
    assert compare(exp(-exp(x)), exp(x), x) == ">"
    assert compare(exp(exp(-exp(x)) + x), exp(-exp(x)), x) == "<"


def test_compare3():
    assert compare(exp(exp(x)), exp(x + exp(-exp(x))), x) == ">"


def test_sign1():
    assert sign(Rational(0), x) == 0
    assert sign(Rational(3), x) == 1
    assert sign(Rational(-5), x) == -1
    assert sign(log(x), x) == 1
    assert sign(exp(-x), x) == 1
    assert sign(exp(x), x) == 1
    assert sign(-exp(x), x) == -1
    assert sign(3 - 1/x, x) == 1
    assert sign(-3 - 1/x, x) == -1
    assert sign(sin(1/x), x) == 1
    assert sign((x**Integer(2)), x) == 1
    assert sign(x**2, x) == 1
    assert sign(x**5, x) == 1


def test_sign2():
    assert sign(x, x) == 1
    assert sign(-x, x) == -1
    y = Symbol("y", positive=True)
    assert sign(y, x) == 1
    assert sign(-y, x) == -1
    assert sign(y*x, x) == 1
    assert sign(-y*x, x) == -1


def mmrv(a, b):
    return set(mrv(a, b)[0].keys())


def test_mrv1():
    assert mmrv(x, x) == {x}
    assert mmrv(x + 1/x, x) == {x}
    assert mmrv(x**2, x) == {x}
    assert mmrv(log(x), x) == {x}
    assert mmrv(exp(x), x) == {exp(x)}
    assert mmrv(exp(-x), x) == {exp(-x)}
    assert mmrv(exp(x**2), x) == {exp(x**2)}
    assert mmrv(-exp(1/x), x) == {x}
    assert mmrv(exp(x + 1/x), x) == {exp(x + 1/x)}


def test_mrv2a():
    assert mmrv(exp(x + exp(-exp(x))), x) == {exp(-exp(x))}
    assert mmrv(exp(x + exp(-x)), x) == {exp(x + exp(-x)), exp(-x)}
    assert mmrv(exp(1/x + exp(-x)), x) == {exp(-x)}

#sometimes infinite recursion due to log(exp(x**2)) not simplifying


def test_mrv2b():
    assert mmrv(exp(x + exp(-x**2)), x) == {exp(-x**2)}

#sometimes infinite recursion due to log(exp(x**2)) not simplifying


def test_mrv2c():
    assert mmrv(
        exp(-x + 1/x**2) - exp(x + 1/x), x) == {exp(x + 1/x), exp(1/x**2 - x)}

#sometimes infinite recursion due to log(exp(x**2)) not simplifying


def test_mrv3():
    assert mmrv(exp(x**2) + x*exp(x) + log(x)**x/x, x) == {exp(x**2)}
    assert mmrv(
        exp(x)*(exp(1/x + exp(-x)) - exp(1/x)), x) == {exp(x), exp(-x)}
    assert mmrv(log(
        x**2 + 2*exp(exp(3*x**3*log(x)))), x) == {exp(exp(3*x**3*log(x)))}
    assert mmrv(log(x - log(x))/log(x), x) == {x}
    assert mmrv(
        (exp(1/x - exp(-x)) - exp(1/x))*exp(x), x) == {exp(x), exp(-x)}
    assert mmrv(
        1/exp(-x + exp(-x)) - exp(x), x) == {exp(x), exp(-x), exp(x - exp(-x))}
    assert mmrv(log(log(x*exp(x*exp(x)) + 1)), x) == {exp(x*exp(x))}
    assert mmrv(exp(exp(log(log(x) + 1/x))), x) == {x}


def test_mrv4():
    ln = log
    assert mmrv((ln(ln(x) + ln(ln(x))) - ln(ln(x)))/ln(ln(x) + ln(ln(ln(x))))*ln(x),
            x) == {x}
    assert mmrv(log(log(x*exp(x*exp(x)) + 1)) - exp(exp(log(log(x) + 1/x))), x) == \
        {exp(x*exp(x))}


def mrewrite(a, b, c):
    return rewrite(a[1], a[0], b, c)


def test_rewrite1():
    e = exp(x)
    assert mrewrite(mrv(e, x), x, m) == (1/m, -x)
    e = exp(x**2)
    assert mrewrite(mrv(e, x), x, m) == (1/m, -x**2)
    e = exp(x + 1/x)
    assert mrewrite(mrv(e, x), x, m) == (1/m, -x - 1/x)
    e = 1/exp(-x + exp(-x)) - exp(x)
    assert mrewrite(mrv(e, x), x, m) == (1/(m*exp(m)) - 1/m, -x)


def test_rewrite2():
    e = exp(x)*log(log(exp(x)))
    assert mmrv(e, x) == {exp(x)}
    assert mrewrite(mrv(e, x), x, m) == (1/m*log(x), -x)

#sometimes infinite recursion due to log(exp(x**2)) not simplifying


def test_rewrite3():
    e = exp(-x + 1/x**2) - exp(x + 1/x)
    #both of these are correct and should be equivalent:
    assert mrewrite(mrv(e, x), x, m) in [(-1/m + m*exp(
        1/x + 1/x**2), -x - 1/x), (m - 1/m*exp(1/x + x**(-2)), x**(-2) - x)]


def test_mrv_leadterm1():
    assert mrv_leadterm(-exp(1/x), x) == (-1, 0)
    assert mrv_leadterm(1/exp(-x + exp(-x)) - exp(x), x) == (-1, 0)
    assert mrv_leadterm(
        (exp(1/x - exp(-x)) - exp(1/x))*exp(x), x) == (-exp(1/x), 0)


def test_mrv_leadterm2():
    #Gruntz: p51, 3.25
    assert mrv_leadterm((log(exp(x) + x) - x)/log(exp(x) + log(x))*exp(x), x) == \
        (1, 0)


def test_mrv_leadterm3():
    #Gruntz: p56, 3.27
    assert mmrv(exp(-x + exp(-x)*exp(-x*log(x))), x) == {exp(-x - x*log(x))}
    assert mrv_leadterm(exp(-x + exp(-x)*exp(-x*log(x))), x) == (exp(-x), 0)


def test_limit1():
    assert gruntz(x, x, oo) is oo
    assert gruntz(x, x, -oo) is -oo
    assert gruntz(-x, x, oo) is -oo
    assert gruntz(x**2, x, -oo) is oo
    assert gruntz(-x**2, x, oo) is -oo
    assert gruntz(x*log(x), x, 0, dir="+") == 0
    assert gruntz(1/x, x, oo) == 0
    assert gruntz(exp(x), x, oo) is oo
    assert gruntz(-exp(x), x, oo) is -oo
    assert gruntz(exp(x)/x, x, oo) is oo
    assert gruntz(1/x - exp(-x), x, oo) == 0
    assert gruntz(x + 1/x, x, oo) is oo


def test_limit2():
    assert gruntz(x**x, x, 0, dir="+") == 1
    assert gruntz((exp(x) - 1)/x, x, 0) == 1
    assert gruntz(1 + 1/x, x, oo) == 1
    assert gruntz(-exp(1/x), x, oo) == -1
    assert gruntz(x + exp(-x), x, oo) is oo
    assert gruntz(x + exp(-x**2), x, oo) is oo
    assert gruntz(x + exp(-exp(x)), x, oo) is oo
    assert gruntz(13 + 1/x - exp(-x), x, oo) == 13


def test_limit3():
    a = Symbol('a')
    assert gruntz(x - log(1 + exp(x)), x, oo) == 0
    assert gruntz(x - log(a + exp(x)), x, oo) == 0
    assert gruntz(exp(x)/(1 + exp(x)), x, oo) == 1
    assert gruntz(exp(x)/(a + exp(x)), x, oo) == 1


def test_limit4():
    #issue 3463
    assert gruntz((3**x + 5**x)**(1/x), x, oo) == 5
    #issue 3463
    assert gruntz((3**(1/x) + 5**(1/x))**x, x, 0) == 5


@XFAIL
def test_MrvTestCase_page47_ex3_21():
    h = exp(-x/(1 + exp(-x)))
    expr = exp(h)*exp(-x/(1 + h))*exp(exp(-x + h))/h**2 - exp(x) + x
    assert mmrv(expr, x) == {1/h, exp(-x), exp(x), exp(x - h), exp(x/(1 + h))}


def test_gruntz_I():
    y = Symbol("y")
    assert gruntz(I*x, x, oo) == I*oo
    assert gruntz(y*I*x, x, oo) == y*I*oo
    assert gruntz(y*3*I*x, x, oo) == y*I*oo
    assert gruntz(y*3*sin(I)*x, x, oo).simplify().rewrite(_sign) == _sign(y)*I*oo


def test_issue_4814():
    assert gruntz((x + 1)**(1/log(x + 1)), x, oo) == E


def test_intractable():
    assert gruntz(1/gamma(x), x, oo) == 0
    assert gruntz(1/loggamma(x), x, oo) == 0
    assert gruntz(gamma(x)/loggamma(x), x, oo) is oo
    assert gruntz(exp(gamma(x))/gamma(x), x, oo) is oo
    assert gruntz(gamma(x), x, 3) == 2
    assert gruntz(gamma(Rational(1, 7) + 1/x), x, oo) == gamma(Rational(1, 7))
    assert gruntz(log(x**x)/log(gamma(x)), x, oo) == 1
    assert gruntz(log(gamma(gamma(x)))/exp(x), x, oo) is oo


def test_aseries_trig():
    assert cancel(gruntz(1/log(atan(x)), x, oo)
           - 1/(log(pi) + log(S.Half))) == 0
    assert gruntz(1/acot(x), x, -oo) is -oo


def test_exp_log_series():
    assert gruntz(x/log(log(x*exp(x))), x, oo) is oo


def test_issue_3644():
    assert gruntz(((x**7 + x + 1)/(2**x + x**2))**(-1/x), x, oo) == 2


def test_issue_6843():
    n = Symbol('n', integer=True, positive=True)
    r = (n + 1)*x**(n + 1)/(x**(n + 1) - 1) - x/(x - 1)
    assert gruntz(r, x, 1).simplify() == n/2


def test_issue_4190():
    assert gruntz(x - gamma(1/x), x, oo) == S.EulerGamma


@XFAIL
def test_issue_5172():
    n = Symbol('n')
    r = Symbol('r', positive=True)
    c = Symbol('c')
    p = Symbol('p', positive=True)
    m = Symbol('m', negative=True)
    expr = ((2*n*(n - r + 1)/(n + r*(n - r + 1)))**c + \
        (r - 1)*(n*(n - r + 2)/(n + r*(n - r + 1)))**c - n)/(n**c - n)
    expr = expr.subs(c, c + 1)
    assert gruntz(expr.subs(c, m), n, oo) == 1
    # fail:
    assert gruntz(expr.subs(c, p), n, oo).simplify() == \
        (2**(p + 1) + r - 1)/(r + 1)**(p + 1)


def test_issue_4109():
    assert gruntz(1/gamma(x), x, 0) == 0
    assert gruntz(x*gamma(x), x, 0) == 1


def test_issue_6682():
    assert gruntz(exp(2*Ei(-x))/x**2, x, 0) == exp(2*EulerGamma)


def test_issue_7096():
    from sympy.functions import sign
    assert gruntz(x**-pi, x, 0, dir='-') == oo*sign((-1)**(-pi))

def test_issue_24210_25885():
    eq = exp(x)/(1+1/x)**x**2
    ans = sqrt(E)
    assert gruntz(eq, x, oo) == ans
    assert gruntz(1/eq, x, oo) == 1/ans
