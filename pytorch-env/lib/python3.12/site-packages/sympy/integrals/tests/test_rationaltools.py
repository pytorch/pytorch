from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan
from sympy.integrals.integrals import integrate
from sympy.polys.polytools import Poly
from sympy.simplify.simplify import simplify

from sympy.integrals.rationaltools import ratint, ratint_logpart, log_to_atan

from sympy.abc import a, b, x, t

half = S.Half


def test_ratint():
    assert ratint(S.Zero, x) == 0
    assert ratint(S(7), x) == 7*x

    assert ratint(x, x) == x**2/2
    assert ratint(2*x, x) == x**2
    assert ratint(-2*x, x) == -x**2

    assert ratint(8*x**7 + 2*x + 1, x) == x**8 + x**2 + x

    f = S.One
    g = x + 1

    assert ratint(f / g, x) == log(x + 1)
    assert ratint((f, g), x) == log(x + 1)

    f = x**3 - x
    g = x - 1

    assert ratint(f/g, x) == x**3/3 + x**2/2

    f = x
    g = (x - a)*(x + a)

    assert ratint(f/g, x) == log(x**2 - a**2)/2

    f = S.One
    g = x**2 + 1

    assert ratint(f/g, x, real=None) == atan(x)
    assert ratint(f/g, x, real=True) == atan(x)

    assert ratint(f/g, x, real=False) == I*log(x + I)/2 - I*log(x - I)/2

    f = S(36)
    g = x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2

    assert ratint(f/g, x) == \
        -4*log(x + 1) + 4*log(x - 2) + (12*x + 6)/(x**2 - 1)

    f = x**4 - 3*x**2 + 6
    g = x**6 - 5*x**4 + 5*x**2 + 4

    assert ratint(f/g, x) == \
        atan(x) + atan(x**3) + atan(x/2 - Rational(3, 2)*x**3 + S.Half*x**5)

    f = x**7 - 24*x**4 - 4*x**2 + 8*x - 8
    g = x**8 + 6*x**6 + 12*x**4 + 8*x**2

    assert ratint(f/g, x) == \
        (4 + 6*x + 8*x**2 + 3*x**3)/(4*x + 4*x**3 + x**5) + log(x)

    assert ratint((x**3*f)/(x*g), x) == \
        -(12 - 16*x + 6*x**2 - 14*x**3)/(4 + 4*x**2 + x**4) - \
        5*sqrt(2)*atan(x*sqrt(2)/2) + S.Half*x**2 - 3*log(2 + x**2)

    f = x**5 - x**4 + 4*x**3 + x**2 - x + 5
    g = x**4 - 2*x**3 + 5*x**2 - 4*x + 4

    assert ratint(f/g, x) == \
        x + S.Half*x**2 + S.Half*log(2 - x + x**2) + (9 - 4*x)/(7*x**2 - 7*x + 14) + \
        13*sqrt(7)*atan(Rational(-1, 7)*sqrt(7) + 2*x*sqrt(7)/7)/49

    assert ratint(1/(x**2 + x + 1), x) == \
        2*sqrt(3)*atan(sqrt(3)/3 + 2*x*sqrt(3)/3)/3

    assert ratint(1/(x**3 + 1), x) == \
        -log(1 - x + x**2)/6 + log(1 + x)/3 + sqrt(3)*atan(-sqrt(3)
             /3 + 2*x*sqrt(3)/3)/3

    assert ratint(1/(x**2 + x + 1), x, real=False) == \
        -I*3**half*log(half + x - half*I*3**half)/3 + \
        I*3**half*log(half + x + half*I*3**half)/3

    assert ratint(1/(x**3 + 1), x, real=False) == log(1 + x)/3 + \
        (Rational(-1, 6) + I*3**half/6)*log(-half + x + I*3**half/2) + \
        (Rational(-1, 6) - I*3**half/6)*log(-half + x - I*3**half/2)

    # issue 4991
    assert ratint(1/(x*(a + b*x)**3), x) == \
        (3*a + 2*b*x)/(2*a**4 + 4*a**3*b*x + 2*a**2*b**2*x**2) + (
            log(x) - log(a/b + x))/a**3

    assert ratint(x/(1 - x**2), x) == -log(x**2 - 1)/2
    assert ratint(-x/(1 - x**2), x) == log(x**2 - 1)/2

    assert ratint((x/4 - 4/(1 - x)).diff(x), x) == x/4 + 4/(x - 1)

    ans = atan(x)
    assert ratint(1/(x**2 + 1), x, symbol=x) == ans
    assert ratint(1/(x**2 + 1), x, symbol='x') == ans
    assert ratint(1/(x**2 + 1), x, symbol=a) == ans
    # this asserts that as_dummy must return a unique symbol
    # even if the symbol is already a Dummy
    d = Dummy()
    assert ratint(1/(d**2 + 1), d, symbol=d) == atan(d)


def test_ratint_logpart():
    assert ratint_logpart(x, x**2 - 9, x, t) == \
        [(Poly(x**2 - 9, x), Poly(-2*t + 1, t))]
    assert ratint_logpart(x**2, x**3 - 5, x, t) == \
        [(Poly(x**3 - 5, x), Poly(-3*t + 1, t))]


def test_issue_5414():
    assert ratint(1/(x**2 + 16), x) == atan(x/4)/4


def test_issue_5249():
    assert ratint(
        1/(x**2 + a**2), x) == (-I*log(-I*a + x)/2 + I*log(I*a + x)/2)/a


def test_issue_5817():
    a, b, c = symbols('a,b,c', positive=True)

    assert simplify(ratint(a/(b*c*x**2 + a**2 + b*a), x)) == \
        sqrt(a)*atan(sqrt(
            b)*sqrt(c)*x/(sqrt(a)*sqrt(a + b)))/(sqrt(b)*sqrt(c)*sqrt(a + b))


def test_issue_5981():
    u = symbols('u')
    assert integrate(1/(u**2 + 1)) == atan(u)

def test_issue_10488():
    a,b,c,x = symbols('a b c x', positive=True)
    assert integrate(x/(a*x+b),x) == x/a - b*log(a*x + b)/a**2


def test_issues_8246_12050_13501_14080():
    a = symbols('a', nonzero=True)
    assert integrate(a/(x**2 + a**2), x) == atan(x/a)
    assert integrate(1/(x**2 + a**2), x) == atan(x/a)/a
    assert integrate(1/(1 + a**2*x**2), x) == atan(a*x)/a


def test_issue_6308():
    k, a0 = symbols('k a0', real=True)
    assert integrate((x**2 + 1 - k**2)/(x**2 + 1 + a0**2), x) == \
        x - (a0**2 + k**2)*atan(x/sqrt(a0**2 + 1))/sqrt(a0**2 + 1)


def test_issue_5907():
    a = symbols('a', nonzero=True)
    assert integrate(1/(x**2 + a**2)**2, x) == \
         x/(2*a**4 + 2*a**2*x**2) + atan(x/a)/(2*a**3)


def test_log_to_atan():
    f, g = (Poly(x + S.Half, x, domain='QQ'), Poly(sqrt(3)/2, x, domain='EX'))
    fg_ans = 2*atan(2*sqrt(3)*x/3 + sqrt(3)/3)
    assert log_to_atan(f, g) == fg_ans
    assert log_to_atan(g, f) == -fg_ans


def test_issue_25896():
    # for both tests, C = 0 in log_to_real
    # but this only has a log result
    e = (2*x + 1)/(x**2 + x + 1) + 1/x
    assert ratint(e, x) == log(x**3 + x**2 + x)
    # while this has more
    assert ratint((4*x + 7)/(x**2 + 4*x + 6) + 2/x, x) == (
        2*log(x) + 2*log(x**2 + 4*x + 6) - sqrt(2)*atan(
        sqrt(2)*x/2 + sqrt(2))/2)
