from sympy.calculus.util import AccumBounds
from sympy.core.function import (Derivative, PoleError)
from sympy.core.numbers import (E, I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, cot, sin, tan)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.abc import x, y, z

from sympy.testing.pytest import raises, XFAIL


def test_simple_1():
    assert x.nseries(x, n=5) == x
    assert y.nseries(x, n=5) == y
    assert (1/(x*y)).nseries(y, n=5) == 1/(x*y)
    assert Rational(3, 4).nseries(x, n=5) == Rational(3, 4)
    assert x.nseries() == x


def test_mul_0():
    assert (x*log(x)).nseries(x, n=5) == x*log(x)


def test_mul_1():
    assert (x*log(2 + x)).nseries(x, n=5) == x*log(2) + x**2/2 - x**3/8 + \
        x**4/24 + O(x**5)
    assert (x*log(1 + x)).nseries(
        x, n=5) == x**2 - x**3/2 + x**4/3 + O(x**5)


def test_pow_0():
    assert (x**2).nseries(x, n=5) == x**2
    assert (1/x).nseries(x, n=5) == 1/x
    assert (1/x**2).nseries(x, n=5) == 1/x**2
    assert (x**Rational(2, 3)).nseries(x, n=5) == (x**Rational(2, 3))
    assert (sqrt(x)**3).nseries(x, n=5) == (sqrt(x)**3)


def test_pow_1():
    assert ((1 + x)**2).nseries(x, n=5) == x**2 + 2*x + 1

    # https://github.com/sympy/sympy/issues/21075
    assert ((sqrt(x) + 1)**2).nseries(x) == 2*sqrt(x) + x + 1
    assert ((sqrt(x) + cbrt(x))**2).nseries(x) == 2*x**Rational(5, 6)\
        + x**Rational(2, 3) + x


def test_geometric_1():
    assert (1/(1 - x)).nseries(x, n=5) == 1 + x + x**2 + x**3 + x**4 + O(x**5)
    assert (x/(1 - x)).nseries(x, n=6) == x + x**2 + x**3 + x**4 + x**5 + O(x**6)
    assert (x**3/(1 - x)).nseries(x, n=8) == x**3 + x**4 + x**5 + x**6 + \
        x**7 + O(x**8)


def test_sqrt_1():
    assert sqrt(1 + x).nseries(x, n=5) == 1 + x/2 - x**2/8 + x**3/16 - 5*x**4/128 + O(x**5)


def test_exp_1():
    assert exp(x).nseries(x, n=5) == 1 + x + x**2/2 + x**3/6 + x**4/24 + O(x**5)
    assert exp(x).nseries(x, n=12) == 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 +  \
        x**6/720 + x**7/5040 + x**8/40320 + x**9/362880 + x**10/3628800 +  \
        x**11/39916800 + O(x**12)
    assert exp(1/x).nseries(x, n=5) == exp(1/x)
    assert exp(1/(1 + x)).nseries(x, n=4) ==  \
        (E*(1 - x - 13*x**3/6 + 3*x**2/2)).expand() + O(x**4)
    assert exp(2 + x).nseries(x, n=5) ==  \
        (exp(2)*(1 + x + x**2/2 + x**3/6 + x**4/24)).expand() + O(x**5)


def test_exp_sqrt_1():
    assert exp(1 + sqrt(x)).nseries(x, n=3) ==  \
        (exp(1)*(1 + sqrt(x) + x/2 + sqrt(x)*x/6)).expand() + O(sqrt(x)**3)


def test_power_x_x1():
    assert (exp(x*log(x))).nseries(x, n=4) == \
        1 + x*log(x) + x**2*log(x)**2/2 + x**3*log(x)**3/6 + O(x**4*log(x)**4)


def test_power_x_x2():
    assert (x**x).nseries(x, n=4) == \
        1 + x*log(x) + x**2*log(x)**2/2 + x**3*log(x)**3/6 + O(x**4*log(x)**4)


def test_log_singular1():
    assert log(1 + 1/x).nseries(x, n=5) == x - log(x) - x**2/2 + x**3/3 - \
        x**4/4 + O(x**5)


def test_log_power1():
    e = 1 / (1/x + x ** (log(3)/log(2)))
    assert e.nseries(x, n=5) == -x**(log(3)/log(2) + 2) + x + O(x**5)


def test_log_series():
    l = Symbol('l')
    e = 1/(1 - log(x))
    assert e.nseries(x, n=5, logx=l) == 1/(1 - l)


def test_log2():
    e = log(-1/x)
    assert e.nseries(x, n=5) == -log(x) + log(-1)


def test_log3():
    l = Symbol('l')
    e = 1/log(-1/x)
    assert e.nseries(x, n=4, logx=l) == 1/(-l + log(-1))


def test_series1():
    e = sin(x)
    assert e.nseries(x, 0, 0) != 0
    assert e.nseries(x, 0, 0) == O(1, x)
    assert e.nseries(x, 0, 1) == O(x, x)
    assert e.nseries(x, 0, 2) == x + O(x**2, x)
    assert e.nseries(x, 0, 3) == x + O(x**3, x)
    assert e.nseries(x, 0, 4) == x - x**3/6 + O(x**4, x)

    e = (exp(x) - 1)/x
    assert e.nseries(x, 0, 3) == 1 + x/2 + x**2/6 + O(x**3)

    assert x.nseries(x, 0, 2) == x


@XFAIL
def test_series1_failing():
    assert x.nseries(x, 0, 0) == O(1, x)
    assert x.nseries(x, 0, 1) == O(x, x)


def test_seriesbug1():
    assert (1/x).nseries(x, 0, 3) == 1/x
    assert (x + 1/x).nseries(x, 0, 3) == x + 1/x


def test_series2x():
    assert ((x + 1)**(-2)).nseries(x, 0, 4) == 1 - 2*x + 3*x**2 - 4*x**3 + O(x**4, x)
    assert ((x + 1)**(-1)).nseries(x, 0, 4) == 1 - x + x**2 - x**3 + O(x**4, x)
    assert ((x + 1)**0).nseries(x, 0, 3) == 1
    assert ((x + 1)**1).nseries(x, 0, 3) == 1 + x
    assert ((x + 1)**2).nseries(x, 0, 3) == x**2 + 2*x + 1
    assert ((x + 1)**3).nseries(x, 0, 3) == 1 + 3*x + 3*x**2 + O(x**3)

    assert (1/(1 + x)).nseries(x, 0, 4) == 1 - x + x**2 - x**3 + O(x**4, x)
    assert (x + 3/(1 + 2*x)).nseries(x, 0, 4) == 3 - 5*x + 12*x**2 - 24*x**3 + O(x**4, x)

    assert ((1/x + 1)**3).nseries(x, 0, 3) == 1 + 3/x + 3/x**2 + x**(-3)
    assert (1/(1 + 1/x)).nseries(x, 0, 4) == x - x**2 + x**3 - O(x**4, x)
    assert (1/(1 + 1/x**2)).nseries(x, 0, 6) == x**2 - x**4 + O(x**6, x)


def test_bug2():  # 1/log(0)*log(0) problem
    w = Symbol("w")
    e = (w**(-1) + w**(
        -log(3)*log(2)**(-1)))**(-1)*(3*w**(-log(3)*log(2)**(-1)) + 2*w**(-1))
    e = e.expand()
    assert e.nseries(w, 0, 4).subs(w, 0) == 3


def test_exp():
    e = (1 + x)**(1/x)
    assert e.nseries(x, n=3) == exp(1) - x*exp(1)/2 + 11*exp(1)*x**2/24 + O(x**3)


def test_exp2():
    w = Symbol("w")
    e = w**(1 - log(x)/(log(2) + log(x)))
    logw = Symbol("logw")
    assert e.nseries(
        w, 0, 1, logx=logw) == exp(logw*log(2)/(log(x) + log(2)))


def test_bug3():
    e = (2/x + 3/x**2)/(1/x + 1/x**2)
    assert e.nseries(x, n=3) == 3 - x + x**2 + O(x**3)


def test_generalexponent():
    p = 2
    e = (2/x + 3/x**p)/(1/x + 1/x**p)
    assert e.nseries(x, 0, 3) == 3 - x + x**2 + O(x**3)
    p = S.Half
    e = (2/x + 3/x**p)/(1/x + 1/x**p)
    assert e.nseries(x, 0, 2) == 2 - x + sqrt(x) + x**(S(3)/2) + O(x**2)

    e = 1 + sqrt(x)
    assert e.nseries(x, 0, 4) == 1 + sqrt(x)

# more complicated example


def test_genexp_x():
    e = 1/(1 + sqrt(x))
    assert e.nseries(x, 0, 2) == \
        1 + x - sqrt(x) - sqrt(x)**3 + O(x**2, x)

# more complicated example


def test_genexp_x2():
    p = Rational(3, 2)
    e = (2/x + 3/x**p)/(1/x + 1/x**p)
    assert e.nseries(x, 0, 3) == 3 + x + x**2 - sqrt(x) - x**(S(3)/2) - x**(S(5)/2) + O(x**3)


def test_seriesbug2():
    w = Symbol("w")
    #simple case (1):
    e = ((2*w)/w)**(1 + w)
    assert e.nseries(w, 0, 1) == 2 + O(w, w)
    assert e.nseries(w, 0, 1).subs(w, 0) == 2


def test_seriesbug2b():
    w = Symbol("w")
    #test sin
    e = sin(2*w)/w
    assert e.nseries(w, 0, 3) == 2 - 4*w**2/3 + O(w**3)


def test_seriesbug2d():
    w = Symbol("w", real=True)
    e = log(sin(2*w)/w)
    assert e.series(w, n=5) == log(2) - 2*w**2/3 - 4*w**4/45 + O(w**5)


def test_seriesbug2c():
    w = Symbol("w", real=True)
    #more complicated case, but sin(x)~x, so the result is the same as in (1)
    e = (sin(2*w)/w)**(1 + w)
    assert e.series(w, 0, 1) == 2 + O(w)
    assert e.series(w, 0, 3) == 2 + 2*w*log(2) + \
        w**2*(Rational(-4, 3) + log(2)**2) + O(w**3)
    assert e.series(w, 0, 2).subs(w, 0) == 2


def test_expbug4():
    x = Symbol("x", real=True)
    assert (log(
        sin(2*x)/x)*(1 + x)).series(x, 0, 2) == log(2) + x*log(2) + O(x**2, x)
    assert exp(
        log(sin(2*x)/x)*(1 + x)).series(x, 0, 2) == 2 + 2*x*log(2) + O(x**2)

    assert exp(log(2) + O(x)).nseries(x, 0, 2) == 2 + O(x)
    assert ((2 + O(x))**(1 + x)).nseries(x, 0, 2) == 2 + O(x)


def test_logbug4():
    assert log(2 + O(x)).nseries(x, 0, 2) == log(2) + O(x, x)


def test_expbug5():
    assert exp(log(1 + x)/x).nseries(x, n=3) == exp(1) + -exp(1)*x/2 + 11*exp(1)*x**2/24 + O(x**3)

    assert exp(O(x)).nseries(x, 0, 2) == 1 + O(x)


def test_sinsinbug():
    assert sin(sin(x)).nseries(x, 0, 8) == x - x**3/3 + x**5/10 - 8*x**7/315 + O(x**8)


def test_issue_3258():
    a = x/(exp(x) - 1)
    assert a.nseries(x, 0, 5) == 1 - x/2 - x**4/720 + x**2/12 + O(x**5)


def test_issue_3204():
    x = Symbol("x", nonnegative=True)
    f = sin(x**3)**Rational(1, 3)
    assert f.nseries(x, 0, 17) == x - x**7/18 - x**13/3240 + O(x**17)


def test_issue_3224():
    f = sqrt(1 - sqrt(y))
    assert f.nseries(y, 0, 2) == 1 - sqrt(y)/2 - y/8 - sqrt(y)**3/16 + O(y**2)


def test_issue_3463():
    w, i = symbols('w,i')
    r = log(5)/log(3)
    p = w**(-1 + r)
    e = 1/x*(-log(w**(1 + r)) + log(w + w**r))
    e_ser = -r*log(w)/x + p/x - p**2/(2*x) + O(w)
    assert e.nseries(w, n=1) == e_ser


def test_sin():
    assert sin(8*x).nseries(x, n=4) == 8*x - 256*x**3/3 + O(x**4)
    assert sin(x + y).nseries(x, n=1) == sin(y) + O(x)
    assert sin(x + y).nseries(x, n=2) == sin(y) + cos(y)*x + O(x**2)
    assert sin(x + y).nseries(x, n=5) == sin(y) + cos(y)*x - sin(y)*x**2/2 - \
        cos(y)*x**3/6 + sin(y)*x**4/24 + O(x**5)


def test_issue_3515():
    e = sin(8*x)/x
    assert e.nseries(x, n=6) == 8 - 256*x**2/3 + 4096*x**4/15 + O(x**6)


def test_issue_3505():
    e = sin(x)**(-4)*(sqrt(cos(x))*sin(x)**2 -
        cos(x)**Rational(1, 3)*sin(x)**2)
    assert e.nseries(x, n=9) == Rational(-1, 12) - 7*x**2/288 - \
        43*x**4/10368 - 1123*x**6/2488320 + 377*x**8/29859840 + O(x**9)


def test_issue_3501():
    a = Symbol("a")
    e = x**(-2)*(x*sin(a + x) - x*sin(a))
    assert e.nseries(x, n=6) == cos(a) - sin(a)*x/2 - cos(a)*x**2/6 + \
        x**3*sin(a)/24 + x**4*cos(a)/120 - x**5*sin(a)/720 + O(x**6)
    e = x**(-2)*(x*cos(a + x) - x*cos(a))
    assert e.nseries(x, n=6) == -sin(a) - cos(a)*x/2 + sin(a)*x**2/6 + \
        cos(a)*x**3/24 - x**4*sin(a)/120 - x**5*cos(a)/720 + O(x**6)


def test_issue_3502():
    e = sin(5*x)/sin(2*x)
    assert e.nseries(x, n=2) == Rational(5, 2) + O(x**2)
    assert e.nseries(x, n=6) == \
        Rational(5, 2) - 35*x**2/4 + 329*x**4/48 + O(x**6)


def test_issue_3503():
    e = sin(2 + x)/(2 + x)
    assert e.nseries(x, n=2) == sin(2)/2 + x*cos(2)/2 - x*sin(2)/4 + O(x**2)


def test_issue_3506():
    e = (x + sin(3*x))**(-2)*(x*(x + sin(3*x)) - (x + sin(3*x))*sin(2*x))
    assert e.nseries(x, n=7) == \
        Rational(-1, 4) + 5*x**2/96 + 91*x**4/768 + 11117*x**6/129024 + O(x**7)


def test_issue_3508():
    x = Symbol("x", real=True)
    assert log(sin(x)).series(x, n=5) == log(x) - x**2/6 - x**4/180 + O(x**5)
    e = -log(x) + x*(-log(x) + log(sin(2*x))) + log(sin(2*x))
    assert e.series(x, n=5) == \
        log(2) + log(2)*x - 2*x**2/3 - 2*x**3/3 - 4*x**4/45 + O(x**5)


def test_issue_3507():
    e = x**(-4)*(x**2 - x**2*sqrt(cos(x)))
    assert e.nseries(x, n=9) == \
        Rational(1, 4) + x**2/96 + 19*x**4/5760 + 559*x**6/645120 + 29161*x**8/116121600 + O(x**9)


def test_issue_3639():
    assert sin(cos(x)).nseries(x, n=5) == \
        sin(1) - x**2*cos(1)/2 - x**4*sin(1)/8 + x**4*cos(1)/24 + O(x**5)


def test_hyperbolic():
    assert sinh(x).nseries(x, n=6) == x + x**3/6 + x**5/120 + O(x**6)
    assert cosh(x).nseries(x, n=5) == 1 + x**2/2 + x**4/24 + O(x**5)
    assert tanh(x).nseries(x, n=6) == x - x**3/3 + 2*x**5/15 + O(x**6)
    assert coth(x).nseries(x, n=6) == \
        1/x - x**3/45 + x/3 + 2*x**5/945 + O(x**6)
    assert asinh(x).nseries(x, n=6) == x - x**3/6 + 3*x**5/40 + O(x**6)
    assert acosh(x).nseries(x, n=6) == \
        pi*I/2 - I*x - 3*I*x**5/40 - I*x**3/6 + O(x**6)
    assert atanh(x).nseries(x, n=6) == x + x**3/3 + x**5/5 + O(x**6)
    assert acoth(x).nseries(x, n=6) == -I*pi/2 + x + x**3/3 + x**5/5 + O(x**6)


def test_series2():
    w = Symbol("w", real=True)
    x = Symbol("x", real=True)
    e = w**(-2)*(w*exp(1/x - w) - w*exp(1/x))
    assert e.nseries(w, n=4) == -exp(1/x) + w*exp(1/x)/2 - w**2*exp(1/x)/6 + w**3*exp(1/x)/24 + O(w**4)


def test_series3():
    w = Symbol("w", real=True)
    e = w**(-6)*(w**3*tan(w) - w**3*sin(w))
    assert e.nseries(w, n=8) == Integer(1)/2 + w**2/8 + 13*w**4/240 + 529*w**6/24192 + O(w**8)


def test_bug4():
    w = Symbol("w")
    e = x/(w**4 + x**2*w**4 + 2*x*w**4)*w**4
    assert e.nseries(w, n=2).removeO().expand() in [x/(1 + 2*x + x**2),
        1/(1 + x/2 + 1/x/2)/2, 1/x/(1 + 2/x + x**(-2))]


def test_bug5():
    w = Symbol("w")
    l = Symbol('l')
    e = (-log(w) + log(1 + w*log(x)))**(-2)*w**(-2)*((-log(w) +
        log(1 + x*w))*(-log(w) + log(1 + w*log(x)))*w - x*(-log(w) +
        log(1 + w*log(x)))*w)
    assert e.nseries(w, n=0, logx=l) == x/w/l + 1/w + O(1, w)
    assert e.nseries(w, n=1, logx=l) == x/w/l + 1/w - x/l + 1/l*log(x) \
        + x*log(x)/l**2 + O(w)


def test_issue_4115():
    assert (sin(x)/(1 - cos(x))).nseries(x, n=1) == 2/x + O(x)
    assert (sin(x)**2/(1 - cos(x))).nseries(x, n=1) == 2 + O(x)


def test_pole():
    raises(PoleError, lambda: sin(1/x).series(x, 0, 5))
    raises(PoleError, lambda: sin(1 + 1/x).series(x, 0, 5))
    raises(PoleError, lambda: (x*sin(1/x)).series(x, 0, 5))


def test_expsinbug():
    assert exp(sin(x)).series(x, 0, 0) == O(1, x)
    assert exp(sin(x)).series(x, 0, 1) == 1 + O(x)
    assert exp(sin(x)).series(x, 0, 2) == 1 + x + O(x**2)
    assert exp(sin(x)).series(x, 0, 3) == 1 + x + x**2/2 + O(x**3)
    assert exp(sin(x)).series(x, 0, 4) == 1 + x + x**2/2 + O(x**4)
    assert exp(sin(x)).series(x, 0, 5) == 1 + x + x**2/2 - x**4/8 + O(x**5)


def test_floor():
    x = Symbol('x')
    assert floor(x).series(x) == 0
    assert floor(-x).series(x) == -1
    assert floor(sin(x)).series(x) == 0
    assert floor(sin(-x)).series(x) == -1
    assert floor(x**3).series(x) == 0
    assert floor(-x**3).series(x) == -1
    assert floor(cos(x)).series(x) == 0
    assert floor(cos(-x)).series(x) == 0
    assert floor(5 + sin(x)).series(x) == 5
    assert floor(5 + sin(-x)).series(x) == 4

    assert floor(x).series(x, 2) == 2
    assert floor(-x).series(x, 2) == -3

    x = Symbol('x', negative=True)
    assert floor(x + 1.5).series(x) == 1


def test_frac():
    assert frac(x).series(x, cdir=1) == x
    assert frac(x).series(x, cdir=-1) == 1 + x
    assert frac(2*x + 1).series(x, cdir=1) == 2*x
    assert frac(2*x + 1).series(x, cdir=-1) == 1 + 2*x
    assert frac(x**2).series(x, cdir=1) == x**2
    assert frac(x**2).series(x, cdir=-1) == x**2
    assert frac(sin(x) + 5).series(x, cdir=1) == x - x**3/6 + x**5/120 + O(x**6)
    assert frac(sin(x) + 5).series(x, cdir=-1) == 1 + x - x**3/6 + x**5/120 + O(x**6)
    assert frac(sin(x) + S.Half).series(x) == S.Half + x - x**3/6 + x**5/120 + O(x**6)
    assert frac(x**8).series(x, cdir=1) == O(x**6)
    assert frac(1/x).series(x) == AccumBounds(0, 1) + O(x**6)


def test_ceiling():
    assert ceiling(x).series(x) == 1
    assert ceiling(-x).series(x) == 0
    assert ceiling(sin(x)).series(x) == 1
    assert ceiling(sin(-x)).series(x) == 0
    assert ceiling(1 - cos(x)).series(x) == 1
    assert ceiling(1 - cos(-x)).series(x) == 1
    assert ceiling(x).series(x, 2) == 3
    assert ceiling(-x).series(x, 2) == -2


def test_abs():
    a = Symbol('a')
    assert abs(x).nseries(x, n=4) == x
    assert abs(-x).nseries(x, n=4) == x
    assert abs(x + 1).nseries(x, n=4) == x + 1
    assert abs(sin(x)).nseries(x, n=4) == x - Rational(1, 6)*x**3 + O(x**4)
    assert abs(sin(-x)).nseries(x, n=4) == x - Rational(1, 6)*x**3 + O(x**4)
    assert abs(x - a).nseries(x, 1) == -a*sign(1 - a) + (x - 1)*sign(1 - a) + sign(1 - a)


def test_dir():
    assert abs(x).series(x, 0, dir="+") == x
    assert abs(x).series(x, 0, dir="-") == -x
    assert floor(x + 2).series(x, 0, dir='+') == 2
    assert floor(x + 2).series(x, 0, dir='-') == 1
    assert floor(x + 2.2).series(x, 0, dir='-') == 2
    assert ceiling(x + 2.2).series(x, 0, dir='-') == 3
    assert sin(x + y).series(x, 0, dir='-') == sin(x + y).series(x, 0, dir='+')


def test_cdir():
    assert abs(x).series(x, 0, cdir=1) == x
    assert abs(x).series(x, 0, cdir=-1) == -x
    assert floor(x + 2).series(x, 0, cdir=1) == 2
    assert floor(x + 2).series(x, 0, cdir=-1) == 1
    assert floor(x + 2.2).series(x, 0, cdir=1) == 2
    assert ceiling(x + 2.2).series(x, 0, cdir=-1) == 3
    assert sin(x + y).series(x, 0, cdir=-1) == sin(x + y).series(x, 0, cdir=1)


def test_issue_3504():
    a = Symbol("a")
    e = asin(a*x)/x
    assert e.series(x, 4, n=2).removeO() == \
        (x - 4)*(a/(4*sqrt(-16*a**2 + 1)) - asin(4*a)/16) + asin(4*a)/4


def test_issue_4441():
    a, b = symbols('a,b')
    f = 1/(1 + a*x)
    assert f.series(x, 0, 5) == 1 - a*x + a**2*x**2 - a**3*x**3 + \
        a**4*x**4 + O(x**5)
    f = 1/(1 + (a + b)*x)
    assert f.series(x, 0, 3) == 1 + x*(-a - b)\
        + x**2*(a + b)**2 + O(x**3)


def test_issue_4329():
    assert tan(x).series(x, pi/2, n=3).removeO() == \
        -pi/6 + x/3 - 1/(x - pi/2)
    assert cot(x).series(x, pi, n=3).removeO() == \
        -x/3 + pi/3 + 1/(x - pi)
    assert limit(tan(x)**tan(2*x), x, pi/4) == exp(-1)


def test_issue_5183():
    assert abs(x + x**2).series(n=1) == O(x)
    assert abs(x + x**2).series(n=2) == x + O(x**2)
    assert ((1 + x)**2).series(x, n=6) == x**2 + 2*x + 1
    assert (1 + 1/x).series() == 1 + 1/x
    assert Derivative(exp(x).series(), x).doit() == \
        1 + x + x**2/2 + x**3/6 + x**4/24 + O(x**5)


def test_issue_5654():
    a = Symbol('a')
    assert (1/(x**2+a**2)**2).nseries(x, x0=I*a, n=0) == \
        -I/(4*a**3*(-I*a + x)) - 1/(4*a**2*(-I*a + x)**2) + O(1, (x, I*a))
    assert (1/(x**2+a**2)**2).nseries(x, x0=I*a, n=1) == 3/(16*a**4) \
        -I/(4*a**3*(-I*a + x)) - 1/(4*a**2*(-I*a + x)**2) + O(-I*a + x, (x, I*a))


def test_issue_5925():
    sx = sqrt(x + z).series(z, 0, 1)
    sxy = sqrt(x + y + z).series(z, 0, 1)
    s1, s2 = sx.subs(x, x + y), sxy
    assert (s1 - s2).expand().removeO().simplify() == 0

    sx = sqrt(x + z).series(z, 0, 1)
    sxy = sqrt(x + y + z).series(z, 0, 1)
    assert sxy.subs({x:1, y:2}) == sx.subs(x, 3)


def test_exp_2():
    assert exp(x**3).nseries(x, 0, 14) == 1 + x**3 + x**6/2 + x**9/6 + x**12/24 + O(x**14)
