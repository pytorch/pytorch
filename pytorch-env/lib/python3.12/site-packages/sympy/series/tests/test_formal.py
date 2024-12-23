from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, asech)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin)
from sympy.functions.special.bessel import airyai
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.series.formal import fps
from sympy.series.order import O
from sympy.series.formal import (rational_algorithm, FormalPowerSeries,
                                 FormalPowerSeriesProduct, FormalPowerSeriesCompose,
                                 FormalPowerSeriesInverse, simpleDE,
                                 rational_independent, exp_re, hyper_re)
from sympy.testing.pytest import raises, XFAIL, slow

x, y, z = symbols('x y z')
n, m, k = symbols('n m k', integer=True)
f, r = Function('f'), Function('r')


def test_rational_algorithm():
    f = 1 / ((x - 1)**2 * (x - 2))
    assert rational_algorithm(f, x, k) == \
        (-2**(-k - 1) + 1 - (factorial(k + 1) / factorial(k)), 0, 0)

    f = (1 + x + x**2 + x**3) / ((x - 1) * (x - 2))
    assert rational_algorithm(f, x, k) == \
        (-15*2**(-k - 1) + 4, x + 4, 0)

    f = z / (y*m - m*x - y*x + x**2)
    assert rational_algorithm(f, x, k) == \
        (((-y**(-k - 1)*z) / (y - m)) + ((m**(-k - 1)*z) / (y - m)), 0, 0)

    f = x / (1 - x - x**2)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        (((Rational(-1, 2) + sqrt(5)/2)**(-k - 1) *
         (-sqrt(5)/10 + S.Half)) +
         ((-sqrt(5)/2 - S.Half)**(-k - 1) *
         (sqrt(5)/10 + S.Half)), 0, 0)

    f = 1 / (x**2 + 2*x + 2)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        ((I*(-1 + I)**(-k - 1)) / 2 - (I*(-1 - I)**(-k - 1)) / 2, 0, 0)

    f = log(1 + x)
    assert rational_algorithm(f, x, k) == \
        (-(-1)**(-k) / k, 0, 1)

    f = atan(x)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        (((I*I**(-k)) / 2 - (I*(-I)**(-k)) / 2) / k, 0, 1)

    f = x*atan(x) - log(1 + x**2) / 2
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        (((I*I**(-k + 1)) / 2 - (I*(-I)**(-k + 1)) / 2) /
         (k*(k - 1)), 0, 2)

    f = log((1 + x) / (1 - x)) / 2 - atan(x)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        ((-(-1)**(-k) / 2 - (I*I**(-k)) / 2 + (I*(-I)**(-k)) / 2 +
          S.Half) / k, 0, 1)

    assert rational_algorithm(cos(x), x, k) is None


def test_rational_independent():
    ri = rational_independent
    assert ri([], x) == []
    assert ri([cos(x), sin(x)], x) == [cos(x), sin(x)]
    assert ri([x**2, sin(x), x*sin(x), x**3], x) == \
        [x**3 + x**2, x*sin(x) + sin(x)]
    assert ri([S.One, x*log(x), log(x), sin(x)/x, cos(x), sin(x), x], x) == \
        [x + 1, x*log(x) + log(x), sin(x)/x + sin(x), cos(x)]


def test_simpleDE():
    # Tests just the first valid DE
    for DE in simpleDE(exp(x), x, f):
        assert DE == (-f(x) + Derivative(f(x), x), 1)
        break
    for DE in simpleDE(sin(x), x, f):
        assert DE == (f(x) + Derivative(f(x), x, x), 2)
        break
    for DE in simpleDE(log(1 + x), x, f):
        assert DE == ((x + 1)*Derivative(f(x), x, 2) + Derivative(f(x), x), 2)
        break
    for DE in simpleDE(asin(x), x, f):
        assert DE == (x*Derivative(f(x), x) + (x**2 - 1)*Derivative(f(x), x, x),
                      2)
        break
    for DE in simpleDE(exp(x)*sin(x), x, f):
        assert DE == (2*f(x) - 2*Derivative(f(x)) + Derivative(f(x), x, x), 2)
        break
    for DE in simpleDE(((1 + x)/(1 - x))**n, x, f):
        assert DE == (2*n*f(x) + (x**2 - 1)*Derivative(f(x), x), 1)
        break
    for DE in simpleDE(airyai(x), x, f):
        assert DE == (-x*f(x) + Derivative(f(x), x, x), 2)
        break


def test_exp_re():
    d = -f(x) + Derivative(f(x), x)
    assert exp_re(d, r, k) == -r(k) + r(k + 1)

    d = f(x) + Derivative(f(x), x, x)
    assert exp_re(d, r, k) == r(k) + r(k + 2)

    d = f(x) + Derivative(f(x), x) + Derivative(f(x), x, x)
    assert exp_re(d, r, k) == r(k) + r(k + 1) + r(k + 2)

    d = Derivative(f(x), x) + Derivative(f(x), x, x)
    assert exp_re(d, r, k) == r(k) + r(k + 1)

    d = Derivative(f(x), x, 3) + Derivative(f(x), x, 4) + Derivative(f(x))
    assert exp_re(d, r, k) == r(k) + r(k + 2) + r(k + 3)


def test_hyper_re():
    d = f(x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == r(k) + (k+1)*(k+2)*r(k + 2)

    d = -x*f(x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == (k + 2)*(k + 3)*r(k + 3) - r(k)

    d = 2*f(x) - 2*Derivative(f(x), x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == \
        (-2*k - 2)*r(k + 1) + (k + 1)*(k + 2)*r(k + 2) + 2*r(k)

    d = 2*n*f(x) + (x**2 - 1)*Derivative(f(x), x)
    assert hyper_re(d, r, k) == \
        k*r(k) + 2*n*r(k + 1) + (-k - 2)*r(k + 2)

    d = (x**10 + 4)*Derivative(f(x), x) + x*(x**10 - 1)*Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == \
        (k*(k - 1) + k)*r(k) + (4*k - (k + 9)*(k + 10) + 40)*r(k + 10)

    d = ((x**2 - 1)*Derivative(f(x), x, 3) + 3*x*Derivative(f(x), x, x) +
         Derivative(f(x), x))
    assert hyper_re(d, r, k) == \
        ((k*(k - 2)*(k - 1) + 3*k*(k - 1) + k)*r(k) +
         (-k*(k + 1)*(k + 2))*r(k + 2))


def test_fps():
    assert fps(1) == 1
    assert fps(2, x) == 2
    assert fps(2, x, dir='+') == 2
    assert fps(2, x, dir='-') == 2
    assert fps(1/x + 1/x**2) == 1/x + 1/x**2
    assert fps(log(1 + x), hyper=False, rational=False) == log(1 + x)

    f = fps(x**2 + x + 1)
    assert isinstance(f, FormalPowerSeries)
    assert f.function == x**2 + x + 1
    assert f[0] == 1
    assert f[2] == x**2
    assert f.truncate(4) == x**2 + x + 1 + O(x**4)
    assert f.polynomial() == x**2 + x + 1

    f = fps(log(1 + x))
    assert isinstance(f, FormalPowerSeries)
    assert f.function == log(1 + x)
    assert f.subs(x, y) == f
    assert f[:5] == [0, x, -x**2/2, x**3/3, -x**4/4]
    assert f.as_leading_term(x) == x
    assert f.polynomial(6) == x - x**2/2 + x**3/3 - x**4/4 + x**5/5

    k = f.ak.variables[0]
    assert f.infinite == Sum((-(-1)**(-k)*x**k)/k, (k, 1, oo))

    ft, s = f.truncate(n=None), f[:5]
    for i, t in enumerate(ft):
        if i == 5:
            break
        assert s[i] == t

    f = sin(x).fps(x)
    assert isinstance(f, FormalPowerSeries)
    assert f.truncate() == x - x**3/6 + x**5/120 + O(x**6)

    raises(NotImplementedError, lambda: fps(y*x))
    raises(ValueError, lambda: fps(x, dir=0))


@slow
def test_fps__rational():
    assert fps(1/x) == (1/x)
    assert fps((x**2 + x + 1) / x**3, dir=-1) == (x**2 + x + 1) / x**3

    f = 1 / ((x - 1)**2 * (x - 2))
    assert fps(f, x).truncate() == \
        (Rational(-1, 2) - x*Rational(5, 4) - 17*x**2/8 - 49*x**3/16 - 129*x**4/32 -
         321*x**5/64 + O(x**6))

    f = (1 + x + x**2 + x**3) / ((x - 1) * (x - 2))
    assert fps(f, x).truncate() == \
        (S.Half + x*Rational(5, 4) + 17*x**2/8 + 49*x**3/16 + 113*x**4/32 +
         241*x**5/64 + O(x**6))

    f = x / (1 - x - x**2)
    assert fps(f, x, full=True).truncate() == \
        x + x**2 + 2*x**3 + 3*x**4 + 5*x**5 + O(x**6)

    f = 1 / (x**2 + 2*x + 2)
    assert fps(f, x, full=True).truncate() == \
        S.Half - x/2 + x**2/4 - x**4/8 + x**5/8 + O(x**6)

    f = log(1 + x)
    assert fps(f, x).truncate() == \
        x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)
    assert fps(f, x, dir=1).truncate() == fps(f, x, dir=-1).truncate()
    assert fps(f, x, 2).truncate() == \
        (log(3) - Rational(2, 3) - (x - 2)**2/18 + (x - 2)**3/81 -
         (x - 2)**4/324 + (x - 2)**5/1215 + x/3 + O((x - 2)**6, (x, 2)))
    assert fps(f, x, 2, dir=-1).truncate() == \
        (log(3) - Rational(2, 3) - (-x + 2)**2/18 - (-x + 2)**3/81 -
         (-x + 2)**4/324 - (-x + 2)**5/1215 + x/3 + O((x - 2)**6, (x, 2)))

    f = atan(x)
    assert fps(f, x, full=True).truncate() == x - x**3/3 + x**5/5 + O(x**6)
    assert fps(f, x, full=True, dir=1).truncate() == \
        fps(f, x, full=True, dir=-1).truncate()
    assert fps(f, x, 2, full=True).truncate() == \
        (atan(2) - Rational(2, 5) - 2*(x - 2)**2/25 + 11*(x - 2)**3/375 -
         6*(x - 2)**4/625 + 41*(x - 2)**5/15625 + x/5 + O((x - 2)**6, (x, 2)))
    assert fps(f, x, 2, full=True, dir=-1).truncate() == \
        (atan(2) - Rational(2, 5) - 2*(-x + 2)**2/25 - 11*(-x + 2)**3/375 -
         6*(-x + 2)**4/625 - 41*(-x + 2)**5/15625 + x/5 + O((x - 2)**6, (x, 2)))

    f = x*atan(x) - log(1 + x**2) / 2
    assert fps(f, x, full=True).truncate() == x**2/2 - x**4/12 + O(x**6)

    f = log((1 + x) / (1 - x)) / 2 - atan(x)
    assert fps(f, x, full=True).truncate(n=10) == 2*x**3/3 + 2*x**7/7 + O(x**10)


@slow
def test_fps__hyper():
    f = sin(x)
    assert fps(f, x).truncate() == x - x**3/6 + x**5/120 + O(x**6)

    f = cos(x)
    assert fps(f, x).truncate() == 1 - x**2/2 + x**4/24 + O(x**6)

    f = exp(x)
    assert fps(f, x).truncate() == \
        1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6)

    f = atan(x)
    assert fps(f, x).truncate() == x - x**3/3 + x**5/5 + O(x**6)

    f = exp(acos(x))
    assert fps(f, x).truncate() == \
        (exp(pi/2) - x*exp(pi/2) + x**2*exp(pi/2)/2 - x**3*exp(pi/2)/3 +
         5*x**4*exp(pi/2)/24 - x**5*exp(pi/2)/6 + O(x**6))

    f = exp(acosh(x))
    assert fps(f, x).truncate() == I + x - I*x**2/2 - I*x**4/8 + O(x**6)

    f = atan(1/x)
    assert fps(f, x).truncate() == pi/2 - x + x**3/3 - x**5/5 + O(x**6)

    f = x*atan(x) - log(1 + x**2) / 2
    assert fps(f, x, rational=False).truncate() == x**2/2 - x**4/12 + O(x**6)

    f = log(1 + x)
    assert fps(f, x, rational=False).truncate() == \
        x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)

    f = airyai(x**2)
    assert fps(f, x).truncate() == \
        (3**Rational(5, 6)*gamma(Rational(1, 3))/(6*pi) -
         3**Rational(2, 3)*x**2/(3*gamma(Rational(1, 3))) + O(x**6))

    f = exp(x)*sin(x)
    assert fps(f, x).truncate() == x + x**2 + x**3/3 - x**5/30 + O(x**6)

    f = exp(x)*sin(x)/x
    assert fps(f, x).truncate() == 1 + x + x**2/3 - x**4/30 - x**5/90 + O(x**6)

    f = sin(x) * cos(x)
    assert fps(f, x).truncate() == x - 2*x**3/3 + 2*x**5/15 + O(x**6)


def test_fps_shift():
    f = x**-5*sin(x)
    assert fps(f, x).truncate() == \
        1/x**4 - 1/(6*x**2) + Rational(1, 120) - x**2/5040 + x**4/362880 + O(x**6)

    f = x**2*atan(x)
    assert fps(f, x, rational=False).truncate() == \
        x**3 - x**5/3 + O(x**6)

    f = cos(sqrt(x))*x
    assert fps(f, x).truncate() == \
        x - x**2/2 + x**3/24 - x**4/720 + x**5/40320 + O(x**6)

    f = x**2*cos(sqrt(x))
    assert fps(f, x).truncate() == \
        x**2 - x**3/2 + x**4/24 - x**5/720 + O(x**6)


def test_fps__Add_expr():
    f = x*atan(x) - log(1 + x**2) / 2
    assert fps(f, x).truncate() == x**2/2 - x**4/12 + O(x**6)

    f = sin(x) + cos(x) - exp(x) + log(1 + x)
    assert fps(f, x).truncate() == x - 3*x**2/2 - x**4/4 + x**5/5 + O(x**6)

    f = 1/x + sin(x)
    assert fps(f, x).truncate() == 1/x + x - x**3/6 + x**5/120 + O(x**6)

    f = sin(x) - cos(x) + 1/(x - 1)
    assert fps(f, x).truncate() == \
        -2 - x**2/2 - 7*x**3/6 - 25*x**4/24 - 119*x**5/120 + O(x**6)


def test_fps__asymptotic():
    f = exp(x)
    assert fps(f, x, oo) == f
    assert fps(f, x, -oo).truncate() == O(1/x**6, (x, oo))

    f = erf(x)
    assert fps(f, x, oo).truncate() == 1 + O(1/x**6, (x, oo))
    assert fps(f, x, -oo).truncate() == -1 + O(1/x**6, (x, oo))

    f = atan(x)
    assert fps(f, x, oo, full=True).truncate() == \
        -1/(5*x**5) + 1/(3*x**3) - 1/x + pi/2 + O(1/x**6, (x, oo))
    assert fps(f, x, -oo, full=True).truncate() == \
        -1/(5*x**5) + 1/(3*x**3) - 1/x - pi/2 + O(1/x**6, (x, oo))

    f = log(1 + x)
    assert fps(f, x, oo) != \
        (-1/(5*x**5) - 1/(4*x**4) + 1/(3*x**3) - 1/(2*x**2) + 1/x - log(1/x) +
         O(1/x**6, (x, oo)))
    assert fps(f, x, -oo) != \
        (-1/(5*x**5) - 1/(4*x**4) + 1/(3*x**3) - 1/(2*x**2) + 1/x + I*pi -
         log(-1/x) + O(1/x**6, (x, oo)))


def test_fps__fractional():
    f = sin(sqrt(x)) / x
    assert fps(f, x).truncate() == \
        (1/sqrt(x) - sqrt(x)/6 + x**Rational(3, 2)/120 -
         x**Rational(5, 2)/5040 + x**Rational(7, 2)/362880 -
         x**Rational(9, 2)/39916800 + x**Rational(11, 2)/6227020800 + O(x**6))

    f = sin(sqrt(x)) * x
    assert fps(f, x).truncate() == \
        (x**Rational(3, 2) - x**Rational(5, 2)/6 + x**Rational(7, 2)/120 -
         x**Rational(9, 2)/5040 + x**Rational(11, 2)/362880 + O(x**6))

    f = atan(sqrt(x)) / x**2
    assert fps(f, x).truncate() == \
        (x**Rational(-3, 2) - x**Rational(-1, 2)/3 + x**S.Half/5 -
         x**Rational(3, 2)/7 + x**Rational(5, 2)/9 - x**Rational(7, 2)/11 +
         x**Rational(9, 2)/13 - x**Rational(11, 2)/15 + O(x**6))

    f = exp(sqrt(x))
    assert fps(f, x).truncate().expand() == \
        (1 + x/2 + x**2/24 + x**3/720 + x**4/40320 + x**5/3628800 + sqrt(x) +
         x**Rational(3, 2)/6 + x**Rational(5, 2)/120 + x**Rational(7, 2)/5040 +
         x**Rational(9, 2)/362880 + x**Rational(11, 2)/39916800 + O(x**6))

    f = exp(sqrt(x))*x
    assert fps(f, x).truncate().expand() == \
        (x + x**2/2 + x**3/24 + x**4/720 + x**5/40320 + x**Rational(3, 2) +
         x**Rational(5, 2)/6 + x**Rational(7, 2)/120 + x**Rational(9, 2)/5040 +
         x**Rational(11, 2)/362880 + O(x**6))


def test_fps__logarithmic_singularity():
    f = log(1 + 1/x)
    assert fps(f, x) != \
        -log(x) + x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)
    assert fps(f, x, rational=False) != \
        -log(x) + x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)


@XFAIL
def test_fps__logarithmic_singularity_fail():
    f = asech(x)  # Algorithms for computing limits probably needs improvemnts
    assert fps(f, x) == log(2) - log(x) - x**2/4 - 3*x**4/64 + O(x**6)


def test_fps_symbolic():
    f = x**n*sin(x**2)
    assert fps(f, x).truncate(8) == x**(n + 2) - x**(n + 6)/6 + O(x**(n + 8), x)

    f = x**n*log(1 + x)
    fp = fps(f, x)
    k = fp.ak.variables[0]
    assert fp.infinite == \
        Sum((-(-1)**(-k)*x**(k + n))/k, (k, 1, oo))

    f = (x - 2)**n*log(1 + x)
    assert fps(f, x, 2).truncate() == \
        ((x - 2)**n*log(3) + (x - 2)**(n + 1)/3 - (x - 2)**(n + 2)/18 + (x - 2)**(n + 3)/81 -
         (x - 2)**(n + 4)/324 + (x - 2)**(n + 5)/1215 + O((x - 2)**(n + 6), (x, 2)))

    f = x**(n - 2)*cos(x)
    assert fps(f, x).truncate() == \
        (x**(n - 2) - x**n/2 + x**(n + 2)/24 + O(x**(n + 4), x))

    f = x**(n - 2)*sin(x) + x**n*exp(x)
    assert fps(f, x).truncate() == \
        (x**(n - 1) + x**(n + 1) + x**(n + 2)/2 + x**n +
         x**(n + 4)/24 + x**(n + 5)/60 + O(x**(n + 6), x))

    f = x**n*atan(x)
    assert fps(f, x, oo).truncate() == \
        (-x**(n - 5)/5 + x**(n - 3)/3 + x**n*(pi/2 - 1/x) +
         O((1/x)**(-n)/x**6, (x, oo)))

    f = x**(n/2)*cos(x)
    assert fps(f, x).truncate() == \
        x**(n/2) - x**(n/2 + 2)/2 + x**(n/2 + 4)/24 + O(x**(n/2 + 6), x)

    f = x**(n + m)*sin(x)
    assert fps(f, x).truncate() == \
        x**(m + n + 1) - x**(m + n + 3)/6 + x**(m + n + 5)/120 + O(x**(m + n + 6), x)


def test_fps__slow():
    f = x*exp(x)*sin(2*x)  # TODO: rsolve needs improvement
    assert fps(f, x).truncate() == 2*x**2 + 2*x**3 - x**4/3 - x**5 + O(x**6)


def test_fps__operations():
    f1, f2 = fps(sin(x)), fps(cos(x))

    fsum = f1 + f2
    assert fsum.function == sin(x) + cos(x)
    assert fsum.truncate() == \
        1 + x - x**2/2 - x**3/6 + x**4/24 + x**5/120 + O(x**6)

    fsum = f1 + 1
    assert fsum.function == sin(x) + 1
    assert fsum.truncate() == 1 + x - x**3/6 + x**5/120 + O(x**6)

    fsum = 1 + f2
    assert fsum.function == cos(x) + 1
    assert fsum.truncate() == 2 - x**2/2 + x**4/24 + O(x**6)

    assert (f1 + x) == Add(f1, x)

    assert -f2.truncate() == -1 + x**2/2 - x**4/24 + O(x**6)
    assert (f1 - f1) is S.Zero

    fsub = f1 - f2
    assert fsub.function == sin(x) - cos(x)
    assert fsub.truncate() == \
        -1 + x + x**2/2 - x**3/6 - x**4/24 + x**5/120 + O(x**6)

    fsub = f1 - 1
    assert fsub.function == sin(x) - 1
    assert fsub.truncate() == -1 + x - x**3/6 + x**5/120 + O(x**6)

    fsub = 1 - f2
    assert fsub.function == -cos(x) + 1
    assert fsub.truncate() == x**2/2 - x**4/24 + O(x**6)

    raises(ValueError, lambda: f1 + fps(exp(x), dir=-1))
    raises(ValueError, lambda: f1 + fps(exp(x), x0=1))

    fm = f1 * 3

    assert fm.function == 3*sin(x)
    assert fm.truncate() == 3*x - x**3/2 + x**5/40 + O(x**6)

    fm = 3 * f2

    assert fm.function == 3*cos(x)
    assert fm.truncate() == 3 - 3*x**2/2 + x**4/8 + O(x**6)

    assert (f1 * f2) == Mul(f1, f2)
    assert (f1 * x) == Mul(f1, x)

    fd = f1.diff()
    assert fd.function == cos(x)
    assert fd.truncate() == 1 - x**2/2 + x**4/24 + O(x**6)

    fd = f2.diff()
    assert fd.function == -sin(x)
    assert fd.truncate() == -x + x**3/6 - x**5/120 + O(x**6)

    fd = f2.diff().diff()
    assert fd.function == -cos(x)
    assert fd.truncate() == -1 + x**2/2 - x**4/24 + O(x**6)

    f3 = fps(exp(sqrt(x)))
    fd = f3.diff()
    assert fd.truncate().expand() == \
        (1/(2*sqrt(x)) + S.Half + x/12 + x**2/240 + x**3/10080 + x**4/725760 +
         x**5/79833600 + sqrt(x)/4 + x**Rational(3, 2)/48 + x**Rational(5, 2)/1440 +
         x**Rational(7, 2)/80640 + x**Rational(9, 2)/7257600 + x**Rational(11, 2)/958003200 +
         O(x**6))

    assert f1.integrate((x, 0, 1)) == -cos(1) + 1
    assert integrate(f1, (x, 0, 1)) == -cos(1) + 1

    fi = integrate(f1, x)
    assert fi.function == -cos(x)
    assert fi.truncate() == -1 + x**2/2 - x**4/24 + O(x**6)

    fi = f2.integrate(x)
    assert fi.function == sin(x)
    assert fi.truncate() == x - x**3/6 + x**5/120 + O(x**6)

def test_fps__product():
    f1, f2, f3 = fps(sin(x)), fps(exp(x)), fps(cos(x))

    raises(ValueError, lambda: f1.product(exp(x), x))
    raises(ValueError, lambda: f1.product(fps(exp(x), dir=-1), x, 4))
    raises(ValueError, lambda: f1.product(fps(exp(x), x0=1), x, 4))
    raises(ValueError, lambda: f1.product(fps(exp(y)), x, 4))

    fprod = f1.product(f2, x)
    assert isinstance(fprod, FormalPowerSeriesProduct)
    assert isinstance(fprod.ffps, FormalPowerSeries)
    assert isinstance(fprod.gfps, FormalPowerSeries)
    assert fprod.f == sin(x)
    assert fprod.g == exp(x)
    assert fprod.function == sin(x) * exp(x)
    assert fprod._eval_terms(4) == x + x**2 + x**3/3
    assert fprod.truncate(4) == x + x**2 + x**3/3 + O(x**4)
    assert fprod.polynomial(4) == x + x**2 + x**3/3

    raises(NotImplementedError, lambda: fprod._eval_term(5))
    raises(NotImplementedError, lambda: fprod.infinite)
    raises(NotImplementedError, lambda: fprod._eval_derivative(x))
    raises(NotImplementedError, lambda: fprod.integrate(x))

    assert f1.product(f3, x)._eval_terms(4) == x - 2*x**3/3
    assert f1.product(f3, x).truncate(4) == x - 2*x**3/3 + O(x**4)


def test_fps__compose():
    f1, f2, f3 = fps(exp(x)), fps(sin(x)), fps(cos(x))

    raises(ValueError, lambda: f1.compose(sin(x), x))
    raises(ValueError, lambda: f1.compose(fps(sin(x), dir=-1), x, 4))
    raises(ValueError, lambda: f1.compose(fps(sin(x), x0=1), x, 4))
    raises(ValueError, lambda: f1.compose(fps(sin(y)), x, 4))

    raises(ValueError, lambda: f1.compose(f3, x))
    raises(ValueError, lambda: f2.compose(f3, x))

    fcomp = f1.compose(f2, x)
    assert isinstance(fcomp, FormalPowerSeriesCompose)
    assert isinstance(fcomp.ffps, FormalPowerSeries)
    assert isinstance(fcomp.gfps, FormalPowerSeries)
    assert fcomp.f == exp(x)
    assert fcomp.g == sin(x)
    assert fcomp.function == exp(sin(x))
    assert fcomp._eval_terms(6) == 1 + x + x**2/2 - x**4/8 - x**5/15
    assert fcomp.truncate() == 1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)
    assert fcomp.truncate(5) == 1 + x + x**2/2 - x**4/8 + O(x**5)

    raises(NotImplementedError, lambda: fcomp._eval_term(5))
    raises(NotImplementedError, lambda: fcomp.infinite)
    raises(NotImplementedError, lambda: fcomp._eval_derivative(x))
    raises(NotImplementedError, lambda: fcomp.integrate(x))

    assert f1.compose(f2, x).truncate(4) == 1 + x + x**2/2 + O(x**4)
    assert f1.compose(f2, x).truncate(8) == \
        1 + x + x**2/2 - x**4/8 - x**5/15 - x**6/240 + x**7/90 + O(x**8)
    assert f1.compose(f2, x).truncate(6) == \
        1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)

    assert f2.compose(f2, x).truncate(4) == x - x**3/3 + O(x**4)
    assert f2.compose(f2, x).truncate(8) == x - x**3/3 + x**5/10 - 8*x**7/315 + O(x**8)
    assert f2.compose(f2, x).truncate(6) == x - x**3/3 + x**5/10 + O(x**6)


def test_fps__inverse():
    f1, f2, f3 = fps(sin(x)), fps(exp(x)), fps(cos(x))

    raises(ValueError, lambda: f1.inverse(x))

    finv = f2.inverse(x)
    assert isinstance(finv, FormalPowerSeriesInverse)
    assert isinstance(finv.ffps, FormalPowerSeries)
    raises(ValueError, lambda: finv.gfps)

    assert finv.f == exp(x)
    assert finv.function == exp(-x)
    assert finv._eval_terms(5) == 1 - x + x**2/2 - x**3/6 + x**4/24
    assert finv.truncate() == 1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + O(x**6)
    assert finv.truncate(5) == 1 - x + x**2/2 - x**3/6 + x**4/24 + O(x**5)

    raises(NotImplementedError, lambda: finv._eval_term(5))
    raises(ValueError, lambda: finv.g)
    raises(NotImplementedError, lambda: finv.infinite)
    raises(NotImplementedError, lambda: finv._eval_derivative(x))
    raises(NotImplementedError, lambda: finv.integrate(x))

    assert f2.inverse(x).truncate(8) == \
        1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + x**6/720 - x**7/5040 + O(x**8)

    assert f3.inverse(x).truncate() == 1 + x**2/2 + 5*x**4/24 + O(x**6)
    assert f3.inverse(x).truncate(8) == 1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + O(x**8)
