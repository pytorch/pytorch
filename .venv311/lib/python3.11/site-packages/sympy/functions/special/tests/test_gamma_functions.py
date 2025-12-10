from sympy.core.function import expand_func, Subs
from sympy.core import EulerGamma
from sympy.core.numbers import (I, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, atan)
from sympy.functions.special.error_functions import (Ei, erf, erfc)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma, lowergamma, multigamma, polygamma, trigamma, uppergamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.series.order import O

from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
from sympy.core.random import (test_derivative_numerically as td,
                                      random_complex_number as randcplx,
                                      verify_numerically as tn)

x = Symbol('x')
y = Symbol('y')
n = Symbol('n', integer=True)
w = Symbol('w', real=True)

def test_gamma():
    assert gamma(nan) is nan
    assert gamma(oo) is oo

    assert gamma(-100) is zoo
    assert gamma(0) is zoo
    assert gamma(-100.0) is zoo

    assert gamma(1) == 1
    assert gamma(2) == 1
    assert gamma(3) == 2

    assert gamma(102) == factorial(101)

    assert gamma(S.Half) == sqrt(pi)

    assert gamma(Rational(3, 2)) == sqrt(pi)*S.Half
    assert gamma(Rational(5, 2)) == sqrt(pi)*Rational(3, 4)
    assert gamma(Rational(7, 2)) == sqrt(pi)*Rational(15, 8)

    assert gamma(Rational(-1, 2)) == -2*sqrt(pi)
    assert gamma(Rational(-3, 2)) == sqrt(pi)*Rational(4, 3)
    assert gamma(Rational(-5, 2)) == sqrt(pi)*Rational(-8, 15)

    assert gamma(Rational(-15, 2)) == sqrt(pi)*Rational(256, 2027025)

    assert gamma(Rational(
        -11, 8)).expand(func=True) == Rational(64, 33)*gamma(Rational(5, 8))
    assert gamma(Rational(
        -10, 3)).expand(func=True) == Rational(81, 280)*gamma(Rational(2, 3))
    assert gamma(Rational(
        14, 3)).expand(func=True) == Rational(880, 81)*gamma(Rational(2, 3))
    assert gamma(Rational(
        17, 7)).expand(func=True) == Rational(30, 49)*gamma(Rational(3, 7))
    assert gamma(Rational(
        19, 8)).expand(func=True) == Rational(33, 64)*gamma(Rational(3, 8))

    assert gamma(x).diff(x) == gamma(x)*polygamma(0, x)

    assert gamma(x - 1).expand(func=True) == gamma(x)/(x - 1)
    assert gamma(x + 2).expand(func=True, mul=False) == x*(x + 1)*gamma(x)

    assert conjugate(gamma(x)) == gamma(conjugate(x))

    assert expand_func(gamma(x + Rational(3, 2))) == \
        (x + S.Half)*gamma(x + S.Half)

    assert expand_func(gamma(x - S.Half)) == \
        gamma(S.Half + x)/(x - S.Half)

    # Test a bug:
    assert expand_func(gamma(x + Rational(3, 4))) == gamma(x + Rational(3, 4))

    # XXX: Not sure about these tests. I can fix them by defining e.g.
    # exp_polar.is_integer but I'm not sure if that makes sense.
    assert gamma(3*exp_polar(I*pi)/4).is_nonnegative is False
    assert gamma(3*exp_polar(I*pi)/4).is_extended_nonpositive is True

    y = Symbol('y', nonpositive=True, integer=True)
    assert gamma(y).is_real == False
    y = Symbol('y', positive=True, noninteger=True)
    assert gamma(y).is_real == True

    assert gamma(-1.0, evaluate=False).is_real == False
    assert gamma(0, evaluate=False).is_real == False
    assert gamma(-2, evaluate=False).is_real == False


def test_gamma_rewrite():
    assert gamma(n).rewrite(factorial) == factorial(n - 1)


def test_gamma_series():
    assert gamma(x + 1).series(x, 0, 3) == \
        1 - EulerGamma*x + x**2*(EulerGamma**2/2 + pi**2/12) + O(x**3)
    assert gamma(x).series(x, -1, 3) == \
        -1/(x + 1) + EulerGamma - 1 + (x + 1)*(-1 - pi**2/12 - EulerGamma**2/2 + \
       EulerGamma) + (x + 1)**2*(-1 - pi**2/12 - EulerGamma**2/2 + EulerGamma**3/6 - \
       polygamma(2, 1)/6 + EulerGamma*pi**2/12 + EulerGamma) + O((x + 1)**3, (x, -1))


def tn_branch(s, func):
    from sympy.core.random import uniform
    c = uniform(1, 5)
    expr = func(s, c*exp_polar(I*pi)) - func(s, c*exp_polar(-I*pi))
    eps = 1e-15
    expr2 = func(s + eps, -c + eps*I) - func(s + eps, -c - eps*I)
    return abs(expr.n() - expr2.n()).n() < 1e-10


def test_lowergamma():
    from sympy.functions.special.error_functions import expint
    from sympy.functions.special.hyper import meijerg
    assert lowergamma(x, 0) == 0
    assert lowergamma(x, y).diff(y) == y**(x - 1)*exp(-y)
    assert td(lowergamma(randcplx(), y), y)
    assert td(lowergamma(x, randcplx()), x)
    assert lowergamma(x, y).diff(x) == \
        gamma(x)*digamma(x) - uppergamma(x, y)*log(y) \
        - meijerg([], [1, 1], [0, 0, x], [], y)

    assert lowergamma(S.Half, x) == sqrt(pi)*erf(sqrt(x))
    assert not lowergamma(S.Half - 3, x).has(lowergamma)
    assert not lowergamma(S.Half + 3, x).has(lowergamma)
    assert lowergamma(S.Half, x, evaluate=False).has(lowergamma)
    assert tn(lowergamma(S.Half + 3, x, evaluate=False),
              lowergamma(S.Half + 3, x), x)
    assert tn(lowergamma(S.Half - 3, x, evaluate=False),
              lowergamma(S.Half - 3, x), x)

    assert tn_branch(-3, lowergamma)
    assert tn_branch(-4, lowergamma)
    assert tn_branch(Rational(1, 3), lowergamma)
    assert tn_branch(pi, lowergamma)
    assert lowergamma(3, exp_polar(4*pi*I)*x) == lowergamma(3, x)
    assert lowergamma(y, exp_polar(5*pi*I)*x) == \
        exp(4*I*pi*y)*lowergamma(y, x*exp_polar(pi*I))
    assert lowergamma(-2, exp_polar(5*pi*I)*x) == \
        lowergamma(-2, x*exp_polar(I*pi)) + 2*pi*I

    assert conjugate(lowergamma(x, y)) == lowergamma(conjugate(x), conjugate(y))
    assert conjugate(lowergamma(x, 0)) == 0
    assert unchanged(conjugate, lowergamma(x, -oo))

    assert lowergamma(0, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(S(1)/3, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1, x, evaluate=False)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(x, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(x + 1, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1/x, x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(0, x + 1)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(S(1)/3, x + 1)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(1, x + 1, evaluate=False)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(x, x + 1)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(x + 1, x + 1)._eval_is_meromorphic(x, 0) == True
    assert lowergamma(1/x, x + 1)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(0, 1/x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(S(1)/3, 1/x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1, 1/x, evaluate=False)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(x, 1/x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(x + 1, 1/x)._eval_is_meromorphic(x, 0) == False
    assert lowergamma(1/x, 1/x)._eval_is_meromorphic(x, 0) == False

    assert lowergamma(x, 2).series(x, oo, 3) == \
        2**x*(1 + 2/(x + 1))*exp(-2)/x + O(exp(x*log(2))/x**3, (x, oo))

    assert lowergamma(
        x, y).rewrite(expint) == -y**x*expint(-x + 1, y) + gamma(x)
    k = Symbol('k', integer=True)
    assert lowergamma(
        k, y).rewrite(expint) == -y**k*expint(-k + 1, y) + gamma(k)
    k = Symbol('k', integer=True, positive=False)
    assert lowergamma(k, y).rewrite(expint) == lowergamma(k, y)
    assert lowergamma(x, y).rewrite(uppergamma) == gamma(x) - uppergamma(x, y)

    assert lowergamma(70, 6) == factorial(69) - 69035724522603011058660187038367026272747334489677105069435923032634389419656200387949342530805432320 * exp(-6)
    assert (lowergamma(S(77) / 2, 6) - lowergamma(S(77) / 2, 6, evaluate=False)).evalf() < 1e-16
    assert (lowergamma(-S(77) / 2, 6) - lowergamma(-S(77) / 2, 6, evaluate=False)).evalf() < 1e-16


def test_uppergamma():
    from sympy.functions.special.error_functions import expint
    from sympy.functions.special.hyper import meijerg
    assert uppergamma(4, 0) == 6
    assert uppergamma(x, y).diff(y) == -y**(x - 1)*exp(-y)
    assert td(uppergamma(randcplx(), y), y)
    assert uppergamma(x, y).diff(x) == \
        uppergamma(x, y)*log(y) + meijerg([], [1, 1], [0, 0, x], [], y)
    assert td(uppergamma(x, randcplx()), x)

    p = Symbol('p', positive=True)
    assert uppergamma(0, p) == -Ei(-p)
    assert uppergamma(p, 0) == gamma(p)
    assert uppergamma(S.Half, x) == sqrt(pi)*erfc(sqrt(x))
    assert not uppergamma(S.Half - 3, x).has(uppergamma)
    assert not uppergamma(S.Half + 3, x).has(uppergamma)
    assert uppergamma(S.Half, x, evaluate=False).has(uppergamma)
    assert tn(uppergamma(S.Half + 3, x, evaluate=False),
              uppergamma(S.Half + 3, x), x)
    assert tn(uppergamma(S.Half - 3, x, evaluate=False),
              uppergamma(S.Half - 3, x), x)

    assert unchanged(uppergamma, x, -oo)
    assert unchanged(uppergamma, x, 0)

    assert tn_branch(-3, uppergamma)
    assert tn_branch(-4, uppergamma)
    assert tn_branch(Rational(1, 3), uppergamma)
    assert tn_branch(pi, uppergamma)
    assert uppergamma(3, exp_polar(4*pi*I)*x) == uppergamma(3, x)
    assert uppergamma(y, exp_polar(5*pi*I)*x) == \
        exp(4*I*pi*y)*uppergamma(y, x*exp_polar(pi*I)) + \
        gamma(y)*(1 - exp(4*pi*I*y))
    assert uppergamma(-2, exp_polar(5*pi*I)*x) == \
        uppergamma(-2, x*exp_polar(I*pi)) - 2*pi*I

    assert uppergamma(-2, x) == expint(3, x)/x**2

    assert conjugate(uppergamma(x, y)) == uppergamma(conjugate(x), conjugate(y))
    assert unchanged(conjugate, uppergamma(x, -oo))

    assert uppergamma(x, y).rewrite(expint) == y**x*expint(-x + 1, y)
    assert uppergamma(x, y).rewrite(lowergamma) == gamma(x) - lowergamma(x, y)

    assert uppergamma(70, 6) == 69035724522603011058660187038367026272747334489677105069435923032634389419656200387949342530805432320*exp(-6)
    assert (uppergamma(S(77) / 2, 6) - uppergamma(S(77) / 2, 6, evaluate=False)).evalf() < 1e-16
    assert (uppergamma(-S(77) / 2, 6) - uppergamma(-S(77) / 2, 6, evaluate=False)).evalf() < 1e-16


def test_polygamma():
    assert polygamma(n, nan) is nan

    assert polygamma(0, oo) is oo
    assert polygamma(0, -oo) is oo
    assert polygamma(0, I*oo) is oo
    assert polygamma(0, -I*oo) is oo
    assert polygamma(1, oo) == 0
    assert polygamma(5, oo) == 0

    assert polygamma(0, -9) is zoo

    assert polygamma(0, -9) is zoo
    assert polygamma(0, -1) is zoo
    assert polygamma(Rational(3, 2), -1) is zoo

    assert polygamma(0, 0) is zoo

    assert polygamma(0, 1) == -EulerGamma
    assert polygamma(0, 7) == Rational(49, 20) - EulerGamma

    assert polygamma(1, 1) == pi**2/6
    assert polygamma(1, 2) == pi**2/6 - 1
    assert polygamma(1, 3) == pi**2/6 - Rational(5, 4)
    assert polygamma(3, 1) == pi**4 / 15
    assert polygamma(3, 5) == 6*(Rational(-22369, 20736) + pi**4/90)
    assert polygamma(5, 1) == 8 * pi**6 / 63

    assert polygamma(1, S.Half) == pi**2 / 2
    assert polygamma(2, S.Half) == -14*zeta(3)
    assert polygamma(11, S.Half) == 176896*pi**12

    def t(m, n):
        x = S(m)/n
        r = polygamma(0, x)
        if r.has(polygamma):
            return False
        return abs(polygamma(0, x.n()).n() - r.n()).n() < 1e-10
    assert t(1, 2)
    assert t(3, 2)
    assert t(-1, 2)
    assert t(1, 4)
    assert t(-3, 4)
    assert t(1, 3)
    assert t(4, 3)
    assert t(3, 4)
    assert t(2, 3)
    assert t(123, 5)

    assert polygamma(0, x).rewrite(zeta) == polygamma(0, x)
    assert polygamma(1, x).rewrite(zeta) == zeta(2, x)
    assert polygamma(2, x).rewrite(zeta) == -2*zeta(3, x)
    assert polygamma(I, 2).rewrite(zeta) == polygamma(I, 2)
    n1 = Symbol('n1')
    n2 = Symbol('n2', real=True)
    n3 = Symbol('n3', integer=True)
    n4 = Symbol('n4', positive=True)
    n5 = Symbol('n5', positive=True, integer=True)
    assert polygamma(n1, x).rewrite(zeta) == polygamma(n1, x)
    assert polygamma(n2, x).rewrite(zeta) == polygamma(n2, x)
    assert polygamma(n3, x).rewrite(zeta) == polygamma(n3, x)
    assert polygamma(n4, x).rewrite(zeta) == polygamma(n4, x)
    assert polygamma(n5, x).rewrite(zeta) == (-1)**(n5 + 1) * factorial(n5) * zeta(n5 + 1, x)

    assert polygamma(3, 7*x).diff(x) == 7*polygamma(4, 7*x)

    assert polygamma(0, x).rewrite(harmonic) == harmonic(x - 1) - EulerGamma
    assert polygamma(2, x).rewrite(harmonic) == 2*harmonic(x - 1, 3) - 2*zeta(3)
    ni = Symbol("n", integer=True)
    assert polygamma(ni, x).rewrite(harmonic) == (-1)**(ni + 1)*(-harmonic(x - 1, ni + 1)
                                                                 + zeta(ni + 1))*factorial(ni)

    # Polygamma of non-negative integer order is unbranched:
    k = Symbol('n', integer=True, nonnegative=True)
    assert polygamma(k, exp_polar(2*I*pi)*x) == polygamma(k, x)

    # but negative integers are branched!
    k = Symbol('n', integer=True)
    assert polygamma(k, exp_polar(2*I*pi)*x).args == (k, exp_polar(2*I*pi)*x)

    # Polygamma of order -1 is loggamma:
    assert polygamma(-1, x) == loggamma(x) - log(2*pi) / 2

    # But smaller orders are iterated integrals and don't have a special name
    assert polygamma(-2, x).func is polygamma

    # Test a bug
    assert polygamma(0, -x).expand(func=True) == polygamma(0, -x)

    assert polygamma(2, 2.5).is_positive == False
    assert polygamma(2, -2.5).is_positive == False
    assert polygamma(3, 2.5).is_positive == True
    assert polygamma(3, -2.5).is_positive is True
    assert polygamma(-2, -2.5).is_positive is None
    assert polygamma(-3, -2.5).is_positive is None

    assert polygamma(2, 2.5).is_negative == True
    assert polygamma(3, 2.5).is_negative == False
    assert polygamma(3, -2.5).is_negative == False
    assert polygamma(2, -2.5).is_negative is True
    assert polygamma(-2, -2.5).is_negative is None
    assert polygamma(-3, -2.5).is_negative is None

    assert polygamma(I, 2).is_positive is None
    assert polygamma(I, 3).is_negative is None

    # issue 17350
    assert (I*polygamma(I, pi)).as_real_imag() == \
           (-im(polygamma(I, pi)), re(polygamma(I, pi)))
    assert (tanh(polygamma(I, 1))).rewrite(exp) == \
           (exp(polygamma(I, 1)) - exp(-polygamma(I, 1)))/(exp(polygamma(I, 1)) + exp(-polygamma(I, 1)))
    assert (I / polygamma(I, 4)).rewrite(exp) == \
           I*exp(-I*atan(im(polygamma(I, 4))/re(polygamma(I, 4))))/Abs(polygamma(I, 4))

    # issue 12569
    assert unchanged(im, polygamma(0, I))
    assert polygamma(Symbol('a', positive=True), Symbol('b', positive=True)).is_real is True
    assert polygamma(0, I).is_real is None

    assert str(polygamma(pi, 3).evalf(n=10)) == "0.1169314564"
    assert str(polygamma(2.3, 1.0).evalf(n=10)) == "-3.003302909"
    assert str(polygamma(-1, 1).evalf(n=10)) == "-0.9189385332" # not zero
    assert str(polygamma(I, 1).evalf(n=10)) == "-3.109856569 + 1.89089016*I"
    assert str(polygamma(1, I).evalf(n=10)) == "-0.5369999034 - 0.7942335428*I"
    assert str(polygamma(I, I).evalf(n=10)) == "6.332362889 + 45.92828268*I"


def test_polygamma_expand_func():
    assert polygamma(0, x).expand(func=True) == polygamma(0, x)
    assert polygamma(0, 2*x).expand(func=True) == \
        polygamma(0, x)/2 + polygamma(0, S.Half + x)/2 + log(2)
    assert polygamma(1, 2*x).expand(func=True) == \
        polygamma(1, x)/4 + polygamma(1, S.Half + x)/4
    assert polygamma(2, x).expand(func=True) == \
        polygamma(2, x)
    assert polygamma(0, -1 + x).expand(func=True) == \
        polygamma(0, x) - 1/(x - 1)
    assert polygamma(0, 1 + x).expand(func=True) == \
        1/x + polygamma(0, x )
    assert polygamma(0, 2 + x).expand(func=True) == \
        1/x + 1/(1 + x) + polygamma(0, x)
    assert polygamma(0, 3 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x)
    assert polygamma(0, 4 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x) + 1/(3 + x)
    assert polygamma(1, 1 + x).expand(func=True) == \
        polygamma(1, x) - 1/x**2
    assert polygamma(1, 2 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2
    assert polygamma(1, 3 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - 1/(2 + x)**2
    assert polygamma(1, 4 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - \
        1/(2 + x)**2 - 1/(3 + x)**2
    assert polygamma(0, x + y).expand(func=True) == \
        polygamma(0, x + y)
    assert polygamma(1, x + y).expand(func=True) == \
        polygamma(1, x + y)
    assert polygamma(1, 3 + 4*x + y).expand(func=True, multinomial=False) == \
        polygamma(1, y + 4*x) - 1/(y + 4*x)**2 - \
        1/(1 + y + 4*x)**2 - 1/(2 + y + 4*x)**2
    assert polygamma(3, 3 + 4*x + y).expand(func=True, multinomial=False) == \
        polygamma(3, y + 4*x) - 6/(y + 4*x)**4 - \
        6/(1 + y + 4*x)**4 - 6/(2 + y + 4*x)**4
    assert polygamma(3, 4*x + y + 1).expand(func=True, multinomial=False) == \
        polygamma(3, y + 4*x) - 6/(y + 4*x)**4
    e = polygamma(3, 4*x + y + Rational(3, 2))
    assert e.expand(func=True) == e
    e = polygamma(3, x + y + Rational(3, 4))
    assert e.expand(func=True, basic=False) == e

    assert polygamma(-1, x, evaluate=False).expand(func=True) == \
        loggamma(x) - log(pi)/2 - log(2)/2
    p2 = polygamma(-2, x).expand(func=True) + x**2/2 - x/2 + S(1)/12
    assert isinstance(p2, Subs)
    assert p2.point == (-1,)


def test_digamma():
    assert digamma(nan) == nan

    assert digamma(oo) == oo
    assert digamma(-oo) == oo
    assert digamma(I*oo) == oo
    assert digamma(-I*oo) == oo

    assert digamma(-9) == zoo

    assert digamma(-9) == zoo
    assert digamma(-1) == zoo

    assert digamma(0) == zoo

    assert digamma(1) == -EulerGamma
    assert digamma(7) == Rational(49, 20) - EulerGamma

    def t(m, n):
        x = S(m)/n
        r = digamma(x)
        if r.has(digamma):
            return False
        return abs(digamma(x.n()).n() - r.n()).n() < 1e-10
    assert t(1, 2)
    assert t(3, 2)
    assert t(-1, 2)
    assert t(1, 4)
    assert t(-3, 4)
    assert t(1, 3)
    assert t(4, 3)
    assert t(3, 4)
    assert t(2, 3)
    assert t(123, 5)

    assert digamma(x).rewrite(zeta) == polygamma(0, x)

    assert digamma(x).rewrite(harmonic) == harmonic(x - 1) - EulerGamma

    assert digamma(I).is_real is None

    assert digamma(x,evaluate=False).fdiff() == polygamma(1, x)

    assert digamma(x,evaluate=False).is_real is None

    assert digamma(x,evaluate=False).is_positive is None

    assert digamma(x,evaluate=False).is_negative is None

    assert digamma(x,evaluate=False).rewrite(polygamma) == polygamma(0, x)


def test_digamma_expand_func():
    assert digamma(x).expand(func=True) == polygamma(0, x)
    assert digamma(2*x).expand(func=True) == \
        polygamma(0, x)/2 + polygamma(0, Rational(1, 2) + x)/2 + log(2)
    assert digamma(-1 + x).expand(func=True) == \
        polygamma(0, x) - 1/(x - 1)
    assert digamma(1 + x).expand(func=True) == \
        1/x + polygamma(0, x )
    assert digamma(2 + x).expand(func=True) == \
        1/x + 1/(1 + x) + polygamma(0, x)
    assert digamma(3 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x)
    assert digamma(4 + x).expand(func=True) == \
        polygamma(0, x) + 1/x + 1/(1 + x) + 1/(2 + x) + 1/(3 + x)
    assert digamma(x + y).expand(func=True) == \
        polygamma(0, x + y)

def test_trigamma():
    assert trigamma(nan) == nan

    assert trigamma(oo) == 0

    assert trigamma(1) == pi**2/6
    assert trigamma(2) == pi**2/6 - 1
    assert trigamma(3) == pi**2/6 - Rational(5, 4)

    assert trigamma(x, evaluate=False).rewrite(zeta) == zeta(2, x)
    assert trigamma(x, evaluate=False).rewrite(harmonic) == \
        trigamma(x).rewrite(polygamma).rewrite(harmonic)

    assert trigamma(x,evaluate=False).fdiff() == polygamma(2, x)

    assert trigamma(x,evaluate=False).is_real is None

    assert trigamma(x,evaluate=False).is_positive is None

    assert trigamma(x,evaluate=False).is_negative is None

    assert trigamma(x,evaluate=False).rewrite(polygamma) == polygamma(1, x)

def test_trigamma_expand_func():
    assert trigamma(2*x).expand(func=True) == \
        polygamma(1, x)/4 + polygamma(1, Rational(1, 2) + x)/4
    assert trigamma(1 + x).expand(func=True) == \
        polygamma(1, x) - 1/x**2
    assert trigamma(2 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2
    assert trigamma(3 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - 1/(2 + x)**2
    assert trigamma(4 + x).expand(func=True, multinomial=False) == \
        polygamma(1, x) - 1/x**2 - 1/(1 + x)**2 - \
        1/(2 + x)**2 - 1/(3 + x)**2
    assert trigamma(x + y).expand(func=True) == \
        polygamma(1, x + y)
    assert trigamma(3 + 4*x + y).expand(func=True, multinomial=False) == \
        polygamma(1, y + 4*x) - 1/(y + 4*x)**2 - \
        1/(1 + y + 4*x)**2 - 1/(2 + y + 4*x)**2

def test_loggamma():
    raises(TypeError, lambda: loggamma(2, 3))
    raises(ArgumentIndexError, lambda: loggamma(x).fdiff(2))

    assert loggamma(-1) is oo
    assert loggamma(-2) is oo
    assert loggamma(0) is oo
    assert loggamma(1) == 0
    assert loggamma(2) == 0
    assert loggamma(3) == log(2)
    assert loggamma(4) == log(6)

    n = Symbol("n", integer=True, positive=True)
    assert loggamma(n) == log(gamma(n))
    assert loggamma(-n) is oo
    assert loggamma(n/2) == log(2**(-n + 1)*sqrt(pi)*gamma(n)/gamma(n/2 + S.Half))

    assert loggamma(oo) is oo
    assert loggamma(-oo) is zoo
    assert loggamma(I*oo) is zoo
    assert loggamma(-I*oo) is zoo
    assert loggamma(zoo) is zoo
    assert loggamma(nan) is nan

    L = loggamma(Rational(16, 3))
    E = -5*log(3) + loggamma(Rational(1, 3)) + log(4) + log(7) + log(10) + log(13)
    assert expand_func(L).doit() == E
    assert L.n() == E.n()

    L = loggamma(Rational(19, 4))
    E = -4*log(4) + loggamma(Rational(3, 4)) + log(3) + log(7) + log(11) + log(15)
    assert expand_func(L).doit() == E
    assert L.n() == E.n()

    L = loggamma(Rational(23, 7))
    E = -3*log(7) + log(2) + loggamma(Rational(2, 7)) + log(9) + log(16)
    assert expand_func(L).doit() == E
    assert L.n() == E.n()

    L = loggamma(Rational(19, 4) - 7)
    E = -log(9) - log(5) + loggamma(Rational(3, 4)) + 3*log(4) - 3*I*pi
    assert expand_func(L).doit() == E
    assert L.n() == E.n()

    L = loggamma(Rational(23, 7) - 6)
    E = -log(19) - log(12) - log(5) + loggamma(Rational(2, 7)) + 3*log(7) - 3*I*pi
    assert expand_func(L).doit() == E
    assert L.n() == E.n()

    assert loggamma(x).diff(x) == polygamma(0, x)
    s1 = loggamma(1/(x + sin(x)) + cos(x)).nseries(x, n=4)
    s2 = (-log(2*x) - 1)/(2*x) - log(x/pi)/2 + (4 - log(2*x))*x/24 + O(x**2) + \
        log(x)*x**2/2
    assert (s1 - s2).expand(force=True).removeO() == 0
    s1 = loggamma(1/x).series(x)
    s2 = (1/x - S.Half)*log(1/x) - 1/x + log(2*pi)/2 + \
        x/12 - x**3/360 + x**5/1260 + O(x**7)
    assert ((s1 - s2).expand(force=True)).removeO() == 0

    assert loggamma(x).rewrite('intractable') == log(gamma(x))

    s1 = loggamma(x).series(x).cancel()
    assert s1 == -log(x) - EulerGamma*x + pi**2*x**2/12 + x**3*polygamma(2, 1)/6 + \
        pi**4*x**4/360 + x**5*polygamma(4, 1)/120 + O(x**6)
    assert s1 == loggamma(x).rewrite('intractable').series(x).cancel()

    assert conjugate(loggamma(x)) == loggamma(conjugate(x))
    assert conjugate(loggamma(0)) is oo
    assert conjugate(loggamma(1)) == loggamma(conjugate(1))
    assert conjugate(loggamma(-oo)) == conjugate(zoo)

    assert loggamma(Symbol('v', positive=True)).is_real is True
    assert loggamma(Symbol('v', zero=True)).is_real is False
    assert loggamma(Symbol('v', negative=True)).is_real is False
    assert loggamma(Symbol('v', nonpositive=True)).is_real is False
    assert loggamma(Symbol('v', nonnegative=True)).is_real is None
    assert loggamma(Symbol('v', imaginary=True)).is_real is None
    assert loggamma(Symbol('v', real=True)).is_real is None
    assert loggamma(Symbol('v')).is_real is None

    assert loggamma(S.Half).is_real is True
    assert loggamma(0).is_real is False
    assert loggamma(Rational(-1, 2)).is_real is False
    assert loggamma(I).is_real is None
    assert loggamma(2 + 3*I).is_real is None

    def tN(N, M):
        assert loggamma(1/x)._eval_nseries(x, n=N).getn() == M
    tN(0, 0)
    tN(1, 1)
    tN(2, 2)
    tN(3, 3)
    tN(4, 4)
    tN(5, 5)


def test_polygamma_expansion():
    # A. & S., pa. 259 and 260
    assert polygamma(0, 1/x).nseries(x, n=3) == \
        -log(x) - x/2 - x**2/12 + O(x**3)
    assert polygamma(1, 1/x).series(x, n=5) == \
        x + x**2/2 + x**3/6 + O(x**5)
    assert polygamma(3, 1/x).nseries(x, n=11) == \
        2*x**3 + 3*x**4 + 2*x**5 - x**7 + 4*x**9/3 + O(x**11)


def test_polygamma_leading_term():
    expr = -log(1/x) + polygamma(0, 1 + 1/x) + S.EulerGamma
    assert expr.as_leading_term(x, logx=-y) == S.EulerGamma


def test_issue_8657():
    n = Symbol('n', negative=True, integer=True)
    m = Symbol('m', integer=True)
    o = Symbol('o', positive=True)
    p = Symbol('p', negative=True, integer=False)
    assert gamma(n).is_real is False
    assert gamma(m).is_real is None
    assert gamma(o).is_real is True
    assert gamma(p).is_real is True
    assert gamma(w).is_real is None


def test_issue_8524():
    x = Symbol('x', positive=True)
    y = Symbol('y', negative=True)
    z = Symbol('z', positive=False)
    p = Symbol('p', negative=False)
    q = Symbol('q', integer=True)
    r = Symbol('r', integer=False)
    e = Symbol('e', even=True, negative=True)
    assert gamma(x).is_positive is True
    assert gamma(y).is_positive is None
    assert gamma(z).is_positive is None
    assert gamma(p).is_positive is None
    assert gamma(q).is_positive is None
    assert gamma(r).is_positive is None
    assert gamma(e + S.Half).is_positive is True
    assert gamma(e - S.Half).is_positive is False

def test_issue_14450():
    assert uppergamma(Rational(3, 8), x).evalf() == uppergamma(Rational(3, 8), x)
    assert lowergamma(x, Rational(3, 8)).evalf() == lowergamma(x, Rational(3, 8))
    # some values from Wolfram Alpha for comparison
    assert abs(uppergamma(Rational(3, 8), 2).evalf() - 0.07105675881) < 1e-9
    assert abs(lowergamma(Rational(3, 8), 2).evalf() - 2.2993794256) < 1e-9

def test_issue_14528():
    k = Symbol('k', integer=True, nonpositive=True)
    assert isinstance(gamma(k), gamma)

def test_multigamma():
    from sympy.concrete.products import Product
    p = Symbol('p')
    _k = Dummy('_k')

    assert multigamma(x, p).dummy_eq(pi**(p*(p - 1)/4)*\
        Product(gamma(x + (1 - _k)/2), (_k, 1, p)))

    assert conjugate(multigamma(x, p)).dummy_eq(pi**((conjugate(p) - 1)*\
        conjugate(p)/4)*Product(gamma(conjugate(x) + (1-conjugate(_k))/2), (_k, 1, p)))
    assert conjugate(multigamma(x, 1)) == gamma(conjugate(x))

    p = Symbol('p', positive=True)
    assert conjugate(multigamma(x, p)).dummy_eq(pi**((p - 1)*p/4)*\
        Product(gamma(conjugate(x) + (1-conjugate(_k))/2), (_k, 1, p)))

    assert multigamma(nan, 1) is nan
    assert multigamma(oo, 1).doit() is oo

    assert multigamma(1, 1) == 1
    assert multigamma(2, 1) == 1
    assert multigamma(3, 1) == 2

    assert multigamma(102, 1) == factorial(101)
    assert multigamma(S.Half, 1) == sqrt(pi)

    assert multigamma(1, 2) == pi
    assert multigamma(2, 2) == pi/2

    assert multigamma(1, 3) is zoo
    assert multigamma(2, 3) == pi**2/2
    assert multigamma(3, 3) == 3*pi**2/2

    assert multigamma(x, 1).diff(x) == gamma(x)*polygamma(0, x)
    assert multigamma(x, 2).diff(x) == sqrt(pi)*gamma(x)*gamma(x - S.Half)*\
        polygamma(0, x) + sqrt(pi)*gamma(x)*gamma(x - S.Half)*polygamma(0, x - S.Half)

    assert multigamma(x - 1, 1).expand(func=True) == gamma(x)/(x - 1)
    assert multigamma(x + 2, 1).expand(func=True, mul=False) == x*(x + 1)*\
        gamma(x)
    assert multigamma(x - 1, 2).expand(func=True) == sqrt(pi)*gamma(x)*\
        gamma(x + S.Half)/(x**3 - 3*x**2 + x*Rational(11, 4) - Rational(3, 4))
    assert multigamma(x - 1, 3).expand(func=True) == pi**Rational(3, 2)*gamma(x)**2*\
        gamma(x + S.Half)/(x**5 - 6*x**4 + 55*x**3/4 - 15*x**2 + x*Rational(31, 4) - Rational(3, 2))

    assert multigamma(n, 1).rewrite(factorial) == factorial(n - 1)
    assert multigamma(n, 2).rewrite(factorial) == sqrt(pi)*\
        factorial(n - Rational(3, 2))*factorial(n - 1)
    assert multigamma(n, 3).rewrite(factorial) == pi**Rational(3, 2)*\
        factorial(n - 2)*factorial(n - Rational(3, 2))*factorial(n - 1)

    assert multigamma(Rational(-1, 2), 3, evaluate=False).is_real == False
    assert multigamma(S.Half, 3, evaluate=False).is_real == False
    assert multigamma(0, 1, evaluate=False).is_real == False
    assert multigamma(1, 3, evaluate=False).is_real == False
    assert multigamma(-1.0, 3, evaluate=False).is_real == False
    assert multigamma(0.7, 3, evaluate=False).is_real == True
    assert multigamma(3, 3, evaluate=False).is_real == True

def test_gamma_as_leading_term():
    assert gamma(x).as_leading_term(x) == 1/x
    assert gamma(2 + x).as_leading_term(x) == S(1)
    assert gamma(cos(x)).as_leading_term(x) == S(1)
    assert gamma(sin(x)).as_leading_term(x) == 1/x
