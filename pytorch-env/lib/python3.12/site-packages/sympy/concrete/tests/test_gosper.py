"""Tests for Gosper's algorithm for hypergeometric summation. """

from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.polytools import Poly
from sympy.simplify.simplify import simplify
from sympy.concrete.gosper import gosper_normal, gosper_sum, gosper_term
from sympy.abc import a, b, j, k, m, n, r, x


def test_gosper_normal():
    eq = 4*n + 5, 2*(4*n + 1)*(2*n + 3), n
    assert gosper_normal(*eq) == \
        (Poly(Rational(1, 4), n), Poly(n + Rational(3, 2)), Poly(n + Rational(1, 4)))
    assert gosper_normal(*eq, polys=False) == \
        (Rational(1, 4), n + Rational(3, 2), n + Rational(1, 4))


def test_gosper_term():
    assert gosper_term((4*k + 1)*factorial(
        k)/factorial(2*k + 1), k) == (-k - S.Half)/(k + Rational(1, 4))


def test_gosper_sum():
    assert gosper_sum(1, (k, 0, n)) == 1 + n
    assert gosper_sum(k, (k, 0, n)) == n*(1 + n)/2
    assert gosper_sum(k**2, (k, 0, n)) == n*(1 + n)*(1 + 2*n)/6
    assert gosper_sum(k**3, (k, 0, n)) == n**2*(1 + n)**2/4

    assert gosper_sum(2**k, (k, 0, n)) == 2*2**n - 1

    assert gosper_sum(factorial(k), (k, 0, n)) is None
    assert gosper_sum(binomial(n, k), (k, 0, n)) is None

    assert gosper_sum(factorial(k)/k**2, (k, 0, n)) is None
    assert gosper_sum((k - 3)*factorial(k), (k, 0, n)) is None

    assert gosper_sum(k*factorial(k), k) == factorial(k)
    assert gosper_sum(
        k*factorial(k), (k, 0, n)) == n*factorial(n) + factorial(n) - 1

    assert gosper_sum((-1)**k*binomial(n, k), (k, 0, n)) == 0
    assert gosper_sum((
        -1)**k*binomial(n, k), (k, 0, m)) == -(-1)**m*(m - n)*binomial(n, m)/n

    assert gosper_sum((4*k + 1)*factorial(k)/factorial(2*k + 1), (k, 0, n)) == \
        (2*factorial(2*n + 1) - factorial(n))/factorial(2*n + 1)

    # issue 6033:
    assert gosper_sum(
        n*(n + a + b)*a**n*b**n/(factorial(n + a)*factorial(n + b)), \
        (n, 0, m)).simplify() == -exp(m*log(a) + m*log(b))*gamma(a + 1) \
        *gamma(b + 1)/(gamma(a)*gamma(b)*gamma(a + m + 1)*gamma(b + m + 1)) \
        + 1/(gamma(a)*gamma(b))


def test_gosper_sum_indefinite():
    assert gosper_sum(k, k) == k*(k - 1)/2
    assert gosper_sum(k**2, k) == k*(k - 1)*(2*k - 1)/6

    assert gosper_sum(1/(k*(k + 1)), k) == -1/k
    assert gosper_sum(-(27*k**4 + 158*k**3 + 430*k**2 + 678*k + 445)*gamma(2*k
                      + 4)/(3*(3*k + 7)*gamma(3*k + 6)), k) == \
        (3*k + 5)*(k**2 + 2*k + 5)*gamma(2*k + 4)/gamma(3*k + 6)


def test_gosper_sum_parametric():
    assert gosper_sum(binomial(S.Half, m - j + 1)*binomial(S.Half, m + j), (j, 1, n)) == \
        n*(1 + m - n)*(-1 + 2*m + 2*n)*binomial(S.Half, 1 + m - n)* \
        binomial(S.Half, m + n)/(m*(1 + 2*m))


def test_gosper_sum_algebraic():
    assert gosper_sum(
        n**2 + sqrt(2), (n, 0, m)) == (m + 1)*(2*m**2 + m + 6*sqrt(2))/6


def test_gosper_sum_iterated():
    f1 = binomial(2*k, k)/4**k
    f2 = (1 + 2*n)*binomial(2*n, n)/4**n
    f3 = (1 + 2*n)*(3 + 2*n)*binomial(2*n, n)/(3*4**n)
    f4 = (1 + 2*n)*(3 + 2*n)*(5 + 2*n)*binomial(2*n, n)/(15*4**n)
    f5 = (1 + 2*n)*(3 + 2*n)*(5 + 2*n)*(7 + 2*n)*binomial(2*n, n)/(105*4**n)

    assert gosper_sum(f1, (k, 0, n)) == f2
    assert gosper_sum(f2, (n, 0, n)) == f3
    assert gosper_sum(f3, (n, 0, n)) == f4
    assert gosper_sum(f4, (n, 0, n)) == f5

# the AeqB tests test expressions given in
# www.math.upenn.edu/~wilf/AeqB.pdf


def test_gosper_sum_AeqB_part1():
    f1a = n**4
    f1b = n**3*2**n
    f1c = 1/(n**2 + sqrt(5)*n - 1)
    f1d = n**4*4**n/binomial(2*n, n)
    f1e = factorial(3*n)/(factorial(n)*factorial(n + 1)*factorial(n + 2)*27**n)
    f1f = binomial(2*n, n)**2/((n + 1)*4**(2*n))
    f1g = (4*n - 1)*binomial(2*n, n)**2/((2*n - 1)**2*4**(2*n))
    f1h = n*factorial(n - S.Half)**2/factorial(n + 1)**2

    g1a = m*(m + 1)*(2*m + 1)*(3*m**2 + 3*m - 1)/30
    g1b = 26 + 2**(m + 1)*(m**3 - 3*m**2 + 9*m - 13)
    g1c = (m + 1)*(m*(m**2 - 7*m + 3)*sqrt(5) - (
        3*m**3 - 7*m**2 + 19*m - 6))/(2*m**3*sqrt(5) + m**4 + 5*m**2 - 1)/6
    g1d = Rational(-2, 231) + 2*4**m*(m + 1)*(63*m**4 + 112*m**3 + 18*m**2 -
             22*m + 3)/(693*binomial(2*m, m))
    g1e = Rational(-9, 2) + (81*m**2 + 261*m + 200)*factorial(
        3*m + 2)/(40*27**m*factorial(m)*factorial(m + 1)*factorial(m + 2))
    g1f = (2*m + 1)**2*binomial(2*m, m)**2/(4**(2*m)*(m + 1))
    g1g = -binomial(2*m, m)**2/4**(2*m)
    g1h = 4*pi -(2*m + 1)**2*(3*m + 4)*factorial(m - S.Half)**2/factorial(m + 1)**2

    g = gosper_sum(f1a, (n, 0, m))
    assert g is not None and simplify(g - g1a) == 0
    g = gosper_sum(f1b, (n, 0, m))
    assert g is not None and simplify(g - g1b) == 0
    g = gosper_sum(f1c, (n, 0, m))
    assert g is not None and simplify(g - g1c) == 0
    g = gosper_sum(f1d, (n, 0, m))
    assert g is not None and simplify(g - g1d) == 0
    g = gosper_sum(f1e, (n, 0, m))
    assert g is not None and simplify(g - g1e) == 0
    g = gosper_sum(f1f, (n, 0, m))
    assert g is not None and simplify(g - g1f) == 0
    g = gosper_sum(f1g, (n, 0, m))
    assert g is not None and simplify(g - g1g) == 0
    g = gosper_sum(f1h, (n, 0, m))
    # need to call rewrite(gamma) here because we have terms involving
    # factorial(1/2)
    assert g is not None and simplify(g - g1h).rewrite(gamma) == 0


def test_gosper_sum_AeqB_part2():
    f2a = n**2*a**n
    f2b = (n - r/2)*binomial(r, n)
    f2c = factorial(n - 1)**2/(factorial(n - x)*factorial(n + x))

    g2a = -a*(a + 1)/(a - 1)**3 + a**(
        m + 1)*(a**2*m**2 - 2*a*m**2 + m**2 - 2*a*m + 2*m + a + 1)/(a - 1)**3
    g2b = (m - r)*binomial(r, m)/2
    ff = factorial(1 - x)*factorial(1 + x)
    g2c = 1/ff*(
        1 - 1/x**2) + factorial(m)**2/(x**2*factorial(m - x)*factorial(m + x))

    g = gosper_sum(f2a, (n, 0, m))
    assert g is not None and simplify(g - g2a) == 0
    g = gosper_sum(f2b, (n, 0, m))
    assert g is not None and simplify(g - g2b) == 0
    g = gosper_sum(f2c, (n, 1, m))
    assert g is not None and simplify(g - g2c) == 0


def test_gosper_nan():
    a = Symbol('a', positive=True)
    b = Symbol('b', positive=True)
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True)
    f2d = n*(n + a + b)*a**n*b**n/(factorial(n + a)*factorial(n + b))
    g2d = 1/(factorial(a - 1)*factorial(
        b - 1)) - a**(m + 1)*b**(m + 1)/(factorial(a + m)*factorial(b + m))
    g = gosper_sum(f2d, (n, 0, m))
    assert simplify(g - g2d) == 0


def test_gosper_sum_AeqB_part3():
    f3a = 1/n**4
    f3b = (6*n + 3)/(4*n**4 + 8*n**3 + 8*n**2 + 4*n + 3)
    f3c = 2**n*(n**2 - 2*n - 1)/(n**2*(n + 1)**2)
    f3d = n**2*4**n/((n + 1)*(n + 2))
    f3e = 2**n/(n + 1)
    f3f = 4*(n - 1)*(n**2 - 2*n - 1)/(n**2*(n + 1)**2*(n - 2)**2*(n - 3)**2)
    f3g = (n**4 - 14*n**2 - 24*n - 9)*2**n/(n**2*(n + 1)**2*(n + 2)**2*
           (n + 3)**2)

    # g3a -> no closed form
    g3b = m*(m + 2)/(2*m**2 + 4*m + 3)
    g3c = 2**m/m**2 - 2
    g3d = Rational(2, 3) + 4**(m + 1)*(m - 1)/(m + 2)/3
    # g3e -> no closed form
    g3f = -(Rational(-1, 16) + 1/((m - 2)**2*(m + 1)**2))  # the AeqB key is wrong
    g3g = Rational(-2, 9) + 2**(m + 1)/((m + 1)**2*(m + 3)**2)

    g = gosper_sum(f3a, (n, 1, m))
    assert g is None
    g = gosper_sum(f3b, (n, 1, m))
    assert g is not None and simplify(g - g3b) == 0
    g = gosper_sum(f3c, (n, 1, m - 1))
    assert g is not None and simplify(g - g3c) == 0
    g = gosper_sum(f3d, (n, 1, m))
    assert g is not None and simplify(g - g3d) == 0
    g = gosper_sum(f3e, (n, 0, m - 1))
    assert g is None
    g = gosper_sum(f3f, (n, 4, m))
    assert g is not None and simplify(g - g3f) == 0
    g = gosper_sum(f3g, (n, 1, m))
    assert g is not None and simplify(g - g3g) == 0
