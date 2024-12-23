from itertools import product

from sympy.concrete.summations import Sum
from sympy.core.function import (diff, expand_func)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (conjugate, polar_lift)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.bessel import (besseli, besselj, besselk, bessely, hankel1, hankel2, hn1, hn2, jn, jn_zeros, yn)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.series.series import series
from sympy.functions.special.bessel import (airyai, airybi,
                                            airyaiprime, airybiprime, marcumq)
from sympy.core.random import (random_complex_number as randcplx,
                                      verify_numerically as tn,
                                      test_derivative_numerically as td,
                                      _randint)
from sympy.simplify import besselsimp
from sympy.testing.pytest import raises, slow

from sympy.abc import z, n, k, x

randint = _randint()


def test_bessel_rand():
    for f in [besselj, bessely, besseli, besselk, hankel1, hankel2]:
        assert td(f(randcplx(), z), z)

    for f in [jn, yn, hn1, hn2]:
        assert td(f(randint(-10, 10), z), z)


def test_bessel_twoinputs():
    for f in [besselj, bessely, besseli, besselk, hankel1, hankel2, jn, yn]:
        raises(TypeError, lambda: f(1))
        raises(TypeError, lambda: f(1, 2, 3))


def test_besselj_leading_term():
    assert besselj(0, x).as_leading_term(x) == 1
    assert besselj(1, sin(x)).as_leading_term(x) == x/2
    assert besselj(1, 2*sqrt(x)).as_leading_term(x) == sqrt(x)

    # https://github.com/sympy/sympy/issues/21701
    assert (besselj(z, x)/x**z).as_leading_term(x) == 1/(2**z*gamma(z + 1))


def test_bessely_leading_term():
    assert bessely(0, x).as_leading_term(x) == (2*log(x) - 2*log(2) + 2*S.EulerGamma)/pi
    assert bessely(1, sin(x)).as_leading_term(x) == -2/(pi*x)
    assert bessely(1, 2*sqrt(x)).as_leading_term(x) == -1/(pi*sqrt(x))


def test_besseli_leading_term():
    assert besseli(0, x).as_leading_term(x) == 1
    assert besseli(1, sin(x)).as_leading_term(x) == x/2
    assert besseli(1, 2*sqrt(x)).as_leading_term(x) == sqrt(x)


def test_besselk_leading_term():
    assert besselk(0, x).as_leading_term(x) == -log(x) - S.EulerGamma + log(2)
    assert besselk(1, sin(x)).as_leading_term(x) == 1/x
    assert besselk(1, 2*sqrt(x)).as_leading_term(x) == 1/(2*sqrt(x))


def test_besselj_series():
    assert besselj(0, x).series(x) == 1 - x**2/4 + x**4/64 + O(x**6)
    assert besselj(0, x**(1.1)).series(x) == 1 + x**4.4/64 - x**2.2/4 + O(x**6)
    assert besselj(0, x**2 + x).series(x) == 1 - x**2/4 - x**3/2\
        - 15*x**4/64 + x**5/16 + O(x**6)
    assert besselj(0, sqrt(x) + x).series(x, n=4) == 1 - x/4 - 15*x**2/64\
        + 215*x**3/2304 - x**Rational(3, 2)/2 + x**Rational(5, 2)/16\
        + 23*x**Rational(7, 2)/384 + O(x**4)
    assert besselj(0, x/(1 - x)).series(x) == 1 - x**2/4 - x**3/2 - 47*x**4/64\
        - 15*x**5/16 + O(x**6)
    assert besselj(0, log(1 + x)).series(x) == 1 - x**2/4 + x**3/4\
        - 41*x**4/192 + 17*x**5/96 + O(x**6)
    assert besselj(1, sin(x)).series(x) == x/2 - 7*x**3/48 + 73*x**5/1920 + O(x**6)
    assert besselj(1, 2*sqrt(x)).series(x) == sqrt(x) - x**Rational(3, 2)/2\
        + x**Rational(5, 2)/12 - x**Rational(7, 2)/144 + x**Rational(9, 2)/2880\
        - x**Rational(11, 2)/86400 + O(x**6)
    assert besselj(-2, sin(x)).series(x, n=4) == besselj(2, sin(x)).series(x, n=4)


def test_bessely_series():
    const = 2*S.EulerGamma/pi - 2*log(2)/pi + 2*log(x)/pi
    assert bessely(0, x).series(x, n=4) == const + x**2*(-log(x)/(2*pi)\
        + (2 - 2*S.EulerGamma)/(4*pi) + log(2)/(2*pi)) + O(x**4*log(x))
    assert bessely(1, x).series(x, n=4) == -2/(pi*x) + x*(log(x)/pi - log(2)/pi - \
        (1 - 2*S.EulerGamma)/(2*pi)) + x**3*(-log(x)/(8*pi) + \
        (S(5)/2 - 2*S.EulerGamma)/(16*pi) + log(2)/(8*pi)) + O(x**4*log(x))
    assert bessely(2, x).series(x, n=4) == -4/(pi*x**2) - 1/pi + x**2*(log(x)/(4*pi) - \
        log(2)/(4*pi) - (S(3)/2 - 2*S.EulerGamma)/(8*pi)) + O(x**4*log(x))
    assert bessely(3, x).series(x, n=4) == -16/(pi*x**3) - 2/(pi*x) - \
        x/(4*pi) + x**3*(log(x)/(24*pi) - log(2)/(24*pi) - \
        (S(11)/6 - 2*S.EulerGamma)/(48*pi)) + O(x**4*log(x))
    assert bessely(0, x**(1.1)).series(x, n=4) == 2*S.EulerGamma/pi\
        - 2*log(2)/pi + 2.2*log(x)/pi + x**2.2*(-0.55*log(x)/pi\
        + (2 - 2*S.EulerGamma)/(4*pi) + log(2)/(2*pi)) + O(x**4*log(x))
    assert bessely(0, x**2 + x).series(x, n=4) == \
        const - (2 - 2*S.EulerGamma)*(-x**3/(2*pi) - x**2/(4*pi)) + 2*x/pi\
        + x**2*(-log(x)/(2*pi) - 1/pi + log(2)/(2*pi))\
        + x**3*(-log(x)/pi + 1/(6*pi) + log(2)/pi) + O(x**4*log(x))
    assert bessely(0, x/(1 - x)).series(x, n=3) == const\
        + 2*x/pi + x**2*(-log(x)/(2*pi) + (2 - 2*S.EulerGamma)/(4*pi)\
        + log(2)/(2*pi) + 1/pi) + O(x**3*log(x))
    assert bessely(0, log(1 + x)).series(x, n=3) == const\
        - x/pi + x**2*(-log(x)/(2*pi) + (2 - 2*S.EulerGamma)/(4*pi)\
        + log(2)/(2*pi) + 5/(12*pi)) + O(x**3*log(x))
    assert bessely(1, sin(x)).series(x, n=4) == -1/(pi*(-x**3/12 + x/2)) - \
        (1 - 2*S.EulerGamma)*(-x**3/12 + x/2)/pi + x*(log(x)/pi - log(2)/pi) + \
        x**3*(-7*log(x)/(24*pi) - 1/(6*pi) + (S(5)/2 - 2*S.EulerGamma)/(16*pi) +
        7*log(2)/(24*pi)) + O(x**4*log(x))
    assert bessely(1, 2*sqrt(x)).series(x, n=3) == -1/(pi*sqrt(x)) + \
        sqrt(x)*(log(x)/pi - (1 - 2*S.EulerGamma)/pi) + x**(S(3)/2)*(-log(x)/(2*pi) + \
        (S(5)/2 - 2*S.EulerGamma)/(2*pi)) + x**(S(5)/2)*(log(x)/(12*pi) - \
        (S(10)/3 - 2*S.EulerGamma)/(12*pi)) + O(x**3*log(x))
    assert bessely(-2, sin(x)).series(x, n=4) == bessely(2, sin(x)).series(x, n=4)


def test_besseli_series():
    assert besseli(0, x).series(x) == 1 + x**2/4 + x**4/64 + O(x**6)
    assert besseli(0, x**(1.1)).series(x) == 1 + x**4.4/64 + x**2.2/4 + O(x**6)
    assert besseli(0, x**2 + x).series(x) == 1 + x**2/4 + x**3/2 + 17*x**4/64 + \
        x**5/16 + O(x**6)
    assert besseli(0, sqrt(x) + x).series(x, n=4) == 1 + x/4 + 17*x**2/64 + \
        217*x**3/2304 + x**(S(3)/2)/2 + x**(S(5)/2)/16 + 25*x**(S(7)/2)/384 + O(x**4)
    assert besseli(0, x/(1 - x)).series(x) == 1 + x**2/4 + x**3/2 + 49*x**4/64 + \
        17*x**5/16 + O(x**6)
    assert besseli(0, log(1 + x)).series(x) == 1 + x**2/4 - x**3/4 + 47*x**4/192 - \
        23*x**5/96 + O(x**6)
    assert besseli(1, sin(x)).series(x) == x/2 - x**3/48 - 47*x**5/1920 + O(x**6)
    assert besseli(1, 2*sqrt(x)).series(x) == sqrt(x) + x**(S(3)/2)/2 + x**(S(5)/2)/12 + \
        x**(S(7)/2)/144 + x**(S(9)/2)/2880 + x**(S(11)/2)/86400 + O(x**6)
    assert besseli(-2, sin(x)).series(x, n=4) == besseli(2, sin(x)).series(x, n=4)


def test_besselk_series():
    const = log(2) - S.EulerGamma - log(x)
    assert besselk(0, x).series(x, n=4) == const + \
        x**2*(-log(x)/4 - S.EulerGamma/4 + log(2)/4 + S(1)/4) + O(x**4*log(x))
    assert besselk(1, x).series(x, n=4) == 1/x + x*(log(x)/2 - log(2)/2 - \
        S(1)/4 + S.EulerGamma/2) + x**3*(log(x)/16 - S(5)/64 - log(2)/16 + \
        S.EulerGamma/16) + O(x**4*log(x))
    assert besselk(2, x).series(x, n=4) == 2/x**2 - S(1)/2 + x**2*(-log(x)/8 - \
        S.EulerGamma/8 + log(2)/8 + S(3)/32) + O(x**4*log(x))
    assert besselk(0, x**(1.1)).series(x, n=4) == log(2) - S.EulerGamma - \
        1.1*log(x) + x**2.2*(-0.275*log(x) - S.EulerGamma/4 + \
        log(2)/4 + S(1)/4) + O(x**4*log(x))
    assert besselk(0, x**2 + x).series(x, n=4) == const + \
        (2 - 2*S.EulerGamma)*(x**3/4 + x**2/8) - x + x**2*(-log(x)/4 + \
        log(2)/4 + S(1)/2) + x**3*(-log(x)/2 - S(7)/12 + log(2)/2) + O(x**4*log(x))
    assert besselk(0, x/(1 - x)).series(x, n=3) == const - x + x**2*(-log(x)/4 - \
        S(1)/4 - S.EulerGamma/4 + log(2)/4) + O(x**3*log(x))
    assert besselk(0, log(1 + x)).series(x, n=3) == const + x/2 + \
        x**2*(-log(x)/4 - S.EulerGamma/4 + S(1)/24 + log(2)/4) + O(x**3*log(x))
    assert besselk(1, 2*sqrt(x)).series(x, n=3) == 1/(2*sqrt(x)) + \
        sqrt(x)*(log(x)/2 - S(1)/2 + S.EulerGamma) + x**(S(3)/2)*(log(x)/4 - S(5)/8 + \
        S.EulerGamma/2) + x**(S(5)/2)*(log(x)/24 - S(5)/36 + S.EulerGamma/12) + O(x**3*log(x))
    assert besselk(-2, sin(x)).series(x, n=4) == besselk(2, sin(x)).series(x, n=4)


def test_diff():
    assert besselj(n, z).diff(z) == besselj(n - 1, z)/2 - besselj(n + 1, z)/2
    assert bessely(n, z).diff(z) == bessely(n - 1, z)/2 - bessely(n + 1, z)/2
    assert besseli(n, z).diff(z) == besseli(n - 1, z)/2 + besseli(n + 1, z)/2
    assert besselk(n, z).diff(z) == -besselk(n - 1, z)/2 - besselk(n + 1, z)/2
    assert hankel1(n, z).diff(z) == hankel1(n - 1, z)/2 - hankel1(n + 1, z)/2
    assert hankel2(n, z).diff(z) == hankel2(n - 1, z)/2 - hankel2(n + 1, z)/2


def test_rewrite():
    assert besselj(n, z).rewrite(jn) == sqrt(2*z/pi)*jn(n - S.Half, z)
    assert bessely(n, z).rewrite(yn) == sqrt(2*z/pi)*yn(n - S.Half, z)
    assert besseli(n, z).rewrite(besselj) == \
        exp(-I*n*pi/2)*besselj(n, polar_lift(I)*z)
    assert besselj(n, z).rewrite(besseli) == \
        exp(I*n*pi/2)*besseli(n, polar_lift(-I)*z)

    nu = randcplx()

    assert tn(besselj(nu, z), besselj(nu, z).rewrite(besseli), z)
    assert tn(besselj(nu, z), besselj(nu, z).rewrite(bessely), z)

    assert tn(besseli(nu, z), besseli(nu, z).rewrite(besselj), z)
    assert tn(besseli(nu, z), besseli(nu, z).rewrite(bessely), z)

    assert tn(bessely(nu, z), bessely(nu, z).rewrite(besselj), z)
    assert tn(bessely(nu, z), bessely(nu, z).rewrite(besseli), z)

    assert tn(besselk(nu, z), besselk(nu, z).rewrite(besselj), z)
    assert tn(besselk(nu, z), besselk(nu, z).rewrite(besseli), z)
    assert tn(besselk(nu, z), besselk(nu, z).rewrite(bessely), z)

    # check that a rewrite was triggered, when the order is set to a generic
    # symbol 'nu'
    assert yn(nu, z) != yn(nu, z).rewrite(jn)
    assert hn1(nu, z) != hn1(nu, z).rewrite(jn)
    assert hn2(nu, z) != hn2(nu, z).rewrite(jn)
    assert jn(nu, z) != jn(nu, z).rewrite(yn)
    assert hn1(nu, z) != hn1(nu, z).rewrite(yn)
    assert hn2(nu, z) != hn2(nu, z).rewrite(yn)

    # rewriting spherical bessel functions (SBFs) w.r.t. besselj, bessely is
    # not allowed if a generic symbol 'nu' is used as the order of the SBFs
    # to avoid inconsistencies (the order of bessel[jy] is allowed to be
    # complex-valued, whereas SBFs are defined only for integer orders)
    order = nu
    for f in (besselj, bessely):
        assert hn1(order, z) == hn1(order, z).rewrite(f)
        assert hn2(order, z) == hn2(order, z).rewrite(f)

    assert jn(order, z).rewrite(besselj) == sqrt(2)*sqrt(pi)*sqrt(1/z)*besselj(order + S.Half, z)/2
    assert jn(order, z).rewrite(bessely) == (-1)**nu*sqrt(2)*sqrt(pi)*sqrt(1/z)*bessely(-order - S.Half, z)/2

    # for integral orders rewriting SBFs w.r.t bessel[jy] is allowed
    N = Symbol('n', integer=True)
    ri = randint(-11, 10)
    for order in (ri, N):
        for f in (besselj, bessely):
            assert yn(order, z) != yn(order, z).rewrite(f)
            assert jn(order, z) != jn(order, z).rewrite(f)
            assert hn1(order, z) != hn1(order, z).rewrite(f)
            assert hn2(order, z) != hn2(order, z).rewrite(f)

    for func, refunc in product((yn, jn, hn1, hn2),
                                (jn, yn, besselj, bessely)):
        assert tn(func(ri, z), func(ri, z).rewrite(refunc), z)


def test_expand():
    assert expand_func(besselj(S.Half, z).rewrite(jn)) == \
        sqrt(2)*sin(z)/(sqrt(pi)*sqrt(z))
    assert expand_func(bessely(S.Half, z).rewrite(yn)) == \
        -sqrt(2)*cos(z)/(sqrt(pi)*sqrt(z))

    # XXX: teach sin/cos to work around arguments like
    # x*exp_polar(I*pi*n/2).  Then change besselsimp -> expand_func
    assert besselsimp(besselj(S.Half, z)) == sqrt(2)*sin(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(besselj(Rational(-1, 2), z)) == sqrt(2)*cos(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(besselj(Rational(5, 2), z)) == \
        -sqrt(2)*(z**2*sin(z) + 3*z*cos(z) - 3*sin(z))/(sqrt(pi)*z**Rational(5, 2))
    assert besselsimp(besselj(Rational(-5, 2), z)) == \
        -sqrt(2)*(z**2*cos(z) - 3*z*sin(z) - 3*cos(z))/(sqrt(pi)*z**Rational(5, 2))

    assert besselsimp(bessely(S.Half, z)) == \
        -(sqrt(2)*cos(z))/(sqrt(pi)*sqrt(z))
    assert besselsimp(bessely(Rational(-1, 2), z)) == sqrt(2)*sin(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(bessely(Rational(5, 2), z)) == \
        sqrt(2)*(z**2*cos(z) - 3*z*sin(z) - 3*cos(z))/(sqrt(pi)*z**Rational(5, 2))
    assert besselsimp(bessely(Rational(-5, 2), z)) == \
        -sqrt(2)*(z**2*sin(z) + 3*z*cos(z) - 3*sin(z))/(sqrt(pi)*z**Rational(5, 2))

    assert besselsimp(besseli(S.Half, z)) == sqrt(2)*sinh(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(besseli(Rational(-1, 2), z)) == \
        sqrt(2)*cosh(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(besseli(Rational(5, 2), z)) == \
        sqrt(2)*(z**2*sinh(z) - 3*z*cosh(z) + 3*sinh(z))/(sqrt(pi)*z**Rational(5, 2))
    assert besselsimp(besseli(Rational(-5, 2), z)) == \
        sqrt(2)*(z**2*cosh(z) - 3*z*sinh(z) + 3*cosh(z))/(sqrt(pi)*z**Rational(5, 2))

    assert besselsimp(besselk(S.Half, z)) == \
        besselsimp(besselk(Rational(-1, 2), z)) == sqrt(pi)*exp(-z)/(sqrt(2)*sqrt(z))
    assert besselsimp(besselk(Rational(5, 2), z)) == \
        besselsimp(besselk(Rational(-5, 2), z)) == \
        sqrt(2)*sqrt(pi)*(z**2 + 3*z + 3)*exp(-z)/(2*z**Rational(5, 2))

    n = Symbol('n', integer=True, positive=True)

    assert expand_func(besseli(n + 2, z)) == \
        besseli(n, z) + (-2*n - 2)*(-2*n*besseli(n, z)/z + besseli(n - 1, z))/z
    assert expand_func(besselj(n + 2, z)) == \
        -besselj(n, z) + (2*n + 2)*(2*n*besselj(n, z)/z - besselj(n - 1, z))/z
    assert expand_func(besselk(n + 2, z)) == \
        besselk(n, z) + (2*n + 2)*(2*n*besselk(n, z)/z + besselk(n - 1, z))/z
    assert expand_func(bessely(n + 2, z)) == \
        -bessely(n, z) + (2*n + 2)*(2*n*bessely(n, z)/z - bessely(n - 1, z))/z

    assert expand_func(besseli(n + S.Half, z).rewrite(jn)) == \
        (sqrt(2)*sqrt(z)*exp(-I*pi*(n + S.Half)/2) *
         exp_polar(I*pi/4)*jn(n, z*exp_polar(I*pi/2))/sqrt(pi))
    assert expand_func(besselj(n + S.Half, z).rewrite(jn)) == \
        sqrt(2)*sqrt(z)*jn(n, z)/sqrt(pi)

    r = Symbol('r', real=True)
    p = Symbol('p', positive=True)
    i = Symbol('i', integer=True)

    for besselx in [besselj, bessely, besseli, besselk]:
        assert besselx(i, p).is_extended_real is True
        assert besselx(i, x).is_extended_real is None
        assert besselx(x, z).is_extended_real is None

    for besselx in [besselj, besseli]:
        assert besselx(i, r).is_extended_real is True
    for besselx in [bessely, besselk]:
        assert besselx(i, r).is_extended_real is None

    for besselx in [besselj, bessely, besseli, besselk]:
        assert expand_func(besselx(oo, x)) == besselx(oo, x, evaluate=False)
        assert expand_func(besselx(-oo, x)) == besselx(-oo, x, evaluate=False)


# Quite varying time, but often really slow
@slow
def test_slow_expand():
    def check(eq, ans):
        return tn(eq, ans) and eq == ans

    rn = randcplx(a=1, b=0, d=0, c=2)

    for besselx in [besselj, bessely, besseli, besselk]:
        ri = S(2*randint(-11, 10) + 1) / 2  # half integer in [-21/2, 21/2]
        assert tn(besselsimp(besselx(ri, z)), besselx(ri, z))

    assert check(expand_func(besseli(rn, x)),
                 besseli(rn - 2, x) - 2*(rn - 1)*besseli(rn - 1, x)/x)
    assert check(expand_func(besseli(-rn, x)),
                 besseli(-rn + 2, x) + 2*(-rn + 1)*besseli(-rn + 1, x)/x)

    assert check(expand_func(besselj(rn, x)),
                 -besselj(rn - 2, x) + 2*(rn - 1)*besselj(rn - 1, x)/x)
    assert check(expand_func(besselj(-rn, x)),
                 -besselj(-rn + 2, x) + 2*(-rn + 1)*besselj(-rn + 1, x)/x)

    assert check(expand_func(besselk(rn, x)),
                 besselk(rn - 2, x) + 2*(rn - 1)*besselk(rn - 1, x)/x)
    assert check(expand_func(besselk(-rn, x)),
                 besselk(-rn + 2, x) - 2*(-rn + 1)*besselk(-rn + 1, x)/x)

    assert check(expand_func(bessely(rn, x)),
                 -bessely(rn - 2, x) + 2*(rn - 1)*bessely(rn - 1, x)/x)
    assert check(expand_func(bessely(-rn, x)),
                 -bessely(-rn + 2, x) + 2*(-rn + 1)*bessely(-rn + 1, x)/x)


def mjn(n, z):
    return expand_func(jn(n, z))


def myn(n, z):
    return expand_func(yn(n, z))


def test_jn():
    z = symbols("z")
    assert jn(0, 0) == 1
    assert jn(1, 0) == 0
    assert jn(-1, 0) == S.ComplexInfinity
    assert jn(z, 0) == jn(z, 0, evaluate=False)
    assert jn(0, oo) == 0
    assert jn(0, -oo) == 0

    assert mjn(0, z) == sin(z)/z
    assert mjn(1, z) == sin(z)/z**2 - cos(z)/z
    assert mjn(2, z) == (3/z**3 - 1/z)*sin(z) - (3/z**2) * cos(z)
    assert mjn(3, z) == (15/z**4 - 6/z**2)*sin(z) + (1/z - 15/z**3)*cos(z)
    assert mjn(4, z) == (1/z + 105/z**5 - 45/z**3)*sin(z) + \
        (-105/z**4 + 10/z**2)*cos(z)
    assert mjn(5, z) == (945/z**6 - 420/z**4 + 15/z**2)*sin(z) + \
        (-1/z - 945/z**5 + 105/z**3)*cos(z)
    assert mjn(6, z) == (-1/z + 10395/z**7 - 4725/z**5 + 210/z**3)*sin(z) + \
        (-10395/z**6 + 1260/z**4 - 21/z**2)*cos(z)

    assert expand_func(jn(n, z)) == jn(n, z)

    # SBFs not defined for complex-valued orders
    assert jn(2+3j, 5.2+0.3j).evalf() == jn(2+3j, 5.2+0.3j)

    assert eq([jn(2, 5.2+0.3j).evalf(10)],
              [0.09941975672 - 0.05452508024*I])


def test_yn():
    z = symbols("z")
    assert myn(0, z) == -cos(z)/z
    assert myn(1, z) == -cos(z)/z**2 - sin(z)/z
    assert myn(2, z) == -((3/z**3 - 1/z)*cos(z) + (3/z**2)*sin(z))
    assert expand_func(yn(n, z)) == yn(n, z)

    # SBFs not defined for complex-valued orders
    assert yn(2+3j, 5.2+0.3j).evalf() == yn(2+3j, 5.2+0.3j)

    assert eq([yn(2, 5.2+0.3j).evalf(10)],
              [0.185250342 + 0.01489557397*I])


def test_sympify_yn():
    assert S(15) in myn(3, pi).atoms()
    assert myn(3, pi) == 15/pi**4 - 6/pi**2


def eq(a, b, tol=1e-6):
    for u, v in zip(a, b):
        if not (abs(u - v) < tol):
            return False
    return True


def test_jn_zeros():
    assert eq(jn_zeros(0, 4), [3.141592, 6.283185, 9.424777, 12.566370])
    assert eq(jn_zeros(1, 4), [4.493409, 7.725251, 10.904121, 14.066193])
    assert eq(jn_zeros(2, 4), [5.763459, 9.095011, 12.322940, 15.514603])
    assert eq(jn_zeros(3, 4), [6.987932, 10.417118, 13.698023, 16.923621])
    assert eq(jn_zeros(4, 4), [8.182561, 11.704907, 15.039664, 18.301255])


def test_bessel_eval():
    n, m, k = Symbol('n', integer=True), Symbol('m'), Symbol('k', integer=True, zero=False)

    for f in [besselj, besseli]:
        assert f(0, 0) is S.One
        assert f(2.1, 0) is S.Zero
        assert f(-3, 0) is S.Zero
        assert f(-10.2, 0) is S.ComplexInfinity
        assert f(1 + 3*I, 0) is S.Zero
        assert f(-3 + I, 0) is S.ComplexInfinity
        assert f(-2*I, 0) is S.NaN
        assert f(n, 0) != S.One and f(n, 0) != S.Zero
        assert f(m, 0) != S.One and f(m, 0) != S.Zero
        assert f(k, 0) is S.Zero

    assert bessely(0, 0) is S.NegativeInfinity
    assert besselk(0, 0) is S.Infinity
    for f in [bessely, besselk]:
        assert f(1 + I, 0) is S.ComplexInfinity
        assert f(I, 0) is S.NaN

    for f in [besselj, bessely]:
        assert f(m, S.Infinity) is S.Zero
        assert f(m, S.NegativeInfinity) is S.Zero

    for f in [besseli, besselk]:
        assert f(m, I*S.Infinity) is S.Zero
        assert f(m, I*S.NegativeInfinity) is S.Zero

    for f in [besseli, besselk]:
        assert f(-4, z) == f(4, z)
        assert f(-3, z) == f(3, z)
        assert f(-n, z) == f(n, z)
        assert f(-m, z) != f(m, z)

    for f in [besselj, bessely]:
        assert f(-4, z) == f(4, z)
        assert f(-3, z) == -f(3, z)
        assert f(-n, z) == (-1)**n*f(n, z)
        assert f(-m, z) != (-1)**m*f(m, z)

    for f in [besselj, besseli]:
        assert f(m, -z) == (-z)**m*z**(-m)*f(m, z)

    assert besseli(2, -z) == besseli(2, z)
    assert besseli(3, -z) == -besseli(3, z)

    assert besselj(0, -z) == besselj(0, z)
    assert besselj(1, -z) == -besselj(1, z)

    assert besseli(0, I*z) == besselj(0, z)
    assert besseli(1, I*z) == I*besselj(1, z)
    assert besselj(3, I*z) == -I*besseli(3, z)


def test_bessel_nan():
    # FIXME: could have these return NaN; for now just fix infinite recursion
    for f in [besselj, bessely, besseli, besselk, hankel1, hankel2, yn, jn]:
        assert f(1, S.NaN) == f(1, S.NaN, evaluate=False)


def test_meromorphic():
    assert besselj(2, x).is_meromorphic(x, 1) == True
    assert besselj(2, x).is_meromorphic(x, 0) == True
    assert besselj(2, x).is_meromorphic(x, oo) == False
    assert besselj(S(2)/3, x).is_meromorphic(x, 1) == True
    assert besselj(S(2)/3, x).is_meromorphic(x, 0) == False
    assert besselj(S(2)/3, x).is_meromorphic(x, oo) == False
    assert besselj(x, 2*x).is_meromorphic(x, 2) == False
    assert besselk(0, x).is_meromorphic(x, 1) == True
    assert besselk(2, x).is_meromorphic(x, 0) == True
    assert besseli(0, x).is_meromorphic(x, 1) == True
    assert besseli(2, x).is_meromorphic(x, 0) == True
    assert bessely(0, x).is_meromorphic(x, 1) == True
    assert bessely(0, x).is_meromorphic(x, 0) == False
    assert bessely(2, x).is_meromorphic(x, 0) == True
    assert hankel1(3, x**2 + 2*x).is_meromorphic(x, 1) == True
    assert hankel1(0, x).is_meromorphic(x, 0) == False
    assert hankel2(11, 4).is_meromorphic(x, 5) == True
    assert hn1(6, 7*x**3 + 4).is_meromorphic(x, 7) == True
    assert hn2(3, 2*x).is_meromorphic(x, 9) == True
    assert jn(5, 2*x + 7).is_meromorphic(x, 4) == True
    assert yn(8, x**2 + 11).is_meromorphic(x, 6) == True


def test_conjugate():
    n = Symbol('n')
    z = Symbol('z', extended_real=False)
    x = Symbol('x', extended_real=True)
    y = Symbol('y', positive=True)
    t = Symbol('t', negative=True)

    for f in [besseli, besselj, besselk, bessely, hankel1, hankel2]:
        assert f(n, -1).conjugate() != f(conjugate(n), -1)
        assert f(n, x).conjugate() != f(conjugate(n), x)
        assert f(n, t).conjugate() != f(conjugate(n), t)

    rz = randcplx(b=0.5)

    for f in [besseli, besselj, besselk, bessely]:
        assert f(n, 1 + I).conjugate() == f(conjugate(n), 1 - I)
        assert f(n, 0).conjugate() == f(conjugate(n), 0)
        assert f(n, 1).conjugate() == f(conjugate(n), 1)
        assert f(n, z).conjugate() == f(conjugate(n), conjugate(z))
        assert f(n, y).conjugate() == f(conjugate(n), y)
        assert tn(f(n, rz).conjugate(), f(conjugate(n), conjugate(rz)))

    assert hankel1(n, 1 + I).conjugate() == hankel2(conjugate(n), 1 - I)
    assert hankel1(n, 0).conjugate() == hankel2(conjugate(n), 0)
    assert hankel1(n, 1).conjugate() == hankel2(conjugate(n), 1)
    assert hankel1(n, y).conjugate() == hankel2(conjugate(n), y)
    assert hankel1(n, z).conjugate() == hankel2(conjugate(n), conjugate(z))
    assert tn(hankel1(n, rz).conjugate(), hankel2(conjugate(n), conjugate(rz)))

    assert hankel2(n, 1 + I).conjugate() == hankel1(conjugate(n), 1 - I)
    assert hankel2(n, 0).conjugate() == hankel1(conjugate(n), 0)
    assert hankel2(n, 1).conjugate() == hankel1(conjugate(n), 1)
    assert hankel2(n, y).conjugate() == hankel1(conjugate(n), y)
    assert hankel2(n, z).conjugate() == hankel1(conjugate(n), conjugate(z))
    assert tn(hankel2(n, rz).conjugate(), hankel1(conjugate(n), conjugate(rz)))


def test_branching():
    assert besselj(polar_lift(k), x) == besselj(k, x)
    assert besseli(polar_lift(k), x) == besseli(k, x)

    n = Symbol('n', integer=True)
    assert besselj(n, exp_polar(2*pi*I)*x) == besselj(n, x)
    assert besselj(n, polar_lift(x)) == besselj(n, x)
    assert besseli(n, exp_polar(2*pi*I)*x) == besseli(n, x)
    assert besseli(n, polar_lift(x)) == besseli(n, x)

    def tn(func, s):
        from sympy.core.random import uniform
        c = uniform(1, 5)
        expr = func(s, c*exp_polar(I*pi)) - func(s, c*exp_polar(-I*pi))
        eps = 1e-15
        expr2 = func(s + eps, -c + eps*I) - func(s + eps, -c - eps*I)
        return abs(expr.n() - expr2.n()).n() < 1e-10

    nu = Symbol('nu')
    assert besselj(nu, exp_polar(2*pi*I)*x) == exp(2*pi*I*nu)*besselj(nu, x)
    assert besseli(nu, exp_polar(2*pi*I)*x) == exp(2*pi*I*nu)*besseli(nu, x)
    assert tn(besselj, 2)
    assert tn(besselj, pi)
    assert tn(besselj, I)
    assert tn(besseli, 2)
    assert tn(besseli, pi)
    assert tn(besseli, I)


def test_airy_base():
    z = Symbol('z')
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)

    assert conjugate(airyai(z)) == airyai(conjugate(z))
    assert airyai(x).is_extended_real

    assert airyai(x+I*y).as_real_imag() == (
            airyai(x - I*y)/2 + airyai(x + I*y)/2,
            I*(airyai(x - I*y) - airyai(x + I*y))/2)


def test_airyai():
    z = Symbol('z', real=False)
    t = Symbol('t', negative=True)
    p = Symbol('p', positive=True)

    assert isinstance(airyai(z), airyai)

    assert airyai(0) == 3**Rational(1, 3)/(3*gamma(Rational(2, 3)))
    assert airyai(oo) == 0
    assert airyai(-oo) == 0

    assert diff(airyai(z), z) == airyaiprime(z)

    assert series(airyai(z), z, 0, 3) == (
        3**Rational(5, 6)*gamma(Rational(1, 3))/(6*pi) - 3**Rational(1, 6)*z*gamma(Rational(2, 3))/(2*pi) + O(z**3))

    assert airyai(z).rewrite(hyper) == (
        -3**Rational(2, 3)*z*hyper((), (Rational(4, 3),), z**3/9)/(3*gamma(Rational(1, 3))) +
         3**Rational(1, 3)*hyper((), (Rational(2, 3),), z**3/9)/(3*gamma(Rational(2, 3))))

    assert isinstance(airyai(z).rewrite(besselj), airyai)
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)
    assert airyai(z).rewrite(besseli) == (
        -z*besseli(Rational(1, 3), 2*z**Rational(3, 2)/3)/(3*(z**Rational(3, 2))**Rational(1, 3)) +
         (z**Rational(3, 2))**Rational(1, 3)*besseli(Rational(-1, 3), 2*z**Rational(3, 2)/3)/3)
    assert airyai(p).rewrite(besseli) == (
        sqrt(p)*(besseli(Rational(-1, 3), 2*p**Rational(3, 2)/3) -
                 besseli(Rational(1, 3), 2*p**Rational(3, 2)/3))/3)

    assert expand_func(airyai(2*(3*z**5)**Rational(1, 3))) == (
        -sqrt(3)*(-1 + (z**5)**Rational(1, 3)/z**Rational(5, 3))*airybi(2*3**Rational(1, 3)*z**Rational(5, 3))/6 +
         (1 + (z**5)**Rational(1, 3)/z**Rational(5, 3))*airyai(2*3**Rational(1, 3)*z**Rational(5, 3))/2)


def test_airybi():
    z = Symbol('z', real=False)
    t = Symbol('t', negative=True)
    p = Symbol('p', positive=True)

    assert isinstance(airybi(z), airybi)

    assert airybi(0) == 3**Rational(5, 6)/(3*gamma(Rational(2, 3)))
    assert airybi(oo) is oo
    assert airybi(-oo) == 0

    assert diff(airybi(z), z) == airybiprime(z)

    assert series(airybi(z), z, 0, 3) == (
        3**Rational(1, 3)*gamma(Rational(1, 3))/(2*pi) + 3**Rational(2, 3)*z*gamma(Rational(2, 3))/(2*pi) + O(z**3))

    assert airybi(z).rewrite(hyper) == (
        3**Rational(1, 6)*z*hyper((), (Rational(4, 3),), z**3/9)/gamma(Rational(1, 3)) +
        3**Rational(5, 6)*hyper((), (Rational(2, 3),), z**3/9)/(3*gamma(Rational(2, 3))))

    assert isinstance(airybi(z).rewrite(besselj), airybi)
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)
    assert airybi(z).rewrite(besseli) == (
        sqrt(3)*(z*besseli(Rational(1, 3), 2*z**Rational(3, 2)/3)/(z**Rational(3, 2))**Rational(1, 3) +
                 (z**Rational(3, 2))**Rational(1, 3)*besseli(Rational(-1, 3), 2*z**Rational(3, 2)/3))/3)
    assert airybi(p).rewrite(besseli) == (
        sqrt(3)*sqrt(p)*(besseli(Rational(-1, 3), 2*p**Rational(3, 2)/3) +
                         besseli(Rational(1, 3), 2*p**Rational(3, 2)/3))/3)

    assert expand_func(airybi(2*(3*z**5)**Rational(1, 3))) == (
        sqrt(3)*(1 - (z**5)**Rational(1, 3)/z**Rational(5, 3))*airyai(2*3**Rational(1, 3)*z**Rational(5, 3))/2 +
        (1 + (z**5)**Rational(1, 3)/z**Rational(5, 3))*airybi(2*3**Rational(1, 3)*z**Rational(5, 3))/2)


def test_airyaiprime():
    z = Symbol('z', real=False)
    t = Symbol('t', negative=True)
    p = Symbol('p', positive=True)

    assert isinstance(airyaiprime(z), airyaiprime)

    assert airyaiprime(0) == -3**Rational(2, 3)/(3*gamma(Rational(1, 3)))
    assert airyaiprime(oo) == 0

    assert diff(airyaiprime(z), z) == z*airyai(z)

    assert series(airyaiprime(z), z, 0, 3) == (
        -3**Rational(2, 3)/(3*gamma(Rational(1, 3))) + 3**Rational(1, 3)*z**2/(6*gamma(Rational(2, 3))) + O(z**3))

    assert airyaiprime(z).rewrite(hyper) == (
        3**Rational(1, 3)*z**2*hyper((), (Rational(5, 3),), z**3/9)/(6*gamma(Rational(2, 3))) -
        3**Rational(2, 3)*hyper((), (Rational(1, 3),), z**3/9)/(3*gamma(Rational(1, 3))))

    assert isinstance(airyaiprime(z).rewrite(besselj), airyaiprime)
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)
    assert airyaiprime(z).rewrite(besseli) == (
        z**2*besseli(Rational(2, 3), 2*z**Rational(3, 2)/3)/(3*(z**Rational(3, 2))**Rational(2, 3)) -
        (z**Rational(3, 2))**Rational(2, 3)*besseli(Rational(-1, 3), 2*z**Rational(3, 2)/3)/3)
    assert airyaiprime(p).rewrite(besseli) == (
        p*(-besseli(Rational(-2, 3), 2*p**Rational(3, 2)/3) + besseli(Rational(2, 3), 2*p**Rational(3, 2)/3))/3)

    assert expand_func(airyaiprime(2*(3*z**5)**Rational(1, 3))) == (
        sqrt(3)*(z**Rational(5, 3)/(z**5)**Rational(1, 3) - 1)*airybiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/6 +
        (z**Rational(5, 3)/(z**5)**Rational(1, 3) + 1)*airyaiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/2)


def test_airybiprime():
    z = Symbol('z', real=False)
    t = Symbol('t', negative=True)
    p = Symbol('p', positive=True)

    assert isinstance(airybiprime(z), airybiprime)

    assert airybiprime(0) == 3**Rational(1, 6)/gamma(Rational(1, 3))
    assert airybiprime(oo) is oo
    assert airybiprime(-oo) == 0

    assert diff(airybiprime(z), z) == z*airybi(z)

    assert series(airybiprime(z), z, 0, 3) == (
        3**Rational(1, 6)/gamma(Rational(1, 3)) + 3**Rational(5, 6)*z**2/(6*gamma(Rational(2, 3))) + O(z**3))

    assert airybiprime(z).rewrite(hyper) == (
        3**Rational(5, 6)*z**2*hyper((), (Rational(5, 3),), z**3/9)/(6*gamma(Rational(2, 3))) +
        3**Rational(1, 6)*hyper((), (Rational(1, 3),), z**3/9)/gamma(Rational(1, 3)))

    assert isinstance(airybiprime(z).rewrite(besselj), airybiprime)
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)
    assert airybiprime(z).rewrite(besseli) == (
        sqrt(3)*(z**2*besseli(Rational(2, 3), 2*z**Rational(3, 2)/3)/(z**Rational(3, 2))**Rational(2, 3) +
                 (z**Rational(3, 2))**Rational(2, 3)*besseli(Rational(-2, 3), 2*z**Rational(3, 2)/3))/3)
    assert airybiprime(p).rewrite(besseli) == (
        sqrt(3)*p*(besseli(Rational(-2, 3), 2*p**Rational(3, 2)/3) + besseli(Rational(2, 3), 2*p**Rational(3, 2)/3))/3)

    assert expand_func(airybiprime(2*(3*z**5)**Rational(1, 3))) == (
        sqrt(3)*(z**Rational(5, 3)/(z**5)**Rational(1, 3) - 1)*airyaiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/2 +
        (z**Rational(5, 3)/(z**5)**Rational(1, 3) + 1)*airybiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/2)


def test_marcumq():
    m = Symbol('m')
    a = Symbol('a')
    b = Symbol('b')

    assert marcumq(0, 0, 0) == 0
    assert marcumq(m, 0, b) == uppergamma(m, b**2/2)/gamma(m)
    assert marcumq(2, 0, 5) == 27*exp(Rational(-25, 2))/2
    assert marcumq(0, a, 0) == 1 - exp(-a**2/2)
    assert marcumq(0, pi, 0) == 1 - exp(-pi**2/2)
    assert marcumq(1, a, a) == S.Half + exp(-a**2)*besseli(0, a**2)/2
    assert marcumq(2, a, a) == S.Half + exp(-a**2)*besseli(0, a**2)/2 + exp(-a**2)*besseli(1, a**2)

    assert diff(marcumq(1, a, 3), a) == a*(-marcumq(1, a, 3) + marcumq(2, a, 3))
    assert diff(marcumq(2, 3, b), b) == -b**2*exp(-b**2/2 - Rational(9, 2))*besseli(1, 3*b)/3

    x = Symbol('x')
    assert marcumq(2, 3, 4).rewrite(Integral, x=x) == \
           Integral(x**2*exp(-x**2/2 - Rational(9, 2))*besseli(1, 3*x), (x, 4, oo))/3
    assert eq([marcumq(5, -2, 3).rewrite(Integral).evalf(10)],
              [0.7905769565])

    k = Symbol('k')
    assert marcumq(-3, -5, -7).rewrite(Sum, k=k) == \
           exp(-37)*Sum((Rational(5, 7))**k*besseli(k, 35), (k, 4, oo))
    assert eq([marcumq(1, 3, 1).rewrite(Sum).evalf(10)],
              [0.9891705502])

    assert marcumq(1, a, a, evaluate=False).rewrite(besseli) == S.Half + exp(-a**2)*besseli(0, a**2)/2
    assert marcumq(2, a, a, evaluate=False).rewrite(besseli) == S.Half + exp(-a**2)*besseli(0, a**2)/2 + \
           exp(-a**2)*besseli(1, a**2)
    assert marcumq(3, a, a).rewrite(besseli) == (besseli(1, a**2) + besseli(2, a**2))*exp(-a**2) + \
           S.Half + exp(-a**2)*besseli(0, a**2)/2
    assert marcumq(5, 8, 8).rewrite(besseli) == exp(-64)*besseli(0, 64)/2 + \
           (besseli(4, 64) + besseli(3, 64) + besseli(2, 64) + besseli(1, 64))*exp(-64) + S.Half
    assert marcumq(m, a, a).rewrite(besseli) == marcumq(m, a, a)

    x = Symbol('x', integer=True)
    assert marcumq(x, a, a).rewrite(besseli) == marcumq(x, a, a)


def test_issue_26134():
    x = Symbol('x')
    assert marcumq(2, 3, 4).rewrite(Integral, x=x).dummy_eq(
        Integral(x**2*exp(-x**2/2 - Rational(9, 2))*besseli(1, 3*x), (x, 4, oo))/3)
