from sympy.concrete.summations import Sum
from sympy.core.function import expand_func
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (Abs, polar_lift)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.zeta_functions import (dirichlet_eta, lerchphi, polylog, riemann_xi, stieltjes, zeta)
from sympy.series.order import O
from sympy.core.function import ArgumentIndexError
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.testing.pytest import raises
from sympy.core.random import (test_derivative_numerically as td,
                      random_complex_number as randcplx, verify_numerically)

x = Symbol('x')
a = Symbol('a')
b = Symbol('b', negative=True)
z = Symbol('z')
s = Symbol('s')


def test_zeta_eval():

    assert zeta(nan) is nan
    assert zeta(x, nan) is nan

    assert zeta(0) == Rational(-1, 2)
    assert zeta(0, x) == S.Half - x
    assert zeta(0, b) == S.Half - b

    assert zeta(1) is zoo
    assert zeta(1, 2) is zoo
    assert zeta(1, -7) is zoo
    assert zeta(1, x) is zoo

    assert zeta(2, 1) == pi**2/6
    assert zeta(3, 1) == zeta(3)

    assert zeta(2) == pi**2/6
    assert zeta(4) == pi**4/90
    assert zeta(6) == pi**6/945

    assert zeta(4, 3) == pi**4/90 - Rational(17, 16)
    assert zeta(7, 4) == zeta(7) - Rational(282251, 279936)
    assert zeta(S.Half, 2).func == zeta
    assert expand_func(zeta(S.Half, 2)) == zeta(S.Half) - 1
    assert zeta(x, 3).func == zeta
    assert expand_func(zeta(x, 3)) == zeta(x) - 1 - 1/2**x

    assert zeta(2, 0) is nan
    assert zeta(3, -1) is nan
    assert zeta(4, -2) is nan

    assert zeta(oo) == 1

    assert zeta(-1) == Rational(-1, 12)
    assert zeta(-2) == 0
    assert zeta(-3) == Rational(1, 120)
    assert zeta(-4) == 0
    assert zeta(-5) == Rational(-1, 252)

    assert zeta(-1, 3) == Rational(-37, 12)
    assert zeta(-1, 7) == Rational(-253, 12)
    assert zeta(-1, -4) == Rational(-121, 12)
    assert zeta(-1, -9) == Rational(-541, 12)

    assert zeta(-4, 3) == -17
    assert zeta(-4, -8) == 8772

    assert zeta(0, 1) == Rational(-1, 2)
    assert zeta(0, -1) == Rational(3, 2)

    assert zeta(0, 2) == Rational(-3, 2)
    assert zeta(0, -2) == Rational(5, 2)

    assert zeta(
        3).evalf(20).epsilon_eq(Float("1.2020569031595942854", 20), 1e-19)


def test_zeta_series():
    assert zeta(x, a).series(a, z, 2) == \
        zeta(x, z) - x*(a-z)*zeta(x+1, z) + O((a-z)**2, (a, z))


def test_dirichlet_eta_eval():
    assert dirichlet_eta(0) == S.Half
    assert dirichlet_eta(-1) == Rational(1, 4)
    assert dirichlet_eta(1) == log(2)
    assert dirichlet_eta(1, S.Half).simplify() == pi/2
    assert dirichlet_eta(1, 2) == 1 - log(2)
    assert dirichlet_eta(2) == pi**2/12
    assert dirichlet_eta(4) == pi**4*Rational(7, 720)
    assert str(dirichlet_eta(I).evalf(n=10)) == '0.5325931818 + 0.2293848577*I'
    assert str(dirichlet_eta(I, I).evalf(n=10)) == '3.462349253 + 0.220285771*I'


def test_riemann_xi_eval():
    assert riemann_xi(2) == pi/6
    assert riemann_xi(0) == Rational(1, 2)
    assert riemann_xi(1) == Rational(1, 2)
    assert riemann_xi(3).rewrite(zeta) == 3*zeta(3)/(2*pi)
    assert riemann_xi(4) == pi**2/15


def test_rewriting():
    from sympy.functions.elementary.piecewise import Piecewise
    assert isinstance(dirichlet_eta(x).rewrite(zeta), Piecewise)
    assert isinstance(dirichlet_eta(x).rewrite(genocchi), Piecewise)
    assert zeta(x).rewrite(dirichlet_eta) == dirichlet_eta(x)/(1 - 2**(1 - x))
    assert zeta(x).rewrite(dirichlet_eta, a=2) == zeta(x)
    assert verify_numerically(dirichlet_eta(x), dirichlet_eta(x).rewrite(zeta), x)
    assert verify_numerically(dirichlet_eta(x), dirichlet_eta(x).rewrite(genocchi), x)
    assert verify_numerically(zeta(x), zeta(x).rewrite(dirichlet_eta), x)

    assert zeta(x, a).rewrite(lerchphi) == lerchphi(1, x, a)
    assert polylog(s, z).rewrite(lerchphi) == lerchphi(z, s, 1)*z

    assert lerchphi(1, x, a).rewrite(zeta) == zeta(x, a)
    assert z*lerchphi(z, s, 1).rewrite(polylog) == polylog(s, z)


def test_derivatives():
    from sympy.core.function import Derivative
    assert zeta(x, a).diff(x) == Derivative(zeta(x, a), x)
    assert zeta(x, a).diff(a) == -x*zeta(x + 1, a)
    assert lerchphi(
        z, s, a).diff(z) == (lerchphi(z, s - 1, a) - a*lerchphi(z, s, a))/z
    assert lerchphi(z, s, a).diff(a) == -s*lerchphi(z, s + 1, a)
    assert polylog(s, z).diff(z) == polylog(s - 1, z)/z

    b = randcplx()
    c = randcplx()
    assert td(zeta(b, x), x)
    assert td(polylog(b, z), z)
    assert td(lerchphi(c, b, x), x)
    assert td(lerchphi(x, b, c), x)
    raises(ArgumentIndexError, lambda: lerchphi(c, b, x).fdiff(2))
    raises(ArgumentIndexError, lambda: lerchphi(c, b, x).fdiff(4))
    raises(ArgumentIndexError, lambda: polylog(b, z).fdiff(1))
    raises(ArgumentIndexError, lambda: polylog(b, z).fdiff(3))


def myexpand(func, target):
    expanded = expand_func(func)
    if target is not None:
        return expanded == target
    if expanded == func:  # it didn't expand
        return False

    # check to see that the expanded and original evaluate to the same value
    subs = {}
    for a in func.free_symbols:
        subs[a] = randcplx()
    return abs(func.subs(subs).n()
               - expanded.replace(exp_polar, exp).subs(subs).n()) < 1e-10


def test_polylog_expansion():
    assert polylog(s, 0) == 0
    assert polylog(s, 1) == zeta(s)
    assert polylog(s, -1) == -dirichlet_eta(s)
    assert polylog(s, exp_polar(I*pi*Rational(4, 3))) == polylog(s, exp(I*pi*Rational(4, 3)))
    assert polylog(s, exp_polar(I*pi)/3) == polylog(s, exp(I*pi)/3)

    assert myexpand(polylog(1, z), -log(1 - z))
    assert myexpand(polylog(0, z), z/(1 - z))
    assert myexpand(polylog(-1, z), z/(1 - z)**2)
    assert ((1-z)**3 * expand_func(polylog(-2, z))).simplify() == z*(1 + z)
    assert myexpand(polylog(-5, z), None)


def test_polylog_series():
    assert polylog(1, z).series(z, n=5) == z + z**2/2 + z**3/3 + z**4/4 + O(z**5)
    assert polylog(1, sqrt(z)).series(z, n=3) == z/2 + z**2/4 + sqrt(z)\
        + z**(S(3)/2)/3 + z**(S(5)/2)/5 + O(z**3)

    # https://github.com/sympy/sympy/issues/9497
    assert polylog(S(3)/2, -z).series(z, 0, 5) == -z + sqrt(2)*z**2/4\
        - sqrt(3)*z**3/9 + z**4/8 + O(z**5)


def test_issue_8404():
    i = Symbol('i', integer=True)
    assert Abs(Sum(1/(3*i + 1)**2, (i, 0, S.Infinity)).doit().n(4)
        - 1.122) < 0.001


def test_polylog_values():
    assert polylog(2, 2) == pi**2/4 - I*pi*log(2)
    assert polylog(2, S.Half) == pi**2/12 - log(2)**2/2
    for z in [S.Half, 2, (sqrt(5)-1)/2, -(sqrt(5)-1)/2, -(sqrt(5)+1)/2, (3-sqrt(5))/2]:
        assert Abs(polylog(2, z).evalf() - polylog(2, z, evaluate=False).evalf()) < 1e-15
    z = Symbol("z")
    for s in [-1, 0]:
        for _ in range(10):
            assert verify_numerically(polylog(s, z), polylog(s, z, evaluate=False),
                                      z, a=-3, b=-2, c=S.Half, d=2)
            assert verify_numerically(polylog(s, z), polylog(s, z, evaluate=False),
                                      z, a=2, b=-2, c=5, d=2)

    from sympy.integrals.integrals import Integral
    assert polylog(0, Integral(1, (x, 0, 1))) == -S.Half


def test_lerchphi_expansion():
    assert myexpand(lerchphi(1, s, a), zeta(s, a))
    assert myexpand(lerchphi(z, s, 1), polylog(s, z)/z)

    # direct summation
    assert myexpand(lerchphi(z, -1, a), a/(1 - z) + z/(1 - z)**2)
    assert myexpand(lerchphi(z, -3, a), None)
    # polylog reduction
    assert myexpand(lerchphi(z, s, S.Half),
                    2**(s - 1)*(polylog(s, sqrt(z))/sqrt(z)
                              - polylog(s, polar_lift(-1)*sqrt(z))/sqrt(z)))
    assert myexpand(lerchphi(z, s, 2), -1/z + polylog(s, z)/z**2)
    assert myexpand(lerchphi(z, s, Rational(3, 2)), None)
    assert myexpand(lerchphi(z, s, Rational(7, 3)), None)
    assert myexpand(lerchphi(z, s, Rational(-1, 3)), None)
    assert myexpand(lerchphi(z, s, Rational(-5, 2)), None)

    # hurwitz zeta reduction
    assert myexpand(lerchphi(-1, s, a),
                    2**(-s)*zeta(s, a/2) - 2**(-s)*zeta(s, (a + 1)/2))
    assert myexpand(lerchphi(I, s, a), None)
    assert myexpand(lerchphi(-I, s, a), None)
    assert myexpand(lerchphi(exp(I*pi*Rational(2, 5)), s, a), None)


def test_stieltjes():
    assert isinstance(stieltjes(x), stieltjes)
    assert isinstance(stieltjes(x, a), stieltjes)

    # Zero'th constant EulerGamma
    assert stieltjes(0) == S.EulerGamma
    assert stieltjes(0, 1) == S.EulerGamma

    # Not defined
    assert stieltjes(nan) is nan
    assert stieltjes(0, nan) is nan
    assert stieltjes(-1) is S.ComplexInfinity
    assert stieltjes(1.5) is S.ComplexInfinity
    assert stieltjes(z, 0) is S.ComplexInfinity
    assert stieltjes(z, -1) is S.ComplexInfinity


def test_stieltjes_evalf():
    assert abs(stieltjes(0).evalf() - 0.577215664) < 1E-9
    assert abs(stieltjes(0, 0.5).evalf() - 1.963510026) < 1E-9
    assert abs(stieltjes(1, 2).evalf() + 0.072815845) < 1E-9


def test_issue_10475():
    a = Symbol('a', extended_real=True)
    b = Symbol('b', extended_positive=True)
    s = Symbol('s', zero=False)

    assert zeta(2 + I).is_finite
    assert zeta(1).is_finite is False
    assert zeta(x).is_finite is None
    assert zeta(x + I).is_finite is None
    assert zeta(a).is_finite is None
    assert zeta(b).is_finite is None
    assert zeta(-b).is_finite is True
    assert zeta(b**2 - 2*b + 1).is_finite is None
    assert zeta(a + I).is_finite is True
    assert zeta(b + 1).is_finite is True
    assert zeta(s + 1).is_finite is True


def test_issue_14177():
    n = Symbol('n', nonnegative=True, integer=True)

    assert zeta(-n).rewrite(bernoulli) == bernoulli(n+1) / (-n-1)
    assert zeta(-n, a).rewrite(bernoulli) == bernoulli(n+1, a) / (-n-1)
    z2n = -(2*I*pi)**(2*n)*bernoulli(2*n) / (2*factorial(2*n))
    assert zeta(2*n).rewrite(bernoulli) == z2n
    assert expand_func(zeta(s, n+1)) == zeta(s) - harmonic(n, s)
    assert expand_func(zeta(-b, -n)) is nan
    assert expand_func(zeta(-b, n)) == zeta(-b, n)

    n = Symbol('n')

    assert zeta(2*n) == zeta(2*n) # As sign of z (= 2*n) is not determined
