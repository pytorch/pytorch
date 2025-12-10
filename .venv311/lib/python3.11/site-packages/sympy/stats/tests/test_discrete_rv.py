from sympy.concrete.summations import Sum
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besseli
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.zeta_functions import zeta
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.exponential import exp
from sympy.logic.boolalg import Or
from sympy.sets.fancysets import Range
from sympy.stats import (P, E, variance, density, characteristic_function,
                         where, moment_generating_function, skewness, cdf,
                         kurtosis, coskewness)
from sympy.stats.drv_types import (PoissonDistribution, GeometricDistribution,
                                   FlorySchulz, Poisson, Geometric, Hermite, Logarithmic,
                                    NegativeBinomial, Skellam, YuleSimon, Zeta,
                                    DiscreteRV)
from sympy.testing.pytest import slow, nocache_fail, raises, skip
from sympy.stats.symbolic_probability import Expectation
from sympy.functions.combinatorial.factorials import FallingFactorial

x = Symbol('x')


def test_PoissonDistribution():
    l = 3
    p = PoissonDistribution(l)
    assert abs(p.cdf(10).evalf() - 1) < .001
    assert abs(p.cdf(10.4).evalf() - 1) < .001
    assert p.expectation(x, x) == l
    assert p.expectation(x**2, x) - p.expectation(x, x)**2 == l


def test_Poisson():
    l = 3
    x = Poisson('x', l)
    assert E(x) == l
    assert E(2*x) == 2*l
    assert variance(x) == l
    assert density(x) == PoissonDistribution(l)
    assert isinstance(E(x, evaluate=False), Expectation)
    assert isinstance(E(2*x, evaluate=False), Expectation)
    # issue 8248
    assert x.pspace.compute_expectation(1) == 1
    # issue 27344
    try:
        import numpy as np
    except ImportError:
        skip("numpy not installed")
    y = Poisson('y', np.float64(4.72544290380919e-11))
    assert E(y) == 4.72544290380919e-11
    y = Poisson('y', np.float64(4.725442903809197e-11))
    assert E(y) == 4.725442903809197e-11
    l2 = 5
    z = Poisson('z', l2)
    assert E(z) == l2
    assert E(FallingFactorial(z, 3)) == l2**3
    assert E(z**2) == l2 + l2**2


def test_FlorySchulz():
    a = Symbol("a")
    z = Symbol("z")
    x = FlorySchulz('x', a)
    assert E(x) == (2 - a)/a
    assert (variance(x) - 2*(1 - a)/a**2).simplify() == S(0)
    assert density(x)(z) == a**2*z*(1 - a)**(z - 1)


@slow
def test_GeometricDistribution():
    p = S.One / 5
    d = GeometricDistribution(p)
    assert d.expectation(x, x) == 1/p
    assert d.expectation(x**2, x) - d.expectation(x, x)**2 == (1-p)/p**2
    assert abs(d.cdf(20000).evalf() - 1) < .001
    assert abs(d.cdf(20000.8).evalf() - 1) < .001
    G = Geometric('G', p=S(1)/4)
    assert cdf(G)(S(7)/2) == P(G <= S(7)/2)

    X = Geometric('X', Rational(1, 5))
    Y = Geometric('Y', Rational(3, 10))
    assert coskewness(X, X + Y, X + 2*Y).simplify() == sqrt(230)*Rational(81, 1150)


def test_Hermite():
    a1 = Symbol("a1", positive=True)
    a2 = Symbol("a2", negative=True)
    raises(ValueError, lambda: Hermite("H", a1, a2))

    a1 = Symbol("a1", negative=True)
    a2 = Symbol("a2", positive=True)
    raises(ValueError, lambda: Hermite("H", a1, a2))

    a1 = Symbol("a1", positive=True)
    x = Symbol("x")
    H = Hermite("H", a1, a2)
    assert moment_generating_function(H)(x) == exp(a1*(exp(x) - 1)
                                            + a2*(exp(2*x) - 1))
    assert characteristic_function(H)(x) == exp(a1*(exp(I*x) - 1)
                                            + a2*(exp(2*I*x) - 1))
    assert E(H) == a1 + 2*a2

    H = Hermite("H", a1=5, a2=4)
    assert density(H)(2) == 33*exp(-9)/2
    assert E(H) == 13
    assert variance(H) == 21
    assert kurtosis(H) == Rational(464,147)
    assert skewness(H) == 37*sqrt(21)/441

def test_Logarithmic():
    p = S.Half
    x = Logarithmic('x', p)
    assert E(x) == -p / ((1 - p) * log(1 - p))
    assert variance(x) == -1/log(2)**2 + 2/log(2)
    assert E(2*x**2 + 3*x + 4) == 4 + 7 / log(2)
    assert isinstance(E(x, evaluate=False), Expectation)


@nocache_fail
def test_negative_binomial():
    r = 5
    p = S.One / 3
    x = NegativeBinomial('x', r, p)
    assert E(x) == r * (1 - p) / p
    # This hangs when run with the cache disabled:
    assert variance(x) == r * (1 - p) / p**2
    assert E(x**5 + 2*x + 3) == E(x**5) + 2*E(x) + 3 == Rational(796473, 1)
    assert isinstance(E(x, evaluate=False), Expectation)


def test_skellam():
    mu1 = Symbol('mu1')
    mu2 = Symbol('mu2')
    z = Symbol('z')
    X = Skellam('x', mu1, mu2)

    assert density(X)(z) == (mu1/mu2)**(z/2) * \
        exp(-mu1 - mu2)*besseli(z, 2*sqrt(mu1*mu2))
    assert skewness(X).expand() == mu1/(mu1*sqrt(mu1 + mu2) + mu2 *
                sqrt(mu1 + mu2)) - mu2/(mu1*sqrt(mu1 + mu2) + mu2*sqrt(mu1 + mu2))
    assert variance(X).expand() == mu1 + mu2
    assert E(X) == mu1 - mu2
    assert characteristic_function(X)(z) == exp(
        mu1*exp(I*z) - mu1 - mu2 + mu2*exp(-I*z))
    assert moment_generating_function(X)(z) == exp(
        mu1*exp(z) - mu1 - mu2 + mu2*exp(-z))


def test_yule_simon():
    from sympy.core.singleton import S
    rho = S(3)
    x = YuleSimon('x', rho)
    assert simplify(E(x)) == rho / (rho - 1)
    assert simplify(variance(x)) == rho**2 / ((rho - 1)**2 * (rho - 2))
    assert isinstance(E(x, evaluate=False), Expectation)
    # To test the cdf function
    assert cdf(x)(x) == Piecewise((-beta(floor(x), 4)*floor(x) + 1, x >= 1), (0, True))


def test_zeta():
    s = S(5)
    x = Zeta('x', s)
    assert E(x) == zeta(s-1) / zeta(s)
    assert simplify(variance(x)) == (
        zeta(s) * zeta(s-2) - zeta(s-1)**2) / zeta(s)**2


def test_discrete_probability():
    X = Geometric('X', Rational(1, 5))
    Y = Poisson('Y', 4)
    G = Geometric('e', x)
    assert P(Eq(X, 3)) == Rational(16, 125)
    assert P(X < 3) == Rational(9, 25)
    assert P(X > 3) == Rational(64, 125)
    assert P(X >= 3) == Rational(16, 25)
    assert P(X <= 3) == Rational(61, 125)
    assert P(Ne(X, 3)) == Rational(109, 125)
    assert P(Eq(Y, 3)) == 32*exp(-4)/3
    assert P(Y < 3) == 13*exp(-4)
    assert P(Y > 3).equals(32*(Rational(-71, 32) + 3*exp(4)/32)*exp(-4)/3)
    assert P(Y >= 3).equals(32*(Rational(-39, 32) + 3*exp(4)/32)*exp(-4)/3)
    assert P(Y <= 3) == 71*exp(-4)/3
    assert P(Ne(Y, 3)).equals(
        13*exp(-4) + 32*(Rational(-71, 32) + 3*exp(4)/32)*exp(-4)/3)
    assert P(X < S.Infinity) is S.One
    assert P(X > S.Infinity) is S.Zero
    assert P(G < 3) == x*(2-x)
    assert P(Eq(G, 3)) == x*(-x + 1)**2


def test_DiscreteRV():
    p = S(1)/2
    x = Symbol('x', integer=True, positive=True)
    pdf = p*(1 - p)**(x - 1) # pdf of Geometric Distribution
    D = DiscreteRV(x, pdf, set=S.Naturals, check=True)
    assert E(D) == E(Geometric('G', S(1)/2)) == 2
    assert P(D > 3) == S(1)/8
    assert D.pspace.domain.set == S.Naturals
    raises(ValueError, lambda: DiscreteRV(x, x, FiniteSet(*range(4)), check=True))

    # purposeful invalid pmf but it should not raise since check=False
    # see test_drv_types.test_ContinuousRV for explanation
    X = DiscreteRV(x, 1/x, S.Naturals)
    assert P(X < 2) == 1
    assert E(X) == oo

def test_precomputed_characteristic_functions():
    import mpmath

    def test_cf(dist, support_lower_limit, support_upper_limit):
        pdf = density(dist)
        t = S('t')
        x = S('x')

        # first function is the hardcoded CF of the distribution
        cf1 = lambdify([t], characteristic_function(dist)(t), 'mpmath')

        # second function is the Fourier transform of the density function
        f = lambdify([x, t], pdf(x)*exp(I*x*t), 'mpmath')
        cf2 = lambda t: mpmath.nsum(lambda x: f(x, t), [
            support_lower_limit, support_upper_limit], maxdegree=10)

        # compare the two functions at various points
        for test_point in [2, 5, 8, 11]:
            n1 = cf1(test_point)
            n2 = cf2(test_point)

            assert abs(re(n1) - re(n2)) < 1e-12
            assert abs(im(n1) - im(n2)) < 1e-12

    test_cf(Geometric('g', Rational(1, 3)), 1, mpmath.inf)
    test_cf(Logarithmic('l', Rational(1, 5)), 1, mpmath.inf)
    test_cf(NegativeBinomial('n', 5, Rational(1, 7)), 0, mpmath.inf)
    test_cf(Poisson('p', 5), 0, mpmath.inf)
    test_cf(YuleSimon('y', 5), 1, mpmath.inf)
    test_cf(Zeta('z', 5), 1, mpmath.inf)


def test_moment_generating_functions():
    t = S('t')

    geometric_mgf = moment_generating_function(Geometric('g', S.Half))(t)
    assert geometric_mgf.diff(t).subs(t, 0) == 2

    logarithmic_mgf = moment_generating_function(Logarithmic('l', S.Half))(t)
    assert logarithmic_mgf.diff(t).subs(t, 0) == 1/log(2)

    negative_binomial_mgf = moment_generating_function(
        NegativeBinomial('n', 5, Rational(1, 3)))(t)
    assert negative_binomial_mgf.diff(t).subs(t, 0) == Rational(10, 1)

    poisson_mgf = moment_generating_function(Poisson('p', 5))(t)
    assert poisson_mgf.diff(t).subs(t, 0) == 5

    skellam_mgf = moment_generating_function(Skellam('s', 1, 1))(t)
    assert skellam_mgf.diff(t).subs(
        t, 2) == (-exp(-2) + exp(2))*exp(-2 + exp(-2) + exp(2))

    yule_simon_mgf = moment_generating_function(YuleSimon('y', 3))(t)
    assert simplify(yule_simon_mgf.diff(t).subs(t, 0)) == Rational(3, 2)

    zeta_mgf = moment_generating_function(Zeta('z', 5))(t)
    assert zeta_mgf.diff(t).subs(t, 0) == pi**4/(90*zeta(5))


def test_Or():
    X = Geometric('X', S.Half)
    assert P(Or(X < 3, X > 4)) == Rational(13, 16)
    assert P(Or(X > 2, X > 1)) == P(X > 1)
    assert P(Or(X >= 3, X < 3)) == 1


def test_where():
    X = Geometric('X', Rational(1, 5))
    Y = Poisson('Y', 4)
    assert where(X**2 > 4).set == Range(3, S.Infinity, 1)
    assert where(X**2 >= 4).set == Range(2, S.Infinity, 1)
    assert where(Y**2 < 9).set == Range(0, 3, 1)
    assert where(Y**2 <= 9).set == Range(0, 4, 1)


def test_conditional():
    X = Geometric('X', Rational(2, 3))
    Y = Poisson('Y', 3)
    assert P(X > 2, X > 3) == 1
    assert P(X > 3, X > 2) == Rational(1, 3)
    assert P(Y > 2, Y < 2) == 0
    assert P(Eq(Y, 3), Y >= 0) == 9*exp(-3)/2
    assert P(Eq(Y, 3), Eq(Y, 2)) == 0
    assert P(X < 2, Eq(X, 2)) == 0
    assert P(X > 2, Eq(X, 3)) == 1


def test_product_spaces():
    X1 = Geometric('X1', S.Half)
    X2 = Geometric('X2', Rational(1, 3))
    assert str(P(X1 + X2 < 3).rewrite(Sum)) == (
        "Sum(Piecewise((1/(4*2**n), n >= -1), (0, True)), (n, -oo, -1))/3")
    assert str(P(X1 + X2 > 3).rewrite(Sum)) == (
        'Sum(Piecewise((2**(X2 - n - 2)*(2/3)**(X2 - 1)/6, '
        'X2 - n <= 2), (0, True)), (X2, 1, oo), (n, 1, oo))')
    assert P(Eq(X1 + X2, 3)) == Rational(1, 12)
