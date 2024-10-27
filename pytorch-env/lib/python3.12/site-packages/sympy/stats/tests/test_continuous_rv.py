from sympy.concrete.summations import Sum
from sympy.core.function import (Lambda, diff, expand_func)
from sympy.core.mul import Mul
from sympy.core import EulerGamma
from sympy.core.numbers import (E as e, I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (asin, atan, cos, sin, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk)
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import (erf, erfc, erfi, expint)
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.sets.sets import Interval
from sympy.simplify.simplify import simplify
from sympy.utilities.lambdify import lambdify
from sympy.functions.special.error_functions import erfinv
from sympy.functions.special.hyper import meijerg
from sympy.sets.sets import FiniteSet, Complement, Intersection
from sympy.stats import (P, E, where, density, variance, covariance, skewness, kurtosis, median,
                         given, pspace, cdf, characteristic_function, moment_generating_function,
                         ContinuousRV, Arcsin, Benini, Beta, BetaNoncentral, BetaPrime,
                         Cauchy, Chi, ChiSquared, ChiNoncentral, Dagum, Davis, Erlang, ExGaussian,
                         Exponential, ExponentialPower, FDistribution, FisherZ, Frechet, Gamma,
                         GammaInverse, Gompertz, Gumbel, Kumaraswamy, Laplace, Levy, Logistic, LogCauchy,
                         LogLogistic, LogitNormal, LogNormal, Maxwell, Moyal, Nakagami, Normal, GaussianInverse,
                         Pareto, PowerFunction, QuadraticU, RaisedCosine, Rayleigh, Reciprocal, ShiftedGompertz, StudentT,
                         Trapezoidal, Triangular, Uniform, UniformSum, VonMises, Weibull, coskewness,
                         WignerSemicircle, Wald, correlation, moment, cmoment, smoment, quantile,
                         Lomax, BoundedPareto)

from sympy.stats.crv_types import NormalDistribution, ExponentialDistribution, ContinuousDistributionHandmade
from sympy.stats.joint_rv_types import MultivariateLaplaceDistribution, MultivariateNormalDistribution
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDomain
from sympy.stats.compound_rv import CompoundPSpace
from sympy.stats.symbolic_probability import Probability
from sympy.testing.pytest import raises, XFAIL, slow, ignore_warnings
from sympy.core.random import verify_numerically as tn

oo = S.Infinity

x, y, z = map(Symbol, 'xyz')

def test_single_normal():
    mu = Symbol('mu', real=True)
    sigma = Symbol('sigma', positive=True)
    X = Normal('x', 0, 1)
    Y = X*sigma + mu

    assert E(Y) == mu
    assert variance(Y) == sigma**2
    pdf = density(Y)
    x = Symbol('x', real=True)
    assert (pdf(x) ==
            2**S.Half*exp(-(x - mu)**2/(2*sigma**2))/(2*pi**S.Half*sigma))

    assert P(X**2 < 1) == erf(2**S.Half/2)
    ans = quantile(Y)(x)
    assert ans == Complement(Intersection(FiniteSet(
        sqrt(2)*sigma*(sqrt(2)*mu/(2*sigma)+ erfinv(2*x - 1))),
        Interval(-oo, oo)), FiniteSet(mu))
    assert E(X, Eq(X, mu)) == mu

    assert median(X) == FiniteSet(0)
    # issue 8248
    assert X.pspace.compute_expectation(1).doit() == 1


def test_conditional_1d():
    X = Normal('x', 0, 1)
    Y = given(X, X >= 0)
    z = Symbol('z')

    assert density(Y)(z) == 2 * density(X)(z)

    assert Y.pspace.domain.set == Interval(0, oo)
    assert E(Y) == sqrt(2) / sqrt(pi)

    assert E(X**2) == E(Y**2)


def test_ContinuousDomain():
    X = Normal('x', 0, 1)
    assert where(X**2 <= 1).set == Interval(-1, 1)
    assert where(X**2 <= 1).symbol == X.symbol
    assert where(And(X**2 <= 1, X >= 0)).set == Interval(0, 1)
    raises(ValueError, lambda: where(sin(X) > 1))

    Y = given(X, X >= 0)

    assert Y.pspace.domain.set == Interval(0, oo)


def test_multiple_normal():
    X, Y = Normal('x', 0, 1), Normal('y', 0, 1)
    p = Symbol("p", positive=True)

    assert E(X + Y) == 0
    assert variance(X + Y) == 2
    assert variance(X + X) == 4
    assert covariance(X, Y) == 0
    assert covariance(2*X + Y, -X) == -2*variance(X)
    assert skewness(X) == 0
    assert skewness(X + Y) == 0
    assert kurtosis(X) == 3
    assert kurtosis(X+Y) == 3
    assert correlation(X, Y) == 0
    assert correlation(X, X + Y) == correlation(X, X - Y)
    assert moment(X, 2) == 1
    assert cmoment(X, 3) == 0
    assert moment(X + Y, 4) == 12
    assert cmoment(X, 2) == variance(X)
    assert smoment(X*X, 2) == 1
    assert smoment(X + Y, 3) == skewness(X + Y)
    assert smoment(X + Y, 4) == kurtosis(X + Y)
    assert E(X, Eq(X + Y, 0)) == 0
    assert variance(X, Eq(X + Y, 0)) == S.Half
    assert quantile(X)(p) == sqrt(2)*erfinv(2*p - S.One)


def test_symbolic():
    mu1, mu2 = symbols('mu1 mu2', real=True)
    s1, s2 = symbols('sigma1 sigma2', positive=True)
    rate = Symbol('lambda', positive=True)
    X = Normal('x', mu1, s1)
    Y = Normal('y', mu2, s2)
    Z = Exponential('z', rate)
    a, b, c = symbols('a b c', real=True)

    assert E(X) == mu1
    assert E(X + Y) == mu1 + mu2
    assert E(a*X + b) == a*E(X) + b
    assert variance(X) == s1**2
    assert variance(X + a*Y + b) == variance(X) + a**2*variance(Y)

    assert E(Z) == 1/rate
    assert E(a*Z + b) == a*E(Z) + b
    assert E(X + a*Z + b) == mu1 + a/rate + b
    assert median(X) == FiniteSet(mu1)


def test_cdf():
    X = Normal('x', 0, 1)

    d = cdf(X)
    assert P(X < 1) == d(1).rewrite(erfc)
    assert d(0) == S.Half

    d = cdf(X, X > 0)  # given X>0
    assert d(0) == 0

    Y = Exponential('y', 10)
    d = cdf(Y)
    assert d(-5) == 0
    assert P(Y > 3) == 1 - d(3)

    raises(ValueError, lambda: cdf(X + Y))

    Z = Exponential('z', 1)
    f = cdf(Z)
    assert f(z) == Piecewise((1 - exp(-z), z >= 0), (0, True))


def test_characteristic_function():
    X = Uniform('x', 0, 1)

    cf = characteristic_function(X)
    assert cf(1) == -I*(-1 + exp(I))

    Y = Normal('y', 1, 1)
    cf = characteristic_function(Y)
    assert cf(0) == 1
    assert cf(1) == exp(I - S.Half)

    Z = Exponential('z', 5)
    cf = characteristic_function(Z)
    assert cf(0) == 1
    assert cf(1).expand() == Rational(25, 26) + I*5/26

    X = GaussianInverse('x', 1, 1)
    cf = characteristic_function(X)
    assert cf(0) == 1
    assert cf(1) == exp(1 - sqrt(1 - 2*I))

    X = ExGaussian('x', 0, 1, 1)
    cf = characteristic_function(X)
    assert cf(0) == 1
    assert cf(1) == (1 + I)*exp(Rational(-1, 2))/2

    L = Levy('x', 0, 1)
    cf = characteristic_function(L)
    assert cf(0) == 1
    assert cf(1) == exp(-sqrt(2)*sqrt(-I))


def test_moment_generating_function():
    t = symbols('t', positive=True)

    # Symbolic tests
    a, b, c = symbols('a b c')

    mgf = moment_generating_function(Beta('x', a, b))(t)
    assert mgf == hyper((a,), (a + b,), t)

    mgf = moment_generating_function(Chi('x', a))(t)
    assert mgf == sqrt(2)*t*gamma(a/2 + S.Half)*\
        hyper((a/2 + S.Half,), (Rational(3, 2),), t**2/2)/gamma(a/2) +\
        hyper((a/2,), (S.Half,), t**2/2)

    mgf = moment_generating_function(ChiSquared('x', a))(t)
    assert mgf == (1 - 2*t)**(-a/2)

    mgf = moment_generating_function(Erlang('x', a, b))(t)
    assert mgf == (1 - t/b)**(-a)

    mgf = moment_generating_function(ExGaussian("x", a, b, c))(t)
    assert mgf == exp(a*t + b**2*t**2/2)/(1 - t/c)

    mgf = moment_generating_function(Exponential('x', a))(t)
    assert mgf == a/(a - t)

    mgf = moment_generating_function(Gamma('x', a, b))(t)
    assert mgf == (-b*t + 1)**(-a)

    mgf = moment_generating_function(Gumbel('x', a, b))(t)
    assert mgf == exp(b*t)*gamma(-a*t + 1)

    mgf = moment_generating_function(Gompertz('x', a, b))(t)
    assert mgf == b*exp(b)*expint(t/a, b)

    mgf = moment_generating_function(Laplace('x', a, b))(t)
    assert mgf == exp(a*t)/(-b**2*t**2 + 1)

    mgf = moment_generating_function(Logistic('x', a, b))(t)
    assert mgf == exp(a*t)*beta(-b*t + 1, b*t + 1)

    mgf = moment_generating_function(Normal('x', a, b))(t)
    assert mgf == exp(a*t + b**2*t**2/2)

    mgf = moment_generating_function(Pareto('x', a, b))(t)
    assert mgf == b*(-a*t)**b*uppergamma(-b, -a*t)

    mgf = moment_generating_function(QuadraticU('x', a, b))(t)
    assert str(mgf) == ("(3*(t*(-4*b + (a + b)**2) + 4)*exp(b*t) - "
    "3*(t*(a**2 + 2*a*(b - 2) + b**2) + 4)*exp(a*t))/(t**2*(a - b)**3)")

    mgf = moment_generating_function(RaisedCosine('x', a, b))(t)
    assert mgf == pi**2*exp(a*t)*sinh(b*t)/(b*t*(b**2*t**2 + pi**2))

    mgf = moment_generating_function(Rayleigh('x', a))(t)
    assert mgf == sqrt(2)*sqrt(pi)*a*t*(erf(sqrt(2)*a*t/2) + 1)\
        *exp(a**2*t**2/2)/2 + 1

    mgf = moment_generating_function(Triangular('x', a, b, c))(t)
    assert str(mgf) == ("(-2*(-a + b)*exp(c*t) + 2*(-a + c)*exp(b*t) + "
    "2*(b - c)*exp(a*t))/(t**2*(-a + b)*(-a + c)*(b - c))")

    mgf = moment_generating_function(Uniform('x', a, b))(t)
    assert mgf == (-exp(a*t) + exp(b*t))/(t*(-a + b))

    mgf = moment_generating_function(UniformSum('x', a))(t)
    assert mgf == ((exp(t) - 1)/t)**a

    mgf = moment_generating_function(WignerSemicircle('x', a))(t)
    assert mgf == 2*besseli(1, a*t)/(a*t)

    # Numeric tests

    mgf = moment_generating_function(Beta('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 1) == hyper((2,), (3,), 1)/2

    mgf = moment_generating_function(Chi('x', 1))(t)
    assert mgf.diff(t).subs(t, 1) == sqrt(2)*hyper((1,), (Rational(3, 2),), S.Half
    )/sqrt(pi) + hyper((Rational(3, 2),), (Rational(3, 2),), S.Half) + 2*sqrt(2)*hyper((2,),
    (Rational(5, 2),), S.Half)/(3*sqrt(pi))

    mgf = moment_generating_function(ChiSquared('x', 1))(t)
    assert mgf.diff(t).subs(t, 1) == I

    mgf = moment_generating_function(Erlang('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == 1

    mgf = moment_generating_function(ExGaussian("x", 0, 1, 1))(t)
    assert mgf.diff(t).subs(t, 2) == -exp(2)

    mgf = moment_generating_function(Exponential('x', 1))(t)
    assert mgf.diff(t).subs(t, 0) == 1

    mgf = moment_generating_function(Gamma('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == 1

    mgf = moment_generating_function(Gumbel('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == EulerGamma + 1

    mgf = moment_generating_function(Gompertz('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 1) == -e*meijerg(((), (1, 1)),
    ((0, 0, 0), ()), 1)

    mgf = moment_generating_function(Laplace('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == 1

    mgf = moment_generating_function(Logistic('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == beta(1, 1)

    mgf = moment_generating_function(Normal('x', 0, 1))(t)
    assert mgf.diff(t).subs(t, 1) == exp(S.Half)

    mgf = moment_generating_function(Pareto('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == expint(1, 0)

    mgf = moment_generating_function(QuadraticU('x', 1, 2))(t)
    assert mgf.diff(t).subs(t, 1) == -12*e - 3*exp(2)

    mgf = moment_generating_function(RaisedCosine('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 1) == -2*e*pi**2*sinh(1)/\
    (1 + pi**2)**2 + e*pi**2*cosh(1)/(1 + pi**2)

    mgf = moment_generating_function(Rayleigh('x', 1))(t)
    assert mgf.diff(t).subs(t, 0) == sqrt(2)*sqrt(pi)/2

    mgf = moment_generating_function(Triangular('x', 1, 3, 2))(t)
    assert mgf.diff(t).subs(t, 1) == -e + exp(3)

    mgf = moment_generating_function(Uniform('x', 0, 1))(t)
    assert mgf.diff(t).subs(t, 1) == 1

    mgf = moment_generating_function(UniformSum('x', 1))(t)
    assert mgf.diff(t).subs(t, 1) == 1

    mgf = moment_generating_function(WignerSemicircle('x', 1))(t)
    assert mgf.diff(t).subs(t, 1) == -2*besseli(1, 1) + besseli(2, 1) +\
        besseli(0, 1)


def test_ContinuousRV():
    pdf = sqrt(2)*exp(-x**2/2)/(2*sqrt(pi))  # Normal distribution
    # X and Y should be equivalent
    X = ContinuousRV(x, pdf, check=True)
    Y = Normal('y', 0, 1)

    assert variance(X) == variance(Y)
    assert P(X > 0) == P(Y > 0)
    Z = ContinuousRV(z, exp(-z), set=Interval(0, oo))
    assert Z.pspace.domain.set == Interval(0, oo)
    assert E(Z) == 1
    assert P(Z > 5) == exp(-5)
    raises(ValueError, lambda: ContinuousRV(z, exp(-z), set=Interval(0, 10), check=True))

    # the correct pdf for Gamma(k, theta) but the integral in `check`
    # integrates to something equivalent to 1 and not to 1 exactly
    _x, k, theta = symbols("x k theta", positive=True)
    pdf = 1/(gamma(k)*theta**k)*_x**(k-1)*exp(-_x/theta)
    X = ContinuousRV(_x, pdf, set=Interval(0, oo))
    Y = Gamma('y', k, theta)
    assert (E(X) - E(Y)).simplify() == 0
    assert (variance(X) - variance(Y)).simplify() == 0


def test_arcsin():

    a = Symbol("a", real=True)
    b = Symbol("b", real=True)

    X = Arcsin('x', a, b)
    assert density(X)(x) == 1/(pi*sqrt((-x + b)*(x - a)))
    assert cdf(X)(x) == Piecewise((0, a > x),
                            (2*asin(sqrt((-a + x)/(-a + b)))/pi, b >= x),
                            (1, True))
    assert pspace(X).domain.set == Interval(a, b)

def test_benini():
    alpha = Symbol("alpha", positive=True)
    beta = Symbol("beta", positive=True)
    sigma = Symbol("sigma", positive=True)
    X = Benini('x', alpha, beta, sigma)

    assert density(X)(x) == ((alpha/x + 2*beta*log(x/sigma)/x)
                          *exp(-alpha*log(x/sigma) - beta*log(x/sigma)**2))

    assert pspace(X).domain.set == Interval(sigma, oo)
    raises(NotImplementedError, lambda: moment_generating_function(X))
    alpha = Symbol("alpha", nonpositive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))

    beta = Symbol("beta", nonpositive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))

    alpha = Symbol("alpha", positive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))

    beta = Symbol("beta", positive=True)
    sigma = Symbol("sigma", nonpositive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))

def test_beta():
    a, b = symbols('alpha beta', positive=True)
    B = Beta('x', a, b)

    assert pspace(B).domain.set == Interval(0, 1)
    assert characteristic_function(B)(x) == hyper((a,), (a + b,), I*x)
    assert density(B)(x) == x**(a - 1)*(1 - x)**(b - 1)/beta(a, b)

    assert simplify(E(B)) == a / (a + b)
    assert simplify(variance(B)) == a*b / (a**3 + 3*a**2*b + a**2 + 3*a*b**2 + 2*a*b + b**3 + b**2)

    # Full symbolic solution is too much, test with numeric version
    a, b = 1, 2
    B = Beta('x', a, b)
    assert expand_func(E(B)) == a / S(a + b)
    assert expand_func(variance(B)) == (a*b) / S((a + b)**2 * (a + b + 1))
    assert median(B) == FiniteSet(1 - 1/sqrt(2))

def test_beta_noncentral():
    a, b = symbols('a b', positive=True)
    c = Symbol('c', nonnegative=True)
    _k = Dummy('k')

    X = BetaNoncentral('x', a, b, c)

    assert pspace(X).domain.set == Interval(0, 1)

    dens = density(X)
    z = Symbol('z')

    res = Sum( z**(_k + a - 1)*(c/2)**_k*(1 - z)**(b - 1)*exp(-c/2)/
               (beta(_k + a, b)*factorial(_k)), (_k, 0, oo))
    assert dens(z).dummy_eq(res)

    # BetaCentral should not raise if the assumptions
    # on the symbols can not be determined
    a, b, c = symbols('a b c')
    assert BetaNoncentral('x', a, b, c)

    a = Symbol('a', positive=False, real=True)
    raises(ValueError, lambda: BetaNoncentral('x', a, b, c))

    a = Symbol('a', positive=True)
    b = Symbol('b', positive=False, real=True)
    raises(ValueError, lambda: BetaNoncentral('x', a, b, c))

    a = Symbol('a', positive=True)
    b = Symbol('b', positive=True)
    c = Symbol('c', nonnegative=False, real=True)
    raises(ValueError, lambda: BetaNoncentral('x', a, b, c))

def test_betaprime():
    alpha = Symbol("alpha", positive=True)

    betap = Symbol("beta", positive=True)

    X = BetaPrime('x', alpha, betap)
    assert density(X)(x) == x**(alpha - 1)*(x + 1)**(-alpha - betap)/beta(alpha, betap)

    alpha = Symbol("alpha", nonpositive=True)
    raises(ValueError, lambda: BetaPrime('x', alpha, betap))

    alpha = Symbol("alpha", positive=True)
    betap = Symbol("beta", nonpositive=True)
    raises(ValueError, lambda: BetaPrime('x', alpha, betap))
    X = BetaPrime('x', 1, 1)
    assert median(X) == FiniteSet(1)


def test_BoundedPareto():
    L, H = symbols('L, H', negative=True)
    raises(ValueError, lambda: BoundedPareto('X', 1, L, H))
    L, H = symbols('L, H', real=False)
    raises(ValueError, lambda: BoundedPareto('X', 1, L, H))
    L, H = symbols('L, H', positive=True)
    raises(ValueError, lambda: BoundedPareto('X', -1, L, H))

    X = BoundedPareto('X', 2, L, H)
    assert X.pspace.domain.set == Interval(L, H)
    assert density(X)(x) == 2*L**2/(x**3*(1 - L**2/H**2))
    assert cdf(X)(x) == Piecewise((-H**2*L**2/(x**2*(H**2 - L**2)) \
                            + H**2/(H**2 - L**2), L <= x), (0, True))
    assert E(X).simplify() == 2*H*L/(H + L)
    X = BoundedPareto('X', 1, 2, 4)
    assert E(X).simplify() == log(16)
    assert median(X) == FiniteSet(Rational(8, 3))
    assert variance(X).simplify() == 8 - 16*log(2)**2


def test_cauchy():
    x0 = Symbol("x0", real=True)
    gamma = Symbol("gamma", positive=True)
    p = Symbol("p", positive=True)

    X = Cauchy('x', x0, gamma)
    # Tests the characteristic function
    assert characteristic_function(X)(x) == exp(-gamma*Abs(x) + I*x*x0)
    raises(NotImplementedError, lambda: moment_generating_function(X))
    assert density(X)(x) == 1/(pi*gamma*(1 + (x - x0)**2/gamma**2))
    assert diff(cdf(X)(x), x) == density(X)(x)
    assert quantile(X)(p) == gamma*tan(pi*(p - S.Half)) + x0

    x1 = Symbol("x1", real=False)
    raises(ValueError, lambda: Cauchy('x', x1, gamma))
    gamma = Symbol("gamma", nonpositive=True)
    raises(ValueError, lambda: Cauchy('x', x0, gamma))
    assert median(X) == FiniteSet(x0)

def test_chi():
    from sympy.core.numbers import I
    k = Symbol("k", integer=True)

    X = Chi('x', k)
    assert density(X)(x) == 2**(-k/2 + 1)*x**(k - 1)*exp(-x**2/2)/gamma(k/2)

    # Tests the characteristic function
    assert characteristic_function(X)(x) == sqrt(2)*I*x*gamma(k/2 + S(1)/2)*hyper((k/2 + S(1)/2,),
                                            (S(3)/2,), -x**2/2)/gamma(k/2) + hyper((k/2,), (S(1)/2,), -x**2/2)

    # Tests the moment generating function
    assert moment_generating_function(X)(x) == sqrt(2)*x*gamma(k/2 + S(1)/2)*hyper((k/2 + S(1)/2,),
                                                (S(3)/2,), x**2/2)/gamma(k/2) + hyper((k/2,), (S(1)/2,), x**2/2)

    k = Symbol("k", integer=True, positive=False)
    raises(ValueError, lambda: Chi('x', k))

    k = Symbol("k", integer=False, positive=True)
    raises(ValueError, lambda: Chi('x', k))

def test_chi_noncentral():
    k = Symbol("k", integer=True)
    l = Symbol("l")

    X = ChiNoncentral("x", k, l)
    assert density(X)(x) == (x**k*l*(x*l)**(-k/2)*
                          exp(-x**2/2 - l**2/2)*besseli(k/2 - 1, x*l))

    k = Symbol("k", integer=True, positive=False)
    raises(ValueError, lambda: ChiNoncentral('x', k, l))

    k = Symbol("k", integer=True, positive=True)
    l = Symbol("l", nonpositive=True)
    raises(ValueError, lambda: ChiNoncentral('x', k, l))

    k = Symbol("k", integer=False)
    l = Symbol("l", positive=True)
    raises(ValueError, lambda: ChiNoncentral('x', k, l))


def test_chi_squared():
    k = Symbol("k", integer=True)
    X = ChiSquared('x', k)

    # Tests the characteristic function
    assert characteristic_function(X)(x) == ((-2*I*x + 1)**(-k/2))

    assert density(X)(x) == 2**(-k/2)*x**(k/2 - 1)*exp(-x/2)/gamma(k/2)
    assert cdf(X)(x) == Piecewise((lowergamma(k/2, x/2)/gamma(k/2), x >= 0), (0, True))
    assert E(X) == k
    assert variance(X) == 2*k

    X = ChiSquared('x', 15)
    assert cdf(X)(3) == -14873*sqrt(6)*exp(Rational(-3, 2))/(5005*sqrt(pi)) + erf(sqrt(6)/2)

    k = Symbol("k", integer=True, positive=False)
    raises(ValueError, lambda: ChiSquared('x', k))

    k = Symbol("k", integer=False, positive=True)
    raises(ValueError, lambda: ChiSquared('x', k))


def test_dagum():
    p = Symbol("p", positive=True)
    b = Symbol("b", positive=True)
    a = Symbol("a", positive=True)

    X = Dagum('x', p, a, b)
    assert density(X)(x) == a*p*(x/b)**(a*p)*((x/b)**a + 1)**(-p - 1)/x
    assert cdf(X)(x) == Piecewise(((1 + (x/b)**(-a))**(-p), x >= 0),
                                    (0, True))

    p = Symbol("p", nonpositive=True)
    raises(ValueError, lambda: Dagum('x', p, a, b))

    p = Symbol("p", positive=True)
    b = Symbol("b", nonpositive=True)
    raises(ValueError, lambda: Dagum('x', p, a, b))

    b = Symbol("b", positive=True)
    a = Symbol("a", nonpositive=True)
    raises(ValueError, lambda: Dagum('x', p, a, b))
    X = Dagum('x', 1, 1, 1)
    assert median(X) == FiniteSet(1)

def test_davis():
    b = Symbol("b", positive=True)
    n = Symbol("n", positive=True)
    mu = Symbol("mu", positive=True)

    X = Davis('x', b, n, mu)
    dividend = b**n*(x - mu)**(-1-n)
    divisor = (exp(b/(x-mu))-1)*(gamma(n)*zeta(n))
    assert density(X)(x) == dividend/divisor


def test_erlang():
    k = Symbol("k", integer=True, positive=True)
    l = Symbol("l", positive=True)

    X = Erlang("x", k, l)
    assert density(X)(x) == x**(k - 1)*l**k*exp(-x*l)/gamma(k)
    assert cdf(X)(x) == Piecewise((lowergamma(k, l*x)/gamma(k), x > 0),
                               (0, True))


def test_exgaussian():
    m, z = symbols("m, z")
    s, l = symbols("s, l", positive=True)
    X = ExGaussian("x", m, s, l)

    assert density(X)(z) == l*exp(l*(l*s**2 + 2*m - 2*z)/2) *\
        erfc(sqrt(2)*(l*s**2 + m - z)/(2*s))/2

    # Note: actual_output simplifies to expected_output.
    # Ideally cdf(X)(z) would return expected_output
    # expected_output = (erf(sqrt(2)*(l*s**2 + m - z)/(2*s)) - 1)*exp(l*(l*s**2 + 2*m - 2*z)/2)/2 - erf(sqrt(2)*(m - z)/(2*s))/2 + S.Half
    u = l*(z - m)
    v = l*s
    GaussianCDF1 = cdf(Normal('x', 0, v))(u)
    GaussianCDF2 = cdf(Normal('x', v**2, v))(u)
    actual_output = GaussianCDF1 - exp(-u + (v**2/2) + log(GaussianCDF2))
    assert cdf(X)(z) == actual_output
    # assert simplify(actual_output) == expected_output

    assert variance(X).expand() == s**2 + l**(-2)

    assert skewness(X).expand() == 2/(l**3*s**2*sqrt(s**2 + l**(-2)) + l *
                                      sqrt(s**2 + l**(-2)))


@slow
def test_exponential():
    rate = Symbol('lambda', positive=True)
    X = Exponential('x', rate)
    p = Symbol("p", positive=True, real=True)

    assert E(X) == 1/rate
    assert variance(X) == 1/rate**2
    assert skewness(X) == 2
    assert skewness(X) == smoment(X, 3)
    assert kurtosis(X) == 9
    assert kurtosis(X) == smoment(X, 4)
    assert smoment(2*X, 4) == smoment(X, 4)
    assert moment(X, 3) == 3*2*1/rate**3
    assert P(X > 0) is S.One
    assert P(X > 1) == exp(-rate)
    assert P(X > 10) == exp(-10*rate)
    assert quantile(X)(p) == -log(1-p)/rate

    assert where(X <= 1).set == Interval(0, 1)
    Y = Exponential('y', 1)
    assert median(Y) == FiniteSet(log(2))
    #Test issue 9970
    z = Dummy('z')
    assert P(X > z) == exp(-z*rate)
    assert P(X < z) == 0
    #Test issue 10076 (Distribution with interval(0,oo))
    x = Symbol('x')
    _z = Dummy('_z')
    b = SingleContinuousPSpace(x, ExponentialDistribution(2))

    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        expected1 = Integral(2*exp(-2*_z), (_z, 3, oo))
        assert b.probability(x > 3, evaluate=False).rewrite(Integral).dummy_eq(expected1)

        expected2 = Integral(2*exp(-2*_z), (_z, 0, 4))
        assert b.probability(x < 4, evaluate=False).rewrite(Integral).dummy_eq(expected2)
    Y = Exponential('y', 2*rate)
    assert coskewness(X, X, X) == skewness(X)
    assert coskewness(X, Y + rate*X, Y + 2*rate*X) == \
                        4/(sqrt(1 + 1/(4*rate**2))*sqrt(4 + 1/(4*rate**2)))
    assert coskewness(X + 2*Y, Y + X, Y + 2*X, X > 3) == \
                        sqrt(170)*Rational(9, 85)

def test_exponential_power():
    mu = Symbol('mu')
    z = Symbol('z')
    alpha = Symbol('alpha', positive=True)
    beta = Symbol('beta', positive=True)

    X = ExponentialPower('x', mu, alpha, beta)

    assert density(X)(z) == beta*exp(-(Abs(mu - z)/alpha)
                                     ** beta)/(2*alpha*gamma(1/beta))
    assert cdf(X)(z) == S.Half + lowergamma(1/beta,
                            (Abs(mu - z)/alpha)**beta)*sign(-mu + z)/\
                                (2*gamma(1/beta))


def test_f_distribution():
    d1 = Symbol("d1", positive=True)
    d2 = Symbol("d2", positive=True)

    X = FDistribution("x", d1, d2)

    assert density(X)(x) == (d2**(d2/2)*sqrt((d1*x)**d1*(d1*x + d2)**(-d1 - d2))
                             /(x*beta(d1/2, d2/2)))

    raises(NotImplementedError, lambda: moment_generating_function(X))
    d1 = Symbol("d1", nonpositive=True)
    raises(ValueError, lambda: FDistribution('x', d1, d1))

    d1 = Symbol("d1", positive=True, integer=False)
    raises(ValueError, lambda: FDistribution('x', d1, d1))

    d1 = Symbol("d1", positive=True)
    d2 = Symbol("d2", nonpositive=True)
    raises(ValueError, lambda: FDistribution('x', d1, d2))

    d2 = Symbol("d2", positive=True, integer=False)
    raises(ValueError, lambda: FDistribution('x', d1, d2))


def test_fisher_z():
    d1 = Symbol("d1", positive=True)
    d2 = Symbol("d2", positive=True)

    X = FisherZ("x", d1, d2)
    assert density(X)(x) == (2*d1**(d1/2)*d2**(d2/2)*(d1*exp(2*x) + d2)
                             **(-d1/2 - d2/2)*exp(d1*x)/beta(d1/2, d2/2))

def test_frechet():
    a = Symbol("a", positive=True)
    s = Symbol("s", positive=True)
    m = Symbol("m", real=True)

    X = Frechet("x", a, s=s, m=m)
    assert density(X)(x) == a*((x - m)/s)**(-a - 1)*exp(-((x - m)/s)**(-a))/s
    assert cdf(X)(x) == Piecewise((exp(-((-m + x)/s)**(-a)), m <= x), (0, True))

@slow
def test_gamma():
    k = Symbol("k", positive=True)
    theta = Symbol("theta", positive=True)

    X = Gamma('x', k, theta)

    # Tests characteristic function
    assert characteristic_function(X)(x) == ((-I*theta*x + 1)**(-k))

    assert density(X)(x) == x**(k - 1)*theta**(-k)*exp(-x/theta)/gamma(k)
    assert cdf(X, meijerg=True)(z) == Piecewise(
            (-k*lowergamma(k, 0)/gamma(k + 1) +
                k*lowergamma(k, z/theta)/gamma(k + 1), z >= 0),
            (0, True))

    # assert simplify(variance(X)) == k*theta**2  # handled numerically below
    assert E(X) == moment(X, 1)

    k, theta = symbols('k theta', positive=True)
    X = Gamma('x', k, theta)
    assert E(X) == k*theta
    assert variance(X) == k*theta**2
    assert skewness(X).expand() == 2/sqrt(k)
    assert kurtosis(X).expand() == 3 + 6/k

    Y = Gamma('y', 2*k, 3*theta)
    assert coskewness(X, theta*X + Y, k*X + Y).simplify() == \
        2*531441**(-k)*sqrt(k)*theta*(3*3**(12*k) - 2*531441**k) \
        /(sqrt(k**2 + 18)*sqrt(theta**2 + 18))

def test_gamma_inverse():
    a = Symbol("a", positive=True)
    b = Symbol("b", positive=True)
    X = GammaInverse("x", a, b)
    assert density(X)(x) == x**(-a - 1)*b**a*exp(-b/x)/gamma(a)
    assert cdf(X)(x) == Piecewise((uppergamma(a, b/x)/gamma(a), x > 0), (0, True))
    assert characteristic_function(X)(x) == 2 * (-I*b*x)**(a/2) \
            * besselk(a, 2*sqrt(b)*sqrt(-I*x))/gamma(a)
    raises(NotImplementedError, lambda: moment_generating_function(X))

def test_gompertz():
    b = Symbol("b", positive=True)
    eta = Symbol("eta", positive=True)

    X = Gompertz("x", b, eta)

    assert density(X)(x) == b*eta*exp(eta)*exp(b*x)*exp(-eta*exp(b*x))
    assert cdf(X)(x) == 1 - exp(eta)*exp(-eta*exp(b*x))
    assert diff(cdf(X)(x), x) == density(X)(x)


def test_gumbel():
    beta = Symbol("beta", positive=True)
    mu = Symbol("mu")
    x = Symbol("x")
    y = Symbol("y")
    X = Gumbel("x", beta, mu)
    Y = Gumbel("y", beta, mu, minimum=True)
    assert density(X)(x).expand() == \
    exp(mu/beta)*exp(-x/beta)*exp(-exp(mu/beta)*exp(-x/beta))/beta
    assert density(Y)(y).expand() == \
    exp(-mu/beta)*exp(y/beta)*exp(-exp(-mu/beta)*exp(y/beta))/beta
    assert cdf(X)(x).expand() == \
    exp(-exp(mu/beta)*exp(-x/beta))
    assert characteristic_function(X)(x) == exp(I*mu*x)*gamma(-I*beta*x + 1)

def test_kumaraswamy():
    a = Symbol("a", positive=True)
    b = Symbol("b", positive=True)

    X = Kumaraswamy("x", a, b)
    assert density(X)(x) == x**(a - 1)*a*b*(-x**a + 1)**(b - 1)
    assert cdf(X)(x) == Piecewise((0, x < 0),
                                (-(-x**a + 1)**b + 1, x <= 1),
                                (1, True))


def test_laplace():
    mu = Symbol("mu")
    b = Symbol("b", positive=True)

    X = Laplace('x', mu, b)

    #Tests characteristic_function
    assert characteristic_function(X)(x) == (exp(I*mu*x)/(b**2*x**2 + 1))

    assert density(X)(x) == exp(-Abs(x - mu)/b)/(2*b)
    assert cdf(X)(x) == Piecewise((exp((-mu + x)/b)/2, mu > x),
                            (-exp((mu - x)/b)/2 + 1, True))
    X = Laplace('x', [1, 2], [[1, 0], [0, 1]])
    assert isinstance(pspace(X).distribution, MultivariateLaplaceDistribution)

def test_levy():
    mu = Symbol("mu", real=True)
    c = Symbol("c", positive=True)

    X = Levy('x', mu, c)
    assert X.pspace.domain.set == Interval(mu, oo)
    assert density(X)(x) == sqrt(c/(2*pi))*exp(-c/(2*(x - mu)))/((x - mu)**(S.One + S.Half))
    assert cdf(X)(x) == erfc(sqrt(c/(2*(x - mu))))

    raises(NotImplementedError, lambda: moment_generating_function(X))
    mu = Symbol("mu", real=False)
    raises(ValueError, lambda: Levy('x',mu,c))

    c = Symbol("c", nonpositive=True)
    raises(ValueError, lambda: Levy('x',mu,c))

    mu = Symbol("mu", real=True)
    raises(ValueError, lambda: Levy('x',mu,c))

def test_logcauchy():
    mu = Symbol("mu", positive=True)
    sigma = Symbol("sigma", positive=True)

    X = LogCauchy("x", mu, sigma)

    assert density(X)(x) == sigma/(x*pi*(sigma**2 + (-mu + log(x))**2))
    assert cdf(X)(x) == atan((log(x) - mu)/sigma)/pi + S.Half


def test_logistic():
    mu = Symbol("mu", real=True)
    s = Symbol("s", positive=True)
    p = Symbol("p", positive=True)

    X = Logistic('x', mu, s)

    #Tests characteristics_function
    assert characteristic_function(X)(x) == \
           (Piecewise((pi*s*x*exp(I*mu*x)/sinh(pi*s*x), Ne(x, 0)), (1, True)))

    assert density(X)(x) == exp((-x + mu)/s)/(s*(exp((-x + mu)/s) + 1)**2)
    assert cdf(X)(x) == 1/(exp((mu - x)/s) + 1)
    assert quantile(X)(p) == mu - s*log(-S.One + 1/p)

def test_loglogistic():
    a, b = symbols('a b')
    assert LogLogistic('x', a, b)

    a = Symbol('a', negative=True)
    b = Symbol('b', positive=True)
    raises(ValueError, lambda: LogLogistic('x', a, b))

    a = Symbol('a', positive=True)
    b = Symbol('b', negative=True)
    raises(ValueError, lambda: LogLogistic('x', a, b))

    a, b, z, p = symbols('a b z p', positive=True)
    X = LogLogistic('x', a, b)
    assert density(X)(z) == b*(z/a)**(b - 1)/(a*((z/a)**b + 1)**2)
    assert cdf(X)(z) == 1/(1 + (z/a)**(-b))
    assert quantile(X)(p) == a*(p/(1 - p))**(1/b)

    # Expectation
    assert E(X) == Piecewise((S.NaN, b <= 1), (pi*a/(b*sin(pi/b)), True))
    b = symbols('b', prime=True) # b > 1
    X = LogLogistic('x', a, b)
    assert E(X) == pi*a/(b*sin(pi/b))
    X = LogLogistic('x', 1, 2)
    assert median(X) == FiniteSet(1)

def test_logitnormal():
    mu = Symbol('mu', real=True)
    s = Symbol('s', positive=True)
    X = LogitNormal('x', mu, s)
    x = Symbol('x')

    assert density(X)(x) == sqrt(2)*exp(-(-mu + log(x/(1 - x)))**2/(2*s**2))/(2*sqrt(pi)*s*x*(1 - x))
    assert cdf(X)(x) == erf(sqrt(2)*(-mu + log(x/(1 - x)))/(2*s))/2 + S(1)/2

def test_lognormal():
    mean = Symbol('mu', real=True)
    std = Symbol('sigma', positive=True)
    X = LogNormal('x', mean, std)
    # The sympy integrator can't do this too well
    #assert E(X) == exp(mean+std**2/2)
    #assert variance(X) == (exp(std**2)-1) * exp(2*mean + std**2)

    # The sympy integrator can't do this too well
    #assert E(X) ==
    raises(NotImplementedError, lambda: moment_generating_function(X))
    mu = Symbol("mu", real=True)
    sigma = Symbol("sigma", positive=True)

    X = LogNormal('x', mu, sigma)
    assert density(X)(x) == (sqrt(2)*exp(-(-mu + log(x))**2
                                    /(2*sigma**2))/(2*x*sqrt(pi)*sigma))
    # Tests cdf
    assert cdf(X)(x) == Piecewise(
                        (erf(sqrt(2)*(-mu + log(x))/(2*sigma))/2
                        + S(1)/2, x > 0), (0, True))

    X = LogNormal('x', 0, 1)  # Mean 0, standard deviation 1
    assert density(X)(x) == sqrt(2)*exp(-log(x)**2/2)/(2*x*sqrt(pi))


def test_Lomax():
    a, l = symbols('a, l', negative=True)
    raises(ValueError, lambda: Lomax('X', a, l))
    a, l = symbols('a, l', real=False)
    raises(ValueError, lambda: Lomax('X', a, l))

    a, l = symbols('a, l', positive=True)
    X = Lomax('X', a, l)
    assert X.pspace.domain.set == Interval(0, oo)
    assert density(X)(x) == a*(1 + x/l)**(-a - 1)/l
    assert cdf(X)(x) == Piecewise((1 - (1 + x/l)**(-a), x >= 0), (0, True))
    a = 3
    X = Lomax('X', a, l)
    assert E(X) == l/2
    assert median(X) == FiniteSet(l*(-1 + 2**Rational(1, 3)))
    assert variance(X) == 3*l**2/4


def test_maxwell():
    a = Symbol("a", positive=True)

    X = Maxwell('x', a)

    assert density(X)(x) == (sqrt(2)*x**2*exp(-x**2/(2*a**2))/
        (sqrt(pi)*a**3))
    assert E(X) == 2*sqrt(2)*a/sqrt(pi)
    assert variance(X) == -8*a**2/pi + 3*a**2
    assert cdf(X)(x) == erf(sqrt(2)*x/(2*a)) - sqrt(2)*x*exp(-x**2/(2*a**2))/(sqrt(pi)*a)
    assert diff(cdf(X)(x), x) == density(X)(x)


@slow
def test_Moyal():
    mu = Symbol('mu',real=False)
    sigma = Symbol('sigma', positive=True)
    raises(ValueError, lambda: Moyal('M',mu, sigma))

    mu = Symbol('mu', real=True)
    sigma = Symbol('sigma', negative=True)
    raises(ValueError, lambda: Moyal('M',mu, sigma))

    sigma = Symbol('sigma', positive=True)
    M = Moyal('M', mu, sigma)
    assert density(M)(z) == sqrt(2)*exp(-exp((mu - z)/sigma)/2
                        - (-mu + z)/(2*sigma))/(2*sqrt(pi)*sigma)
    assert cdf(M)(z).simplify() == 1 - erf(sqrt(2)*exp((mu - z)/(2*sigma))/2)
    assert characteristic_function(M)(z) == 2**(-I*sigma*z)*exp(I*mu*z) \
                        *gamma(-I*sigma*z + Rational(1, 2))/sqrt(pi)
    assert E(M) == mu + EulerGamma*sigma + sigma*log(2)
    assert moment_generating_function(M)(z) == 2**(-sigma*z)*exp(mu*z) \
                        *gamma(-sigma*z + Rational(1, 2))/sqrt(pi)


def test_nakagami():
    mu = Symbol("mu", positive=True)
    omega = Symbol("omega", positive=True)

    X = Nakagami('x', mu, omega)
    assert density(X)(x) == (2*x**(2*mu - 1)*mu**mu*omega**(-mu)
                                *exp(-x**2*mu/omega)/gamma(mu))
    assert simplify(E(X)) == (sqrt(mu)*sqrt(omega)
                                            *gamma(mu + S.Half)/gamma(mu + 1))
    assert simplify(variance(X)) == (
    omega - omega*gamma(mu + S.Half)**2/(gamma(mu)*gamma(mu + 1)))
    assert cdf(X)(x) == Piecewise(
                                (lowergamma(mu, mu*x**2/omega)/gamma(mu), x > 0),
                                (0, True))
    X = Nakagami('x', 1, 1)
    assert median(X) == FiniteSet(sqrt(log(2)))

def test_gaussian_inverse():
    # test for symbolic parameters
    a, b = symbols('a b')
    assert GaussianInverse('x', a, b)

    # Inverse Gaussian distribution is also known as Wald distribution
    # `GaussianInverse` can also be referred by the name `Wald`
    a, b, z = symbols('a b z')
    X = Wald('x', a, b)
    assert density(X)(z) == sqrt(2)*sqrt(b/z**3)*exp(-b*(-a + z)**2/(2*a**2*z))/(2*sqrt(pi))

    a, b = symbols('a b', positive=True)
    z = Symbol('z', positive=True)

    X = GaussianInverse('x', a, b)
    assert density(X)(z) == sqrt(2)*sqrt(b)*sqrt(z**(-3))*exp(-b*(-a + z)**2/(2*a**2*z))/(2*sqrt(pi))
    assert E(X) == a
    assert variance(X).expand() == a**3/b
    assert cdf(X)(z) == (S.Half - erf(sqrt(2)*sqrt(b)*(1 + z/a)/(2*sqrt(z)))/2)*exp(2*b/a) +\
         erf(sqrt(2)*sqrt(b)*(-1 + z/a)/(2*sqrt(z)))/2 + S.Half

    a = symbols('a', nonpositive=True)
    raises(ValueError, lambda: GaussianInverse('x', a, b))

    a = symbols('a', positive=True)
    b = symbols('b', nonpositive=True)
    raises(ValueError, lambda: GaussianInverse('x', a, b))

def test_pareto():
    xm, beta = symbols('xm beta', positive=True)
    alpha = beta + 5
    X = Pareto('x', xm, alpha)

    dens = density(X)

    #Tests cdf function
    assert cdf(X)(x) == \
           Piecewise((-x**(-beta - 5)*xm**(beta + 5) + 1, x >= xm), (0, True))

    #Tests characteristic_function
    assert characteristic_function(X)(x) == \
           ((-I*x*xm)**(beta + 5)*(beta + 5)*uppergamma(-beta - 5, -I*x*xm))

    assert dens(x) == x**(-(alpha + 1))*xm**(alpha)*(alpha)

    assert simplify(E(X)) == alpha*xm/(alpha-1)

    # computation of taylor series for MGF still too slow
    #assert simplify(variance(X)) == xm**2*alpha / ((alpha-1)**2*(alpha-2))


def test_pareto_numeric():
    xm, beta = 3, 2
    alpha = beta + 5
    X = Pareto('x', xm, alpha)

    assert E(X) == alpha*xm/S(alpha - 1)
    assert variance(X) == xm**2*alpha / S((alpha - 1)**2*(alpha - 2))
    assert median(X) == FiniteSet(3*2**Rational(1, 7))
    # Skewness tests too slow. Try shortcutting function?


def test_PowerFunction():
    alpha = Symbol("alpha", nonpositive=True)
    a, b = symbols('a, b', real=True)
    raises (ValueError, lambda: PowerFunction('x', alpha, a, b))

    a, b = symbols('a, b', real=False)
    raises (ValueError, lambda: PowerFunction('x', alpha, a, b))

    alpha = Symbol("alpha", positive=True)
    a, b = symbols('a, b', real=True)
    raises (ValueError, lambda: PowerFunction('x', alpha, 5, 2))

    X = PowerFunction('X', 2, a, b)
    assert density(X)(z) == (-2*a + 2*z)/(-a + b)**2
    assert cdf(X)(z) == Piecewise((a**2/(a**2 - 2*a*b + b**2) -
        2*a*z/(a**2 - 2*a*b + b**2) + z**2/(a**2 - 2*a*b + b**2), a <= z), (0, True))

    X = PowerFunction('X', 2, 0, 1)
    assert density(X)(z) == 2*z
    assert cdf(X)(z) == Piecewise((z**2, z >= 0), (0,True))
    assert E(X) == Rational(2,3)
    assert P(X < 0) == 0
    assert P(X < 1) == 1
    assert median(X) == FiniteSet(1/sqrt(2))

def test_raised_cosine():
    mu = Symbol("mu", real=True)
    s = Symbol("s", positive=True)

    X = RaisedCosine("x", mu, s)

    assert pspace(X).domain.set == Interval(mu - s, mu + s)
    #Tests characteristics_function
    assert characteristic_function(X)(x) == \
           Piecewise((exp(-I*pi*mu/s)/2, Eq(x, -pi/s)), (exp(I*pi*mu/s)/2, Eq(x, pi/s)), (pi**2*exp(I*mu*x)*sin(s*x)/(s*x*(-s**2*x**2 + pi**2)), True))

    assert density(X)(x) == (Piecewise(((cos(pi*(x - mu)/s) + 1)/(2*s),
                          And(x <= mu + s, mu - s <= x)), (0, True)))


def test_rayleigh():
    sigma = Symbol("sigma", positive=True)

    X = Rayleigh('x', sigma)

    #Tests characteristic_function
    assert characteristic_function(X)(x) == (-sqrt(2)*sqrt(pi)*sigma*x*(erfi(sqrt(2)*sigma*x/2) - I)*exp(-sigma**2*x**2/2)/2 + 1)

    assert density(X)(x) ==  x*exp(-x**2/(2*sigma**2))/sigma**2
    assert E(X) == sqrt(2)*sqrt(pi)*sigma/2
    assert variance(X) == -pi*sigma**2/2 + 2*sigma**2
    assert cdf(X)(x) == 1 - exp(-x**2/(2*sigma**2))
    assert diff(cdf(X)(x), x) == density(X)(x)

def test_reciprocal():
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)

    X = Reciprocal('x', a, b)
    assert density(X)(x) == 1/(x*(-log(a) + log(b)))
    assert cdf(X)(x) == Piecewise((log(a)/(log(a) - log(b)) - log(x)/(log(a) - log(b)), a <= x), (0, True))
    X = Reciprocal('x', 5, 30)

    assert E(X) == 25/(log(30) - log(5))
    assert P(X < 4) == S.Zero
    assert P(X < 20) == log(20) / (log(30) - log(5)) - log(5) / (log(30) - log(5))
    assert cdf(X)(10) == log(10) / (log(30) - log(5)) - log(5) / (log(30) - log(5))

    a = symbols('a', nonpositive=True)
    raises(ValueError, lambda: Reciprocal('x', a, b))

    a = symbols('a', positive=True)
    b = symbols('b', positive=True)
    raises(ValueError, lambda: Reciprocal('x', a + b, a))

def test_shiftedgompertz():
    b = Symbol("b", positive=True)
    eta = Symbol("eta", positive=True)
    X = ShiftedGompertz("x", b, eta)
    assert density(X)(x) == b*(eta*(1 - exp(-b*x)) + 1)*exp(-b*x)*exp(-eta*exp(-b*x))


def test_studentt():
    nu = Symbol("nu", positive=True)

    X = StudentT('x', nu)
    assert density(X)(x) == (1 + x**2/nu)**(-nu/2 - S.Half)/(sqrt(nu)*beta(S.Half, nu/2))
    assert cdf(X)(x) == S.Half + x*gamma(nu/2 + S.Half)*hyper((S.Half, nu/2 + S.Half),
                                (Rational(3, 2),), -x**2/nu)/(sqrt(pi)*sqrt(nu)*gamma(nu/2))
    raises(NotImplementedError, lambda: moment_generating_function(X))

def test_trapezoidal():
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    c = Symbol("c", real=True)
    d = Symbol("d", real=True)

    X = Trapezoidal('x', a, b, c, d)
    assert density(X)(x) == Piecewise(((-2*a + 2*x)/((-a + b)*(-a - b + c + d)), (a <= x) & (x < b)),
                                      (2/(-a - b + c + d), (b <= x) & (x < c)),
                                      ((2*d - 2*x)/((-c + d)*(-a - b + c + d)), (c <= x) & (x <= d)),
                                      (0, True))

    X = Trapezoidal('x', 0, 1, 2, 3)
    assert E(X) == Rational(3, 2)
    assert variance(X) == Rational(5, 12)
    assert P(X < 2) == Rational(3, 4)
    assert median(X) == FiniteSet(Rational(3, 2))

def test_triangular():
    a = Symbol("a")
    b = Symbol("b")
    c = Symbol("c")

    X = Triangular('x', a, b, c)
    assert pspace(X).domain.set == Interval(a, b)
    assert str(density(X)(x)) == ("Piecewise(((-2*a + 2*x)/((-a + b)*(-a + c)), (a <= x) & (c > x)), "
    "(2/(-a + b), Eq(c, x)), ((2*b - 2*x)/((-a + b)*(b - c)), (b >= x) & (c < x)), (0, True))")

    #Tests moment_generating_function
    assert moment_generating_function(X)(x).expand() == \
    ((-2*(-a + b)*exp(c*x) + 2*(-a + c)*exp(b*x) + 2*(b - c)*exp(a*x))/(x**2*(-a + b)*(-a + c)*(b - c))).expand()
    assert str(characteristic_function(X)(x)) == \
    '(2*(-a + b)*exp(I*c*x) - 2*(-a + c)*exp(I*b*x) - 2*(b - c)*exp(I*a*x))/(x**2*(-a + b)*(-a + c)*(b - c))'

def test_quadratic_u():
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)

    X = QuadraticU("x", a, b)
    Y = QuadraticU("x", 1, 2)

    assert pspace(X).domain.set == Interval(a, b)
    # Tests _moment_generating_function
    assert moment_generating_function(Y)(1)  == -15*exp(2) + 27*exp(1)
    assert moment_generating_function(Y)(2) == -9*exp(4)/2 + 21*exp(2)/2

    assert characteristic_function(Y)(1) == 3*I*(-1 + 4*I)*exp(I*exp(2*I))
    assert density(X)(x) == (Piecewise((12*(x - a/2 - b/2)**2/(-a + b)**3,
                          And(x <= b, a <= x)), (0, True)))


def test_uniform():
    l = Symbol('l', real=True)
    w = Symbol('w', positive=True)
    X = Uniform('x', l, l + w)

    assert E(X) == l + w/2
    assert variance(X).expand() == w**2/12

    # With numbers all is well
    X = Uniform('x', 3, 5)
    assert P(X < 3) == 0 and P(X > 5) == 0
    assert P(X < 4) == P(X > 4) == S.Half
    assert median(X) == FiniteSet(4)

    z = Symbol('z')
    p = density(X)(z)
    assert p.subs(z, 3.7) == S.Half
    assert p.subs(z, -1) == 0
    assert p.subs(z, 6) == 0

    c = cdf(X)
    assert c(2) == 0 and c(3) == 0
    assert c(Rational(7, 2)) == Rational(1, 4)
    assert c(5) == 1 and c(6) == 1


@XFAIL
@slow
def test_uniform_P():
    """ This stopped working because SingleContinuousPSpace.compute_density no
    longer calls integrate on a DiracDelta but rather just solves directly.
    integrate used to call UniformDistribution.expectation which special-cased
    subsed out the Min and Max terms that Uniform produces

    I decided to regress on this class for general cleanliness (and I suspect
    speed) of the algorithm.
    """
    l = Symbol('l', real=True)
    w = Symbol('w', positive=True)
    X = Uniform('x', l, l + w)
    assert P(X < l) == 0 and P(X > l + w) == 0


def test_uniformsum():
    n = Symbol("n", integer=True)
    _k = Dummy("k")
    x = Symbol("x")

    X = UniformSum('x', n)
    res = Sum((-1)**_k*(-_k + x)**(n - 1)*binomial(n, _k), (_k, 0, floor(x)))/factorial(n - 1)
    assert density(X)(x).dummy_eq(res)

    #Tests set functions
    assert X.pspace.domain.set == Interval(0, n)

    #Tests the characteristic_function
    assert characteristic_function(X)(x) == (-I*(exp(I*x) - 1)/x)**n

    #Tests the moment_generating_function
    assert moment_generating_function(X)(x) == ((exp(x) - 1)/x)**n


def test_von_mises():
    mu = Symbol("mu")
    k = Symbol("k", positive=True)

    X = VonMises("x", mu, k)
    assert density(X)(x) == exp(k*cos(x - mu))/(2*pi*besseli(0, k))


def test_weibull():
    a, b = symbols('a b', positive=True)
    # FIXME: simplify(E(X)) seems to hang without extended_positive=True
    # On a Linux machine this had a rapid memory leak...
    # a, b = symbols('a b', positive=True)
    X = Weibull('x', a, b)

    assert E(X).expand() == a * gamma(1 + 1/b)
    assert variance(X).expand() == (a**2 * gamma(1 + 2/b) - E(X)**2).expand()
    assert simplify(skewness(X)) == (2*gamma(1 + 1/b)**3 - 3*gamma(1 + 1/b)*gamma(1 + 2/b) + gamma(1 + 3/b))/(-gamma(1 + 1/b)**2 + gamma(1 + 2/b))**Rational(3, 2)
    assert simplify(kurtosis(X)) == (-3*gamma(1 + 1/b)**4 +\
        6*gamma(1 + 1/b)**2*gamma(1 + 2/b) - 4*gamma(1 + 1/b)*gamma(1 + 3/b) + gamma(1 + 4/b))/(gamma(1 + 1/b)**2 - gamma(1 + 2/b))**2

def test_weibull_numeric():
    # Test for integers and rationals
    a = 1
    bvals = [S.Half, 1, Rational(3, 2), 5]
    for b in bvals:
        X = Weibull('x', a, b)
        assert simplify(E(X)) == expand_func(a * gamma(1 + 1/S(b)))
        assert simplify(variance(X)) == simplify(
            a**2 * gamma(1 + 2/S(b)) - E(X)**2)
        # Not testing Skew... it's slow with int/frac values > 3/2


def test_wignersemicircle():
    R = Symbol("R", positive=True)

    X = WignerSemicircle('x', R)
    assert pspace(X).domain.set == Interval(-R, R)
    assert density(X)(x) == 2*sqrt(-x**2 + R**2)/(pi*R**2)
    assert E(X) == 0


    #Tests ChiNoncentralDistribution
    assert characteristic_function(X)(x) == \
           Piecewise((2*besselj(1, R*x)/(R*x), Ne(x, 0)), (1, True))


def test_input_value_assertions():
    a, b = symbols('a b')
    p, q = symbols('p q', positive=True)
    m, n = symbols('m n', positive=False, real=True)

    raises(ValueError, lambda: Normal('x', 3, 0))
    raises(ValueError, lambda: Normal('x', m, n))
    Normal('X', a, p)  # No error raised
    raises(ValueError, lambda: Exponential('x', m))
    Exponential('Ex', p)  # No error raised
    for fn in [Pareto, Weibull, Beta, Gamma]:
        raises(ValueError, lambda: fn('x', m, p))
        raises(ValueError, lambda: fn('x', p, n))
        fn('x', p, q)  # No error raised


def test_unevaluated():
    X = Normal('x', 0, 1)
    k = Dummy('k')
    expr1 = Integral(sqrt(2)*k*exp(-k**2/2)/(2*sqrt(pi)), (k, -oo, oo))
    expr2 = Integral(sqrt(2)*exp(-k**2/2)/(2*sqrt(pi)), (k, 0, oo))
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert E(X, evaluate=False).rewrite(Integral).dummy_eq(expr1)
        assert E(X + 1, evaluate=False).rewrite(Integral).dummy_eq(expr1 + 1)
        assert P(X > 0, evaluate=False).rewrite(Integral).dummy_eq(expr2)

    assert P(X > 0, X**2 < 1) == S.Half


def test_probability_unevaluated():
    T = Normal('T', 30, 3)
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert type(P(T > 33, evaluate=False)) == Probability


def test_density_unevaluated():
    X = Normal('X', 0, 1)
    Y = Normal('Y', 0, 2)
    assert isinstance(density(X+Y, evaluate=False)(z), Integral)


def test_NormalDistribution():
    nd = NormalDistribution(0, 1)
    x = Symbol('x')
    assert nd.cdf(x) == erf(sqrt(2)*x/2)/2 + S.Half
    assert nd.expectation(1, x) == 1
    assert nd.expectation(x, x) == 0
    assert nd.expectation(x**2, x) == 1
    #Test issue 10076
    a = SingleContinuousPSpace(x, NormalDistribution(2, 4))
    _z = Dummy('_z')

    expected1 = Integral(sqrt(2)*exp(-(_z - 2)**2/32)/(8*sqrt(pi)),(_z, -oo, 1))
    assert a.probability(x < 1, evaluate=False).dummy_eq(expected1) is True

    expected2 = Integral(sqrt(2)*exp(-(_z - 2)**2/32)/(8*sqrt(pi)),(_z, 1, oo))
    assert a.probability(x > 1, evaluate=False).dummy_eq(expected2) is True

    b = SingleContinuousPSpace(x, NormalDistribution(1, 9))

    expected3 = Integral(sqrt(2)*exp(-(_z - 1)**2/162)/(18*sqrt(pi)),(_z, 6, oo))
    assert b.probability(x > 6, evaluate=False).dummy_eq(expected3) is True

    expected4 = Integral(sqrt(2)*exp(-(_z - 1)**2/162)/(18*sqrt(pi)),(_z, -oo, 6))
    assert b.probability(x < 6, evaluate=False).dummy_eq(expected4) is True


def test_random_parameters():
    mu = Normal('mu', 2, 3)
    meas = Normal('T', mu, 1)
    assert density(meas, evaluate=False)(z)
    assert isinstance(pspace(meas), CompoundPSpace)
    X = Normal('x', [1, 2], [[1, 0], [0, 1]])
    assert isinstance(pspace(X).distribution, MultivariateNormalDistribution)
    assert density(meas)(z).simplify() == sqrt(5)*exp(-z**2/20 + z/5 - S(1)/5)/(10*sqrt(pi))


def test_random_parameters_given():
    mu = Normal('mu', 2, 3)
    meas = Normal('T', mu, 1)
    assert given(meas, Eq(mu, 5)) == Normal('T', 5, 1)


def test_conjugate_priors():
    mu = Normal('mu', 2, 3)
    x = Normal('x', mu, 1)
    assert isinstance(simplify(density(mu, Eq(x, y), evaluate=False)(z)),
            Mul)


def test_difficult_univariate():
    """ Since using solve in place of deltaintegrate we're able to perform
    substantially more complex density computations on single continuous random
    variables """
    x = Normal('x', 0, 1)
    assert density(x**3)
    assert density(exp(x**2))
    assert density(log(x))


def test_issue_10003():
    X = Exponential('x', 3)
    G = Gamma('g', 1, 2)
    assert P(X < -1) is S.Zero
    assert P(G < -1) is S.Zero


def test_precomputed_cdf():
    x = symbols("x", real=True)
    mu = symbols("mu", real=True)
    sigma, xm, alpha = symbols("sigma xm alpha", positive=True)
    n = symbols("n", integer=True, positive=True)
    distribs = [
            Normal("X", mu, sigma),
            Pareto("P", xm, alpha),
            ChiSquared("C", n),
            Exponential("E", sigma),
            # LogNormal("L", mu, sigma),
    ]
    for X in distribs:
        compdiff = cdf(X)(x) - simplify(X.pspace.density.compute_cdf()(x))
        compdiff = simplify(compdiff.rewrite(erfc))
        assert compdiff == 0


@slow
def test_precomputed_characteristic_functions():
    import mpmath

    def test_cf(dist, support_lower_limit, support_upper_limit):
        pdf = density(dist)
        t = Symbol('t')

        # first function is the hardcoded CF of the distribution
        cf1 = lambdify([t], characteristic_function(dist)(t), 'mpmath')

        # second function is the Fourier transform of the density function
        f = lambdify([x, t], pdf(x)*exp(I*x*t), 'mpmath')
        cf2 = lambda t: mpmath.quad(lambda x: f(x, t), [support_lower_limit, support_upper_limit], maxdegree=10)

        # compare the two functions at various points
        for test_point in [2, 5, 8, 11]:
            n1 = cf1(test_point)
            n2 = cf2(test_point)

            assert abs(re(n1) - re(n2)) < 1e-12
            assert abs(im(n1) - im(n2)) < 1e-12

    test_cf(Beta('b', 1, 2), 0, 1)
    test_cf(Chi('c', 3), 0, mpmath.inf)
    test_cf(ChiSquared('c', 2), 0, mpmath.inf)
    test_cf(Exponential('e', 6), 0, mpmath.inf)
    test_cf(Logistic('l', 1, 2), -mpmath.inf, mpmath.inf)
    test_cf(Normal('n', -1, 5), -mpmath.inf, mpmath.inf)
    test_cf(RaisedCosine('r', 3, 1), 2, 4)
    test_cf(Rayleigh('r', 0.5), 0, mpmath.inf)
    test_cf(Uniform('u', -1, 1), -1, 1)
    test_cf(WignerSemicircle('w', 3), -3, 3)


def test_long_precomputed_cdf():
    x = symbols("x", real=True)
    distribs = [
            Arcsin("A", -5, 9),
            Dagum("D", 4, 10, 3),
            Erlang("E", 14, 5),
            Frechet("F", 2, 6, -3),
            Gamma("G", 2, 7),
            GammaInverse("GI", 3, 5),
            Kumaraswamy("K", 6, 8),
            Laplace("LA", -5, 4),
            Logistic("L", -6, 7),
            Nakagami("N", 2, 7),
            StudentT("S", 4)
            ]
    for distr in distribs:
        for _ in range(5):
            assert tn(diff(cdf(distr)(x), x), density(distr)(x), x, a=0, b=0, c=1, d=0)

    US = UniformSum("US", 5)
    pdf01 = density(US)(x).subs(floor(x), 0).doit()   # pdf on (0, 1)
    cdf01 = cdf(US, evaluate=False)(x).subs(floor(x), 0).doit()   # cdf on (0, 1)
    assert tn(diff(cdf01, x), pdf01, x, a=0, b=0, c=1, d=0)


def test_issue_13324():
    X = Uniform('X', 0, 1)
    assert E(X, X > S.Half) == Rational(3, 4)
    assert E(X, X > 0) == S.Half

def test_issue_20756():
    X = Uniform('X', -1, +1)
    Y = Uniform('Y', -1, +1)
    assert E(X * Y) == S.Zero
    assert E(X * ((Y + 1) - 1)) == S.Zero
    assert E(Y * (X*(X + 1) - X*X)) == S.Zero

def test_FiniteSet_prob():
    E = Exponential('E', 3)
    N = Normal('N', 5, 7)
    assert P(Eq(E, 1)) is S.Zero
    assert P(Eq(N, 2)) is S.Zero
    assert P(Eq(N, x)) is S.Zero

def test_prob_neq():
    E = Exponential('E', 4)
    X = ChiSquared('X', 4)
    assert P(Ne(E, 2)) == 1
    assert P(Ne(X, 4)) == 1
    assert P(Ne(X, 4)) == 1
    assert P(Ne(X, 5)) == 1
    assert P(Ne(E, x)) == 1

def test_union():
    N = Normal('N', 3, 2)
    assert simplify(P(N**2 - N > 2)) == \
        -erf(sqrt(2))/2 - erfc(sqrt(2)/4)/2 + Rational(3, 2)
    assert simplify(P(N**2 - 4 > 0)) == \
        -erf(5*sqrt(2)/4)/2 - erfc(sqrt(2)/4)/2 + Rational(3, 2)

def test_Or():
    N = Normal('N', 0, 1)
    assert simplify(P(Or(N > 2, N < 1))) == \
        -erf(sqrt(2))/2 - erfc(sqrt(2)/2)/2 + Rational(3, 2)
    assert P(Or(N < 0, N < 1)) == P(N < 1)
    assert P(Or(N > 0, N < 0)) == 1


def test_conditional_eq():
    E = Exponential('E', 1)
    assert P(Eq(E, 1), Eq(E, 1)) == 1
    assert P(Eq(E, 1), Eq(E, 2)) == 0
    assert P(E > 1, Eq(E, 2)) == 1
    assert P(E < 1, Eq(E, 2)) == 0

def test_ContinuousDistributionHandmade():
    x = Symbol('x')
    z = Dummy('z')
    dens = Lambda(x, Piecewise((S.Half, (0<=x)&(x<1)), (0, (x>=1)&(x<2)),
        (S.Half, (x>=2)&(x<3)), (0, True)))
    dens = ContinuousDistributionHandmade(dens, set=Interval(0, 3))
    space = SingleContinuousPSpace(z, dens)
    assert dens.pdf == Lambda(x, Piecewise((S(1)/2, (x >= 0) & (x < 1)),
        (0, (x >= 1) & (x < 2)), (S(1)/2, (x >= 2) & (x < 3)), (0, True)))
    assert median(space.value) == Interval(1, 2)
    assert E(space.value) == Rational(3, 2)
    assert variance(space.value) == Rational(13, 12)


def test_issue_16318():
    # test compute_expectation function of the SingleContinuousDomain
    N = SingleContinuousDomain(x, Interval(0, 1))
    raises(ValueError, lambda: SingleContinuousDomain.compute_expectation(N, x+1, {x, y}))

def test_compute_density():
    X = Normal('X', 0, Symbol("sigma")**2)
    raises(ValueError, lambda: density(X**5 + X))
