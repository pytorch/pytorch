"""

Contains
========
FlorySchulz
Geometric
Hermite
Logarithmic
NegativeBinomial
Poisson
Skellam
YuleSimon
Zeta
"""



from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial, FallingFactorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besseli
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.hyper import hyper
from sympy.functions.special.zeta_functions import (polylog, zeta)
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import _value_check, is_random


__all__ = ['FlorySchulz',
'Geometric',
'Hermite',
'Logarithmic',
'NegativeBinomial',
'Poisson',
'Skellam',
'YuleSimon',
'Zeta'
]


def rv(symbol, cls, *args, **kwargs):
    args = list(map(sympify, args))
    dist = cls(*args)
    if kwargs.pop('check', True):
        dist.check(*args)
    pspace = SingleDiscretePSpace(symbol, dist)
    if any(is_random(arg) for arg in args):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(symbol, CompoundDistribution(dist))
    return pspace.value


class DiscreteDistributionHandmade(SingleDiscreteDistribution):
    _argnames = ('pdf',)

    def __new__(cls, pdf, set=S.Integers):
        return Basic.__new__(cls, pdf, set)

    @property
    def set(self):
        return self.args[1]

    @staticmethod
    def check(pdf, set):
        x = Dummy('x')
        val = Sum(pdf(x), (x, set._inf, set._sup)).doit()
        _value_check(Eq(val, 1) != S.false, "The pdf is incorrect on the given set.")



def DiscreteRV(symbol, density, set=S.Integers, **kwargs):
    """
    Create a Discrete Random Variable given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents name of the random variable.
    density : Expression containing symbol
        Represents probability density function.
    set : set
        Represents the region where the pdf is valid, by default is real line.
    check : bool
        If True, it will check whether the given density
        integrates to 1 over the given set. If False, it
        will not perform this check. Default is False.

    Examples
    ========

    >>> from sympy.stats import DiscreteRV, P, E
    >>> from sympy import Rational, Symbol
    >>> x = Symbol('x')
    >>> n = 10
    >>> density = Rational(1, 10)
    >>> X = DiscreteRV(x, density, set=set(range(n)))
    >>> E(X)
    9/2
    >>> P(X>3)
    3/5

    Returns
    =======

    RandomSymbol

    """
    set = sympify(set)
    pdf = Piecewise((density, set.as_relational(symbol)), (0, True))
    pdf = Lambda(symbol, pdf)
    # have a default of False while `rv` should have a default of True
    kwargs['check'] = kwargs.pop('check', False)
    return rv(symbol.name, DiscreteDistributionHandmade, pdf, set, **kwargs)


#-------------------------------------------------------------------------------
# Flory-Schulz distribution ------------------------------------------------------------

class FlorySchulzDistribution(SingleDiscreteDistribution):
    _argnames = ('a',)
    set = S.Naturals

    @staticmethod
    def check(a):
        _value_check((0 < a, a < 1), "a must be between 0 and 1")

    def pdf(self, k):
        a = self.a
        return (a**2 * k * (1 - a)**(k - 1))

    def _characteristic_function(self, t):
        a = self.a
        return a**2*exp(I*t)/((1 + (a - 1)*exp(I*t))**2)

    def _moment_generating_function(self, t):
        a = self.a
        return a**2*exp(t)/((1 + (a - 1)*exp(t))**2)


def FlorySchulz(name, a):
    r"""
    Create a discrete random variable with a FlorySchulz distribution.

    The density of the FlorySchulz distribution is given by

    .. math::
        f(k) := (a^2) k (1 - a)^{k-1}

    Parameters
    ==========

    a : A real number between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, E, variance, FlorySchulz
    >>> from sympy import Symbol, S

    >>> a = S.One / 5
    >>> z = Symbol("z")

    >>> X = FlorySchulz("x", a)

    >>> density(X)(z)
    (4/5)**(z - 1)*z/25

    >>> E(X)
    9

    >>> variance(X)
    40

    References
    ==========

    https://en.wikipedia.org/wiki/Flory%E2%80%93Schulz_distribution
    """
    return rv(name, FlorySchulzDistribution, a)


#-------------------------------------------------------------------------------
# Geometric distribution ------------------------------------------------------------

class GeometricDistribution(SingleDiscreteDistribution):
    _argnames = ('p',)
    set = S.Naturals

    @staticmethod
    def check(p):
        _value_check((0 < p, p <= 1), "p must be between 0 and 1")

    def pdf(self, k):
        return (1 - self.p)**(k - 1) * self.p

    def _characteristic_function(self, t):
        p = self.p
        return p * exp(I*t) / (1 - (1 - p)*exp(I*t))

    def _moment_generating_function(self, t):
        p = self.p
        return p * exp(t) / (1 - (1 - p) * exp(t))


def Geometric(name, p):
    r"""
    Create a discrete random variable with a Geometric distribution.

    Explanation
    ===========

    The density of the Geometric distribution is given by

    .. math::
        f(k) := p (1 - p)^{k - 1}

    Parameters
    ==========

    p : A probability between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Geometric, density, E, variance
    >>> from sympy import Symbol, S

    >>> p = S.One / 5
    >>> z = Symbol("z")

    >>> X = Geometric("x", p)

    >>> density(X)(z)
    (4/5)**(z - 1)/5

    >>> E(X)
    5

    >>> variance(X)
    20

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Geometric_distribution
    .. [2] https://mathworld.wolfram.com/GeometricDistribution.html

    """
    return rv(name, GeometricDistribution, p)


#-------------------------------------------------------------------------------
# Hermite distribution ---------------------------------------------------------


class HermiteDistribution(SingleDiscreteDistribution):
    _argnames = ('a1', 'a2')
    set = S.Naturals0

    @staticmethod
    def check(a1, a2):
        _value_check(a1.is_nonnegative, 'Parameter a1 must be >= 0.')
        _value_check(a2.is_nonnegative, 'Parameter a2 must be >= 0.')

    def pdf(self, k):
        a1, a2 = self.a1, self.a2
        term1 = exp(-(a1 + a2))
        j = Dummy("j", integer=True)
        num = a1**(k - 2*j) * a2**j
        den = factorial(k - 2*j) * factorial(j)
        return term1 * Sum(num/den, (j, 0, k//2)).doit()

    def _moment_generating_function(self, t):
        a1, a2 = self.a1, self.a2
        term1 = a1 * (exp(t) - 1)
        term2 = a2 * (exp(2*t) - 1)
        return exp(term1 + term2)

    def _characteristic_function(self, t):
        a1, a2 = self.a1, self.a2
        term1 = a1 * (exp(I*t) - 1)
        term2 = a2 * (exp(2*I*t) - 1)
        return exp(term1 + term2)

def Hermite(name, a1, a2):
    r"""
    Create a discrete random variable with a Hermite distribution.

    Explanation
    ===========

    The density of the Hermite distribution is given by

    .. math::
        f(x):= e^{-a_1 -a_2}\sum_{j=0}^{\left \lfloor x/2 \right \rfloor}
                    \frac{a_{1}^{x-2j}a_{2}^{j}}{(x-2j)!j!}

    Parameters
    ==========

    a1 : A Positive number greater than equal to 0.
    a2 : A Positive number greater than equal to 0.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Hermite, density, E, variance
    >>> from sympy import Symbol

    >>> a1 = Symbol("a1", positive=True)
    >>> a2 = Symbol("a2", positive=True)
    >>> x = Symbol("x")

    >>> H = Hermite("H", a1=5, a2=4)

    >>> density(H)(2)
    33*exp(-9)/2

    >>> E(H)
    13

    >>> variance(H)
    21

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_distribution

    """

    return rv(name, HermiteDistribution, a1, a2)


#-------------------------------------------------------------------------------
# Logarithmic distribution ------------------------------------------------------------

class LogarithmicDistribution(SingleDiscreteDistribution):
    _argnames = ('p',)

    set = S.Naturals

    @staticmethod
    def check(p):
        _value_check((p > 0, p < 1), "p should be between 0 and 1")

    def pdf(self, k):
        p = self.p
        return (-1) * p**k / (k * log(1 - p))

    def _characteristic_function(self, t):
        p = self.p
        return log(1 - p * exp(I*t)) / log(1 - p)

    def _moment_generating_function(self, t):
        p = self.p
        return log(1 - p * exp(t)) / log(1 - p)


def Logarithmic(name, p):
    r"""
    Create a discrete random variable with a Logarithmic distribution.

    Explanation
    ===========

    The density of the Logarithmic distribution is given by

    .. math::
        f(k) := \frac{-p^k}{k \ln{(1 - p)}}

    Parameters
    ==========

    p : A value between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Logarithmic, density, E, variance
    >>> from sympy import Symbol, S

    >>> p = S.One / 5
    >>> z = Symbol("z")

    >>> X = Logarithmic("x", p)

    >>> density(X)(z)
    -1/(5**z*z*log(4/5))

    >>> E(X)
    -1/(-4*log(5) + 8*log(2))

    >>> variance(X)
    -1/((-4*log(5) + 8*log(2))*(-2*log(5) + 4*log(2))) + 1/(-64*log(2)*log(5) + 64*log(2)**2 + 16*log(5)**2) - 10/(-32*log(5) + 64*log(2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_distribution
    .. [2] https://mathworld.wolfram.com/LogarithmicDistribution.html

    """
    return rv(name, LogarithmicDistribution, p)


#-------------------------------------------------------------------------------
# Negative binomial distribution ------------------------------------------------------------

class NegativeBinomialDistribution(SingleDiscreteDistribution):
    _argnames = ('r', 'p')
    set = S.Naturals0

    @staticmethod
    def check(r, p):
        _value_check(r > 0, 'r should be positive')
        _value_check((p > 0, p < 1), 'p should be between 0 and 1')

    def pdf(self, k):
        r = self.r
        p = self.p

        return binomial(k + r - 1, k) * (1 - p)**k * p**r

    def _characteristic_function(self, t):
        r = self.r
        p = self.p

        return (p / (1 - (1 - p) * exp(I*t)))**r

    def _moment_generating_function(self, t):
        r = self.r
        p = self.p

        return (p / (1 - (1 - p) * exp(t)))**r

def NegativeBinomial(name, r, p):
    r"""
    Create a discrete random variable with a Negative Binomial distribution.

    Explanation
    ===========

    The density of the Negative Binomial distribution is given by

    .. math::
        f(k) := \binom{k + r - 1}{k} (1 - p)^k p^r

    Parameters
    ==========

    r : A positive value
        Number of successes until the experiment is stopped.
    p : A value between 0 and 1
        Probability of success.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import NegativeBinomial, density, E, variance
    >>> from sympy import Symbol, S

    >>> r = 5
    >>> p = S.One / 3
    >>> z = Symbol("z")

    >>> X = NegativeBinomial("x", r, p)

    >>> density(X)(z)
    (2/3)**z*binomial(z + 4, z)/243

    >>> E(X)
    10

    >>> variance(X)
    30

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Negative_binomial_distribution
    .. [2] https://mathworld.wolfram.com/NegativeBinomialDistribution.html

    """
    return rv(name, NegativeBinomialDistribution, r, p)


#-------------------------------------------------------------------------------
# Poisson distribution ------------------------------------------------------------

class PoissonDistribution(SingleDiscreteDistribution):
    _argnames = ('lamda',)

    set = S.Naturals0

    @staticmethod
    def check(lamda):
        _value_check(lamda > 0, "Lambda must be positive")

    def pdf(self, k):
        return self.lamda**k / factorial(k) * exp(-self.lamda)

    def _characteristic_function(self, t):
        return exp(self.lamda * (exp(I*t) - 1))

    def _moment_generating_function(self, t):
        return exp(self.lamda * (exp(t) - 1))

    def expectation(self, expr, var, evaluate=True, **kwargs):
        if evaluate:
            if expr == var:
                return self.lamda
            if (
                isinstance(expr, FallingFactorial)
                and expr.args[1].is_integer
                and expr.args[1].is_positive
                and expr.args[0] == var
            ):
                return self.lamda ** expr.args[1]
        return super().expectation(expr, var, evaluate, **kwargs)

def Poisson(name, lamda):
    r"""
    Create a discrete random variable with a Poisson distribution.

    Explanation
    ===========

    The density of the Poisson distribution is given by

    .. math::
        f(k) := \frac{\lambda^{k} e^{- \lambda}}{k!}

    Parameters
    ==========

    lamda : Positive number, a rate

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Poisson, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> rate = Symbol("lambda", positive=True)
    >>> z = Symbol("z")

    >>> X = Poisson("x", rate)

    >>> density(X)(z)
    lambda**z*exp(-lambda)/factorial(z)

    >>> E(X)
    lambda

    >>> simplify(variance(X))
    lambda

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution
    .. [2] https://mathworld.wolfram.com/PoissonDistribution.html

    """
    return rv(name, PoissonDistribution, lamda)


# -----------------------------------------------------------------------------
# Skellam distribution --------------------------------------------------------


class SkellamDistribution(SingleDiscreteDistribution):
    _argnames = ('mu1', 'mu2')
    set = S.Integers

    @staticmethod
    def check(mu1, mu2):
        _value_check(mu1 >= 0, 'Parameter mu1 must be >= 0')
        _value_check(mu2 >= 0, 'Parameter mu2 must be >= 0')

    def pdf(self, k):
        (mu1, mu2) = (self.mu1, self.mu2)
        term1 = exp(-(mu1 + mu2)) * (mu1 / mu2) ** (k / 2)
        term2 = besseli(k, 2 * sqrt(mu1 * mu2))
        return term1 * term2

    def _cdf(self, x):
        raise NotImplementedError(
            "Skellam doesn't have closed form for the CDF.")

    def _characteristic_function(self, t):
        (mu1, mu2) = (self.mu1, self.mu2)
        return exp(-(mu1 + mu2) + mu1 * exp(I * t) + mu2 * exp(-I * t))

    def _moment_generating_function(self, t):
        (mu1, mu2) = (self.mu1, self.mu2)
        return exp(-(mu1 + mu2) + mu1 * exp(t) + mu2 * exp(-t))


def Skellam(name, mu1, mu2):
    r"""
    Create a discrete random variable with a Skellam distribution.

    Explanation
    ===========

    The Skellam is the distribution of the difference N1 - N2
    of two statistically independent random variables N1 and N2
    each Poisson-distributed with respective expected values mu1 and mu2.

    The density of the Skellam distribution is given by

    .. math::
        f(k) := e^{-(\mu_1+\mu_2)}(\frac{\mu_1}{\mu_2})^{k/2}I_k(2\sqrt{\mu_1\mu_2})

    Parameters
    ==========

    mu1 : A non-negative value
    mu2 : A non-negative value

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Skellam, density, E, variance
    >>> from sympy import Symbol, pprint

    >>> z = Symbol("z", integer=True)
    >>> mu1 = Symbol("mu1", positive=True)
    >>> mu2 = Symbol("mu2", positive=True)
    >>> X = Skellam("x", mu1, mu2)

    >>> pprint(density(X)(z), use_unicode=False)
         z
         -
         2
    /mu1\   -mu1 - mu2        /       _____   _____\
    |---| *e          *besseli\z, 2*\/ mu1 *\/ mu2 /
    \mu2/
    >>> E(X)
    mu1 - mu2
    >>> variance(X).expand()
    mu1 + mu2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Skellam_distribution

    """
    return rv(name, SkellamDistribution, mu1, mu2)


#-------------------------------------------------------------------------------
# Yule-Simon distribution ------------------------------------------------------------

class YuleSimonDistribution(SingleDiscreteDistribution):
    _argnames = ('rho',)
    set = S.Naturals

    @staticmethod
    def check(rho):
        _value_check(rho > 0, 'rho should be positive')

    def pdf(self, k):
        rho = self.rho
        return rho * beta(k, rho + 1)

    def _cdf(self, x):
        return Piecewise((1 - floor(x) * beta(floor(x), self.rho + 1), x >= 1), (0, True))

    def _characteristic_function(self, t):
        rho = self.rho
        return rho * hyper((1, 1), (rho + 2,), exp(I*t)) * exp(I*t) / (rho + 1)

    def _moment_generating_function(self, t):
        rho = self.rho
        return rho * hyper((1, 1), (rho + 2,), exp(t)) * exp(t) / (rho + 1)


def YuleSimon(name, rho):
    r"""
    Create a discrete random variable with a Yule-Simon distribution.

    Explanation
    ===========

    The density of the Yule-Simon distribution is given by

    .. math::
        f(k) := \rho B(k, \rho + 1)

    Parameters
    ==========

    rho : A positive value

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import YuleSimon, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> p = 5
    >>> z = Symbol("z")

    >>> X = YuleSimon("x", p)

    >>> density(X)(z)
    5*beta(z, 6)

    >>> simplify(E(X))
    5/4

    >>> simplify(variance(X))
    25/48

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Yule%E2%80%93Simon_distribution

    """
    return rv(name, YuleSimonDistribution, rho)


#-------------------------------------------------------------------------------
# Zeta distribution ------------------------------------------------------------

class ZetaDistribution(SingleDiscreteDistribution):
    _argnames = ('s',)
    set = S.Naturals

    @staticmethod
    def check(s):
        _value_check(s > 1, 's should be greater than 1')

    def pdf(self, k):
        s = self.s
        return 1 / (k**s * zeta(s))

    def _characteristic_function(self, t):
        return polylog(self.s, exp(I*t)) / zeta(self.s)

    def _moment_generating_function(self, t):
        return polylog(self.s, exp(t)) / zeta(self.s)


def Zeta(name, s):
    r"""
    Create a discrete random variable with a Zeta distribution.

    Explanation
    ===========

    The density of the Zeta distribution is given by

    .. math::
        f(k) := \frac{1}{k^s \zeta{(s)}}

    Parameters
    ==========

    s : A value greater than 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Zeta, density, E, variance
    >>> from sympy import Symbol

    >>> s = 5
    >>> z = Symbol("z")

    >>> X = Zeta("x", s)

    >>> density(X)(z)
    1/(z**5*zeta(5))

    >>> E(X)
    pi**4/(90*zeta(5))

    >>> variance(X)
    -pi**8/(8100*zeta(5)**2) + zeta(3)/zeta(5)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Zeta_distribution

    """
    return rv(name, ZetaDistribution, s)
