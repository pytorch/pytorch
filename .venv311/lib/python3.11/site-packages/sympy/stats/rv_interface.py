from sympy.sets import FiniteSet
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import FallingFactorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import piecewise_fold
from sympy.integrals.integrals import Integral
from sympy.solvers.solveset import solveset
from .rv import (probability, expectation, density, where, given, pspace, cdf, PSpace,
                 characteristic_function, sample, sample_iter, random_symbols, independent, dependent,
                 sampling_density, moment_generating_function, quantile, is_random,
                 sample_stochastic_process)


__all__ = ['P', 'E', 'H', 'density', 'where', 'given', 'sample', 'cdf',
        'characteristic_function', 'pspace', 'sample_iter', 'variance', 'std',
        'skewness', 'kurtosis', 'covariance', 'dependent', 'entropy', 'median',
        'independent', 'random_symbols', 'correlation', 'factorial_moment',
        'moment', 'cmoment', 'sampling_density', 'moment_generating_function',
        'smoment', 'quantile', 'sample_stochastic_process']



def moment(X, n, c=0, condition=None, *, evaluate=True, **kwargs):
    """
    Return the nth moment of a random expression about c.

    .. math::
        moment(X, c, n) = E((X-c)^{n})

    Default value of c is 0.

    Examples
    ========

    >>> from sympy.stats import Die, moment, E
    >>> X = Die('X', 6)
    >>> moment(X, 1, 6)
    -5/2
    >>> moment(X, 2)
    91/6
    >>> moment(X, 1) == E(X)
    True
    """
    from sympy.stats.symbolic_probability import Moment
    if evaluate:
        return Moment(X, n, c, condition).doit()
    return Moment(X, n, c, condition).rewrite(Integral)


def variance(X, condition=None, **kwargs):
    """
    Variance of a random expression.

    .. math::
        variance(X) = E((X-E(X))^{2})

    Examples
    ========

    >>> from sympy.stats import Die, Bernoulli, variance
    >>> from sympy import simplify, Symbol

    >>> X = Die('X', 6)
    >>> p = Symbol('p')
    >>> B = Bernoulli('B', p, 1, 0)

    >>> variance(2*X)
    35/3

    >>> simplify(variance(B))
    p*(1 - p)
    """
    if is_random(X) and pspace(X) == PSpace():
        from sympy.stats.symbolic_probability import Variance
        return Variance(X, condition)

    return cmoment(X, 2, condition, **kwargs)


def standard_deviation(X, condition=None, **kwargs):
    r"""
    Standard Deviation of a random expression

    .. math::
        std(X) = \sqrt(E((X-E(X))^{2}))

    Examples
    ========

    >>> from sympy.stats import Bernoulli, std
    >>> from sympy import Symbol, simplify

    >>> p = Symbol('p')
    >>> B = Bernoulli('B', p, 1, 0)

    >>> simplify(std(B))
    sqrt(p*(1 - p))
    """
    return sqrt(variance(X, condition, **kwargs))
std = standard_deviation

def entropy(expr, condition=None, **kwargs):
    """
    Calculates entropy of a probability distribution.

    Parameters
    ==========

    expression : the random expression whose entropy is to be calculated
    condition : optional, to specify conditions on random expression
    b: base of the logarithm, optional
       By default, it is taken as Euler's number

    Returns
    =======

    result : Entropy of the expression, a constant

    Examples
    ========

    >>> from sympy.stats import Normal, Die, entropy
    >>> X = Normal('X', 0, 1)
    >>> entropy(X)
    log(2)/2 + 1/2 + log(pi)/2

    >>> D = Die('D', 4)
    >>> entropy(D)
    log(4)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Entropy_%28information_theory%29
    .. [2] https://www.crmarsh.com/static/pdf/Charles_Marsh_Continuous_Entropy.pdf
    .. [3] https://kconrad.math.uconn.edu/blurbs/analysis/entropypost.pdf
    """
    pdf = density(expr, condition, **kwargs)
    base = kwargs.get('b', exp(1))
    if isinstance(pdf, dict):
            return sum(-prob*log(prob, base) for prob in pdf.values())
    return expectation(-log(pdf(expr), base))

def covariance(X, Y, condition=None, **kwargs):
    """
    Covariance of two random expressions.

    Explanation
    ===========

    The expectation that the two variables will rise and fall together

    .. math::
        covariance(X,Y) = E((X-E(X)) (Y-E(Y)))

    Examples
    ========

    >>> from sympy.stats import Exponential, covariance
    >>> from sympy import Symbol

    >>> rate = Symbol('lambda', positive=True, real=True)
    >>> X = Exponential('X', rate)
    >>> Y = Exponential('Y', rate)

    >>> covariance(X, X)
    lambda**(-2)
    >>> covariance(X, Y)
    0
    >>> covariance(X, Y + rate*X)
    1/lambda
    """
    if (is_random(X) and pspace(X) == PSpace()) or (is_random(Y) and pspace(Y) == PSpace()):
        from sympy.stats.symbolic_probability import Covariance
        return Covariance(X, Y, condition)

    return expectation(
        (X - expectation(X, condition, **kwargs)) *
        (Y - expectation(Y, condition, **kwargs)),
        condition, **kwargs)


def correlation(X, Y, condition=None, **kwargs):
    r"""
    Correlation of two random expressions, also known as correlation
    coefficient or Pearson's correlation.

    Explanation
    ===========

    The normalized expectation that the two variables will rise
    and fall together

    .. math::
        correlation(X,Y) = E((X-E(X))(Y-E(Y)) / (\sigma_x  \sigma_y))

    Examples
    ========

    >>> from sympy.stats import Exponential, correlation
    >>> from sympy import Symbol

    >>> rate = Symbol('lambda', positive=True, real=True)
    >>> X = Exponential('X', rate)
    >>> Y = Exponential('Y', rate)

    >>> correlation(X, X)
    1
    >>> correlation(X, Y)
    0
    >>> correlation(X, Y + rate*X)
    1/sqrt(1 + lambda**(-2))
    """
    return covariance(X, Y, condition, **kwargs)/(std(X, condition, **kwargs)
     * std(Y, condition, **kwargs))


def cmoment(X, n, condition=None, *, evaluate=True, **kwargs):
    """
    Return the nth central moment of a random expression about its mean.

    .. math::
        cmoment(X, n) = E((X - E(X))^{n})

    Examples
    ========

    >>> from sympy.stats import Die, cmoment, variance
    >>> X = Die('X', 6)
    >>> cmoment(X, 3)
    0
    >>> cmoment(X, 2)
    35/12
    >>> cmoment(X, 2) == variance(X)
    True
    """
    from sympy.stats.symbolic_probability import CentralMoment
    if evaluate:
        return CentralMoment(X, n, condition).doit()
    return CentralMoment(X, n, condition).rewrite(Integral)


def smoment(X, n, condition=None, **kwargs):
    r"""
    Return the nth Standardized moment of a random expression.

    .. math::
        smoment(X, n) = E(((X - \mu)/\sigma_X)^{n})

    Examples
    ========

    >>> from sympy.stats import skewness, Exponential, smoment
    >>> from sympy import Symbol
    >>> rate = Symbol('lambda', positive=True, real=True)
    >>> Y = Exponential('Y', rate)
    >>> smoment(Y, 4)
    9
    >>> smoment(Y, 4) == smoment(3*Y, 4)
    True
    >>> smoment(Y, 3) == skewness(Y)
    True
    """
    sigma = std(X, condition, **kwargs)
    return (1/sigma)**n*cmoment(X, n, condition, **kwargs)

def skewness(X, condition=None, **kwargs):
    r"""
    Measure of the asymmetry of the probability distribution.

    Explanation
    ===========

    Positive skew indicates that most of the values lie to the right of
    the mean.

    .. math::
        skewness(X) = E(((X - E(X))/\sigma_X)^{3})

    Parameters
    ==========

    condition : Expr containing RandomSymbols
            A conditional expression. skewness(X, X>0) is skewness of X given X > 0

    Examples
    ========

    >>> from sympy.stats import skewness, Exponential, Normal
    >>> from sympy import Symbol
    >>> X = Normal('X', 0, 1)
    >>> skewness(X)
    0
    >>> skewness(X, X > 0) # find skewness given X > 0
    (-sqrt(2)/sqrt(pi) + 4*sqrt(2)/pi**(3/2))/(1 - 2/pi)**(3/2)

    >>> rate = Symbol('lambda', positive=True, real=True)
    >>> Y = Exponential('Y', rate)
    >>> skewness(Y)
    2
    """
    return smoment(X, 3, condition=condition, **kwargs)

def kurtosis(X, condition=None, **kwargs):
    r"""
    Characterizes the tails/outliers of a probability distribution.

    Explanation
    ===========

    Kurtosis of any univariate normal distribution is 3. Kurtosis less than
    3 means that the distribution produces fewer and less extreme outliers
    than the normal distribution.

    .. math::
        kurtosis(X) = E(((X - E(X))/\sigma_X)^{4})

    Parameters
    ==========

    condition : Expr containing RandomSymbols
            A conditional expression. kurtosis(X, X>0) is kurtosis of X given X > 0

    Examples
    ========

    >>> from sympy.stats import kurtosis, Exponential, Normal
    >>> from sympy import Symbol
    >>> X = Normal('X', 0, 1)
    >>> kurtosis(X)
    3
    >>> kurtosis(X, X > 0) # find kurtosis given X > 0
    (-4/pi - 12/pi**2 + 3)/(1 - 2/pi)**2

    >>> rate = Symbol('lamda', positive=True, real=True)
    >>> Y = Exponential('Y', rate)
    >>> kurtosis(Y)
    9

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kurtosis
    .. [2] https://mathworld.wolfram.com/Kurtosis.html
    """
    return smoment(X, 4, condition=condition, **kwargs)


def factorial_moment(X, n, condition=None, **kwargs):
    """
    The factorial moment is a mathematical quantity defined as the expectation
    or average of the falling factorial of a random variable.

    .. math::
        factorial-moment(X, n) = E(X(X - 1)(X - 2)...(X - n + 1))

    Parameters
    ==========

    n: A natural number, n-th factorial moment.

    condition : Expr containing RandomSymbols
            A conditional expression.

    Examples
    ========

    >>> from sympy.stats import factorial_moment, Poisson, Binomial
    >>> from sympy import Symbol, S
    >>> lamda = Symbol('lamda')
    >>> X = Poisson('X', lamda)
    >>> factorial_moment(X, 2)
    lamda**2
    >>> Y = Binomial('Y', 2, S.Half)
    >>> factorial_moment(Y, 2)
    1/2
    >>> factorial_moment(Y, 2, Y > 1) # find factorial moment for Y > 1
    2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Factorial_moment
    .. [2] https://mathworld.wolfram.com/FactorialMoment.html
    """
    return expectation(FallingFactorial(X, n), condition=condition, **kwargs)

def median(X, evaluate=True, **kwargs):
    r"""
    Calculates the median of the probability distribution.

    Explanation
    ===========

    Mathematically, median of Probability distribution is defined as all those
    values of `m` for which the following condition is satisfied

    .. math::
        P(X\leq m) \geq  \frac{1}{2} \text{ and} \text{ } P(X\geq m)\geq \frac{1}{2}

    Parameters
    ==========

    X: The random expression whose median is to be calculated.

    Returns
    =======

    The FiniteSet or an Interval which contains the median of the
    random expression.

    Examples
    ========

    >>> from sympy.stats import Normal, Die, median
    >>> N = Normal('N', 3, 1)
    >>> median(N)
    {3}
    >>> D = Die('D')
    >>> median(D)
    {3, 4}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Median#Probability_distributions

    """
    if not is_random(X):
        return X

    from sympy.stats.crv import ContinuousPSpace
    from sympy.stats.drv import DiscretePSpace
    from sympy.stats.frv import FinitePSpace

    if isinstance(pspace(X), FinitePSpace):
        cdf = pspace(X).compute_cdf(X)
        result = []
        for key, value in cdf.items():
            if value>= Rational(1, 2) and (1 - value) + \
            pspace(X).probability(Eq(X, key)) >= Rational(1, 2):
                result.append(key)
        return FiniteSet(*result)
    if isinstance(pspace(X), (ContinuousPSpace, DiscretePSpace)):
        cdf = pspace(X).compute_cdf(X)
        x = Dummy('x')
        result = solveset(piecewise_fold(cdf(x) - Rational(1, 2)), x, pspace(X).set)
        return result
    raise NotImplementedError("The median of %s is not implemented."%str(pspace(X)))


def coskewness(X, Y, Z, condition=None, **kwargs):
    r"""
    Calculates the co-skewness of three random variables.

    Explanation
    ===========

    Mathematically Coskewness is defined as

    .. math::
        coskewness(X,Y,Z)=\frac{E[(X-E[X]) * (Y-E[Y]) * (Z-E[Z])]} {\sigma_{X}\sigma_{Y}\sigma_{Z}}

    Parameters
    ==========

    X : RandomSymbol
            Random Variable used to calculate coskewness
    Y : RandomSymbol
            Random Variable used to calculate coskewness
    Z : RandomSymbol
            Random Variable used to calculate coskewness
    condition : Expr containing RandomSymbols
            A conditional expression

    Examples
    ========

    >>> from sympy.stats import coskewness, Exponential, skewness
    >>> from sympy import symbols
    >>> p = symbols('p', positive=True)
    >>> X = Exponential('X', p)
    >>> Y = Exponential('Y', 2*p)
    >>> coskewness(X, Y, Y)
    0
    >>> coskewness(X, Y + X, Y + 2*X)
    16*sqrt(85)/85
    >>> coskewness(X + 2*Y, Y + X, Y + 2*X, X > 3)
    9*sqrt(170)/85
    >>> coskewness(Y, Y, Y) == skewness(Y)
    True
    >>> coskewness(X, Y + p*X, Y + 2*p*X)
    4/(sqrt(1 + 1/(4*p**2))*sqrt(4 + 1/(4*p**2)))

    Returns
    =======

    coskewness : The coskewness of the three random variables

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Coskewness

    """
    num = expectation((X - expectation(X, condition, **kwargs)) \
         * (Y - expectation(Y, condition, **kwargs)) \
         * (Z - expectation(Z, condition, **kwargs)), condition, **kwargs)
    den = std(X, condition, **kwargs) * std(Y, condition, **kwargs) \
         * std(Z, condition, **kwargs)
    return num/den


P = probability
E = expectation
H = entropy
