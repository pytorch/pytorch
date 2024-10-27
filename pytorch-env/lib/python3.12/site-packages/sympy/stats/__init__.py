"""
SymPy statistics module

Introduces a random variable type into the SymPy language.

Random variables may be declared using prebuilt functions such as
Normal, Exponential, Coin, Die, etc...  or built with functions like FiniteRV.

Queries on random expressions can be made using the functions

========================= =============================
    Expression                    Meaning
------------------------- -----------------------------
 ``P(condition)``          Probability
 ``E(expression)``         Expected value
 ``H(expression)``         Entropy
 ``variance(expression)``  Variance
 ``density(expression)``   Probability Density Function
 ``sample(expression)``    Produce a realization
 ``where(condition)``      Where the condition is true
========================= =============================

Examples
========

>>> from sympy.stats import P, E, variance, Die, Normal
>>> from sympy import simplify
>>> X, Y = Die('X', 6), Die('Y', 6) # Define two six sided dice
>>> Z = Normal('Z', 0, 1) # Declare a Normal random variable with mean 0, std 1
>>> P(X>3) # Probability X is greater than 3
1/2
>>> E(X+Y) # Expectation of the sum of two dice
7
>>> variance(X+Y) # Variance of the sum of two dice
35/6
>>> simplify(P(Z>1)) # Probability of Z being greater than 1
1/2 - erf(sqrt(2)/2)/2


One could also create custom distribution and define custom random variables
as follows:

1. If you want to create a Continuous Random Variable:

>>> from sympy.stats import ContinuousRV, P, E
>>> from sympy import exp, Symbol, Interval, oo
>>> x = Symbol('x')
>>> pdf = exp(-x) # pdf of the Continuous Distribution
>>> Z = ContinuousRV(x, pdf, set=Interval(0, oo))
>>> E(Z)
1
>>> P(Z > 5)
exp(-5)

1.1 To create an instance of Continuous Distribution:

>>> from sympy.stats import ContinuousDistributionHandmade
>>> from sympy import Lambda
>>> dist = ContinuousDistributionHandmade(Lambda(x, pdf), set=Interval(0, oo))
>>> dist.pdf(x)
exp(-x)

2. If you want to create a Discrete Random Variable:

>>> from sympy.stats import DiscreteRV, P, E
>>> from sympy import Symbol, S
>>> p = S(1)/2
>>> x = Symbol('x', integer=True, positive=True)
>>> pdf = p*(1 - p)**(x - 1)
>>> D = DiscreteRV(x, pdf, set=S.Naturals)
>>> E(D)
2
>>> P(D > 3)
1/8

2.1 To create an instance of Discrete Distribution:

>>> from sympy.stats import DiscreteDistributionHandmade
>>> from sympy import Lambda
>>> dist = DiscreteDistributionHandmade(Lambda(x, pdf), set=S.Naturals)
>>> dist.pdf(x)
2**(1 - x)/2

3. If you want to create a Finite Random Variable:

>>> from sympy.stats import FiniteRV, P, E
>>> from sympy import Rational, Eq
>>> pmf = {1: Rational(1, 3), 2: Rational(1, 6), 3: Rational(1, 4), 4: Rational(1, 4)}
>>> X = FiniteRV('X', pmf)
>>> E(X)
29/12
>>> P(X > 3)
1/4

3.1 To create an instance of Finite Distribution:

>>> from sympy.stats import FiniteDistributionHandmade
>>> dist = FiniteDistributionHandmade(pmf)
>>> dist.pmf(x)
Lambda(x, Piecewise((1/3, Eq(x, 1)), (1/6, Eq(x, 2)), (1/4, Eq(x, 3) | Eq(x, 4)), (0, True)))
"""

__all__ = [
    'P', 'E', 'H', 'density', 'where', 'given', 'sample', 'cdf','median',
    'characteristic_function', 'pspace', 'sample_iter', 'variance', 'std',
    'skewness', 'kurtosis', 'covariance', 'dependent', 'entropy', 'independent',
    'random_symbols', 'correlation', 'factorial_moment', 'moment', 'cmoment',
    'sampling_density', 'moment_generating_function', 'smoment', 'quantile',
    'coskewness', 'sample_stochastic_process',

    'FiniteRV', 'DiscreteUniform', 'Die', 'Bernoulli', 'Coin', 'Binomial',
    'BetaBinomial', 'Hypergeometric', 'Rademacher', 'IdealSoliton', 'RobustSoliton',
    'FiniteDistributionHandmade',

    'ContinuousRV', 'Arcsin', 'Benini', 'Beta', 'BetaNoncentral', 'BetaPrime',
    'BoundedPareto', 'Cauchy', 'Chi', 'ChiNoncentral', 'ChiSquared', 'Dagum', 'Davis', 'Erlang',
    'ExGaussian', 'Exponential', 'ExponentialPower', 'FDistribution',
    'FisherZ', 'Frechet', 'Gamma', 'GammaInverse', 'Gompertz', 'Gumbel',
    'Kumaraswamy', 'Laplace', 'Levy', 'Logistic','LogCauchy', 'LogLogistic', 'LogitNormal', 'LogNormal', 'Lomax',
    'Moyal', 'Maxwell', 'Nakagami', 'Normal', 'GaussianInverse', 'Pareto', 'PowerFunction',
    'QuadraticU', 'RaisedCosine', 'Rayleigh','Reciprocal', 'StudentT', 'ShiftedGompertz',
    'Trapezoidal', 'Triangular', 'Uniform', 'UniformSum', 'VonMises', 'Wald',
    'Weibull', 'WignerSemicircle', 'ContinuousDistributionHandmade',

    'FlorySchulz', 'Geometric','Hermite', 'Logarithmic', 'NegativeBinomial', 'Poisson', 'Skellam',
    'YuleSimon', 'Zeta', 'DiscreteRV', 'DiscreteDistributionHandmade',

    'JointRV', 'Dirichlet', 'GeneralizedMultivariateLogGamma',
    'GeneralizedMultivariateLogGammaOmega', 'Multinomial', 'MultivariateBeta',
    'MultivariateEwens', 'MultivariateT', 'NegativeMultinomial',
    'NormalGamma', 'MultivariateNormal', 'MultivariateLaplace', 'marginal_distribution',

    'StochasticProcess', 'DiscreteTimeStochasticProcess',
    'DiscreteMarkovChain', 'TransitionMatrixOf', 'StochasticStateSpaceOf',
    'GeneratorMatrixOf', 'ContinuousMarkovChain', 'BernoulliProcess',
    'PoissonProcess', 'WienerProcess', 'GammaProcess',

    'CircularEnsemble', 'CircularUnitaryEnsemble',
    'CircularOrthogonalEnsemble', 'CircularSymplecticEnsemble',
    'GaussianEnsemble', 'GaussianUnitaryEnsemble',
    'GaussianOrthogonalEnsemble', 'GaussianSymplecticEnsemble',
    'joint_eigen_distribution', 'JointEigenDistribution',
    'level_spacing_distribution',

    'MatrixGamma', 'Wishart', 'MatrixNormal', 'MatrixStudentT',

    'Probability', 'Expectation', 'Variance', 'Covariance', 'Moment',
    'CentralMoment',

    'ExpectationMatrix', 'VarianceMatrix', 'CrossCovarianceMatrix'

]
from .rv_interface import (P, E, H, density, where, given, sample, cdf, median,
        characteristic_function, pspace, sample_iter, variance, std, skewness,
        kurtosis, covariance, dependent, entropy, independent, random_symbols,
        correlation, factorial_moment, moment, cmoment, sampling_density,
        moment_generating_function, smoment, quantile, coskewness,
        sample_stochastic_process)

from .frv_types import (FiniteRV, DiscreteUniform, Die, Bernoulli, Coin,
        Binomial, BetaBinomial, Hypergeometric, Rademacher,
        FiniteDistributionHandmade, IdealSoliton, RobustSoliton)

from .crv_types import (ContinuousRV, Arcsin, Benini, Beta, BetaNoncentral,
        BetaPrime, BoundedPareto, Cauchy, Chi, ChiNoncentral, ChiSquared,
        Dagum, Davis, Erlang, ExGaussian, Exponential, ExponentialPower,
        FDistribution, FisherZ, Frechet, Gamma, GammaInverse, GaussianInverse,
        Gompertz, Gumbel, Kumaraswamy, Laplace, Levy, Logistic, LogCauchy,
        LogLogistic, LogitNormal, LogNormal, Lomax, Maxwell, Moyal, Nakagami,
        Normal, Pareto, QuadraticU, RaisedCosine, Rayleigh, Reciprocal,
        StudentT, PowerFunction, ShiftedGompertz, Trapezoidal, Triangular,
        Uniform, UniformSum, VonMises, Wald, Weibull, WignerSemicircle,
        ContinuousDistributionHandmade)

from .drv_types import (FlorySchulz, Geometric, Hermite, Logarithmic, NegativeBinomial, Poisson,
        Skellam, YuleSimon, Zeta, DiscreteRV, DiscreteDistributionHandmade)

from .joint_rv_types import (JointRV, Dirichlet,
        GeneralizedMultivariateLogGamma, GeneralizedMultivariateLogGammaOmega,
        Multinomial, MultivariateBeta, MultivariateEwens, MultivariateT,
        NegativeMultinomial, NormalGamma, MultivariateNormal, MultivariateLaplace,
        marginal_distribution)

from .stochastic_process_types import (StochasticProcess,
        DiscreteTimeStochasticProcess, DiscreteMarkovChain,
        TransitionMatrixOf, StochasticStateSpaceOf, GeneratorMatrixOf,
        ContinuousMarkovChain, BernoulliProcess, PoissonProcess, WienerProcess,
        GammaProcess)

from .random_matrix_models import (CircularEnsemble, CircularUnitaryEnsemble,
        CircularOrthogonalEnsemble, CircularSymplecticEnsemble,
        GaussianEnsemble, GaussianUnitaryEnsemble, GaussianOrthogonalEnsemble,
        GaussianSymplecticEnsemble, joint_eigen_distribution,
        JointEigenDistribution, level_spacing_distribution)

from .matrix_distributions import MatrixGamma, Wishart, MatrixNormal, MatrixStudentT

from .symbolic_probability import (Probability, Expectation, Variance,
        Covariance, Moment, CentralMoment)

from .symbolic_multivariate_probability import (ExpectationMatrix, VarianceMatrix,
        CrossCovarianceMatrix)
