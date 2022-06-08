r"""
The ``distributions`` package contains parameterizable probability distributions
and sampling functions. This allows the construction of stochastic computation
graphs and stochastic gradient estimators for optimization. This package
generally follows the design of the `TensorFlow Distributions`_ package.

.. _`TensorFlow Distributions`:
    https://arxiv.org/abs/1711.10604

It is not possible to directly backpropagate through random samples. However,
there are two main methods for creating surrogate functions that can be
backpropagated through. These are the score function estimator/likelihood ratio
estimator/REINFORCE and the pathwise derivative estimator. REINFORCE is commonly
seen as the basis for policy gradient methods in reinforcement learning, and the
pathwise derivative estimator is commonly seen in the reparameterization trick
in variational autoencoders. Whilst the score function only requires the value
of samples :math:`f(x)`, the pathwise derivative requires the derivative
:math:`f'(x)`. The next sections discuss these two in a reinforcement learning
example. For more details see
`Gradient Estimation Using Stochastic Computation Graphs`_ .

.. _`Gradient Estimation Using Stochastic Computation Graphs`:
     https://arxiv.org/abs/1506.05254

Score function
^^^^^^^^^^^^^^

When the probability density function is differentiable with respect to its
parameters, we only need :meth:`~torch.distributions.Distribution.sample` and
:meth:`~torch.distributions.Distribution.log_prob` to implement REINFORCE:

.. math::

    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,
:math:`r` is the reward and :math:`p(a|\pi^\theta(s))` is the probability of
taking action :math:`a` in state :math:`s` given policy :math:`\pi^\theta`.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss function. Note that we use a negative because optimizers use gradient
descent, whilst the rule above assumes gradient ascent. With a categorical
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    # Note that this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()

Pathwise derivative
^^^^^^^^^^^^^^^^^^^

The other way to implement these stochastic/policy gradients would be to use the
reparameterization trick from the
:meth:`~torch.distributions.Distribution.rsample` method, where the
parameterized random variable can be constructed via a parameterized
deterministic function of a parameter-free random variable. The reparameterized
sample therefore becomes differentiable. The code for implementing the pathwise
derivative would be as follows::

    params = policy_network(state)
    m = Normal(*params)
    # Any distribution with .has_rsample == True could work based on the application
    action = m.rsample()
    next_state, reward = env.step(action)  # Assuming that reward is differentiable
    loss = -reward
    loss.backward()
"""

from . import transforms
from .arcsine import Arcsine
from .bates import Bates
from .benford import Benford
from .benini import Benini
from .benktander_type_i import BenktanderTypeI
from .benktander_type_ii import BenktanderTypeII
from .bernoulli import Bernoulli
from .beta import Beta
from .beta_binomial import BetaBinomial
from .beta_negative_binomial import BetaNegativeBinomial
from .beta_prime import BetaPrime
from .binomial import Binomial
from .borel import Borel
from .borel_tanner import BorelTanner
from .burr import Burr
from .categorical import Categorical
from .cauchy import Cauchy
from .chi import Chi
from .chi2 import Chi2
from .constraint_registry import biject_to, transform_to
from .continuous_bernoulli import ContinuousBernoulli
from .copula import Copula
from .coxian import Coxian
from .dagum import Dagum
from .davis import Davis
from .dirichlet import Dirichlet
from .distribution import Distribution
from .erlang import Erlang
from .exp_family import ExponentialFamily
from .exponential import Exponential
from .exponential_power import ExponentialPower
from .f import F
from .fisher_noncentral_hypergeometric import FisherNoncentralHypergeometric
from .fisher_z import FisherZ
from .fishersnedecor import FisherSnedecor
from .frechet import Frechet
from .gamma import Gamma
from .generalized_extreme_value import GeneralizedExtremeValue
from .generalized_hyperbolic import GeneralizedHyperbolic
from .geometric import Geometric
from .geometric_poisson import GeometricPoisson
from .gompertz import Gompertz
from .gumbel import Gumbel
from .half_cauchy import HalfCauchy
from .half_normal import HalfNormal
from .hotelling_t_squared import HotellingTSquared
from .hyperbolic_secant import HyperbolicSecant
from .hyperexponential import Hyperexponential
from .hypergeometric import Hypergeometric
from .independent import Independent
from .inverse_chi_squared import InverseChiSquared
from .inverse_gamma import InverseGamma
from .irwin_hall import IrwinHall
from .kl import kl_divergence, register_kl, _add_kl_info
from .kumaraswamy import Kumaraswamy
from .landau import Landau
from .laplace import Laplace
from .levy import Levy
from .lkj_cholesky import LKJCholesky
from .log_logistic import LogLogistic
from .log_normal import LogNormal
from .logarithmic import Logarithmic
from .logistic import Logistic
from .logistic_normal import LogisticNormal
from .lowrank_multivariate_normal import LowRankMultivariateNormal
from .marchenko_pastur import MarchenkoPastur
from .maxwell_boltzmann import MaxwellBoltzmann
from .mixture_same_family import MixtureSameFamily
from .multinomial import Multinomial
from .multivariate_hypergeometric import MultivariateHypergeometric
from .multivariate_laplace import MultivariateLaplace
from .multivariate_normal import MultivariateNormal
from .multivariate_t import MultivariateT
from .nakagami import Nakagami
from .negative_binomial import NegativeBinomial
from .negative_multinomial import NegativeMultinomial
from .noncentral_beta import NoncentralBeta
from .noncentral_chi_squared import NoncentralChiSquared
from .noncentral_f import NoncentralF
from .noncentral_t import NoncentralT
from .normal import Normal
from .normal_exponential_gamma import NormalExponentialGamma
from .normal_inverse_gaussian import NormalInverseGaussian
from .one_hot_categorical import (
    OneHotCategorical,
    OneHotCategoricalStraightThrough,
)
from .pareto import Pareto
from .pert import PERT
from .poisson import Poisson
from .q_exponential import QExponential
from .q_gaussian import QGaussian
from .q_weibull import QWeibull
from .rayleigh import Rayleigh
from .relaxed_bernoulli import RelaxedBernoulli
from .relaxed_categorical import RelaxedOneHotCategorical
from .skellam import Skellam
from .skew_normal import SkewNormal
from .stable import Stable
from .studentT import StudentT
from .tracy_widom import TracyWidom
from .transformed_distribution import TransformedDistribution
from .transforms import *  # noqa: F403
from .triangular import Triangular
from .tukey_lambda import TukeyLambda
from .uniform import Uniform
from .variance_gamma import VarianceGamma
from .voigt import Voigt
from .von_mises import VonMises
from .wakeby import Wakeby
from .wallenius_noncentral_hypergeometric import (
    WalleniusNoncentralHypergeometric,
)
from .weibull import Weibull
from .wigner_semicircle import WignerSemicircle
from .wishart import Wishart
from .yule_simon import YuleSimon
from .zipf import Zipf

_add_kl_info()

del _add_kl_info

__all__ = [
    'Arcsine',
    'Bates',
    'Benford',
    'Benini',
    'BenktanderTypeI',
    'BenktanderTypeII',
    'Bernoulli',
    'Beta',
    'BetaBinomial',
    'BetaNegativeBinomial',
    'BetaPrime',
    'Binomial',
    'Borel',
    'BorelTanner',
    'Burr',
    'Categorical',
    'Cauchy',
    'Chi',
    'Chi2',
    'ContinuousBernoulli',
    'Copula',
    'Coxian',
    'Dagum',
    'Davis',
    'Dirichlet',
    'Distribution',
    'Erlang',
    'Exponential',
    'ExponentialFamily',
    'ExponentialPower',
    'F',
    'FisherNoncentralHypergeometric',
    'FisherSnedecor',
    'FisherZ',
    'Frechet',
    'Gamma',
    'GeneralizedExtremeValue',
    'GeneralizedHyperbolic',
    'Geometric',
    'GeometricPoisson',
    'Gompertz',
    'Gumbel',
    'HalfCauchy',
    'HalfNormal',
    'HotellingTSquared',
    'HyperbolicSecant',
    'Hyperexponential',
    'Hypergeometric',
    'Independent',
    'InverseChiSquared',
    'InverseGamma',
    'IrwinHall',
    'Kumaraswamy',
    'Landau',
    'Laplace',
    'Levy',
    'LKJCholesky',
    'Logarithmic',
    'Logistic',
    'LogisticNormal',
    'LogLogistic',
    'LogNormal',
    'LowRankMultivariateNormal',
    'MarchenkoPastur',
    'MaxwellBoltzmann',
    'MixtureSameFamily',
    'Multinomial',
    'MultivariateHypergeometric',
    'MultivariateLaplace',
    'MultivariateNormal',
    'MultivariateT',
    'Nakagami',
    'NegativeBinomial',
    'NegativeMultinomial',
    'NoncentralBeta',
    'NoncentralChiSquared',
    'NoncentralF',
    'NoncentralT',
    'Normal',
    'NormalExponentialGamma',
    'NormalInverseGaussian',
    'OneHotCategorical',
    'OneHotCategoricalStraightThrough',
    'Pareto',
    'PERT',
    'Poisson',
    'QExponential',
    'QGaussian',
    'QWeibull',
    'Rayleigh',
    'RelaxedBernoulli',
    'RelaxedOneHotCategorical',
    'Skellam',
    'SkewNormal',
    'Stable',
    'StudentT',
    'TracyWidom',
    'TransformedDistribution',
    'Triangular',
    'TukeyLambda',
    'Uniform',
    'VarianceGamma',
    'Voigt',
    'VonMises',
    'Wakeby',
    'WalleniusNoncentralHypergeometric',
    'Weibull',
    'WignerSemicircle',
    'Wishart',
    'YuleSimon',
    'Zipf',
    'biject_to',
    'kl_divergence',
    'register_kl',
    'transform_to',
]

__all__.extend(transforms.__all__)
