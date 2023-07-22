# Owner(s): ["module: distributions"]

"""
Note [Randomized statistical tests]
-----------------------------------

This note describes how to maintain tests in this file as random sources
change. This file contains two types of randomized tests:

1. The easier type of randomized test are tests that should always pass but are
   initialized with random data. If these fail something is wrong, but it's
   fine to use a fixed seed by inheriting from common.TestCase.

2. The trickier tests are statistical tests. These tests explicitly call
   set_rng_seed(n) and are marked "see Note [Randomized statistical tests]".
   These statistical tests have a known positive failure rate
   (we set failure_rate=1e-3 by default). We need to balance strength of these
   tests with annoyance of false alarms. One way that works is to specifically
   set seeds in each of the randomized tests. When a random generator
   occasionally changes (as in #4312 vectorizing the Box-Muller sampler), some
   of these statistical tests may (rarely) fail. If one fails in this case,
   it's fine to increment the seed of the failing test (but you shouldn't need
   to increment it more than once; otherwise something is probably actually
   wrong).

3. `test_geometric_sample`, `test_binomial_sample` and `test_poisson_sample`
   are validated against `scipy.stats.` which are not guaranteed to be identical
   across different versions of scipy (namely, they yield invalid results in 1.7+)
"""

import math
import numbers
import unittest
from collections import namedtuple
from itertools import product
from random import shuffle
from packaging import version

import torch

# TODO: remove this global setting
# Distributions tests use double as the default dtype
torch.set_default_dtype(torch.double)

from torch import inf, nan
from torch.testing._internal.common_utils import \
    (TestCase, run_tests, set_rng_seed, TEST_WITH_UBSAN, load_tests,
     gradcheck, skipIfTorchDynamo)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.autograd import grad
import torch.autograd.forward_ad as fwAD
from torch.autograd.functional import jacobian
from torch.distributions import (Bernoulli, Beta, Binomial, Categorical,
                                 Cauchy, Chi2, ContinuousBernoulli, Dirichlet,
                                 Distribution, Exponential, ExponentialFamily,
                                 FisherSnedecor, Gamma, Geometric, Gumbel,
                                 HalfCauchy, HalfNormal, Independent, Kumaraswamy,
                                 LKJCholesky, Laplace, LogisticNormal,
                                 LogNormal, LowRankMultivariateNormal,
                                 MixtureSameFamily, Multinomial, MultivariateNormal,
                                 NegativeBinomial, Normal,
                                 OneHotCategorical, OneHotCategoricalStraightThrough,
                                 Pareto, Poisson, RelaxedBernoulli, RelaxedOneHotCategorical,
                                 StudentT, TransformedDistribution, Uniform,
                                 VonMises, Weibull, Wishart, constraints, kl_divergence)
from torch.distributions.constraint_registry import transform_to
from torch.distributions.constraints import Constraint, is_dependent
from torch.distributions.dirichlet import _Dirichlet_backward
from torch.distributions.kl import _kl_expfamily_expfamily
from torch.distributions.transforms import (AffineTransform, CatTransform, ExpTransform,
                                            StackTransform, identity_transform)
from torch.distributions.utils import (probs_to_logits, lazy_property, tril_matrix_to_vec,
                                       vec_to_tril_matrix)
from torch.nn.functional import softmax

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

TEST_NUMPY = True
try:
    import numpy as np
    import scipy.stats
    import scipy.special
except ImportError:
    TEST_NUMPY = False


def pairwise(Dist, *params):
    """
    Creates a pair of distributions `Dist` initialized to test each element of
    param with each other.
    """
    params1 = [torch.tensor([p] * len(p)) for p in params]
    params2 = [p.transpose(0, 1) for p in params1]
    return Dist(*params1), Dist(*params2)


def is_all_nan(tensor):
    """
    Checks if all entries of a tensor is nan.
    """
    return (tensor != tensor).all()


# Register all distributions for generic tests.
Example = namedtuple('Example', ['Dist', 'params'])
EXAMPLES = [
    Example(Bernoulli, [
        {'probs': torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([0.3], requires_grad=True)},
        {'probs': 0.3},
        {'logits': torch.tensor([0.], requires_grad=True)},
    ]),
    Example(Geometric, [
        {'probs': torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([0.3], requires_grad=True)},
        {'probs': 0.3},
    ]),
    Example(Beta, [
        {
            'concentration1': torch.randn(2, 3).exp().requires_grad_(),
            'concentration0': torch.randn(2, 3).exp().requires_grad_(),
        },
        {
            'concentration1': torch.randn(4).exp().requires_grad_(),
            'concentration0': torch.randn(4).exp().requires_grad_(),
        },
    ]),
    Example(Categorical, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
        {'logits': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(Binomial, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True), 'total_count': torch.tensor([10])},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True), 'total_count': torch.tensor([10, 8])},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
         'total_count': torch.tensor([[10., 8.], [5., 3.]])},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
         'total_count': torch.tensor(0.)},
    ]),
    Example(NegativeBinomial, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True), 'total_count': torch.tensor([10])},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True), 'total_count': torch.tensor([10, 8])},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
         'total_count': torch.tensor([[10., 8.], [5., 3.]])},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
         'total_count': torch.tensor(0.)},
    ]),
    Example(Multinomial, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True), 'total_count': 10},
    ]),
    Example(Cauchy, [
        {'loc': 0.0, 'scale': 1.0},
        {'loc': torch.tensor([0.0]), 'scale': 1.0},
        {'loc': torch.tensor([[0.0], [0.0]]),
         'scale': torch.tensor([[1.0], [1.0]])}
    ]),
    Example(Chi2, [
        {'df': torch.randn(2, 3).exp().requires_grad_()},
        {'df': torch.randn(1).exp().requires_grad_()},
    ]),
    Example(StudentT, [
        {'df': torch.randn(2, 3).exp().requires_grad_()},
        {'df': torch.randn(1).exp().requires_grad_()},
    ]),
    Example(Dirichlet, [
        {'concentration': torch.randn(2, 3).exp().requires_grad_()},
        {'concentration': torch.randn(4).exp().requires_grad_()},
    ]),
    Example(Exponential, [
        {'rate': torch.randn(5, 5).abs().requires_grad_()},
        {'rate': torch.randn(1).abs().requires_grad_()},
    ]),
    Example(FisherSnedecor, [
        {
            'df1': torch.randn(5, 5).abs().requires_grad_(),
            'df2': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'df1': torch.randn(1).abs().requires_grad_(),
            'df2': torch.randn(1).abs().requires_grad_(),
        },
        {
            'df1': torch.tensor([1.0]),
            'df2': 1.0,
        }
    ]),
    Example(Gamma, [
        {
            'concentration': torch.randn(2, 3).exp().requires_grad_(),
            'rate': torch.randn(2, 3).exp().requires_grad_(),
        },
        {
            'concentration': torch.randn(1).exp().requires_grad_(),
            'rate': torch.randn(1).exp().requires_grad_(),
        },
    ]),
    Example(Gumbel, [
        {
            'loc': torch.randn(5, 5, requires_grad=True),
            'scale': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'loc': torch.randn(1, requires_grad=True),
            'scale': torch.randn(1).abs().requires_grad_(),
        },
    ]),
    Example(HalfCauchy, [
        {'scale': 1.0},
        {'scale': torch.tensor([[1.0], [1.0]])}
    ]),
    Example(HalfNormal, [
        {'scale': torch.randn(5, 5).abs().requires_grad_()},
        {'scale': torch.randn(1).abs().requires_grad_()},
        {'scale': torch.tensor([1e-5, 1e-5], requires_grad=True)}
    ]),
    Example(Independent, [
        {
            'base_distribution': Normal(torch.randn(2, 3, requires_grad=True),
                                        torch.randn(2, 3).abs().requires_grad_()),
            'reinterpreted_batch_ndims': 0,
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, requires_grad=True),
                                        torch.randn(2, 3).abs().requires_grad_()),
            'reinterpreted_batch_ndims': 1,
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, requires_grad=True),
                                        torch.randn(2, 3).abs().requires_grad_()),
            'reinterpreted_batch_ndims': 2,
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, 5, requires_grad=True),
                                        torch.randn(2, 3, 5).abs().requires_grad_()),
            'reinterpreted_batch_ndims': 2,
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, 5, requires_grad=True),
                                        torch.randn(2, 3, 5).abs().requires_grad_()),
            'reinterpreted_batch_ndims': 3,
        },
    ]),
    Example(Kumaraswamy, [
        {
            'concentration1': torch.empty(2, 3).uniform_(1, 2).requires_grad_(),
            'concentration0': torch.empty(2, 3).uniform_(1, 2).requires_grad_(),
        },
        {
            'concentration1': torch.rand(4).uniform_(1, 2).requires_grad_(),
            'concentration0': torch.rand(4).uniform_(1, 2).requires_grad_(),
        },
    ]),
    Example(LKJCholesky, [
        {
            'dim': 2,
            'concentration': 0.5
        },
        {
            'dim': 3,
            'concentration': torch.tensor([0.5, 1., 2.]),
        },
        {
            'dim': 100,
            'concentration': 4.
        },
    ]),
    Example(Laplace, [
        {
            'loc': torch.randn(5, 5, requires_grad=True),
            'scale': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'loc': torch.randn(1, requires_grad=True),
            'scale': torch.randn(1).abs().requires_grad_(),
        },
        {
            'loc': torch.tensor([1.0, 0.0], requires_grad=True),
            'scale': torch.tensor([1e-5, 1e-5], requires_grad=True),
        },
    ]),
    Example(LogNormal, [
        {
            'loc': torch.randn(5, 5, requires_grad=True),
            'scale': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'loc': torch.randn(1, requires_grad=True),
            'scale': torch.randn(1).abs().requires_grad_(),
        },
        {
            'loc': torch.tensor([1.0, 0.0], requires_grad=True),
            'scale': torch.tensor([1e-5, 1e-5], requires_grad=True),
        },
    ]),
    Example(LogisticNormal, [
        {
            'loc': torch.randn(5, 5).requires_grad_(),
            'scale': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'loc': torch.randn(1).requires_grad_(),
            'scale': torch.randn(1).abs().requires_grad_(),
        },
        {
            'loc': torch.tensor([1.0, 0.0], requires_grad=True),
            'scale': torch.tensor([1e-5, 1e-5], requires_grad=True),
        },
    ]),
    Example(LowRankMultivariateNormal, [
        {
            'loc': torch.randn(5, 2, requires_grad=True),
            'cov_factor': torch.randn(5, 2, 1, requires_grad=True),
            'cov_diag': torch.tensor([2.0, 0.25], requires_grad=True),
        },
        {
            'loc': torch.randn(4, 3, requires_grad=True),
            'cov_factor': torch.randn(3, 2, requires_grad=True),
            'cov_diag': torch.tensor([5.0, 1.5, 3.], requires_grad=True),
        }
    ]),
    Example(MultivariateNormal, [
        {
            'loc': torch.randn(5, 2, requires_grad=True),
            'covariance_matrix': torch.tensor([[2.0, 0.3], [0.3, 0.25]], requires_grad=True),
        },
        {
            'loc': torch.randn(2, 3, requires_grad=True),
            'precision_matrix': torch.tensor([[2.0, 0.1, 0.0],
                                              [0.1, 0.25, 0.0],
                                              [0.0, 0.0, 0.3]], requires_grad=True),
        },
        {
            'loc': torch.randn(5, 3, 2, requires_grad=True),
            'scale_tril': torch.tensor([[[2.0, 0.0], [-0.5, 0.25]],
                                        [[2.0, 0.0], [0.3, 0.25]],
                                        [[5.0, 0.0], [-0.5, 1.5]]], requires_grad=True),
        },
        {
            'loc': torch.tensor([1.0, -1.0]),
            'covariance_matrix': torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
        },
    ]),
    Example(Normal, [
        {
            'loc': torch.randn(5, 5, requires_grad=True),
            'scale': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'loc': torch.randn(1, requires_grad=True),
            'scale': torch.randn(1).abs().requires_grad_(),
        },
        {
            'loc': torch.tensor([1.0, 0.0], requires_grad=True),
            'scale': torch.tensor([1e-5, 1e-5], requires_grad=True),
        },
    ]),
    Example(OneHotCategorical, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
        {'logits': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(OneHotCategoricalStraightThrough, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
        {'logits': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(Pareto, [
        {
            'scale': 1.0,
            'alpha': 1.0
        },
        {
            'scale': torch.randn(5, 5).abs().requires_grad_(),
            'alpha': torch.randn(5, 5).abs().requires_grad_()
        },
        {
            'scale': torch.tensor([1.0]),
            'alpha': 1.0
        }
    ]),
    Example(Poisson, [
        {
            'rate': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'rate': torch.randn(3).abs().requires_grad_(),
        },
        {
            'rate': 0.2,
        },
        {
            'rate': torch.tensor([0.0], requires_grad=True),
        },
        {
            'rate': 0.0,
        }
    ]),
    Example(RelaxedBernoulli, [
        {
            'temperature': torch.tensor([0.5], requires_grad=True),
            'probs': torch.tensor([0.7, 0.2, 0.4], requires_grad=True),
        },
        {
            'temperature': torch.tensor([2.0]),
            'probs': torch.tensor([0.3]),
        },
        {
            'temperature': torch.tensor([7.2]),
            'logits': torch.tensor([-2.0, 2.0, 1.0, 5.0])
        }
    ]),
    Example(RelaxedOneHotCategorical, [
        {
            'temperature': torch.tensor([0.5], requires_grad=True),
            'probs': torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], requires_grad=True)
        },
        {
            'temperature': torch.tensor([2.0]),
            'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        },
        {
            'temperature': torch.tensor([7.2]),
            'logits': torch.tensor([[-2.0, 2.0], [1.0, 5.0]])
        }
    ]),
    Example(TransformedDistribution, [
        {
            'base_distribution': Normal(torch.randn(2, 3, requires_grad=True),
                                        torch.randn(2, 3).abs().requires_grad_()),
            'transforms': [],
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, requires_grad=True),
                                        torch.randn(2, 3).abs().requires_grad_()),
            'transforms': ExpTransform(),
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, 5, requires_grad=True),
                                        torch.randn(2, 3, 5).abs().requires_grad_()),
            'transforms': [AffineTransform(torch.randn(3, 5), torch.randn(3, 5)),
                           ExpTransform()],
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, 5, requires_grad=True),
                                        torch.randn(2, 3, 5).abs().requires_grad_()),
            'transforms': AffineTransform(1, 2),
        },
        {
            'base_distribution': Uniform(torch.tensor(1e8).log(), torch.tensor(1e10).log()),
            'transforms': ExpTransform(),
        },
    ]),
    Example(Uniform, [
        {
            'low': torch.zeros(5, 5, requires_grad=True),
            'high': torch.ones(5, 5, requires_grad=True),
        },
        {
            'low': torch.zeros(1, requires_grad=True),
            'high': torch.ones(1, requires_grad=True),
        },
        {
            'low': torch.tensor([1.0, 1.0], requires_grad=True),
            'high': torch.tensor([2.0, 3.0], requires_grad=True),
        },
    ]),
    Example(Weibull, [
        {
            'scale': torch.randn(5, 5).abs().requires_grad_(),
            'concentration': torch.randn(1).abs().requires_grad_()
        }
    ]),
    Example(Wishart, [
        {
            'covariance_matrix': torch.tensor([[2.0, 0.3], [0.3, 0.25]], requires_grad=True),
            'df': torch.tensor([3.], requires_grad=True),
        },
        {
            'precision_matrix': torch.tensor([[2.0, 0.1, 0.0],
                                              [0.1, 0.25, 0.0],
                                              [0.0, 0.0, 0.3]], requires_grad=True),
            'df': torch.tensor([5., 4], requires_grad=True),
        },
        {
            'scale_tril': torch.tensor([[[2.0, 0.0], [-0.5, 0.25]],
                                        [[2.0, 0.0], [0.3, 0.25]],
                                        [[5.0, 0.0], [-0.5, 1.5]]], requires_grad=True),
            'df': torch.tensor([5., 3.5, 3], requires_grad=True),
        },
        {
            'covariance_matrix': torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
            'df': torch.tensor([3.0]),
        },
        {
            'covariance_matrix': torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
            'df': 3.0,
        },
    ]),
    Example(MixtureSameFamily, [
        {
            'mixture_distribution': Categorical(torch.rand(5, requires_grad=True)),
            'component_distribution': Normal(torch.randn(5, requires_grad=True),
                                             torch.rand(5, requires_grad=True)),
        },
        {
            'mixture_distribution': Categorical(torch.rand(5, requires_grad=True)),
            'component_distribution': MultivariateNormal(
                loc=torch.randn(5, 2, requires_grad=True),
                covariance_matrix=torch.tensor([[2.0, 0.3], [0.3, 0.25]], requires_grad=True)),
        },
    ]),
    Example(VonMises, [
        {
            'loc': torch.tensor(1.0, requires_grad=True),
            'concentration': torch.tensor(10.0, requires_grad=True)
        },
        {
            'loc': torch.tensor([0.0, math.pi / 2], requires_grad=True),
            'concentration': torch.tensor([1.0, 10.0], requires_grad=True)
        },
    ]),
    Example(ContinuousBernoulli, [
        {'probs': torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([0.3], requires_grad=True)},
        {'probs': 0.3},
        {'logits': torch.tensor([0.], requires_grad=True)},
    ])
]

BAD_EXAMPLES = [
    Example(Bernoulli, [
        {'probs': torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([-0.5], requires_grad=True)},
        {'probs': 1.00001},
    ]),
    Example(Beta, [
        {
            'concentration1': torch.tensor([0.0], requires_grad=True),
            'concentration0': torch.tensor([0.0], requires_grad=True),
        },
        {
            'concentration1': torch.tensor([-1.0], requires_grad=True),
            'concentration0': torch.tensor([-2.0], requires_grad=True),
        },
    ]),
    Example(Geometric, [
        {'probs': torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([-0.3], requires_grad=True)},
        {'probs': 1.00000001},
    ]),
    Example(Categorical, [
        {'probs': torch.tensor([[-0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[-1.0, 10.0], [0.0, -1.0]], requires_grad=True)},
    ]),
    Example(Binomial, [
        {'probs': torch.tensor([[-0.0000001, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True),
         'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True),
         'total_count': 10},
    ]),
    Example(NegativeBinomial, [
        {'probs': torch.tensor([[-0.0000001, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True),
         'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True),
         'total_count': 10},
    ]),
    Example(Cauchy, [
        {'loc': 0.0, 'scale': -1.0},
        {'loc': torch.tensor([0.0]), 'scale': 0.0},
        {'loc': torch.tensor([[0.0], [-2.0]]),
         'scale': torch.tensor([[-0.000001], [1.0]])}
    ]),
    Example(Chi2, [
        {'df': torch.tensor([0.], requires_grad=True)},
        {'df': torch.tensor([-2.], requires_grad=True)},
    ]),
    Example(StudentT, [
        {'df': torch.tensor([0.], requires_grad=True)},
        {'df': torch.tensor([-2.], requires_grad=True)},
    ]),
    Example(Dirichlet, [
        {'concentration': torch.tensor([0.], requires_grad=True)},
        {'concentration': torch.tensor([-2.], requires_grad=True)}
    ]),
    Example(Exponential, [
        {'rate': torch.tensor([0., 0.], requires_grad=True)},
        {'rate': torch.tensor([-2.], requires_grad=True)}
    ]),
    Example(FisherSnedecor, [
        {
            'df1': torch.tensor([0., 0.], requires_grad=True),
            'df2': torch.tensor([-1., -100.], requires_grad=True),
        },
        {
            'df1': torch.tensor([1., 1.], requires_grad=True),
            'df2': torch.tensor([0., 0.], requires_grad=True),
        }
    ]),
    Example(Gamma, [
        {
            'concentration': torch.tensor([0., 0.], requires_grad=True),
            'rate': torch.tensor([-1., -100.], requires_grad=True),
        },
        {
            'concentration': torch.tensor([1., 1.], requires_grad=True),
            'rate': torch.tensor([0., 0.], requires_grad=True),
        }
    ]),
    Example(Gumbel, [
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([0., 1.], requires_grad=True),
        },
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([1., -1.], requires_grad=True),
        },
    ]),
    Example(HalfCauchy, [
        {'scale': -1.0},
        {'scale': 0.0},
        {'scale': torch.tensor([[-0.000001], [1.0]])}
    ]),
    Example(HalfNormal, [
        {'scale': torch.tensor([0., 1.], requires_grad=True)},
        {'scale': torch.tensor([1., -1.], requires_grad=True)},
    ]),
    Example(LKJCholesky, [
        {
            'dim': -2,
            'concentration': 0.1
        },
        {
            'dim': 1,
            'concentration': 2.,
        },
        {
            'dim': 2,
            'concentration': 0.,
        },
    ]),
    Example(Laplace, [
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([0., 1.], requires_grad=True),
        },
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([1., -1.], requires_grad=True),
        },
    ]),
    Example(LogNormal, [
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([0., 1.], requires_grad=True),
        },
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([1., -1.], requires_grad=True),
        },
    ]),
    Example(MultivariateNormal, [
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'covariance_matrix': torch.tensor([[1.0, 0.0], [0.0, -2.0]], requires_grad=True),
        },
    ]),
    Example(Normal, [
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([0., 1.], requires_grad=True),
        },
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([1., -1.], requires_grad=True),
        },
        {
            'loc': torch.tensor([1.0, 0.0], requires_grad=True),
            'scale': torch.tensor([1e-5, -1e-5], requires_grad=True),
        },
    ]),
    Example(OneHotCategorical, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(OneHotCategoricalStraightThrough, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(Pareto, [
        {
            'scale': 0.0,
            'alpha': 0.0
        },
        {
            'scale': torch.tensor([0.0, 0.0], requires_grad=True),
            'alpha': torch.tensor([-1e-5, 0.0], requires_grad=True)
        },
        {
            'scale': torch.tensor([1.0]),
            'alpha': -1.0
        }
    ]),
    Example(Poisson, [
        {
            'rate': torch.tensor([-0.1], requires_grad=True),
        },
        {
            'rate': -1.0,
        }
    ]),
    Example(RelaxedBernoulli, [
        {
            'temperature': torch.tensor([1.5], requires_grad=True),
            'probs': torch.tensor([1.7, 0.2, 0.4], requires_grad=True),
        },
        {
            'temperature': torch.tensor([2.0]),
            'probs': torch.tensor([-1.0]),
        }
    ]),
    Example(RelaxedOneHotCategorical, [
        {
            'temperature': torch.tensor([0.5], requires_grad=True),
            'probs': torch.tensor([[-0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], requires_grad=True)
        },
        {
            'temperature': torch.tensor([2.0]),
            'probs': torch.tensor([[-1.0, 0.0], [-1.0, 1.1]])
        }
    ]),
    Example(TransformedDistribution, [
        {
            'base_distribution': Normal(0, 1),
            'transforms': lambda x: x,
        },
        {
            'base_distribution': Normal(0, 1),
            'transforms': [lambda x: x],
        },
    ]),
    Example(Uniform, [
        {
            'low': torch.tensor([2.0], requires_grad=True),
            'high': torch.tensor([2.0], requires_grad=True),
        },
        {
            'low': torch.tensor([0.0], requires_grad=True),
            'high': torch.tensor([0.0], requires_grad=True),
        },
        {
            'low': torch.tensor([1.0], requires_grad=True),
            'high': torch.tensor([0.0], requires_grad=True),
        }
    ]),
    Example(Weibull, [
        {
            'scale': torch.tensor([0.0], requires_grad=True),
            'concentration': torch.tensor([0.0], requires_grad=True)
        },
        {
            'scale': torch.tensor([1.0], requires_grad=True),
            'concentration': torch.tensor([-1.0], requires_grad=True)
        }
    ]),
    Example(Wishart, [
        {
            'covariance_matrix': torch.tensor([[1.0, 0.0], [0.0, -2.0]], requires_grad=True),
            'df': torch.tensor([1.5], requires_grad=True),
        },
        {
            'covariance_matrix': torch.tensor([[1.0, 1.0], [1.0, -2.0]], requires_grad=True),
            'df': torch.tensor([3.], requires_grad=True),
        },
        {
            'covariance_matrix': torch.tensor([[1.0, 1.0], [1.0, -2.0]], requires_grad=True),
            'df': 3.,
        },
    ]),
    Example(ContinuousBernoulli, [
        {'probs': torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([-0.5], requires_grad=True)},
        {'probs': 1.00001},
    ])
]


class DistributionsTestCase(TestCase):
    def setUp(self):
        """The tests assume that the validation flag is set."""
        torch.distributions.Distribution.set_default_validate_args(True)
        super().setUp()


@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestDistributions(DistributionsTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _gradcheck_log_prob(self, dist_ctor, ctor_params):
        # performs gradient checks on log_prob
        distribution = dist_ctor(*ctor_params)
        s = distribution.sample()
        if not distribution.support.is_discrete:
            s = s.detach().requires_grad_()

        expected_shape = distribution.batch_shape + distribution.event_shape
        self.assertEqual(s.size(), expected_shape)

        def apply_fn(s, *params):
            return dist_ctor(*params).log_prob(s)

        gradcheck(apply_fn, (s,) + tuple(ctor_params), raise_exception=True)

    def _check_forward_ad(self, fn):
        with fwAD.dual_level():
            x = torch.tensor(1.)
            t = torch.tensor(1.)
            dual = fwAD.make_dual(x, t)
            dual_out = fn(dual)
            self.assertEqual(torch.count_nonzero(fwAD.unpack_dual(dual_out).tangent).item(), 0)

    def _check_log_prob(self, dist, asset_fn):
        # checks that the log_prob matches a reference function
        s = dist.sample()
        log_probs = dist.log_prob(s)
        log_probs_data_flat = log_probs.view(-1)
        s_data_flat = s.view(len(log_probs_data_flat), -1)
        for i, (val, log_prob) in enumerate(zip(s_data_flat, log_probs_data_flat)):
            asset_fn(i, val.squeeze(), log_prob)

    def _check_sampler_sampler(self, torch_dist, ref_dist, message, multivariate=False,
                               circular=False, num_samples=10000, failure_rate=1e-3):
        # Checks that the .sample() method matches a reference function.
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        torch_samples = torch_samples.cpu().numpy()
        ref_samples = ref_dist.rvs(num_samples).astype(np.float64)
        if multivariate:
            # Project onto a random axis.
            axis = np.random.normal(size=(1,) + torch_samples.shape[1:])
            axis /= np.linalg.norm(axis)
            torch_samples = (axis * torch_samples).reshape(num_samples, -1).sum(-1)
            ref_samples = (axis * ref_samples).reshape(num_samples, -1).sum(-1)
        samples = [(x, +1) for x in torch_samples] + [(x, -1) for x in ref_samples]
        if circular:
            samples = [(np.cos(x), v) for (x, v) in samples]
        shuffle(samples)  # necessary to prevent stable sort from making uneven bins for discrete
        samples.sort(key=lambda x: x[0])
        samples = np.array(samples)[:, 1]

        # Aggregate into bins filled with roughly zero-mean unit-variance RVs.
        num_bins = 10
        samples_per_bin = len(samples) // num_bins
        bins = samples.reshape((num_bins, samples_per_bin)).mean(axis=1)
        stddev = samples_per_bin ** -0.5
        threshold = stddev * scipy.special.erfinv(1 - 2 * failure_rate / num_bins)
        message = f'{message}.sample() is biased:\n{bins}'
        for bias in bins:
            self.assertLess(-threshold, bias, message)
            self.assertLess(bias, threshold, message)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def _check_sampler_discrete(self, torch_dist, ref_dist, message,
                                num_samples=10000, failure_rate=1e-3):
        """Runs a Chi2-test for the support, but ignores tail instead of combining"""
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        torch_samples = torch_samples.cpu().numpy()
        unique, counts = np.unique(torch_samples, return_counts=True)
        pmf = ref_dist.pmf(unique)
        pmf = pmf / pmf.sum()  # renormalize to 1.0 for chisq test
        msk = (counts > 5) & ((pmf * num_samples) > 5)
        self.assertGreater(pmf[msk].sum(), 0.9, "Distribution is too sparse for test; try increasing num_samples")
        # Add a remainder bucket that combines counts for all values
        # below threshold, if such values exist (i.e. mask has False entries).
        if not msk.all():
            counts = np.concatenate([counts[msk], np.sum(counts[~msk], keepdims=True)])
            pmf = np.concatenate([pmf[msk], np.sum(pmf[~msk], keepdims=True)])
        chisq, p = scipy.stats.chisquare(counts, pmf * num_samples)
        self.assertGreater(p, failure_rate, message)

    def _check_enumerate_support(self, dist, examples):
        for params, expected in examples:
            params = {k: torch.tensor(v) for k, v in params.items()}
            d = dist(**params)
            actual = d.enumerate_support(expand=False)
            expected = torch.tensor(expected, dtype=actual.dtype)
            self.assertEqual(actual, expected)
            actual = d.enumerate_support(expand=True)
            expected_with_expand = expected.expand((-1,) + d.batch_shape + d.event_shape)
            self.assertEqual(actual, expected_with_expand)

    def test_repr(self):
        for Dist, params in EXAMPLES:
            for param in params:
                dist = Dist(**param)
                self.assertTrue(repr(dist).startswith(dist.__class__.__name__))

    def test_sample_detached(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                variable_params = [p for p in param.values() if getattr(p, 'requires_grad', False)]
                if not variable_params:
                    continue
                dist = Dist(**param)
                sample = dist.sample()
                self.assertFalse(sample.requires_grad,
                                 msg=f'{Dist.__name__} example {i + 1}/{len(params)}, .sample() is not detached')

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_rsample_requires_grad(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                if not any(getattr(p, 'requires_grad', False) for p in param.values()):
                    continue
                dist = Dist(**param)
                if not dist.has_rsample:
                    continue
                sample = dist.rsample()
                self.assertTrue(sample.requires_grad,
                                msg=f'{Dist.__name__} example {i + 1}/{len(params)}, .rsample() does not require grad')

    def test_enumerate_support_type(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    self.assertTrue(type(dist.sample()) is type(dist.enumerate_support()),
                                    msg=('{} example {}/{}, return type mismatch between ' +
                                         'sample and enumerate_support.').format(Dist.__name__, i + 1, len(params)))
                except NotImplementedError:
                    pass

    def test_lazy_property_grad(self):
        x = torch.randn(1, requires_grad=True)

        class Dummy:
            @lazy_property
            def y(self):
                return x + 1

        def test():
            x.grad = None
            Dummy().y.backward()
            self.assertEqual(x.grad, torch.ones(1))

        test()
        with torch.no_grad():
            test()

        mean = torch.randn(2)
        cov = torch.eye(2, requires_grad=True)
        distn = MultivariateNormal(mean, cov)
        with torch.no_grad():
            distn.scale_tril
        distn.scale_tril.sum().backward()
        self.assertIsNotNone(cov.grad)

    def test_has_examples(self):
        distributions_with_examples = {e.Dist for e in EXAMPLES}
        for Dist in globals().values():
            if isinstance(Dist, type) and issubclass(Dist, Distribution) \
                    and Dist is not Distribution and Dist is not ExponentialFamily:
                self.assertIn(Dist, distributions_with_examples,
                              f"Please add {Dist.__name__} to the EXAMPLES list in test_distributions.py")

    def test_support_attributes(self):
        for Dist, params in EXAMPLES:
            for param in params:
                d = Dist(**param)
                event_dim = len(d.event_shape)
                self.assertEqual(d.support.event_dim, event_dim)
                try:
                    self.assertEqual(Dist.support.event_dim, event_dim)
                except NotImplementedError:
                    pass
                is_discrete = d.support.is_discrete
                try:
                    self.assertEqual(Dist.support.is_discrete, is_discrete)
                except NotImplementedError:
                    pass

    def test_distribution_expand(self):
        shapes = [torch.Size(), torch.Size((2,)), torch.Size((2, 1))]
        for Dist, params in EXAMPLES:
            for param in params:
                for shape in shapes:
                    d = Dist(**param)
                    expanded_shape = shape + d.batch_shape
                    original_shape = d.batch_shape + d.event_shape
                    expected_shape = shape + original_shape
                    expanded = d.expand(batch_shape=list(expanded_shape))
                    sample = expanded.sample()
                    actual_shape = expanded.sample().shape
                    self.assertEqual(expanded.__class__, d.__class__)
                    self.assertEqual(d.sample().shape, original_shape)
                    self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                    self.assertEqual(actual_shape, expected_shape)
                    self.assertEqual(expanded.batch_shape, expanded_shape)
                    try:
                        self.assertEqual(expanded.mean,
                                         d.mean.expand(expanded_shape + d.event_shape))
                        self.assertEqual(expanded.variance,
                                         d.variance.expand(expanded_shape + d.event_shape))
                    except NotImplementedError:
                        pass

    def test_distribution_subclass_expand(self):
        expand_by = torch.Size((2,))
        for Dist, params in EXAMPLES:

            class SubClass(Dist):
                pass

            for param in params:
                d = SubClass(**param)
                expanded_shape = expand_by + d.batch_shape
                original_shape = d.batch_shape + d.event_shape
                expected_shape = expand_by + original_shape
                expanded = d.expand(batch_shape=expanded_shape)
                sample = expanded.sample()
                actual_shape = expanded.sample().shape
                self.assertEqual(expanded.__class__, d.__class__)
                self.assertEqual(d.sample().shape, original_shape)
                self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                self.assertEqual(actual_shape, expected_shape)

    def test_bernoulli(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        r = torch.tensor(0.3, requires_grad=True)
        s = 0.3
        self.assertEqual(Bernoulli(p).sample((8,)).size(), (8, 3))
        self.assertFalse(Bernoulli(p).sample().requires_grad)
        self.assertEqual(Bernoulli(r).sample((8,)).size(), (8,))
        self.assertEqual(Bernoulli(r).sample().size(), ())
        self.assertEqual(Bernoulli(r).sample((3, 2)).size(), (3, 2,))
        self.assertEqual(Bernoulli(s).sample().size(), ())
        self._gradcheck_log_prob(Bernoulli, (p,))

        def ref_log_prob(idx, val, log_prob):
            prob = p[idx]
            self.assertEqual(log_prob, math.log(prob if val else 1 - prob))

        self._check_log_prob(Bernoulli(p), ref_log_prob)
        self._check_log_prob(Bernoulli(logits=p.log() - (-p).log1p()), ref_log_prob)
        self.assertRaises(NotImplementedError, Bernoulli(r).rsample)

        # check entropy computation
        self.assertEqual(Bernoulli(p).entropy(), torch.tensor([0.6108, 0.5004, 0.6730]), atol=1e-4, rtol=0)
        self.assertEqual(Bernoulli(torch.tensor([0.0])).entropy(), torch.tensor([0.0]))
        self.assertEqual(Bernoulli(s).entropy(), torch.tensor(0.6108), atol=1e-4, rtol=0)

        self._check_forward_ad(torch.bernoulli)
        self._check_forward_ad(lambda x: x.bernoulli_())
        self._check_forward_ad(lambda x: x.bernoulli_(x.clone().detach()))
        self._check_forward_ad(lambda x: x.bernoulli_(x))

    def test_bernoulli_enumerate_support(self):
        examples = [
            ({"probs": [0.1]}, [[0], [1]]),
            ({"probs": [0.1, 0.9]}, [[0], [1]]),
            ({"probs": [[0.1, 0.2], [0.3, 0.4]]}, [[[0]], [[1]]]),
        ]
        self._check_enumerate_support(Bernoulli, examples)

    def test_bernoulli_3d(self):
        p = torch.full((2, 3, 5), 0.5).requires_grad_()
        self.assertEqual(Bernoulli(p).sample().size(), (2, 3, 5))
        self.assertEqual(Bernoulli(p).sample(sample_shape=(2, 5)).size(),
                         (2, 5, 2, 3, 5))
        self.assertEqual(Bernoulli(p).sample((2,)).size(), (2, 2, 3, 5))

    def test_geometric(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        r = torch.tensor(0.3, requires_grad=True)
        s = 0.3
        self.assertEqual(Geometric(p).sample((8,)).size(), (8, 3))
        self.assertEqual(Geometric(1).sample(), 0)
        self.assertEqual(Geometric(1).log_prob(torch.tensor(1.)), -inf)
        self.assertEqual(Geometric(1).log_prob(torch.tensor(0.)), 0)
        self.assertFalse(Geometric(p).sample().requires_grad)
        self.assertEqual(Geometric(r).sample((8,)).size(), (8,))
        self.assertEqual(Geometric(r).sample().size(), ())
        self.assertEqual(Geometric(r).sample((3, 2)).size(), (3, 2))
        self.assertEqual(Geometric(s).sample().size(), ())
        self._gradcheck_log_prob(Geometric, (p,))
        self.assertRaises(ValueError, lambda: Geometric(0))
        self.assertRaises(NotImplementedError, Geometric(r).rsample)

        self._check_forward_ad(lambda x: x.geometric_(0.2))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_geometric_log_prob_and_entropy(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        s = 0.3

        def ref_log_prob(idx, val, log_prob):
            prob = p[idx].detach()
            self.assertEqual(log_prob, scipy.stats.geom(prob, loc=-1).logpmf(val))

        self._check_log_prob(Geometric(p), ref_log_prob)
        self._check_log_prob(Geometric(logits=p.log() - (-p).log1p()), ref_log_prob)

        # check entropy computation
        self.assertEqual(Geometric(p).entropy(), scipy.stats.geom(p.detach().numpy(), loc=-1).entropy(), atol=1e-3, rtol=0)
        self.assertEqual(float(Geometric(s).entropy()), scipy.stats.geom(s, loc=-1).entropy().item(), atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_geometric_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for prob in [0.01, 0.18, 0.8]:
            self._check_sampler_discrete(Geometric(prob),
                                         scipy.stats.geom(p=prob, loc=-1),
                                         f'Geometric(prob={prob})')

    def test_binomial(self):
        p = torch.arange(0.05, 1, 0.1).requires_grad_()
        for total_count in [1, 2, 10]:
            self._gradcheck_log_prob(lambda p: Binomial(total_count, p), [p])
            self._gradcheck_log_prob(lambda p: Binomial(total_count, None, p.log()), [p])
        self.assertRaises(NotImplementedError, Binomial(10, p).rsample)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_binomial_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for prob in [0.01, 0.1, 0.5, 0.8, 0.9]:
            for count in [2, 10, 100, 500]:
                self._check_sampler_discrete(Binomial(total_count=count, probs=prob),
                                             scipy.stats.binom(count, prob),
                                             f'Binomial(total_count={count}, probs={prob})')

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_binomial_log_prob_and_entropy(self):
        probs = torch.arange(0.05, 1, 0.1)
        for total_count in [1, 2, 10]:

            def ref_log_prob(idx, x, log_prob):
                p = probs.view(-1)[idx].item()
                expected = scipy.stats.binom(total_count, p).logpmf(x)
                self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)
            self._check_log_prob(Binomial(total_count, probs), ref_log_prob)
            logits = probs_to_logits(probs, is_binary=True)
            self._check_log_prob(Binomial(total_count, logits=logits), ref_log_prob)

            bin = Binomial(total_count, logits=logits)
            self.assertEqual(
                bin.entropy(),
                scipy.stats.binom(total_count, bin.probs.detach().numpy(), loc=-1).entropy(),
                atol=1e-3, rtol=0)

    def test_binomial_stable(self):
        logits = torch.tensor([-100., 100.], dtype=torch.float)
        total_count = 1.
        x = torch.tensor([0., 0.], dtype=torch.float)
        log_prob = Binomial(total_count, logits=logits).log_prob(x)
        self.assertTrue(torch.isfinite(log_prob).all())

        # make sure that the grad at logits=0, value=0 is 0.5
        x = torch.tensor(0., requires_grad=True)
        y = Binomial(total_count, logits=x).log_prob(torch.tensor(0.))
        self.assertEqual(grad(y, x)[0], torch.tensor(-0.5))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_binomial_log_prob_vectorized_count(self):
        probs = torch.tensor([0.2, 0.7, 0.9])
        for total_count, sample in [(torch.tensor([10]), torch.tensor([7., 3., 9.])),
                                    (torch.tensor([1, 2, 10]), torch.tensor([0., 1., 9.]))]:
            log_prob = Binomial(total_count, probs).log_prob(sample)
            expected = scipy.stats.binom(total_count.cpu().numpy(), probs.cpu().numpy()).logpmf(sample)
            self.assertEqual(log_prob, expected, atol=1e-4, rtol=0)

    def test_binomial_enumerate_support(self):
        examples = [
            ({"probs": [0.1], "total_count": 2}, [[0], [1], [2]]),
            ({"probs": [0.1, 0.9], "total_count": 2}, [[0], [1], [2]]),
            ({"probs": [[0.1, 0.2], [0.3, 0.4]], "total_count": 3}, [[[0]], [[1]], [[2]], [[3]]]),
        ]
        self._check_enumerate_support(Binomial, examples)

    def test_binomial_extreme_vals(self):
        total_count = 100
        bin0 = Binomial(total_count, 0)
        self.assertEqual(bin0.sample(), 0)
        self.assertEqual(bin0.log_prob(torch.tensor([0.]))[0], 0, atol=1e-3, rtol=0)
        self.assertEqual(float(bin0.log_prob(torch.tensor([1.])).exp()), 0)
        bin1 = Binomial(total_count, 1)
        self.assertEqual(bin1.sample(), total_count)
        self.assertEqual(bin1.log_prob(torch.tensor([float(total_count)]))[0], 0, atol=1e-3, rtol=0)
        self.assertEqual(float(bin1.log_prob(torch.tensor([float(total_count - 1)])).exp()), 0)
        zero_counts = torch.zeros(torch.Size((2, 2)))
        bin2 = Binomial(zero_counts, 1)
        self.assertEqual(bin2.sample(), zero_counts)
        self.assertEqual(bin2.log_prob(zero_counts), zero_counts)

    def test_binomial_vectorized_count(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        total_count = torch.tensor([[4, 7], [3, 8]], dtype=torch.float64)
        bin0 = Binomial(total_count, torch.tensor(1.))
        self.assertEqual(bin0.sample(), total_count)
        bin1 = Binomial(total_count, torch.tensor(0.5))
        samples = bin1.sample(torch.Size((100000,)))
        self.assertTrue((samples <= total_count.type_as(samples)).all())
        self.assertEqual(samples.mean(dim=0), bin1.mean, atol=0.02, rtol=0)
        self.assertEqual(samples.var(dim=0), bin1.variance, atol=0.02, rtol=0)

    def test_negative_binomial(self):
        p = torch.arange(0.05, 1, 0.1).requires_grad_()
        for total_count in [1, 2, 10]:
            self._gradcheck_log_prob(lambda p: NegativeBinomial(total_count, p), [p])
            self._gradcheck_log_prob(lambda p: NegativeBinomial(total_count, None, p.log()), [p])
        self.assertRaises(NotImplementedError, NegativeBinomial(10, p).rsample)
        self.assertRaises(NotImplementedError, NegativeBinomial(10, p).entropy)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_negative_binomial_log_prob(self):
        probs = torch.arange(0.05, 1, 0.1)
        for total_count in [1, 2, 10]:

            def ref_log_prob(idx, x, log_prob):
                p = probs.view(-1)[idx].item()
                expected = scipy.stats.nbinom(total_count, 1 - p).logpmf(x)
                self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

            self._check_log_prob(NegativeBinomial(total_count, probs), ref_log_prob)
            logits = probs_to_logits(probs, is_binary=True)
            self._check_log_prob(NegativeBinomial(total_count, logits=logits), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_negative_binomial_log_prob_vectorized_count(self):
        probs = torch.tensor([0.2, 0.7, 0.9])
        for total_count, sample in [(torch.tensor([10]), torch.tensor([7., 3., 9.])),
                                    (torch.tensor([1, 2, 10]), torch.tensor([0., 1., 9.]))]:
            log_prob = NegativeBinomial(total_count, probs).log_prob(sample)
            expected = scipy.stats.nbinom(total_count.cpu().numpy(), 1 - probs.cpu().numpy()).logpmf(sample)
            self.assertEqual(log_prob, expected, atol=1e-4, rtol=0)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_zero_excluded_binomial(self):
        vals = Binomial(total_count=torch.tensor(1.0).cuda(),
                        probs=torch.tensor(0.9).cuda()
                        ).sample(torch.Size((100000000,)))
        self.assertTrue((vals >= 0).all())
        vals = Binomial(total_count=torch.tensor(1.0).cuda(),
                        probs=torch.tensor(0.1).cuda()
                        ).sample(torch.Size((100000000,)))
        self.assertTrue((vals < 2).all())
        vals = Binomial(total_count=torch.tensor(1.0).cuda(),
                        probs=torch.tensor(0.5).cuda()
                        ).sample(torch.Size((10000,)))
        # vals should be roughly half zeroes, half ones
        assert (vals == 0.0).sum() > 4000
        assert (vals == 1.0).sum() > 4000

    def test_multinomial_1d(self):
        total_count = 10
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        self.assertEqual(Multinomial(total_count, p).sample().size(), (3,))
        self.assertEqual(Multinomial(total_count, p).sample((2, 2)).size(), (2, 2, 3))
        self.assertEqual(Multinomial(total_count, p).sample((1,)).size(), (1, 3))
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])
        self.assertRaises(NotImplementedError, Multinomial(10, p).rsample)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_multinomial_1d_log_prob_and_entropy(self):
        total_count = 10
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        dist = Multinomial(total_count, probs=p)
        x = dist.sample()
        log_prob = dist.log_prob(x)
        expected = torch.tensor(scipy.stats.multinomial.logpmf(x.numpy(), n=total_count, p=dist.probs.detach().numpy()))
        self.assertEqual(log_prob, expected)

        dist = Multinomial(total_count, logits=p.log())
        x = dist.sample()
        log_prob = dist.log_prob(x)
        expected = torch.tensor(scipy.stats.multinomial.logpmf(x.numpy(), n=total_count, p=dist.probs.detach().numpy()))
        self.assertEqual(log_prob, expected)

        expected = scipy.stats.multinomial.entropy(total_count, dist.probs.detach().numpy())
        self.assertEqual(dist.entropy(), expected, atol=1e-3, rtol=0)

    def test_multinomial_2d(self):
        total_count = 10
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        self.assertEqual(Multinomial(total_count, p).sample().size(), (2, 3))
        self.assertEqual(Multinomial(total_count, p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3))
        self.assertEqual(Multinomial(total_count, p).sample((6,)).size(), (6, 2, 3))
        set_rng_seed(0)
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])

        # sample check for extreme value of probs
        self.assertEqual(Multinomial(total_count, s).sample(),
                         torch.tensor([[total_count, 0], [0, total_count]], dtype=torch.float64))

    def test_categorical_1d(self):
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        self.assertTrue(is_all_nan(Categorical(p).mean))
        self.assertTrue(is_all_nan(Categorical(p).variance))
        self.assertEqual(Categorical(p).sample().size(), ())
        self.assertFalse(Categorical(p).sample().requires_grad)
        self.assertEqual(Categorical(p).sample((2, 2)).size(), (2, 2))
        self.assertEqual(Categorical(p).sample((1,)).size(), (1,))
        self._gradcheck_log_prob(Categorical, (p,))
        self.assertRaises(NotImplementedError, Categorical(p).rsample)

    def test_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        self.assertEqual(Categorical(p).mean.size(), (2,))
        self.assertEqual(Categorical(p).variance.size(), (2,))
        self.assertTrue(is_all_nan(Categorical(p).mean))
        self.assertTrue(is_all_nan(Categorical(p).variance))
        self.assertEqual(Categorical(p).sample().size(), (2,))
        self.assertEqual(Categorical(p).sample(sample_shape=(3, 4)).size(), (3, 4, 2))
        self.assertEqual(Categorical(p).sample((6,)).size(), (6, 2))
        self._gradcheck_log_prob(Categorical, (p,))

        # sample check for extreme value of probs
        set_rng_seed(0)
        self.assertEqual(Categorical(s).sample(sample_shape=(2,)),
                         torch.tensor([[0, 1], [0, 1]]))

        def ref_log_prob(idx, val, log_prob):
            sample_prob = p[idx][val] / p[idx].sum()
            self.assertEqual(log_prob, math.log(sample_prob))

        self._check_log_prob(Categorical(p), ref_log_prob)
        self._check_log_prob(Categorical(logits=p.log()), ref_log_prob)

        # check entropy computation
        self.assertEqual(Categorical(p).entropy(), torch.tensor([1.0114, 1.0297]), atol=1e-4, rtol=0)
        self.assertEqual(Categorical(s).entropy(), torch.tensor([0.0, 0.0]))
        # issue gh-40553
        logits = p.log()
        logits[1, 1] = logits[0, 2] = float('-inf')
        e = Categorical(logits=logits).entropy()
        self.assertEqual(e, torch.tensor([0.6365, 0.5983]), atol=1e-4, rtol=0)

    def test_categorical_enumerate_support(self):
        examples = [
            ({"probs": [0.1, 0.2, 0.7]}, [0, 1, 2]),
            ({"probs": [[0.1, 0.9], [0.3, 0.7]]}, [[0], [1]]),
        ]
        self._check_enumerate_support(Categorical, examples)

    def test_one_hot_categorical_1d(self):
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        self.assertEqual(OneHotCategorical(p).sample().size(), (3,))
        self.assertFalse(OneHotCategorical(p).sample().requires_grad)
        self.assertEqual(OneHotCategorical(p).sample((2, 2)).size(), (2, 2, 3))
        self.assertEqual(OneHotCategorical(p).sample((1,)).size(), (1, 3))
        self._gradcheck_log_prob(OneHotCategorical, (p,))
        self.assertRaises(NotImplementedError, OneHotCategorical(p).rsample)

    def test_one_hot_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        self.assertEqual(OneHotCategorical(p).sample().size(), (2, 3))
        self.assertEqual(OneHotCategorical(p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3))
        self.assertEqual(OneHotCategorical(p).sample((6,)).size(), (6, 2, 3))
        self._gradcheck_log_prob(OneHotCategorical, (p,))

        dist = OneHotCategorical(p)
        x = dist.sample()
        self.assertEqual(dist.log_prob(x), Categorical(p).log_prob(x.max(-1)[1]))

    def test_one_hot_categorical_enumerate_support(self):
        examples = [
            ({"probs": [0.1, 0.2, 0.7]}, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ({"probs": [[0.1, 0.9], [0.3, 0.7]]}, [[[1, 0]], [[0, 1]]]),
        ]
        self._check_enumerate_support(OneHotCategorical, examples)

    def test_poisson_forward_ad(self):
        self._check_forward_ad(torch.poisson)

    def test_poisson_shape(self):
        rate = torch.randn(2, 3).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Poisson(rate).sample().size(), (2, 3))
        self.assertEqual(Poisson(rate).sample((7,)).size(), (7, 2, 3))
        self.assertEqual(Poisson(rate_1d).sample().size(), (1,))
        self.assertEqual(Poisson(rate_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Poisson(2.0).sample((2,)).size(), (2,))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_poisson_log_prob(self):
        rate = torch.randn(2, 3).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        rate_zero = torch.zeros([], requires_grad=True)

        def ref_log_prob(ref_rate, idx, x, log_prob):
            l = ref_rate.view(-1)[idx].detach()
            expected = scipy.stats.poisson.logpmf(x, l)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        set_rng_seed(0)
        self._check_log_prob(Poisson(rate), lambda *args: ref_log_prob(rate, *args))
        self._check_log_prob(Poisson(rate_zero), lambda *args: ref_log_prob(rate_zero, *args))
        self._gradcheck_log_prob(Poisson, (rate,))
        self._gradcheck_log_prob(Poisson, (rate_1d,))

        # We cannot check gradients automatically for zero rates because the finite difference
        # approximation enters the forbidden parameter space. We instead compare with the
        # theoretical results.
        dist = Poisson(rate_zero)
        dist.log_prob(torch.ones_like(rate_zero)).backward()
        self.assertEqual(rate_zero.grad, torch.inf)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_poisson_sample(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for rate in [0.1, 1.0, 5.0]:
            self._check_sampler_discrete(Poisson(rate),
                                         scipy.stats.poisson(rate),
                                         f'Poisson(lambda={rate})',
                                         failure_rate=1e-3)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_poisson_gpu_sample(self):
        set_rng_seed(1)
        for rate in [0.12, 0.9, 4.0]:
            self._check_sampler_discrete(Poisson(torch.tensor([rate]).cuda()),
                                         scipy.stats.poisson(rate),
                                         f'Poisson(lambda={rate}, cuda)',
                                         failure_rate=1e-3)

    def test_relaxed_bernoulli(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        r = torch.tensor(0.3, requires_grad=True)
        s = 0.3
        temp = torch.tensor(0.67, requires_grad=True)
        self.assertEqual(RelaxedBernoulli(temp, p).sample((8,)).size(), (8, 3))
        self.assertFalse(RelaxedBernoulli(temp, p).sample().requires_grad)
        self.assertEqual(RelaxedBernoulli(temp, r).sample((8,)).size(), (8,))
        self.assertEqual(RelaxedBernoulli(temp, r).sample().size(), ())
        self.assertEqual(RelaxedBernoulli(temp, r).sample((3, 2)).size(), (3, 2,))
        self.assertEqual(RelaxedBernoulli(temp, s).sample().size(), ())
        self._gradcheck_log_prob(RelaxedBernoulli, (temp, p))
        self._gradcheck_log_prob(RelaxedBernoulli, (temp, r))

        # test that rsample doesn't fail
        s = RelaxedBernoulli(temp, p).rsample()
        s.backward(torch.ones_like(s))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_rounded_relaxed_bernoulli(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]

        class Rounded:
            def __init__(self, dist):
                self.dist = dist

            def sample(self, *args, **kwargs):
                return torch.round(self.dist.sample(*args, **kwargs))

        for probs, temp in product([0.1, 0.2, 0.8], [0.1, 1.0, 10.0]):
            self._check_sampler_discrete(Rounded(RelaxedBernoulli(temp, probs)),
                                         scipy.stats.bernoulli(probs),
                                         f'Rounded(RelaxedBernoulli(temp={temp}, probs={probs}))',
                                         failure_rate=1e-3)

        for probs in [0.001, 0.2, 0.999]:
            equal_probs = torch.tensor(0.5)
            dist = RelaxedBernoulli(1e10, probs)
            s = dist.rsample()
            self.assertEqual(equal_probs, s)

    def test_relaxed_one_hot_categorical_1d(self):
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        temp = torch.tensor(0.67, requires_grad=True)
        self.assertEqual(RelaxedOneHotCategorical(probs=p, temperature=temp).sample().size(), (3,))
        self.assertFalse(RelaxedOneHotCategorical(probs=p, temperature=temp).sample().requires_grad)
        self.assertEqual(RelaxedOneHotCategorical(probs=p, temperature=temp).sample((2, 2)).size(), (2, 2, 3))
        self.assertEqual(RelaxedOneHotCategorical(probs=p, temperature=temp).sample((1,)).size(), (1, 3))
        self._gradcheck_log_prob(lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp, p))

    def test_relaxed_one_hot_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        temp = torch.tensor([3.0], requires_grad=True)
        # The lower the temperature, the more unstable the log_prob gradcheck is
        # w.r.t. the sample. Values below 0.25 empirically fail the default tol.
        temp_2 = torch.tensor([0.25], requires_grad=True)
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        self.assertEqual(RelaxedOneHotCategorical(temp, p).sample().size(), (2, 3))
        self.assertEqual(RelaxedOneHotCategorical(temp, p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3))
        self.assertEqual(RelaxedOneHotCategorical(temp, p).sample((6,)).size(), (6, 2, 3))
        self._gradcheck_log_prob(lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp, p))
        self._gradcheck_log_prob(lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp_2, p))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_argmax_relaxed_categorical(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]

        class ArgMax:
            def __init__(self, dist):
                self.dist = dist

            def sample(self, *args, **kwargs):
                s = self.dist.sample(*args, **kwargs)
                _, idx = torch.max(s, -1)
                return idx

        class ScipyCategorical:
            def __init__(self, dist):
                self.dist = dist

            def pmf(self, samples):
                new_samples = np.zeros(samples.shape + self.dist.p.shape)
                new_samples[np.arange(samples.shape[0]), samples] = 1
                return self.dist.pmf(new_samples)

        for probs, temp in product([torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.2, 0.6])], [0.1, 1.0, 10.0]):
            self._check_sampler_discrete(ArgMax(RelaxedOneHotCategorical(temp, probs)),
                                         ScipyCategorical(scipy.stats.multinomial(1, probs)),
                                         f'Rounded(RelaxedOneHotCategorical(temp={temp}, probs={probs}))',
                                         failure_rate=1e-3)

        for probs in [torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.2, 0.6])]:
            equal_probs = torch.ones(probs.size()) / probs.size()[0]
            dist = RelaxedOneHotCategorical(1e10, probs)
            s = dist.rsample()
            self.assertEqual(equal_probs, s)

    def test_uniform(self):
        low = torch.zeros(5, 5, requires_grad=True)
        high = (torch.ones(5, 5) * 3).requires_grad_()
        low_1d = torch.zeros(1, requires_grad=True)
        high_1d = (torch.ones(1) * 3).requires_grad_()
        self.assertEqual(Uniform(low, high).sample().size(), (5, 5))
        self.assertEqual(Uniform(low, high).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Uniform(low_1d, high_1d).sample().size(), (1,))
        self.assertEqual(Uniform(low_1d, high_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Uniform(0.0, 1.0).sample((1,)).size(), (1,))

        # Check log_prob computation when value outside range
        uniform = Uniform(low_1d, high_1d, validate_args=False)
        above_high = torch.tensor([4.0])
        below_low = torch.tensor([-1.0])
        self.assertEqual(uniform.log_prob(above_high).item(), -inf)
        self.assertEqual(uniform.log_prob(below_low).item(), -inf)

        # check cdf computation when value outside range
        self.assertEqual(uniform.cdf(below_low).item(), 0)
        self.assertEqual(uniform.cdf(above_high).item(), 1)

        set_rng_seed(1)
        self._gradcheck_log_prob(Uniform, (low, high))
        self._gradcheck_log_prob(Uniform, (low, 1.0))
        self._gradcheck_log_prob(Uniform, (0.0, high))

        state = torch.get_rng_state()
        rand = low.new(low.size()).uniform_()
        torch.set_rng_state(state)
        u = Uniform(low, high).rsample()
        u.backward(torch.ones_like(u))
        self.assertEqual(low.grad, 1 - rand)
        self.assertEqual(high.grad, rand)
        low.grad.zero_()
        high.grad.zero_()

        self._check_forward_ad(lambda x: x.uniform_())

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_vonmises_sample(self):
        for loc in [0.0, math.pi / 2.0]:
            for concentration in [0.03, 0.3, 1.0, 10.0, 100.0]:
                self._check_sampler_sampler(VonMises(loc, concentration),
                                            scipy.stats.vonmises(loc=loc, kappa=concentration),
                                            f"VonMises(loc={loc}, concentration={concentration})",
                                            num_samples=int(1e5), circular=True)

    def test_vonmises_logprob(self):
        concentrations = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
        for concentration in concentrations:
            grid = torch.arange(0., 2 * math.pi, 1e-4)
            prob = VonMises(0.0, concentration).log_prob(grid).exp()
            norm = prob.mean().item() * 2 * math.pi
            self.assertLess(abs(norm - 1), 1e-3)

    def test_cauchy(self):
        loc = torch.zeros(5, 5, requires_grad=True)
        scale = torch.ones(5, 5, requires_grad=True)
        loc_1d = torch.zeros(1, requires_grad=True)
        scale_1d = torch.ones(1, requires_grad=True)
        self.assertTrue(is_all_nan(Cauchy(loc_1d, scale_1d).mean))
        self.assertEqual(Cauchy(loc_1d, scale_1d).variance, inf)
        self.assertEqual(Cauchy(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Cauchy(loc, scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Cauchy(0.0, 1.0).sample((1,)).size(), (1,))

        set_rng_seed(1)
        self._gradcheck_log_prob(Cauchy, (loc, scale))
        self._gradcheck_log_prob(Cauchy, (loc, 1.0))
        self._gradcheck_log_prob(Cauchy, (0.0, scale))

        state = torch.get_rng_state()
        eps = loc.new(loc.size()).cauchy_()
        torch.set_rng_state(state)
        c = Cauchy(loc, scale).rsample()
        c.backward(torch.ones_like(c))
        self.assertEqual(loc.grad, torch.ones_like(scale))
        self.assertEqual(scale.grad, eps)
        loc.grad.zero_()
        scale.grad.zero_()

        self._check_forward_ad(lambda x: x.cauchy_())

    def test_halfcauchy(self):
        scale = torch.ones(5, 5, requires_grad=True)
        scale_1d = torch.ones(1, requires_grad=True)
        self.assertTrue(torch.isinf(HalfCauchy(scale_1d).mean).all())
        self.assertEqual(HalfCauchy(scale_1d).variance, inf)
        self.assertEqual(HalfCauchy(scale).sample().size(), (5, 5))
        self.assertEqual(HalfCauchy(scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(HalfCauchy(scale_1d).sample().size(), (1,))
        self.assertEqual(HalfCauchy(scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(HalfCauchy(1.0).sample((1,)).size(), (1,))

        set_rng_seed(1)
        self._gradcheck_log_prob(HalfCauchy, (scale,))
        self._gradcheck_log_prob(HalfCauchy, (1.0,))

        state = torch.get_rng_state()
        eps = scale.new(scale.size()).cauchy_().abs_()
        torch.set_rng_state(state)
        c = HalfCauchy(scale).rsample()
        c.backward(torch.ones_like(c))
        self.assertEqual(scale.grad, eps)
        scale.grad.zero_()

    def test_halfnormal(self):
        std = torch.randn(5, 5).abs().requires_grad_()
        std_1d = torch.randn(1).abs().requires_grad_()
        std_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(HalfNormal(std).sample().size(), (5, 5))
        self.assertEqual(HalfNormal(std).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(HalfNormal(std_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(HalfNormal(std_1d).sample().size(), (1,))
        self.assertEqual(HalfNormal(.6).sample((1,)).size(), (1,))
        self.assertEqual(HalfNormal(50.0).sample((1,)).size(), (1,))

        # sample check for extreme value of std
        set_rng_seed(1)
        self.assertEqual(HalfNormal(std_delta).sample(sample_shape=(1, 2)),
                         torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
                         atol=1e-4, rtol=0)

        self._gradcheck_log_prob(HalfNormal, (std,))
        self._gradcheck_log_prob(HalfNormal, (1.0,))

        # check .log_prob() can broadcast.
        dist = HalfNormal(torch.ones(2, 1, 4))
        log_prob = dist.log_prob(torch.ones(3, 1))
        self.assertEqual(log_prob.shape, (2, 3, 4))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_halfnormal_logprob(self):
        std = torch.randn(5, 1).abs().requires_grad_()

        def ref_log_prob(idx, x, log_prob):
            s = std.view(-1)[idx].detach()
            expected = scipy.stats.halfnorm(scale=s).logpdf(x)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(HalfNormal(std), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_halfnormal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for std in [0.1, 1.0, 10.0]:
            self._check_sampler_sampler(HalfNormal(std),
                                        scipy.stats.halfnorm(scale=std),
                                        f'HalfNormal(scale={std})')

    def test_lognormal(self):
        mean = torch.randn(5, 5, requires_grad=True)
        std = torch.randn(5, 5).abs().requires_grad_()
        mean_1d = torch.randn(1, requires_grad=True)
        std_1d = torch.randn(1).abs().requires_grad_()
        mean_delta = torch.tensor([1.0, 0.0])
        std_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(LogNormal(mean, std).sample().size(), (5, 5))
        self.assertEqual(LogNormal(mean, std).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(LogNormal(mean_1d, std_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(LogNormal(mean_1d, std_1d).sample().size(), (1,))
        self.assertEqual(LogNormal(0.2, .6).sample((1,)).size(), (1,))
        self.assertEqual(LogNormal(-0.7, 50.0).sample((1,)).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(LogNormal(mean_delta, std_delta).sample(sample_shape=(1, 2)),
                         torch.tensor([[[math.exp(1), 1.0], [math.exp(1), 1.0]]]),
                         atol=1e-4, rtol=0)

        self._gradcheck_log_prob(LogNormal, (mean, std))
        self._gradcheck_log_prob(LogNormal, (mean, 1.0))
        self._gradcheck_log_prob(LogNormal, (0.0, std))

        # check .log_prob() can broadcast.
        dist = LogNormal(torch.zeros(4), torch.ones(2, 1, 1))
        log_prob = dist.log_prob(torch.ones(3, 1))
        self.assertEqual(log_prob.shape, (2, 3, 4))

        self._check_forward_ad(lambda x: x.log_normal_())

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lognormal_logprob(self):
        mean = torch.randn(5, 1, requires_grad=True)
        std = torch.randn(5, 1).abs().requires_grad_()

        def ref_log_prob(idx, x, log_prob):
            m = mean.view(-1)[idx].detach()
            s = std.view(-1)[idx].detach()
            expected = scipy.stats.lognorm(s=s, scale=math.exp(m)).logpdf(x)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(LogNormal(mean, std), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lognormal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for mean, std in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(LogNormal(mean, std),
                                        scipy.stats.lognorm(scale=math.exp(mean), s=std),
                                        f'LogNormal(loc={mean}, scale={std})')

    def test_logisticnormal(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        mean = torch.randn(5, 5).requires_grad_()
        std = torch.randn(5, 5).abs().requires_grad_()
        mean_1d = torch.randn(1).requires_grad_()
        std_1d = torch.randn(1).abs().requires_grad_()
        mean_delta = torch.tensor([1.0, 0.0])
        std_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(LogisticNormal(mean, std).sample().size(), (5, 6))
        self.assertEqual(LogisticNormal(mean, std).sample((7,)).size(), (7, 5, 6))
        self.assertEqual(LogisticNormal(mean_1d, std_1d).sample((1,)).size(), (1, 2))
        self.assertEqual(LogisticNormal(mean_1d, std_1d).sample().size(), (2,))
        self.assertEqual(LogisticNormal(0.2, .6).sample().size(), (2,))
        self.assertEqual(LogisticNormal(-0.7, 50.0).sample().size(), (2,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(LogisticNormal(mean_delta, std_delta).sample(),
                         torch.tensor([math.exp(1) / (1. + 1. + math.exp(1)),
                                       1. / (1. + 1. + math.exp(1)),
                                       1. / (1. + 1. + math.exp(1))]),
                         atol=1e-4, rtol=0)

        # TODO: gradcheck seems to mutate the sample values so that the simplex
        # constraint fails by a very small margin.
        self._gradcheck_log_prob(lambda m, s: LogisticNormal(m, s, validate_args=False), (mean, std))
        self._gradcheck_log_prob(lambda m, s: LogisticNormal(m, s, validate_args=False), (mean, 1.0))
        self._gradcheck_log_prob(lambda m, s: LogisticNormal(m, s, validate_args=False), (0.0, std))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_logisticnormal_logprob(self):
        mean = torch.randn(5, 7).requires_grad_()
        std = torch.randn(5, 7).abs().requires_grad_()

        # Smoke test for now
        # TODO: Once _check_log_prob works with multidimensional distributions,
        #       add proper testing of the log probabilities.
        dist = LogisticNormal(mean, std)
        assert dist.log_prob(dist.sample()).detach().cpu().numpy().shape == (5,)

    def _get_logistic_normal_ref_sampler(self, base_dist):

        def _sampler(num_samples):
            x = base_dist.rvs(num_samples)
            offset = np.log((x.shape[-1] + 1) - np.ones_like(x).cumsum(-1))
            z = 1. / (1. + np.exp(offset - x))
            z_cumprod = np.cumprod(1 - z, axis=-1)
            y1 = np.pad(z, ((0, 0), (0, 1)), mode='constant', constant_values=1.)
            y2 = np.pad(z_cumprod, ((0, 0), (1, 0)), mode='constant', constant_values=1.)
            return y1 * y2

        return _sampler

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_logisticnormal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        means = map(np.asarray, [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)])
        covs = map(np.diag, [(0.1, 0.1), (1.0, 1.0), (10.0, 10.0)])
        for mean, cov in product(means, covs):
            base_dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            ref_dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            ref_dist.rvs = self._get_logistic_normal_ref_sampler(base_dist)
            mean_th = torch.tensor(mean)
            std_th = torch.tensor(np.sqrt(np.diag(cov)))
            self._check_sampler_sampler(
                LogisticNormal(mean_th, std_th), ref_dist,
                f'LogisticNormal(loc={mean_th}, scale={std_th})',
                multivariate=True)

    def test_mixture_same_family_shape(self):
        normal_case_1d = MixtureSameFamily(
            Categorical(torch.rand(5)),
            Normal(torch.randn(5), torch.rand(5)))
        normal_case_1d_batch = MixtureSameFamily(
            Categorical(torch.rand(3, 5)),
            Normal(torch.randn(3, 5), torch.rand(3, 5)))
        normal_case_1d_multi_batch = MixtureSameFamily(
            Categorical(torch.rand(4, 3, 5)),
            Normal(torch.randn(4, 3, 5), torch.rand(4, 3, 5)))
        normal_case_2d = MixtureSameFamily(
            Categorical(torch.rand(5)),
            Independent(Normal(torch.randn(5, 2), torch.rand(5, 2)), 1))
        normal_case_2d_batch = MixtureSameFamily(
            Categorical(torch.rand(3, 5)),
            Independent(Normal(torch.randn(3, 5, 2), torch.rand(3, 5, 2)), 1))
        normal_case_2d_multi_batch = MixtureSameFamily(
            Categorical(torch.rand(4, 3, 5)),
            Independent(Normal(torch.randn(4, 3, 5, 2), torch.rand(4, 3, 5, 2)), 1))

        self.assertEqual(normal_case_1d.sample().size(), ())
        self.assertEqual(normal_case_1d.sample((2,)).size(), (2,))
        self.assertEqual(normal_case_1d.sample((2, 7)).size(), (2, 7))
        self.assertEqual(normal_case_1d_batch.sample().size(), (3,))
        self.assertEqual(normal_case_1d_batch.sample((2,)).size(), (2, 3))
        self.assertEqual(normal_case_1d_batch.sample((2, 7)).size(), (2, 7, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample().size(), (4, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample((2,)).size(), (2, 4, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample((2, 7)).size(), (2, 7, 4, 3))

        self.assertEqual(normal_case_2d.sample().size(), (2,))
        self.assertEqual(normal_case_2d.sample((2,)).size(), (2, 2))
        self.assertEqual(normal_case_2d.sample((2, 7)).size(), (2, 7, 2))
        self.assertEqual(normal_case_2d_batch.sample().size(), (3, 2))
        self.assertEqual(normal_case_2d_batch.sample((2,)).size(), (2, 3, 2))
        self.assertEqual(normal_case_2d_batch.sample((2, 7)).size(), (2, 7, 3, 2))
        self.assertEqual(normal_case_2d_multi_batch.sample().size(), (4, 3, 2))
        self.assertEqual(normal_case_2d_multi_batch.sample((2,)).size(), (2, 4, 3, 2))
        self.assertEqual(normal_case_2d_multi_batch.sample((2, 7)).size(), (2, 7, 4, 3, 2))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_mixture_same_family_log_prob(self):
        probs = torch.rand(5, 5).softmax(dim=-1)
        loc = torch.randn(5, 5)
        scale = torch.rand(5, 5)

        def ref_log_prob(idx, x, log_prob):
            p = probs[idx].numpy()
            m = loc[idx].numpy()
            s = scale[idx].numpy()
            mix = scipy.stats.multinomial(1, p)
            comp = scipy.stats.norm(m, s)
            expected = scipy.special.logsumexp(comp.logpdf(x) + np.log(mix.p))
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(
            MixtureSameFamily(Categorical(probs=probs),
                              Normal(loc, scale)), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_mixture_same_family_sample(self):
        probs = torch.rand(5).softmax(dim=-1)
        loc = torch.randn(5)
        scale = torch.rand(5)

        class ScipyMixtureNormal:
            def __init__(self, probs, mu, std):
                self.probs = probs
                self.mu = mu
                self.std = std

            def rvs(self, n_sample):
                comp_samples = [scipy.stats.norm(m, s).rvs(n_sample) for m, s
                                in zip(self.mu, self.std)]
                mix_samples = scipy.stats.multinomial(1, self.probs).rvs(n_sample)
                samples = []
                for i in range(n_sample):
                    samples.append(comp_samples[mix_samples[i].argmax()][i])
                return np.asarray(samples)

        self._check_sampler_sampler(
            MixtureSameFamily(Categorical(probs=probs), Normal(loc, scale)),
            ScipyMixtureNormal(probs.numpy(), loc.numpy(), scale.numpy()),
            '''MixtureSameFamily(Categorical(probs={}),
            Normal(loc={}, scale={}))'''.format(probs, loc, scale))

    def test_normal(self):
        loc = torch.randn(5, 5, requires_grad=True)
        scale = torch.randn(5, 5).abs().requires_grad_()
        loc_1d = torch.randn(1, requires_grad=True)
        scale_1d = torch.randn(1).abs().requires_grad_()
        loc_delta = torch.tensor([1.0, 0.0])
        scale_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(Normal(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Normal(loc, scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Normal(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Normal(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Normal(0.2, .6).sample((1,)).size(), (1,))
        self.assertEqual(Normal(-0.7, 50.0).sample((1,)).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(Normal(loc_delta, scale_delta).sample(sample_shape=(1, 2)),
                         torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
                         atol=1e-4, rtol=0)

        self._gradcheck_log_prob(Normal, (loc, scale))
        self._gradcheck_log_prob(Normal, (loc, 1.0))
        self._gradcheck_log_prob(Normal, (0.0, scale))

        state = torch.get_rng_state()
        eps = torch.normal(torch.zeros_like(loc), torch.ones_like(scale))
        torch.set_rng_state(state)
        z = Normal(loc, scale).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(loc.grad, torch.ones_like(loc))
        self.assertEqual(scale.grad, eps)
        loc.grad.zero_()
        scale.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            m = loc.view(-1)[idx]
            s = scale.view(-1)[idx]
            expected = (math.exp(-(x - m) ** 2 / (2 * s ** 2)) /
                        math.sqrt(2 * math.pi * s ** 2))
            self.assertEqual(log_prob, math.log(expected), atol=1e-3, rtol=0)

        self._check_log_prob(Normal(loc, scale), ref_log_prob)
        self._check_forward_ad(torch.normal)
        self._check_forward_ad(lambda x: torch.normal(x, 0.5))
        self._check_forward_ad(lambda x: torch.normal(0.2, x))
        self._check_forward_ad(lambda x: torch.normal(x, x))
        self._check_forward_ad(lambda x: x.normal_())

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_normal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for loc, scale in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Normal(loc, scale),
                                        scipy.stats.norm(loc=loc, scale=scale),
                                        f'Normal(mean={loc}, std={scale})')

    def test_lowrank_multivariate_normal_shape(self):
        mean = torch.randn(5, 3, requires_grad=True)
        mean_no_batch = torch.randn(3, requires_grad=True)
        mean_multi_batch = torch.randn(6, 5, 3, requires_grad=True)

        # construct PSD covariance
        cov_factor = torch.randn(3, 1, requires_grad=True)
        cov_diag = torch.randn(3).abs().requires_grad_()

        # construct batch of PSD covariances
        cov_factor_batched = torch.randn(6, 5, 3, 2, requires_grad=True)
        cov_diag_batched = torch.randn(6, 5, 3).abs().requires_grad_()

        # ensure that sample, batch, event shapes all handled correctly
        self.assertEqual(LowRankMultivariateNormal(mean, cov_factor, cov_diag)
                         .sample().size(), (5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_no_batch, cov_factor, cov_diag)
                         .sample().size(), (3,))
        self.assertEqual(LowRankMultivariateNormal(mean_multi_batch, cov_factor, cov_diag)
                         .sample().size(), (6, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean, cov_factor, cov_diag)
                         .sample((2,)).size(), (2, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_no_batch, cov_factor, cov_diag)
                         .sample((2,)).size(), (2, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_multi_batch, cov_factor, cov_diag)
                         .sample((2,)).size(), (2, 6, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean, cov_factor, cov_diag)
                         .sample((2, 7)).size(), (2, 7, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_no_batch, cov_factor, cov_diag)
                         .sample((2, 7)).size(), (2, 7, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_multi_batch, cov_factor, cov_diag)
                         .sample((2, 7)).size(), (2, 7, 6, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean, cov_factor_batched, cov_diag_batched)
                         .sample((2, 7)).size(), (2, 7, 6, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_no_batch, cov_factor_batched, cov_diag_batched)
                         .sample((2, 7)).size(), (2, 7, 6, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_multi_batch, cov_factor_batched, cov_diag_batched)
                         .sample((2, 7)).size(), (2, 7, 6, 5, 3))

        # check gradients
        self._gradcheck_log_prob(LowRankMultivariateNormal,
                                 (mean, cov_factor, cov_diag))
        self._gradcheck_log_prob(LowRankMultivariateNormal,
                                 (mean_multi_batch, cov_factor, cov_diag))
        self._gradcheck_log_prob(LowRankMultivariateNormal,
                                 (mean_multi_batch, cov_factor_batched, cov_diag_batched))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_lowrank_multivariate_normal_log_prob(self):
        mean = torch.randn(3, requires_grad=True)
        cov_factor = torch.randn(3, 1, requires_grad=True)
        cov_diag = torch.randn(3).abs().requires_grad_()
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()

        # check that logprob values match scipy logpdf,
        # and that covariance and scale_tril parameters are equivalent
        dist1 = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        ref_dist = scipy.stats.multivariate_normal(mean.detach().numpy(), cov.detach().numpy())

        x = dist1.sample((10,))
        expected = ref_dist.logpdf(x.numpy())

        self.assertEqual(0.0, np.mean((dist1.log_prob(x).detach().numpy() - expected)**2), atol=1e-3, rtol=0)

        # Double-check that batched versions behave the same as unbatched
        mean = torch.randn(5, 3, requires_grad=True)
        cov_factor = torch.randn(5, 3, 2, requires_grad=True)
        cov_diag = torch.randn(5, 3).abs().requires_grad_()

        dist_batched = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        dist_unbatched = [LowRankMultivariateNormal(mean[i], cov_factor[i], cov_diag[i])
                          for i in range(mean.size(0))]

        x = dist_batched.sample((10,))
        batched_prob = dist_batched.log_prob(x)
        unbatched_prob = torch.stack([dist_unbatched[i].log_prob(x[:, i]) for i in range(5)]).t()

        self.assertEqual(batched_prob.shape, unbatched_prob.shape)
        self.assertEqual(batched_prob, unbatched_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lowrank_multivariate_normal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        mean = torch.randn(5, requires_grad=True)
        cov_factor = torch.randn(5, 1, requires_grad=True)
        cov_diag = torch.randn(5).abs().requires_grad_()
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()

        self._check_sampler_sampler(LowRankMultivariateNormal(mean, cov_factor, cov_diag),
                                    scipy.stats.multivariate_normal(mean.detach().numpy(), cov.detach().numpy()),
                                    'LowRankMultivariateNormal(loc={}, cov_factor={}, cov_diag={})'
                                    .format(mean, cov_factor, cov_diag), multivariate=True)

    def test_lowrank_multivariate_normal_properties(self):
        loc = torch.randn(5)
        cov_factor = torch.randn(5, 2)
        cov_diag = torch.randn(5).abs()
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()
        m1 = LowRankMultivariateNormal(loc, cov_factor, cov_diag)
        m2 = MultivariateNormal(loc=loc, covariance_matrix=cov)
        self.assertEqual(m1.mean, m2.mean)
        self.assertEqual(m1.variance, m2.variance)
        self.assertEqual(m1.covariance_matrix, m2.covariance_matrix)
        self.assertEqual(m1.scale_tril, m2.scale_tril)
        self.assertEqual(m1.precision_matrix, m2.precision_matrix)
        self.assertEqual(m1.entropy(), m2.entropy())

    def test_lowrank_multivariate_normal_moments(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        mean = torch.randn(5)
        cov_factor = torch.randn(5, 2)
        cov_diag = torch.randn(5).abs()
        d = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        samples = d.rsample((100000,))
        empirical_mean = samples.mean(0)
        self.assertEqual(d.mean, empirical_mean, atol=0.01, rtol=0)
        empirical_var = samples.var(0)
        self.assertEqual(d.variance, empirical_var, atol=0.02, rtol=0)

    def test_multivariate_normal_shape(self):
        mean = torch.randn(5, 3, requires_grad=True)
        mean_no_batch = torch.randn(3, requires_grad=True)
        mean_multi_batch = torch.randn(6, 5, 3, requires_grad=True)

        # construct PSD covariance
        tmp = torch.randn(3, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # construct batch of PSD covariances
        tmp = torch.randn(6, 5, 3, 10)
        cov_batched = (tmp.unsqueeze(-2) * tmp.unsqueeze(-3)).mean(-1).requires_grad_()
        prec_batched = cov_batched.inverse()
        scale_tril_batched = torch.linalg.cholesky(cov_batched)

        # ensure that sample, batch, event shapes all handled correctly
        self.assertEqual(MultivariateNormal(mean, cov).sample().size(), (5, 3))
        self.assertEqual(MultivariateNormal(mean_no_batch, cov).sample().size(), (3,))
        self.assertEqual(MultivariateNormal(mean_multi_batch, cov).sample().size(), (6, 5, 3))
        self.assertEqual(MultivariateNormal(mean, cov).sample((2,)).size(), (2, 5, 3))
        self.assertEqual(MultivariateNormal(mean_no_batch, cov).sample((2,)).size(), (2, 3))
        self.assertEqual(MultivariateNormal(mean_multi_batch, cov).sample((2,)).size(), (2, 6, 5, 3))
        self.assertEqual(MultivariateNormal(mean, cov).sample((2, 7)).size(), (2, 7, 5, 3))
        self.assertEqual(MultivariateNormal(mean_no_batch, cov).sample((2, 7)).size(), (2, 7, 3))
        self.assertEqual(MultivariateNormal(mean_multi_batch, cov).sample((2, 7)).size(), (2, 7, 6, 5, 3))
        self.assertEqual(MultivariateNormal(mean, cov_batched).sample((2, 7)).size(), (2, 7, 6, 5, 3))
        self.assertEqual(MultivariateNormal(mean_no_batch, cov_batched).sample((2, 7)).size(), (2, 7, 6, 5, 3))
        self.assertEqual(MultivariateNormal(mean_multi_batch, cov_batched).sample((2, 7)).size(), (2, 7, 6, 5, 3))
        self.assertEqual(MultivariateNormal(mean, precision_matrix=prec).sample((2, 7)).size(), (2, 7, 5, 3))
        self.assertEqual(MultivariateNormal(mean, precision_matrix=prec_batched).sample((2, 7)).size(), (2, 7, 6, 5, 3))
        self.assertEqual(MultivariateNormal(mean, scale_tril=scale_tril).sample((2, 7)).size(), (2, 7, 5, 3))
        self.assertEqual(MultivariateNormal(mean, scale_tril=scale_tril_batched).sample((2, 7)).size(), (2, 7, 6, 5, 3))

        # check gradients
        # We write a custom gradcheck function to maintain the symmetry
        # of the perturbed covariances and their inverses (precision)
        def multivariate_normal_log_prob_gradcheck(mean, covariance=None, precision=None, scale_tril=None):
            mvn_samples = MultivariateNormal(mean, covariance, precision, scale_tril).sample().requires_grad_()

            def gradcheck_func(samples, mu, sigma, prec, scale_tril):
                if sigma is not None:
                    sigma = 0.5 * (sigma + sigma.mT)  # Ensure symmetry of covariance
                if prec is not None:
                    prec = 0.5 * (prec + prec.mT)  # Ensure symmetry of precision
                if scale_tril is not None:
                    scale_tril = scale_tril.tril()
                return MultivariateNormal(mu, sigma, prec, scale_tril).log_prob(samples)
            gradcheck(gradcheck_func, (mvn_samples, mean, covariance, precision, scale_tril), raise_exception=True)

        multivariate_normal_log_prob_gradcheck(mean, cov)
        multivariate_normal_log_prob_gradcheck(mean_multi_batch, cov)
        multivariate_normal_log_prob_gradcheck(mean_multi_batch, cov_batched)
        multivariate_normal_log_prob_gradcheck(mean, None, prec)
        multivariate_normal_log_prob_gradcheck(mean_no_batch, None, prec_batched)
        multivariate_normal_log_prob_gradcheck(mean, None, None, scale_tril)
        multivariate_normal_log_prob_gradcheck(mean_no_batch, None, None, scale_tril_batched)

    def test_multivariate_normal_stable_with_precision_matrix(self):
        x = torch.randn(10)
        P = torch.exp(-(x - x.unsqueeze(-1)) ** 2)  # RBF kernel
        MultivariateNormal(x.new_zeros(10), precision_matrix=P)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_multivariate_normal_log_prob(self):
        mean = torch.randn(3, requires_grad=True)
        tmp = torch.randn(3, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # check that logprob values match scipy logpdf,
        # and that covariance and scale_tril parameters are equivalent
        dist1 = MultivariateNormal(mean, cov)
        dist2 = MultivariateNormal(mean, precision_matrix=prec)
        dist3 = MultivariateNormal(mean, scale_tril=scale_tril)
        ref_dist = scipy.stats.multivariate_normal(mean.detach().numpy(), cov.detach().numpy())

        x = dist1.sample((10,))
        expected = ref_dist.logpdf(x.numpy())

        self.assertEqual(0.0, np.mean((dist1.log_prob(x).detach().numpy() - expected)**2), atol=1e-3, rtol=0)
        self.assertEqual(0.0, np.mean((dist2.log_prob(x).detach().numpy() - expected)**2), atol=1e-3, rtol=0)
        self.assertEqual(0.0, np.mean((dist3.log_prob(x).detach().numpy() - expected)**2), atol=1e-3, rtol=0)

        # Double-check that batched versions behave the same as unbatched
        mean = torch.randn(5, 3, requires_grad=True)
        tmp = torch.randn(5, 3, 10)
        cov = (tmp.unsqueeze(-2) * tmp.unsqueeze(-3)).mean(-1).requires_grad_()

        dist_batched = MultivariateNormal(mean, cov)
        dist_unbatched = [MultivariateNormal(mean[i], cov[i]) for i in range(mean.size(0))]

        x = dist_batched.sample((10,))
        batched_prob = dist_batched.log_prob(x)
        unbatched_prob = torch.stack([dist_unbatched[i].log_prob(x[:, i]) for i in range(5)]).t()

        self.assertEqual(batched_prob.shape, unbatched_prob.shape)
        self.assertEqual(batched_prob, unbatched_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_multivariate_normal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        mean = torch.randn(3, requires_grad=True)
        tmp = torch.randn(3, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        self._check_sampler_sampler(MultivariateNormal(mean, cov),
                                    scipy.stats.multivariate_normal(mean.detach().numpy(), cov.detach().numpy()),
                                    f'MultivariateNormal(loc={mean}, cov={cov})',
                                    multivariate=True)
        self._check_sampler_sampler(MultivariateNormal(mean, precision_matrix=prec),
                                    scipy.stats.multivariate_normal(mean.detach().numpy(), cov.detach().numpy()),
                                    f'MultivariateNormal(loc={mean}, atol={prec})',
                                    multivariate=True)
        self._check_sampler_sampler(MultivariateNormal(mean, scale_tril=scale_tril),
                                    scipy.stats.multivariate_normal(mean.detach().numpy(), cov.detach().numpy()),
                                    f'MultivariateNormal(loc={mean}, scale_tril={scale_tril})',
                                    multivariate=True)

    def test_multivariate_normal_properties(self):
        loc = torch.randn(5)
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(5, 5))
        m = MultivariateNormal(loc=loc, scale_tril=scale_tril)
        self.assertEqual(m.covariance_matrix, m.scale_tril.mm(m.scale_tril.t()))
        self.assertEqual(m.covariance_matrix.mm(m.precision_matrix), torch.eye(m.event_shape[0]))
        self.assertEqual(m.scale_tril, torch.linalg.cholesky(m.covariance_matrix))

    def test_multivariate_normal_moments(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        mean = torch.randn(5)
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(5, 5))
        d = MultivariateNormal(mean, scale_tril=scale_tril)
        samples = d.rsample((100000,))
        empirical_mean = samples.mean(0)
        self.assertEqual(d.mean, empirical_mean, atol=0.01, rtol=0)
        empirical_var = samples.var(0)
        self.assertEqual(d.variance, empirical_var, atol=0.05, rtol=0)

    # We applied same tests in Multivariate Normal distribution for Wishart distribution
    def test_wishart_shape(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 3

        df = torch.rand(5, requires_grad=True) + ndim
        df_no_batch = torch.rand([], requires_grad=True) + ndim
        df_multi_batch = torch.rand(6, 5, requires_grad=True) + ndim

        # construct PSD covariance
        tmp = torch.randn(ndim, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # construct batch of PSD covariances
        tmp = torch.randn(6, 5, ndim, 10)
        cov_batched = (tmp.unsqueeze(-2) * tmp.unsqueeze(-3)).mean(-1).requires_grad_()
        prec_batched = cov_batched.inverse()
        scale_tril_batched = torch.linalg.cholesky(cov_batched)

        # ensure that sample, batch, event shapes all handled correctly
        self.assertEqual(Wishart(df, cov).sample().size(), (5, ndim, ndim))
        self.assertEqual(Wishart(df_no_batch, cov).sample().size(), (ndim, ndim))
        self.assertEqual(Wishart(df_multi_batch, cov).sample().size(), (6, 5, ndim, ndim))
        self.assertEqual(Wishart(df, cov).sample((2,)).size(), (2, 5, ndim, ndim))
        self.assertEqual(Wishart(df_no_batch, cov).sample((2,)).size(), (2, ndim, ndim))
        self.assertEqual(Wishart(df_multi_batch, cov).sample((2,)).size(), (2, 6, 5, ndim, ndim))
        self.assertEqual(Wishart(df, cov).sample((2, 7)).size(), (2, 7, 5, ndim, ndim))
        self.assertEqual(Wishart(df_no_batch, cov).sample((2, 7)).size(), (2, 7, ndim, ndim))
        self.assertEqual(Wishart(df_multi_batch, cov).sample((2, 7)).size(), (2, 7, 6, 5, ndim, ndim))
        self.assertEqual(Wishart(df, cov_batched).sample((2, 7)).size(), (2, 7, 6, 5, ndim, ndim))
        self.assertEqual(Wishart(df_no_batch, cov_batched).sample((2, 7)).size(), (2, 7, 6, 5, ndim, ndim))
        self.assertEqual(Wishart(df_multi_batch, cov_batched).sample((2, 7)).size(), (2, 7, 6, 5, ndim, ndim))
        self.assertEqual(Wishart(df, precision_matrix=prec).sample((2, 7)).size(), (2, 7, 5, ndim, ndim))
        self.assertEqual(Wishart(df, precision_matrix=prec_batched).sample((2, 7)).size(), (2, 7, 6, 5, ndim, ndim))
        self.assertEqual(Wishart(df, scale_tril=scale_tril).sample((2, 7)).size(), (2, 7, 5, ndim, ndim))
        self.assertEqual(Wishart(df, scale_tril=scale_tril_batched).sample((2, 7)).size(), (2, 7, 6, 5, ndim, ndim))

        # check gradients
        # Modified and applied the same tests for multivariate_normal
        def wishart_log_prob_gradcheck(df=None, covariance=None, precision=None, scale_tril=None):
            wishart_samples = Wishart(df, covariance, precision, scale_tril).sample().requires_grad_()

            def gradcheck_func(samples, nu, sigma, prec, scale_tril):
                if sigma is not None:
                    sigma = 0.5 * (sigma + sigma.mT)  # Ensure symmetry of covariance
                if prec is not None:
                    prec = 0.5 * (prec + prec.mT)  # Ensure symmetry of precision
                if scale_tril is not None:
                    scale_tril = scale_tril.tril()
                return Wishart(nu, sigma, prec, scale_tril).log_prob(samples)
            gradcheck(gradcheck_func, (wishart_samples, df, covariance, precision, scale_tril), raise_exception=True)

        wishart_log_prob_gradcheck(df, cov)
        wishart_log_prob_gradcheck(df_multi_batch, cov)
        wishart_log_prob_gradcheck(df_multi_batch, cov_batched)
        wishart_log_prob_gradcheck(df, None, prec)
        wishart_log_prob_gradcheck(df_no_batch, None, prec_batched)
        wishart_log_prob_gradcheck(df, None, None, scale_tril)
        wishart_log_prob_gradcheck(df_no_batch, None, None, scale_tril_batched)

    def test_wishart_stable_with_precision_matrix(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 10
        x = torch.randn(ndim)
        P = torch.exp(-(x - x.unsqueeze(-1)) ** 2)  # RBF kernel
        Wishart(torch.tensor(ndim), precision_matrix=P)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_wishart_log_prob(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 3
        df = torch.rand([], requires_grad=True) + ndim - 1
        # SciPy allowed ndim -1 < df < ndim for Wishar distribution after version 1.7.0
        if version.parse(scipy.__version__) < version.parse("1.7.0"):
            df += 1.
        tmp = torch.randn(ndim, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # check that logprob values match scipy logpdf,
        # and that covariance and scale_tril parameters are equivalent
        dist1 = Wishart(df, cov)
        dist2 = Wishart(df, precision_matrix=prec)
        dist3 = Wishart(df, scale_tril=scale_tril)
        ref_dist = scipy.stats.wishart(df.item(), cov.detach().numpy())

        x = dist1.sample((1000,))
        expected = ref_dist.logpdf(x.transpose(0, 2).numpy())

        self.assertEqual(0.0, np.mean((dist1.log_prob(x).detach().numpy() - expected)**2), atol=1e-3, rtol=0)
        self.assertEqual(0.0, np.mean((dist2.log_prob(x).detach().numpy() - expected)**2), atol=1e-3, rtol=0)
        self.assertEqual(0.0, np.mean((dist3.log_prob(x).detach().numpy() - expected)**2), atol=1e-3, rtol=0)

        # Double-check that batched versions behave the same as unbatched
        df = torch.rand(5, requires_grad=True) + ndim - 1
        # SciPy allowed ndim -1 < df < ndim for Wishar distribution after version 1.7.0
        if version.parse(scipy.__version__) < version.parse("1.7.0"):
            df += 1.
        tmp = torch.randn(5, ndim, 10)
        cov = (tmp.unsqueeze(-2) * tmp.unsqueeze(-3)).mean(-1).requires_grad_()

        dist_batched = Wishart(df, cov)
        dist_unbatched = [Wishart(df[i], cov[i]) for i in range(df.size(0))]

        x = dist_batched.sample((1000,))
        batched_prob = dist_batched.log_prob(x)
        unbatched_prob = torch.stack([dist_unbatched[i].log_prob(x[:, i]) for i in range(5)]).t()

        self.assertEqual(batched_prob.shape, unbatched_prob.shape)
        self.assertEqual(batched_prob, unbatched_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_wishart_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 3
        df = torch.rand([], requires_grad=True) + ndim - 1
        # SciPy allowed ndim -1 < df < ndim for Wishar distribution after version 1.7.0
        if version.parse(scipy.__version__) < version.parse("1.7.0"):
            df += 1.
        tmp = torch.randn(ndim, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        ref_dist = scipy.stats.wishart(df.item(), cov.detach().numpy())

        self._check_sampler_sampler(Wishart(df, cov),
                                    ref_dist,
                                    f'Wishart(df={df}, covariance_matrix={cov})',
                                    multivariate=True)
        self._check_sampler_sampler(Wishart(df, precision_matrix=prec),
                                    ref_dist,
                                    f'Wishart(df={df}, precision_matrix={prec})',
                                    multivariate=True)
        self._check_sampler_sampler(Wishart(df, scale_tril=scale_tril),
                                    ref_dist,
                                    f'Wishart(df={df}, scale_tril={scale_tril})',
                                    multivariate=True)

    def test_wishart_properties(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 5
        df = torch.rand([]) + ndim - 1
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(ndim, ndim))
        m = Wishart(df=df, scale_tril=scale_tril)
        self.assertEqual(m.covariance_matrix, m.scale_tril.mm(m.scale_tril.t()))
        self.assertEqual(m.covariance_matrix.mm(m.precision_matrix), torch.eye(m.event_shape[0]))
        self.assertEqual(m.scale_tril, torch.linalg.cholesky(m.covariance_matrix))

    def test_wishart_moments(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 3
        df = torch.rand([]) + ndim - 1
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(ndim, ndim))
        d = Wishart(df=df, scale_tril=scale_tril)
        samples = d.rsample((ndim * ndim * 100000,))
        empirical_mean = samples.mean(0)
        self.assertEqual(d.mean, empirical_mean, atol=0.5, rtol=0)
        empirical_var = samples.var(0)
        self.assertEqual(d.variance, empirical_var, atol=0.5, rtol=0)

    def test_exponential(self):
        rate = torch.randn(5, 5).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Exponential(rate).sample().size(), (5, 5))
        self.assertEqual(Exponential(rate).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Exponential(rate_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Exponential(rate_1d).sample().size(), (1,))
        self.assertEqual(Exponential(0.2).sample((1,)).size(), (1,))
        self.assertEqual(Exponential(50.0).sample((1,)).size(), (1,))

        self._gradcheck_log_prob(Exponential, (rate,))
        state = torch.get_rng_state()
        eps = rate.new(rate.size()).exponential_()
        torch.set_rng_state(state)
        z = Exponential(rate).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(rate.grad, -eps / rate**2)
        rate.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            m = rate.view(-1)[idx]
            expected = math.log(m) - m * x
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Exponential(rate), ref_log_prob)
        self._check_forward_ad(lambda x: x.exponential_())

        def mean_var(lambd, sample):
            sample.exponential_(lambd)
            mean = sample.float().mean()
            var = sample.float().var()
            self.assertEqual((1. / lambd), mean, atol=2e-2, rtol=2e-2)
            self.assertEqual((1. / lambd) ** 2, var, atol=2e-2, rtol=2e-2)

        for dtype in [torch.float, torch.double, torch.bfloat16, torch.float16]:
            for lambd in [0.2, 0.5, 1., 1.5, 2., 5.]:
                sample_len = 50000
                mean_var(lambd, torch.rand(sample_len, dtype=dtype))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_exponential_sample(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for rate in [1e-5, 1.0, 10.]:
            self._check_sampler_sampler(Exponential(rate),
                                        scipy.stats.expon(scale=1. / rate),
                                        f'Exponential(rate={rate})')

    def test_laplace(self):
        loc = torch.randn(5, 5, requires_grad=True)
        scale = torch.randn(5, 5).abs().requires_grad_()
        loc_1d = torch.randn(1, requires_grad=True)
        scale_1d = torch.randn(1, requires_grad=True)
        loc_delta = torch.tensor([1.0, 0.0])
        scale_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(Laplace(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Laplace(loc, scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Laplace(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Laplace(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Laplace(0.2, .6).sample((1,)).size(), (1,))
        self.assertEqual(Laplace(-0.7, 50.0).sample((1,)).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(0)
        self.assertEqual(Laplace(loc_delta, scale_delta).sample(sample_shape=(1, 2)),
                         torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
                         atol=1e-4, rtol=0)

        self._gradcheck_log_prob(Laplace, (loc, scale))
        self._gradcheck_log_prob(Laplace, (loc, 1.0))
        self._gradcheck_log_prob(Laplace, (0.0, scale))

        state = torch.get_rng_state()
        eps = torch.ones_like(loc).uniform_(-.5, .5)
        torch.set_rng_state(state)
        z = Laplace(loc, scale).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(loc.grad, torch.ones_like(loc))
        self.assertEqual(scale.grad, -eps.sign() * torch.log1p(-2 * eps.abs()))
        loc.grad.zero_()
        scale.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            m = loc.view(-1)[idx]
            s = scale.view(-1)[idx]
            expected = (-math.log(2 * s) - abs(x - m) / s)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Laplace(loc, scale), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_laplace_sample(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for loc, scale in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Laplace(loc, scale),
                                        scipy.stats.laplace(loc=loc, scale=scale),
                                        f'Laplace(loc={loc}, scale={scale})')

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma_shape(self):
        alpha = torch.randn(2, 3).exp().requires_grad_()
        beta = torch.randn(2, 3).exp().requires_grad_()
        alpha_1d = torch.randn(1).exp().requires_grad_()
        beta_1d = torch.randn(1).exp().requires_grad_()
        self.assertEqual(Gamma(alpha, beta).sample().size(), (2, 3))
        self.assertEqual(Gamma(alpha, beta).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample().size(), (1,))
        self.assertEqual(Gamma(0.5, 0.5).sample().size(), ())
        self.assertEqual(Gamma(0.5, 0.5).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            a = alpha.view(-1)[idx].detach()
            b = beta.view(-1)[idx].detach()
            expected = scipy.stats.gamma.logpdf(x, a, scale=1 / b)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Gamma(alpha, beta), ref_log_prob)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma_gpu_shape(self):
        alpha = torch.randn(2, 3).cuda().exp().requires_grad_()
        beta = torch.randn(2, 3).cuda().exp().requires_grad_()
        alpha_1d = torch.randn(1).cuda().exp().requires_grad_()
        beta_1d = torch.randn(1).cuda().exp().requires_grad_()
        self.assertEqual(Gamma(alpha, beta).sample().size(), (2, 3))
        self.assertEqual(Gamma(alpha, beta).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample().size(), (1,))
        self.assertEqual(Gamma(0.5, 0.5).sample().size(), ())
        self.assertEqual(Gamma(0.5, 0.5).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            a = alpha.view(-1)[idx].detach().cpu()
            b = beta.view(-1)[idx].detach().cpu()
            expected = scipy.stats.gamma.logpdf(x.cpu(), a, scale=1 / b)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Gamma(alpha, beta), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for alpha, beta in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Gamma(alpha, beta),
                                        scipy.stats.gamma(alpha, scale=1.0 / beta),
                                        f'Gamma(concentration={alpha}, rate={beta})')

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_gamma_gpu_sample(self):
        set_rng_seed(0)
        for alpha, beta in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            a, b = torch.tensor([alpha]).cuda(), torch.tensor([beta]).cuda()
            self._check_sampler_sampler(Gamma(a, b),
                                        scipy.stats.gamma(alpha, scale=1.0 / beta),
                                        f'Gamma(alpha={alpha}, beta={beta})',
                                        failure_rate=1e-4)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_pareto(self):
        scale = torch.randn(2, 3).abs().requires_grad_()
        alpha = torch.randn(2, 3).abs().requires_grad_()
        scale_1d = torch.randn(1).abs().requires_grad_()
        alpha_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Pareto(scale_1d, 0.5).mean, inf)
        self.assertEqual(Pareto(scale_1d, 0.5).variance, inf)
        self.assertEqual(Pareto(scale, alpha).sample().size(), (2, 3))
        self.assertEqual(Pareto(scale, alpha).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Pareto(scale_1d, alpha_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Pareto(scale_1d, alpha_1d).sample().size(), (1,))
        self.assertEqual(Pareto(1.0, 1.0).sample().size(), ())
        self.assertEqual(Pareto(1.0, 1.0).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            s = scale.view(-1)[idx].detach()
            a = alpha.view(-1)[idx].detach()
            expected = scipy.stats.pareto.logpdf(x, a, scale=s)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Pareto(scale, alpha), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_pareto_sample(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for scale, alpha in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Pareto(scale, alpha),
                                        scipy.stats.pareto(alpha, scale=scale),
                                        f'Pareto(scale={scale}, alpha={alpha})')

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gumbel(self):
        loc = torch.randn(2, 3, requires_grad=True)
        scale = torch.randn(2, 3).abs().requires_grad_()
        loc_1d = torch.randn(1, requires_grad=True)
        scale_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Gumbel(loc, scale).sample().size(), (2, 3))
        self.assertEqual(Gumbel(loc, scale).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Gumbel(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Gumbel(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Gumbel(1.0, 1.0).sample().size(), ())
        self.assertEqual(Gumbel(1.0, 1.0).sample((1,)).size(), (1,))
        self.assertEqual(Gumbel(torch.tensor(0.0, dtype=torch.float32),
                                torch.tensor(1.0, dtype=torch.float32),
                                validate_args=False).cdf(20.0), 1.0, atol=1e-4, rtol=0)
        self.assertEqual(Gumbel(torch.tensor(0.0, dtype=torch.float64),
                                torch.tensor(1.0, dtype=torch.float64),
                                validate_args=False).cdf(50.0), 1.0, atol=1e-4, rtol=0)
        self.assertEqual(Gumbel(torch.tensor(0.0, dtype=torch.float32),
                                torch.tensor(1.0, dtype=torch.float32),
                                validate_args=False).cdf(-5.0), 0.0, atol=1e-4, rtol=0)
        self.assertEqual(Gumbel(torch.tensor(0.0, dtype=torch.float64),
                                torch.tensor(1.0, dtype=torch.float64),
                                validate_args=False).cdf(-10.0), 0.0, atol=1e-8, rtol=0)

        def ref_log_prob(idx, x, log_prob):
            l = loc.view(-1)[idx].detach()
            s = scale.view(-1)[idx].detach()
            expected = scipy.stats.gumbel_r.logpdf(x, loc=l, scale=s)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Gumbel(loc, scale), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gumbel_sample(self):
        set_rng_seed(1)  # see note [Randomized statistical tests]
        for loc, scale in product([-5.0, -1.0, -0.1, 0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Gumbel(loc, scale),
                                        scipy.stats.gumbel_r(loc=loc, scale=scale),
                                        f'Gumbel(loc={loc}, scale={scale})')

    def test_kumaraswamy_shape(self):
        concentration1 = torch.randn(2, 3).abs().requires_grad_()
        concentration0 = torch.randn(2, 3).abs().requires_grad_()
        concentration1_1d = torch.randn(1).abs().requires_grad_()
        concentration0_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Kumaraswamy(concentration1, concentration0).sample().size(), (2, 3))
        self.assertEqual(Kumaraswamy(concentration1, concentration0).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Kumaraswamy(concentration1_1d, concentration0_1d).sample().size(), (1,))
        self.assertEqual(Kumaraswamy(concentration1_1d, concentration0_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Kumaraswamy(1.0, 1.0).sample().size(), ())
        self.assertEqual(Kumaraswamy(1.0, 1.0).sample((1,)).size(), (1,))

    # Kumaraswamy distribution is not implemented in SciPy
    # Hence these tests are explicit
    def test_kumaraswamy_mean_variance(self):
        c1_1 = torch.randn(2, 3).abs().requires_grad_()
        c0_1 = torch.randn(2, 3).abs().requires_grad_()
        c1_2 = torch.randn(4).abs().requires_grad_()
        c0_2 = torch.randn(4).abs().requires_grad_()
        cases = [(c1_1, c0_1), (c1_2, c0_2)]
        for i, (a, b) in enumerate(cases):
            m = Kumaraswamy(a, b)
            samples = m.sample((60000, ))
            expected = samples.mean(0)
            actual = m.mean
            error = (expected - actual).abs()
            max_error = max(error[error == error])
            self.assertLess(max_error, 0.01,
                            f"Kumaraswamy example {i + 1}/{len(cases)}, incorrect .mean")
            expected = samples.var(0)
            actual = m.variance
            error = (expected - actual).abs()
            max_error = max(error[error == error])
            self.assertLess(max_error, 0.01,
                            f"Kumaraswamy example {i + 1}/{len(cases)}, incorrect .variance")

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_fishersnedecor(self):
        df1 = torch.randn(2, 3).abs().requires_grad_()
        df2 = torch.randn(2, 3).abs().requires_grad_()
        df1_1d = torch.randn(1).abs()
        df2_1d = torch.randn(1).abs()
        self.assertTrue(is_all_nan(FisherSnedecor(1, 2).mean))
        self.assertTrue(is_all_nan(FisherSnedecor(1, 4).variance))
        self.assertEqual(FisherSnedecor(df1, df2).sample().size(), (2, 3))
        self.assertEqual(FisherSnedecor(df1, df2).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(FisherSnedecor(df1_1d, df2_1d).sample().size(), (1,))
        self.assertEqual(FisherSnedecor(df1_1d, df2_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(FisherSnedecor(1.0, 1.0).sample().size(), ())
        self.assertEqual(FisherSnedecor(1.0, 1.0).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            f1 = df1.view(-1)[idx].detach()
            f2 = df2.view(-1)[idx].detach()
            expected = scipy.stats.f.logpdf(x, f1, f2)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(FisherSnedecor(df1, df2), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_fishersnedecor_sample(self):
        set_rng_seed(1)  # see note [Randomized statistical tests]
        for df1, df2 in product([0.1, 0.5, 1.0, 5.0, 10.0], [0.1, 0.5, 1.0, 5.0, 10.0]):
            self._check_sampler_sampler(FisherSnedecor(df1, df2),
                                        scipy.stats.f(df1, df2),
                                        f'FisherSnedecor(loc={df1}, scale={df2})')

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_chi2_shape(self):
        df = torch.randn(2, 3).exp().requires_grad_()
        df_1d = torch.randn(1).exp().requires_grad_()
        self.assertEqual(Chi2(df).sample().size(), (2, 3))
        self.assertEqual(Chi2(df).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Chi2(df_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Chi2(df_1d).sample().size(), (1,))
        self.assertEqual(Chi2(torch.tensor(0.5, requires_grad=True)).sample().size(), ())
        self.assertEqual(Chi2(0.5).sample().size(), ())
        self.assertEqual(Chi2(0.5).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            d = df.view(-1)[idx].detach()
            expected = scipy.stats.chi2.logpdf(x, d)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Chi2(df), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_chi2_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for df in [0.1, 1.0, 5.0]:
            self._check_sampler_sampler(Chi2(df),
                                        scipy.stats.chi2(df),
                                        f'Chi2(df={df})')

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_studentT(self):
        df = torch.randn(2, 3).exp().requires_grad_()
        df_1d = torch.randn(1).exp().requires_grad_()
        self.assertTrue(is_all_nan(StudentT(1).mean))
        self.assertTrue(is_all_nan(StudentT(1).variance))
        self.assertEqual(StudentT(2).variance, inf)
        self.assertEqual(StudentT(df).sample().size(), (2, 3))
        self.assertEqual(StudentT(df).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(StudentT(df_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(StudentT(df_1d).sample().size(), (1,))
        self.assertEqual(StudentT(torch.tensor(0.5, requires_grad=True)).sample().size(), ())
        self.assertEqual(StudentT(0.5).sample().size(), ())
        self.assertEqual(StudentT(0.5).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            d = df.view(-1)[idx].detach()
            expected = scipy.stats.t.logpdf(x, d)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(StudentT(df), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_studentT_sample(self):
        set_rng_seed(11)  # see Note [Randomized statistical tests]
        for df, loc, scale in product([0.1, 1.0, 5.0, 10.0], [-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(StudentT(df=df, loc=loc, scale=scale),
                                        scipy.stats.t(df=df, loc=loc, scale=scale),
                                        f'StudentT(df={df}, loc={loc}, scale={scale})')

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_studentT_log_prob(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        num_samples = 10
        for df, loc, scale in product([0.1, 1.0, 5.0, 10.0], [-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            dist = StudentT(df=df, loc=loc, scale=scale)
            x = dist.sample((num_samples,))
            actual_log_prob = dist.log_prob(x)
            for i in range(num_samples):
                expected_log_prob = scipy.stats.t.logpdf(x[i], df=df, loc=loc, scale=scale)
                self.assertEqual(float(actual_log_prob[i]), float(expected_log_prob), atol=1e-3, rtol=0)

    def test_dirichlet_shape(self):
        alpha = torch.randn(2, 3).exp().requires_grad_()
        alpha_1d = torch.randn(4).exp().requires_grad_()
        self.assertEqual(Dirichlet(alpha).sample().size(), (2, 3))
        self.assertEqual(Dirichlet(alpha).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Dirichlet(alpha_1d).sample().size(), (4,))
        self.assertEqual(Dirichlet(alpha_1d).sample((1,)).size(), (1, 4))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_dirichlet_log_prob(self):
        num_samples = 10
        alpha = torch.exp(torch.randn(5))
        dist = Dirichlet(alpha)
        x = dist.sample((num_samples,))
        actual_log_prob = dist.log_prob(x)
        for i in range(num_samples):
            expected_log_prob = scipy.stats.dirichlet.logpdf(x[i].numpy(), alpha.numpy())
            self.assertEqual(actual_log_prob[i], expected_log_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_dirichlet_log_prob_zero(self):
        # Specifically test the special case where x=0 and α=1.  The PDF is
        # proportional to x**(α-1), which in this case works out to 0**0=1.
        # The log PDF of this term should therefore be 0.  However, it's easy
        # to accidentally introduce NaNs by calculating log(x) without regard
        # for the value of α-1.
        alpha = torch.tensor([1, 2])
        dist = Dirichlet(alpha)
        x = torch.tensor([0, 1])
        actual_log_prob = dist.log_prob(x)
        expected_log_prob = scipy.stats.dirichlet.logpdf(x.numpy(), alpha.numpy())
        self.assertEqual(actual_log_prob, expected_log_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_dirichlet_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        alpha = torch.exp(torch.randn(3))
        self._check_sampler_sampler(Dirichlet(alpha),
                                    scipy.stats.dirichlet(alpha.numpy()),
                                    f'Dirichlet(alpha={list(alpha)})',
                                    multivariate=True)

    def test_dirichlet_mode(self):
        # Test a few edge cases for the Dirichlet distribution mode. This also covers beta distributions.
        concentrations_and_modes = [
            ([2, 2, 1], [.5, .5, 0.]),
            ([3, 2, 1], [2 / 3, 1 / 3, 0]),
            ([.5, .2, .2], [1., 0., 0.]),
            ([1, 1, 1], [nan, nan, nan]),
        ]
        for concentration, mode in concentrations_and_modes:
            dist = Dirichlet(torch.tensor(concentration))
            self.assertEqual(dist.mode, torch.tensor(mode))

    def test_beta_shape(self):
        con1 = torch.randn(2, 3).exp().requires_grad_()
        con0 = torch.randn(2, 3).exp().requires_grad_()
        con1_1d = torch.randn(4).exp().requires_grad_()
        con0_1d = torch.randn(4).exp().requires_grad_()
        self.assertEqual(Beta(con1, con0).sample().size(), (2, 3))
        self.assertEqual(Beta(con1, con0).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Beta(con1_1d, con0_1d).sample().size(), (4,))
        self.assertEqual(Beta(con1_1d, con0_1d).sample((1,)).size(), (1, 4))
        self.assertEqual(Beta(0.1, 0.3).sample().size(), ())
        self.assertEqual(Beta(0.1, 0.3).sample((5,)).size(), (5,))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_beta_log_prob(self):
        for _ in range(100):
            con1 = np.exp(np.random.normal())
            con0 = np.exp(np.random.normal())
            dist = Beta(con1, con0)
            x = dist.sample()
            actual_log_prob = dist.log_prob(x).sum()
            expected_log_prob = scipy.stats.beta.logpdf(x, con1, con0)
            self.assertEqual(float(actual_log_prob), float(expected_log_prob), atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_beta_sample(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for con1, con0 in product([0.1, 1.0, 10.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Beta(con1, con0),
                                        scipy.stats.beta(con1, con0),
                                        f'Beta(alpha={con1}, beta={con0})')
        # Check that small alphas do not cause NANs.
        for Tensor in [torch.FloatTensor, torch.DoubleTensor]:
            x = Beta(Tensor([1e-6]), Tensor([1e-6])).sample()[0]
            self.assertTrue(np.isfinite(x) and x > 0, f'Invalid Beta.sample(): {x}')

    def test_beta_underflow(self):
        # For low values of (alpha, beta), the gamma samples can underflow
        # with float32 and result in a spurious mode at 0.5. To prevent this,
        # torch._sample_dirichlet works with double precision for intermediate
        # calculations.
        set_rng_seed(1)
        num_samples = 50000
        for dtype in [torch.float, torch.double]:
            conc = torch.tensor(1e-2, dtype=dtype)
            beta_samples = Beta(conc, conc).sample([num_samples])
            self.assertEqual((beta_samples == 0).sum(), 0)
            self.assertEqual((beta_samples == 1).sum(), 0)
            # assert support is concentrated around 0 and 1
            frac_zeros = float((beta_samples < 0.1).sum()) / num_samples
            frac_ones = float((beta_samples > 0.9).sum()) / num_samples
            self.assertEqual(frac_zeros, 0.5, atol=0.05, rtol=0)
            self.assertEqual(frac_ones, 0.5, atol=0.05, rtol=0)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_beta_underflow_gpu(self):
        set_rng_seed(1)
        num_samples = 50000
        conc = torch.tensor(1e-2, dtype=torch.float64).cuda()
        beta_samples = Beta(conc, conc).sample([num_samples])
        self.assertEqual((beta_samples == 0).sum(), 0)
        self.assertEqual((beta_samples == 1).sum(), 0)
        # assert support is concentrated around 0 and 1
        frac_zeros = float((beta_samples < 0.1).sum()) / num_samples
        frac_ones = float((beta_samples > 0.9).sum()) / num_samples
        # TODO: increase precision once imbalance on GPU is fixed.
        self.assertEqual(frac_zeros, 0.5, atol=0.12, rtol=0)
        self.assertEqual(frac_ones, 0.5, atol=0.12, rtol=0)

    def test_continuous_bernoulli(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        r = torch.tensor(0.3, requires_grad=True)
        s = 0.3
        self.assertEqual(ContinuousBernoulli(p).sample((8,)).size(), (8, 3))
        self.assertFalse(ContinuousBernoulli(p).sample().requires_grad)
        self.assertEqual(ContinuousBernoulli(r).sample((8,)).size(), (8,))
        self.assertEqual(ContinuousBernoulli(r).sample().size(), ())
        self.assertEqual(ContinuousBernoulli(r).sample((3, 2)).size(), (3, 2,))
        self.assertEqual(ContinuousBernoulli(s).sample().size(), ())
        self._gradcheck_log_prob(ContinuousBernoulli, (p,))

        def ref_log_prob(idx, val, log_prob):
            prob = p[idx]
            if prob > 0.499 and prob < 0.501:  # using default value of lim here
                log_norm_const = math.log(2.) + 4. / 3. * math.pow(prob - 0.5, 2) + 104. / 45. * math.pow(prob - 0.5, 4)
            else:
                log_norm_const = math.log(2. * math.atanh(1. - 2. * prob) / (1. - 2.0 * prob))
            res = val * math.log(prob) + (1. - val) * math.log1p(-prob) + log_norm_const
            self.assertEqual(log_prob, res)

        self._check_log_prob(ContinuousBernoulli(p), ref_log_prob)
        self._check_log_prob(ContinuousBernoulli(logits=p.log() - (-p).log1p()), ref_log_prob)

        # check entropy computation
        self.assertEqual(ContinuousBernoulli(p).entropy(), torch.tensor([-0.02938, -0.07641, -0.00682]), atol=1e-4, rtol=0)
        # entropy below corresponds to the clamped value of prob when using float 64
        # the value for float32 should be -1.76898
        self.assertEqual(ContinuousBernoulli(torch.tensor([0.0])).entropy(), torch.tensor([-2.58473]), atol=1e-5, rtol=0)
        self.assertEqual(ContinuousBernoulli(s).entropy(), torch.tensor(-0.02938), atol=1e-4, rtol=0)

    def test_continuous_bernoulli_3d(self):
        p = torch.full((2, 3, 5), 0.5).requires_grad_()
        self.assertEqual(ContinuousBernoulli(p).sample().size(), (2, 3, 5))
        self.assertEqual(ContinuousBernoulli(p).sample(sample_shape=(2, 5)).size(),
                         (2, 5, 2, 3, 5))
        self.assertEqual(ContinuousBernoulli(p).sample((2,)).size(), (2, 2, 3, 5))

    def test_lkj_cholesky_log_prob(self):
        def tril_cholesky_to_tril_corr(x):
            x = vec_to_tril_matrix(x, -1)
            diag = (1 - (x * x).sum(-1)).sqrt().diag_embed()
            x = x + diag
            return tril_matrix_to_vec(x @ x.T, -1)

        for dim in range(2, 5):
            log_probs = []
            lkj = LKJCholesky(dim, concentration=1., validate_args=True)
            for i in range(2):
                sample = lkj.sample()
                sample_tril = tril_matrix_to_vec(sample, diag=-1)
                log_prob = lkj.log_prob(sample)
                log_abs_det_jacobian = torch.slogdet(jacobian(tril_cholesky_to_tril_corr, sample_tril)).logabsdet
                log_probs.append(log_prob - log_abs_det_jacobian)
            # for concentration=1., the density is uniform over the space of all
            # correlation matrices.
            if dim == 2:
                # for dim=2, pdf = 0.5 (jacobian adjustment factor is 0.)
                self.assertTrue(all(torch.allclose(x, torch.tensor(0.5).log(), atol=1e-10) for x in log_probs))
            self.assertEqual(log_probs[0], log_probs[1])
            invalid_sample = torch.cat([sample, sample.new_ones(1, dim)], dim=0)
            self.assertRaises(ValueError, lambda: lkj.log_prob(invalid_sample))

    def test_independent_shape(self):
        for Dist, params in EXAMPLES:
            for param in params:
                base_dist = Dist(**param)
                x = base_dist.sample()
                base_log_prob_shape = base_dist.log_prob(x).shape
                for reinterpreted_batch_ndims in range(len(base_dist.batch_shape) + 1):
                    indep_dist = Independent(base_dist, reinterpreted_batch_ndims)
                    indep_log_prob_shape = base_log_prob_shape[:len(base_log_prob_shape) - reinterpreted_batch_ndims]
                    self.assertEqual(indep_dist.log_prob(x).shape, indep_log_prob_shape)
                    self.assertEqual(indep_dist.sample().shape, base_dist.sample().shape)
                    self.assertEqual(indep_dist.has_rsample, base_dist.has_rsample)
                    if indep_dist.has_rsample:
                        self.assertEqual(indep_dist.sample().shape, base_dist.sample().shape)
                    try:
                        self.assertEqual(indep_dist.enumerate_support().shape, base_dist.enumerate_support().shape)
                        self.assertEqual(indep_dist.mean.shape, base_dist.mean.shape)
                    except NotImplementedError:
                        pass
                    try:
                        self.assertEqual(indep_dist.variance.shape, base_dist.variance.shape)
                    except NotImplementedError:
                        pass
                    try:
                        self.assertEqual(indep_dist.entropy().shape, indep_log_prob_shape)
                    except NotImplementedError:
                        pass

    def test_independent_expand(self):
        for Dist, params in EXAMPLES:
            for param in params:
                base_dist = Dist(**param)
                for reinterpreted_batch_ndims in range(len(base_dist.batch_shape) + 1):
                    for s in [torch.Size(), torch.Size((2,)), torch.Size((2, 3))]:
                        indep_dist = Independent(base_dist, reinterpreted_batch_ndims)
                        expanded_shape = s + indep_dist.batch_shape
                        expanded = indep_dist.expand(expanded_shape)
                        expanded_sample = expanded.sample()
                        expected_shape = expanded_shape + indep_dist.event_shape
                        self.assertEqual(expanded_sample.shape, expected_shape)
                        self.assertEqual(expanded.log_prob(expanded_sample),
                                         indep_dist.log_prob(expanded_sample))
                        self.assertEqual(expanded.event_shape, indep_dist.event_shape)
                        self.assertEqual(expanded.batch_shape, expanded_shape)

    def test_cdf_icdf_inverse(self):
        # Tests the invertibility property on the distributions
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                samples = dist.sample(sample_shape=(20,))
                try:
                    cdf = dist.cdf(samples)
                    actual = dist.icdf(cdf)
                except NotImplementedError:
                    continue
                rel_error = torch.abs(actual - samples) / (1e-10 + torch.abs(samples))
                self.assertLess(rel_error.max(), 1e-4, msg='\n'.join([
                    f'{Dist.__name__} example {i + 1}/{len(params)}, icdf(cdf(x)) != x',
                    f'x = {samples}',
                    f'cdf(x) = {cdf}',
                    f'icdf(cdf(x)) = {actual}',
                ]))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma_log_prob_at_boundary(self):
        for concentration, log_prob in [(.5, inf), (1, 0), (2, -inf)]:
            dist = Gamma(concentration, 1)
            scipy_dist = scipy.stats.gamma(concentration)
            self.assertAlmostEqual(dist.log_prob(0), log_prob)
            self.assertAlmostEqual(dist.log_prob(0), scipy_dist.logpdf(0))

    def test_cdf_log_prob(self):
        # Tests if the differentiation of the CDF gives the PDF at a given value
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                # We do not need grads wrt params here, e.g. shape of gamma distribution.
                param = {key: value.detach() if isinstance(value, torch.Tensor) else value
                         for key, value in param.items()}
                dist = Dist(**param)
                samples = dist.sample()
                if not dist.support.is_discrete:
                    samples.requires_grad_()
                try:
                    cdfs = dist.cdf(samples)
                    pdfs = dist.log_prob(samples).exp()
                except NotImplementedError:
                    continue
                cdfs_derivative = grad(cdfs.sum(), [samples])[0]  # this should not be wrapped in torch.abs()
                self.assertEqual(cdfs_derivative, pdfs, msg='\n'.join([
                    f'{Dist.__name__} example {i + 1}/{len(params)}, d(cdf)/dx != pdf(x)',
                    f'x = {samples}',
                    f'cdf = {cdfs}',
                    f'pdf = {pdfs}',
                    f'grad(cdf) = {cdfs_derivative}',
                ]))

    def test_valid_parameter_broadcasting(self):
        # Test correct broadcasting of parameter sizes for distributions that have multiple
        # parameters.
        # example type (distribution instance, expected sample shape)
        valid_examples = [
            (Normal(loc=torch.tensor([0., 0.]), scale=1),
             (2,)),
            (Normal(loc=0, scale=torch.tensor([1., 1.])),
             (2,)),
            (Normal(loc=torch.tensor([0., 0.]), scale=torch.tensor([1.])),
             (2,)),
            (Normal(loc=torch.tensor([0., 0.]), scale=torch.tensor([[1.], [1.]])),
             (2, 2)),
            (Normal(loc=torch.tensor([0., 0.]), scale=torch.tensor([[1.]])),
             (1, 2)),
            (Normal(loc=torch.tensor([0.]), scale=torch.tensor([[1.]])),
             (1, 1)),
            (FisherSnedecor(df1=torch.tensor([1., 1.]), df2=1),
             (2,)),
            (FisherSnedecor(df1=1, df2=torch.tensor([1., 1.])),
             (2,)),
            (FisherSnedecor(df1=torch.tensor([1., 1.]), df2=torch.tensor([1.])),
             (2,)),
            (FisherSnedecor(df1=torch.tensor([1., 1.]), df2=torch.tensor([[1.], [1.]])),
             (2, 2)),
            (FisherSnedecor(df1=torch.tensor([1., 1.]), df2=torch.tensor([[1.]])),
             (1, 2)),
            (FisherSnedecor(df1=torch.tensor([1.]), df2=torch.tensor([[1.]])),
             (1, 1)),
            (Gamma(concentration=torch.tensor([1., 1.]), rate=1),
             (2,)),
            (Gamma(concentration=1, rate=torch.tensor([1., 1.])),
             (2,)),
            (Gamma(concentration=torch.tensor([1., 1.]), rate=torch.tensor([[1.], [1.], [1.]])),
             (3, 2)),
            (Gamma(concentration=torch.tensor([1., 1.]), rate=torch.tensor([[1.], [1.]])),
             (2, 2)),
            (Gamma(concentration=torch.tensor([1., 1.]), rate=torch.tensor([[1.]])),
             (1, 2)),
            (Gamma(concentration=torch.tensor([1.]), rate=torch.tensor([[1.]])),
             (1, 1)),
            (Gumbel(loc=torch.tensor([0., 0.]), scale=1),
             (2,)),
            (Gumbel(loc=0, scale=torch.tensor([1., 1.])),
             (2,)),
            (Gumbel(loc=torch.tensor([0., 0.]), scale=torch.tensor([1.])),
             (2,)),
            (Gumbel(loc=torch.tensor([0., 0.]), scale=torch.tensor([[1.], [1.]])),
             (2, 2)),
            (Gumbel(loc=torch.tensor([0., 0.]), scale=torch.tensor([[1.]])),
             (1, 2)),
            (Gumbel(loc=torch.tensor([0.]), scale=torch.tensor([[1.]])),
             (1, 1)),
            (Kumaraswamy(concentration1=torch.tensor([1., 1.]), concentration0=1.),
             (2,)),
            (Kumaraswamy(concentration1=1, concentration0=torch.tensor([1., 1.])),
             (2, )),
            (Kumaraswamy(concentration1=torch.tensor([1., 1.]), concentration0=torch.tensor([1.])),
             (2,)),
            (Kumaraswamy(concentration1=torch.tensor([1., 1.]), concentration0=torch.tensor([[1.], [1.]])),
             (2, 2)),
            (Kumaraswamy(concentration1=torch.tensor([1., 1.]), concentration0=torch.tensor([[1.]])),
             (1, 2)),
            (Kumaraswamy(concentration1=torch.tensor([1.]), concentration0=torch.tensor([[1.]])),
             (1, 1)),
            (Laplace(loc=torch.tensor([0., 0.]), scale=1),
             (2,)),
            (Laplace(loc=0, scale=torch.tensor([1., 1.])),
             (2,)),
            (Laplace(loc=torch.tensor([0., 0.]), scale=torch.tensor([1.])),
             (2,)),
            (Laplace(loc=torch.tensor([0., 0.]), scale=torch.tensor([[1.], [1.]])),
             (2, 2)),
            (Laplace(loc=torch.tensor([0., 0.]), scale=torch.tensor([[1.]])),
             (1, 2)),
            (Laplace(loc=torch.tensor([0.]), scale=torch.tensor([[1.]])),
             (1, 1)),
            (Pareto(scale=torch.tensor([1., 1.]), alpha=1),
             (2,)),
            (Pareto(scale=1, alpha=torch.tensor([1., 1.])),
             (2,)),
            (Pareto(scale=torch.tensor([1., 1.]), alpha=torch.tensor([1.])),
             (2,)),
            (Pareto(scale=torch.tensor([1., 1.]), alpha=torch.tensor([[1.], [1.]])),
             (2, 2)),
            (Pareto(scale=torch.tensor([1., 1.]), alpha=torch.tensor([[1.]])),
             (1, 2)),
            (Pareto(scale=torch.tensor([1.]), alpha=torch.tensor([[1.]])),
             (1, 1)),
            (StudentT(df=torch.tensor([1., 1.]), loc=1),
             (2,)),
            (StudentT(df=1, scale=torch.tensor([1., 1.])),
             (2,)),
            (StudentT(df=torch.tensor([1., 1.]), loc=torch.tensor([1.])),
             (2,)),
            (StudentT(df=torch.tensor([1., 1.]), scale=torch.tensor([[1.], [1.]])),
             (2, 2)),
            (StudentT(df=torch.tensor([1., 1.]), loc=torch.tensor([[1.]])),
             (1, 2)),
            (StudentT(df=torch.tensor([1.]), scale=torch.tensor([[1.]])),
             (1, 1)),
            (StudentT(df=1., loc=torch.zeros(5, 1), scale=torch.ones(3)),
             (5, 3)),
        ]

        for dist, expected_size in valid_examples:
            actual_size = dist.sample().size()
            self.assertEqual(actual_size, expected_size,
                             msg=f'{dist} actual size: {actual_size} != expected size: {expected_size}')

            sample_shape = torch.Size((2,))
            expected_size = sample_shape + expected_size
            actual_size = dist.sample(sample_shape).size()
            self.assertEqual(actual_size, expected_size,
                             msg=f'{dist} actual size: {actual_size} != expected size: {expected_size}')

    def test_invalid_parameter_broadcasting(self):
        # invalid broadcasting cases; should throw error
        # example type (distribution class, distribution params)
        invalid_examples = [
            (Normal, {
                'loc': torch.tensor([[0, 0]]),
                'scale': torch.tensor([1, 1, 1, 1])
            }),
            (Normal, {
                'loc': torch.tensor([[[0, 0, 0], [0, 0, 0]]]),
                'scale': torch.tensor([1, 1])
            }),
            (FisherSnedecor, {
                'df1': torch.tensor([1, 1]),
                'df2': torch.tensor([1, 1, 1]),
            }),
            (Gumbel, {
                'loc': torch.tensor([[0, 0]]),
                'scale': torch.tensor([1, 1, 1, 1])
            }),
            (Gumbel, {
                'loc': torch.tensor([[[0, 0, 0], [0, 0, 0]]]),
                'scale': torch.tensor([1, 1])
            }),
            (Gamma, {
                'concentration': torch.tensor([0, 0]),
                'rate': torch.tensor([1, 1, 1])
            }),
            (Kumaraswamy, {
                'concentration1': torch.tensor([[1, 1]]),
                'concentration0': torch.tensor([1, 1, 1, 1])
            }),
            (Kumaraswamy, {
                'concentration1': torch.tensor([[[1, 1, 1], [1, 1, 1]]]),
                'concentration0': torch.tensor([1, 1])
            }),
            (Laplace, {
                'loc': torch.tensor([0, 0]),
                'scale': torch.tensor([1, 1, 1])
            }),
            (Pareto, {
                'scale': torch.tensor([1, 1]),
                'alpha': torch.tensor([1, 1, 1])
            }),
            (StudentT, {
                'df': torch.tensor([1., 1.]),
                'scale': torch.tensor([1., 1., 1.])
            }),
            (StudentT, {
                'df': torch.tensor([1., 1.]),
                'loc': torch.tensor([1., 1., 1.])
            })
        ]

        for dist, kwargs in invalid_examples:
            self.assertRaises(RuntimeError, dist, **kwargs)

    def _test_discrete_distribution_mode(self, dist, sanitized_mode, batch_isfinite):
        # We cannot easily check the mode for discrete distributions, but we can look left and right
        # to ensure the log probability is smaller than at the mode.
        for step in [-1, 1]:
            log_prob_mode = dist.log_prob(sanitized_mode)
            if isinstance(dist, OneHotCategorical):
                idx = (dist._categorical.mode + 1) % dist.probs.shape[-1]
                other = torch.nn.functional.one_hot(idx, num_classes=dist.probs.shape[-1]).to(dist.mode)
            else:
                other = dist.mode + step
            mask = batch_isfinite & dist.support.check(other)
            self.assertTrue(mask.any() or dist.mode.unique().numel() == 1)
            # Add a dimension to the right if the event shape is not a scalar, e.g. OneHotCategorical.
            other = torch.where(mask[..., None] if mask.ndim < other.ndim else mask, other, dist.sample())
            log_prob_other = dist.log_prob(other)
            delta = log_prob_mode - log_prob_other
            self.assertTrue((-1e-12 < delta[mask].detach()).all())  # Allow up to 1e-12 rounding error.

    def _test_continuous_distribution_mode(self, dist, sanitized_mode, batch_isfinite):
        # We perturb the mode in the unconstrained space and expect the log probability to decrease.
        num_points = 10
        transform = transform_to(dist.support)
        unconstrained_mode = transform.inv(sanitized_mode)
        perturbation = 1e-5 * (torch.rand((num_points,) + unconstrained_mode.shape) - 0.5)
        perturbed_mode = transform(perturbation + unconstrained_mode)
        log_prob_mode = dist.log_prob(sanitized_mode)
        log_prob_other = dist.log_prob(perturbed_mode)
        delta = log_prob_mode - log_prob_other

        # We pass the test with a small tolerance to allow for rounding and manually set the
        # difference to zero if both log probs are infinite with the same sign.
        both_infinite_with_same_sign = (log_prob_mode == log_prob_other) & (log_prob_mode.abs() == inf)
        delta[both_infinite_with_same_sign] = 0.
        ordering = (delta > -1e-12).all(axis=0)
        self.assertTrue(ordering[batch_isfinite].all())

    def test_mode(self):
        discrete_distributions = (
            Bernoulli, Binomial, Categorical, Geometric, NegativeBinomial, OneHotCategorical, Poisson,
        )
        no_mode_available = (
            ContinuousBernoulli, LKJCholesky, LogisticNormal, MixtureSameFamily, Multinomial,
            RelaxedBernoulli, RelaxedOneHotCategorical,
        )

        for dist_cls, params in EXAMPLES:
            for param in params:
                dist = dist_cls(**param)
                if isinstance(dist, no_mode_available) or type(dist) is TransformedDistribution:
                    with self.assertRaises(NotImplementedError):
                        dist.mode
                    continue

                # Check that either all or no elements in the event shape are nan: the mode cannot be
                # defined for part of an event.
                isfinite = dist.mode.isfinite().reshape(dist.batch_shape + (dist.event_shape.numel(),))
                batch_isfinite = isfinite.all(axis=-1)
                self.assertTrue((batch_isfinite | ~isfinite.any(axis=-1)).all())

                # We sanitize undefined modes by sampling from the distribution.
                sanitized_mode = torch.where(~dist.mode.isnan(), dist.mode, dist.sample())
                if isinstance(dist, discrete_distributions):
                    self._test_discrete_distribution_mode(dist, sanitized_mode, batch_isfinite)
                else:
                    self._test_continuous_distribution_mode(dist, sanitized_mode, batch_isfinite)

                self.assertFalse(dist.log_prob(sanitized_mode).isnan().any())


# These tests are only needed for a few distributions that implement custom
# reparameterized gradients. Most .rsample() implementations simply rely on
# the reparameterization trick and do not need to be tested for accuracy.
@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestRsample(DistributionsTestCase):
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma(self):
        num_samples = 100
        for alpha in [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            alphas = torch.tensor([alpha] * num_samples, dtype=torch.float, requires_grad=True)
            betas = alphas.new_ones(num_samples)
            x = Gamma(alphas, betas).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = alphas.grad[ind].numpy()
            # Compare with expected gradient dx/dalpha along constant cdf(x,alpha).
            cdf = scipy.stats.gamma.cdf
            pdf = scipy.stats.gamma.pdf
            eps = 0.01 * alpha / (1.0 + alpha ** 0.5)
            cdf_alpha = (cdf(x, alpha + eps) - cdf(x, alpha - eps)) / (2 * eps)
            cdf_x = pdf(x, alpha)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(np.max(rel_error), 0.0005, '\n'.join([
                f'Bad gradient dx/alpha for x ~ Gamma({alpha}, 1)',
                f'x {x}',
                f'expected {expected_grad}',
                f'actual {actual_grad}',
                f'rel error {rel_error}',
                f'max error {rel_error.max()}',
                f'at alpha={alpha}, x={x[rel_error.argmax()]}',
            ]))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_chi2(self):
        num_samples = 100
        for df in [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            dfs = torch.tensor([df] * num_samples, dtype=torch.float, requires_grad=True)
            x = Chi2(dfs).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = dfs.grad[ind].numpy()
            # Compare with expected gradient dx/ddf along constant cdf(x,df).
            cdf = scipy.stats.chi2.cdf
            pdf = scipy.stats.chi2.pdf
            eps = 0.01 * df / (1.0 + df ** 0.5)
            cdf_df = (cdf(x, df + eps) - cdf(x, df - eps)) / (2 * eps)
            cdf_x = pdf(x, df)
            expected_grad = -cdf_df / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(np.max(rel_error), 0.001, '\n'.join([
                f'Bad gradient dx/ddf for x ~ Chi2({df})',
                f'x {x}',
                f'expected {expected_grad}',
                f'actual {actual_grad}',
                f'rel error {rel_error}',
                f'max error {rel_error.max()}',
            ]))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_dirichlet_on_diagonal(self):
        num_samples = 20
        grid = [1e-1, 1e0, 1e1]
        for a0, a1, a2 in product(grid, grid, grid):
            alphas = torch.tensor([[a0, a1, a2]] * num_samples, dtype=torch.float, requires_grad=True)
            x = Dirichlet(alphas).rsample()[:, 0]
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = alphas.grad[ind].numpy()[:, 0]
            # Compare with expected gradient dx/dalpha0 along constant cdf(x,alpha).
            # This reduces to a distribution Beta(alpha[0], alpha[1] + alpha[2]).
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            alpha, beta = a0, a1 + a2
            eps = 0.01 * alpha / (1.0 + np.sqrt(alpha))
            cdf_alpha = (cdf(x, alpha + eps, beta) - cdf(x, alpha - eps, beta)) / (2 * eps)
            cdf_x = pdf(x, alpha, beta)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(np.max(rel_error), 0.001, '\n'.join([
                f'Bad gradient dx[0]/dalpha[0] for Dirichlet([{a0}, {a1}, {a2}])',
                f'x {x}',
                f'expected {expected_grad}',
                f'actual {actual_grad}',
                f'rel error {rel_error}',
                f'max error {rel_error.max()}',
                f'at x={x[rel_error.argmax()]}',
            ]))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_beta_wrt_alpha(self):
        num_samples = 20
        grid = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        for con1, con0 in product(grid, grid):
            con1s = torch.tensor([con1] * num_samples, dtype=torch.float, requires_grad=True)
            con0s = con1s.new_tensor([con0] * num_samples)
            x = Beta(con1s, con0s).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = con1s.grad[ind].numpy()
            # Compare with expected gradient dx/dcon1 along constant cdf(x,con1,con0).
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            eps = 0.01 * con1 / (1.0 + np.sqrt(con1))
            cdf_alpha = (cdf(x, con1 + eps, con0) - cdf(x, con1 - eps, con0)) / (2 * eps)
            cdf_x = pdf(x, con1, con0)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(np.max(rel_error), 0.005, '\n'.join([
                f'Bad gradient dx/dcon1 for x ~ Beta({con1}, {con0})',
                f'x {x}',
                f'expected {expected_grad}',
                f'actual {actual_grad}',
                f'rel error {rel_error}',
                f'max error {rel_error.max()}',
                f'at x = {x[rel_error.argmax()]}',
            ]))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_beta_wrt_beta(self):
        num_samples = 20
        grid = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        for con1, con0 in product(grid, grid):
            con0s = torch.tensor([con0] * num_samples, dtype=torch.float, requires_grad=True)
            con1s = con0s.new_tensor([con1] * num_samples)
            x = Beta(con1s, con0s).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = con0s.grad[ind].numpy()
            # Compare with expected gradient dx/dcon0 along constant cdf(x,con1,con0).
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            eps = 0.01 * con0 / (1.0 + np.sqrt(con0))
            cdf_beta = (cdf(x, con1, con0 + eps) - cdf(x, con1, con0 - eps)) / (2 * eps)
            cdf_x = pdf(x, con1, con0)
            expected_grad = -cdf_beta / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(np.max(rel_error), 0.005, '\n'.join([
                f'Bad gradient dx/dcon0 for x ~ Beta({con1}, {con0})',
                f'x {x}',
                f'expected {expected_grad}',
                f'actual {actual_grad}',
                f'rel error {rel_error}',
                f'max error {rel_error.max()}',
                f'at x = {x[rel_error.argmax()]!r}',
            ]))

    def test_dirichlet_multivariate(self):
        alpha_crit = 0.25 * (5.0 ** 0.5 - 1.0)
        num_samples = 100000
        for shift in [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10]:
            alpha = alpha_crit + shift
            alpha = torch.tensor([alpha], dtype=torch.float, requires_grad=True)
            alpha_vec = torch.cat([alpha, alpha, alpha.new([1])])
            z = Dirichlet(alpha_vec.expand(num_samples, 3)).rsample()
            mean_z3 = 1.0 / (2.0 * alpha + 1.0)
            loss = torch.pow(z[:, 2] - mean_z3, 2.0).mean()
            actual_grad = grad(loss, [alpha])[0]
            # Compute expected gradient by hand.
            num = 1.0 - 2.0 * alpha - 4.0 * alpha**2
            den = (1.0 + alpha)**2 * (1.0 + 2.0 * alpha)**3
            expected_grad = num / den
            self.assertEqual(actual_grad, expected_grad, atol=0.002, rtol=0, msg='\n'.join([
                "alpha = alpha_c + %.2g" % shift,
                "expected_grad: %.5g" % expected_grad,
                "actual_grad: %.5g" % actual_grad,
                "error = %.2g" % torch.abs(expected_grad - actual_grad).max(),
            ]))

    def test_dirichlet_tangent_field(self):
        num_samples = 20
        alpha_grid = [0.5, 1.0, 2.0]

        # v = dx/dalpha[0] is the reparameterized gradient aka tangent field.
        def compute_v(x, alpha):
            return torch.stack([
                _Dirichlet_backward(x, alpha, torch.eye(3, 3)[i].expand_as(x))[:, 0]
                for i in range(3)
            ], dim=-1)

        for a1, a2, a3 in product(alpha_grid, alpha_grid, alpha_grid):
            alpha = torch.tensor([a1, a2, a3], requires_grad=True).expand(num_samples, 3)
            x = Dirichlet(alpha).rsample()
            dlogp_da = grad([Dirichlet(alpha).log_prob(x.detach()).sum()],
                            [alpha], retain_graph=True)[0][:, 0]
            dlogp_dx = grad([Dirichlet(alpha.detach()).log_prob(x).sum()],
                            [x], retain_graph=True)[0]
            v = torch.stack([grad([x[:, i].sum()], [alpha], retain_graph=True)[0][:, 0]
                             for i in range(3)], dim=-1)
            # Compute ramaining properties by finite difference.
            self.assertEqual(compute_v(x, alpha), v, msg='Bug in compute_v() helper')
            # dx is an arbitrary orthonormal basis tangent to the simplex.
            dx = torch.tensor([[2., -1., -1.], [0., 1., -1.]])
            dx /= dx.norm(2, -1, True)
            eps = 1e-2 * x.min(-1, True)[0]  # avoid boundary
            dv0 = (compute_v(x + eps * dx[0], alpha) - compute_v(x - eps * dx[0], alpha)) / (2 * eps)
            dv1 = (compute_v(x + eps * dx[1], alpha) - compute_v(x - eps * dx[1], alpha)) / (2 * eps)
            div_v = (dv0 * dx[0] + dv1 * dx[1]).sum(-1)
            # This is a modification of the standard continuity equation, using the product rule to allow
            # expression in terms of log_prob rather than the less numerically stable log_prob.exp().
            error = dlogp_da + (dlogp_dx * v).sum(-1) + div_v
            self.assertLess(torch.abs(error).max(), 0.005, '\n'.join([
                f'Dirichlet([{a1}, {a2}, {a3}]) gradient violates continuity equation:',
                f'error = {error}',
            ]))


class TestDistributionShapes(DistributionsTestCase):
    def setUp(self):
        super().setUp()
        self.scalar_sample = 1
        self.tensor_sample_1 = torch.ones(3, 2)
        self.tensor_sample_2 = torch.ones(3, 2, 3)

    def test_entropy_shape(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(validate_args=False, **param)
                try:
                    actual_shape = dist.entropy().size()
                    expected_shape = dist.batch_shape if dist.batch_shape else torch.Size()
                    message = '{} example {}/{}, shape mismatch. expected {}, actual {}'.format(
                        Dist.__name__, i + 1, len(params), expected_shape, actual_shape)
                    self.assertEqual(actual_shape, expected_shape, msg=message)
                except NotImplementedError:
                    continue

    def test_bernoulli_shape_scalar_params(self):
        bernoulli = Bernoulli(0.3)
        self.assertEqual(bernoulli._batch_shape, torch.Size())
        self.assertEqual(bernoulli._event_shape, torch.Size())
        self.assertEqual(bernoulli.sample().size(), torch.Size())
        self.assertEqual(bernoulli.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, bernoulli.log_prob, self.scalar_sample)
        self.assertEqual(bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(bernoulli.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_bernoulli_shape_tensor_params(self):
        bernoulli = Bernoulli(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(bernoulli._batch_shape, torch.Size((3, 2)))
        self.assertEqual(bernoulli._event_shape, torch.Size(()))
        self.assertEqual(bernoulli.sample().size(), torch.Size((3, 2)))
        self.assertEqual(bernoulli.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, bernoulli.log_prob, self.tensor_sample_2)
        self.assertEqual(bernoulli.log_prob(torch.ones(3, 1, 1)).size(), torch.Size((3, 3, 2)))

    def test_geometric_shape_scalar_params(self):
        geometric = Geometric(0.3)
        self.assertEqual(geometric._batch_shape, torch.Size())
        self.assertEqual(geometric._event_shape, torch.Size())
        self.assertEqual(geometric.sample().size(), torch.Size())
        self.assertEqual(geometric.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, geometric.log_prob, self.scalar_sample)
        self.assertEqual(geometric.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(geometric.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_geometric_shape_tensor_params(self):
        geometric = Geometric(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(geometric._batch_shape, torch.Size((3, 2)))
        self.assertEqual(geometric._event_shape, torch.Size(()))
        self.assertEqual(geometric.sample().size(), torch.Size((3, 2)))
        self.assertEqual(geometric.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(geometric.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, geometric.log_prob, self.tensor_sample_2)
        self.assertEqual(geometric.log_prob(torch.ones(3, 1, 1)).size(), torch.Size((3, 3, 2)))

    def test_beta_shape_scalar_params(self):
        dist = Beta(0.1, 0.1)
        self.assertEqual(dist._batch_shape, torch.Size())
        self.assertEqual(dist._event_shape, torch.Size())
        self.assertEqual(dist.sample().size(), torch.Size())
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, dist.log_prob, self.scalar_sample)
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_beta_shape_tensor_params(self):
        dist = Beta(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                    torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
        self.assertEqual(dist._batch_shape, torch.Size((3, 2)))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        self.assertEqual(dist.log_prob(torch.ones(3, 1, 1)).size(), torch.Size((3, 3, 2)))

    def test_binomial_shape(self):
        dist = Binomial(10, torch.tensor([0.6, 0.3]))
        self.assertEqual(dist._batch_shape, torch.Size((2,)))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size((2,)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)

    def test_binomial_shape_vectorized_n(self):
        dist = Binomial(torch.tensor([[10, 3, 1], [4, 8, 4]]), torch.tensor([0.6, 0.3, 0.1]))
        self.assertEqual(dist._batch_shape, torch.Size((2, 3)))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size((2, 3)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 2, 3)))
        self.assertEqual(dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)

    def test_multinomial_shape(self):
        dist = Multinomial(10, torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        self.assertEqual(dist.log_prob(torch.ones(3, 1, 2)).size(), torch.Size((3, 3)))

    def test_categorical_shape(self):
        # unbatched
        dist = Categorical(torch.tensor([0.6, 0.3, 0.1]))
        self.assertEqual(dist._batch_shape, torch.Size(()))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size())
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2,)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))
        self.assertEqual(dist.log_prob(torch.ones(3, 1)).size(), torch.Size((3, 1)))
        # batched
        dist = Categorical(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size((3,)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)
        self.assertEqual(dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))
        self.assertEqual(dist.log_prob(torch.ones(3, 1)).size(), torch.Size((3, 3)))

    def test_one_hot_categorical_shape(self):
        # unbatched
        dist = OneHotCategorical(torch.tensor([0.6, 0.3, 0.1]))
        self.assertEqual(dist._batch_shape, torch.Size(()))
        self.assertEqual(dist._event_shape, torch.Size((3,)))
        self.assertEqual(dist.sample().size(), torch.Size((3,)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)
        sample = torch.tensor([0., 1., 0.]).expand(3, 2, 3)
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3, 2,)))
        self.assertEqual(dist.log_prob(dist.enumerate_support()).size(), torch.Size((3,)))
        sample = torch.eye(3)
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3,)))
        # batched
        dist = OneHotCategorical(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        sample = torch.tensor([0., 1.])
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        self.assertEqual(dist.log_prob(dist.enumerate_support()).size(), torch.Size((2, 3)))
        sample = torch.tensor([0., 1.]).expand(3, 1, 2)
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3, 3)))

    def test_cauchy_shape_scalar_params(self):
        cauchy = Cauchy(0, 1)
        self.assertEqual(cauchy._batch_shape, torch.Size())
        self.assertEqual(cauchy._event_shape, torch.Size())
        self.assertEqual(cauchy.sample().size(), torch.Size())
        self.assertEqual(cauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, cauchy.log_prob, self.scalar_sample)
        self.assertEqual(cauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(cauchy.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_cauchy_shape_tensor_params(self):
        cauchy = Cauchy(torch.tensor([0., 0.]), torch.tensor([1., 1.]))
        self.assertEqual(cauchy._batch_shape, torch.Size((2,)))
        self.assertEqual(cauchy._event_shape, torch.Size(()))
        self.assertEqual(cauchy.sample().size(), torch.Size((2,)))
        self.assertEqual(cauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2)))
        self.assertEqual(cauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, cauchy.log_prob, self.tensor_sample_2)
        self.assertEqual(cauchy.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_halfcauchy_shape_scalar_params(self):
        halfcauchy = HalfCauchy(1)
        self.assertEqual(halfcauchy._batch_shape, torch.Size())
        self.assertEqual(halfcauchy._event_shape, torch.Size())
        self.assertEqual(halfcauchy.sample().size(), torch.Size())
        self.assertEqual(halfcauchy.sample(torch.Size((3, 2))).size(),
                         torch.Size((3, 2)))
        self.assertRaises(ValueError, halfcauchy.log_prob, self.scalar_sample)
        self.assertEqual(halfcauchy.log_prob(self.tensor_sample_1).size(),
                         torch.Size((3, 2)))
        self.assertEqual(halfcauchy.log_prob(self.tensor_sample_2).size(),
                         torch.Size((3, 2, 3)))

    def test_halfcauchy_shape_tensor_params(self):
        halfcauchy = HalfCauchy(torch.tensor([1., 1.]))
        self.assertEqual(halfcauchy._batch_shape, torch.Size((2,)))
        self.assertEqual(halfcauchy._event_shape, torch.Size(()))
        self.assertEqual(halfcauchy.sample().size(), torch.Size((2,)))
        self.assertEqual(halfcauchy.sample(torch.Size((3, 2))).size(),
                         torch.Size((3, 2, 2)))
        self.assertEqual(halfcauchy.log_prob(self.tensor_sample_1).size(),
                         torch.Size((3, 2)))
        self.assertRaises(ValueError, halfcauchy.log_prob, self.tensor_sample_2)
        self.assertEqual(halfcauchy.log_prob(torch.ones(2, 1)).size(),
                         torch.Size((2, 2)))

    def test_dirichlet_shape(self):
        dist = Dirichlet(torch.tensor([[0.6, 0.3], [1.6, 1.3], [2.6, 2.3]]))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((5, 4)).size(), torch.Size((5, 4, 3, 2)))
        simplex_sample = self.tensor_sample_1 / self.tensor_sample_1.sum(-1, keepdim=True)
        self.assertEqual(dist.log_prob(simplex_sample).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        simplex_sample = torch.ones(3, 1, 2)
        simplex_sample = simplex_sample / simplex_sample.sum(-1).unsqueeze(-1)
        self.assertEqual(dist.log_prob(simplex_sample).size(), torch.Size((3, 3)))

    def test_mixture_same_family_shape(self):
        dist = MixtureSameFamily(Categorical(torch.rand(5)),
                                 Normal(torch.randn(5), torch.rand(5)))
        self.assertEqual(dist._batch_shape, torch.Size())
        self.assertEqual(dist._event_shape, torch.Size())
        self.assertEqual(dist.sample().size(), torch.Size())
        self.assertEqual(dist.sample((5, 4)).size(), torch.Size((5, 4)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_gamma_shape_scalar_params(self):
        gamma = Gamma(1, 1)
        self.assertEqual(gamma._batch_shape, torch.Size())
        self.assertEqual(gamma._event_shape, torch.Size())
        self.assertEqual(gamma.sample().size(), torch.Size())
        self.assertEqual(gamma.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(gamma.log_prob(self.scalar_sample).size(), torch.Size())
        self.assertEqual(gamma.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(gamma.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_gamma_shape_tensor_params(self):
        gamma = Gamma(torch.tensor([1., 1.]), torch.tensor([1., 1.]))
        self.assertEqual(gamma._batch_shape, torch.Size((2,)))
        self.assertEqual(gamma._event_shape, torch.Size(()))
        self.assertEqual(gamma.sample().size(), torch.Size((2,)))
        self.assertEqual(gamma.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(gamma.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, gamma.log_prob, self.tensor_sample_2)
        self.assertEqual(gamma.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_chi2_shape_scalar_params(self):
        chi2 = Chi2(1)
        self.assertEqual(chi2._batch_shape, torch.Size())
        self.assertEqual(chi2._event_shape, torch.Size())
        self.assertEqual(chi2.sample().size(), torch.Size())
        self.assertEqual(chi2.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(chi2.log_prob(self.scalar_sample).size(), torch.Size())
        self.assertEqual(chi2.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(chi2.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_chi2_shape_tensor_params(self):
        chi2 = Chi2(torch.tensor([1., 1.]))
        self.assertEqual(chi2._batch_shape, torch.Size((2,)))
        self.assertEqual(chi2._event_shape, torch.Size(()))
        self.assertEqual(chi2.sample().size(), torch.Size((2,)))
        self.assertEqual(chi2.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(chi2.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, chi2.log_prob, self.tensor_sample_2)
        self.assertEqual(chi2.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_studentT_shape_scalar_params(self):
        st = StudentT(1)
        self.assertEqual(st._batch_shape, torch.Size())
        self.assertEqual(st._event_shape, torch.Size())
        self.assertEqual(st.sample().size(), torch.Size())
        self.assertEqual(st.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, st.log_prob, self.scalar_sample)
        self.assertEqual(st.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(st.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_studentT_shape_tensor_params(self):
        st = StudentT(torch.tensor([1., 1.]))
        self.assertEqual(st._batch_shape, torch.Size((2,)))
        self.assertEqual(st._event_shape, torch.Size(()))
        self.assertEqual(st.sample().size(), torch.Size((2,)))
        self.assertEqual(st.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(st.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, st.log_prob, self.tensor_sample_2)
        self.assertEqual(st.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_pareto_shape_scalar_params(self):
        pareto = Pareto(1, 1)
        self.assertEqual(pareto._batch_shape, torch.Size())
        self.assertEqual(pareto._event_shape, torch.Size())
        self.assertEqual(pareto.sample().size(), torch.Size())
        self.assertEqual(pareto.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(pareto.log_prob(self.tensor_sample_1 + 1).size(), torch.Size((3, 2)))
        self.assertEqual(pareto.log_prob(self.tensor_sample_2 + 1).size(), torch.Size((3, 2, 3)))

    def test_gumbel_shape_scalar_params(self):
        gumbel = Gumbel(1, 1)
        self.assertEqual(gumbel._batch_shape, torch.Size())
        self.assertEqual(gumbel._event_shape, torch.Size())
        self.assertEqual(gumbel.sample().size(), torch.Size())
        self.assertEqual(gumbel.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(gumbel.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(gumbel.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_kumaraswamy_shape_scalar_params(self):
        kumaraswamy = Kumaraswamy(1, 1)
        self.assertEqual(kumaraswamy._batch_shape, torch.Size())
        self.assertEqual(kumaraswamy._event_shape, torch.Size())
        self.assertEqual(kumaraswamy.sample().size(), torch.Size())
        self.assertEqual(kumaraswamy.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(kumaraswamy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(kumaraswamy.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_vonmises_shape_tensor_params(self):
        von_mises = VonMises(torch.tensor([0., 0.]), torch.tensor([1., 1.]))
        self.assertEqual(von_mises._batch_shape, torch.Size((2,)))
        self.assertEqual(von_mises._event_shape, torch.Size(()))
        self.assertEqual(von_mises.sample().size(), torch.Size((2,)))
        self.assertEqual(von_mises.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2)))
        self.assertEqual(von_mises.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(von_mises.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_vonmises_shape_scalar_params(self):
        von_mises = VonMises(0., 1.)
        self.assertEqual(von_mises._batch_shape, torch.Size())
        self.assertEqual(von_mises._event_shape, torch.Size())
        self.assertEqual(von_mises.sample().size(), torch.Size())
        self.assertEqual(von_mises.sample(torch.Size((3, 2))).size(),
                         torch.Size((3, 2)))
        self.assertEqual(von_mises.log_prob(self.tensor_sample_1).size(),
                         torch.Size((3, 2)))
        self.assertEqual(von_mises.log_prob(self.tensor_sample_2).size(),
                         torch.Size((3, 2, 3)))

    def test_weibull_scale_scalar_params(self):
        weibull = Weibull(1, 1)
        self.assertEqual(weibull._batch_shape, torch.Size())
        self.assertEqual(weibull._event_shape, torch.Size())
        self.assertEqual(weibull.sample().size(), torch.Size())
        self.assertEqual(weibull.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(weibull.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(weibull.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_wishart_shape_scalar_params(self):
        wishart = Wishart(torch.tensor(1), torch.tensor([[1.]]))
        self.assertEqual(wishart._batch_shape, torch.Size())
        self.assertEqual(wishart._event_shape, torch.Size((1, 1)))
        self.assertEqual(wishart.sample().size(), torch.Size((1, 1)))
        self.assertEqual(wishart.sample((3, 2)).size(), torch.Size((3, 2, 1, 1)))
        self.assertRaises(ValueError, wishart.log_prob, self.scalar_sample)

    def test_wishart_shape_tensor_params(self):
        wishart = Wishart(torch.tensor([1., 1.]), torch.tensor([[[1.]], [[1.]]]))
        self.assertEqual(wishart._batch_shape, torch.Size((2,)))
        self.assertEqual(wishart._event_shape, torch.Size((1, 1)))
        self.assertEqual(wishart.sample().size(), torch.Size((2, 1, 1)))
        self.assertEqual(wishart.sample((3, 2)).size(), torch.Size((3, 2, 2, 1, 1)))
        self.assertRaises(ValueError, wishart.log_prob, self.tensor_sample_2)
        self.assertEqual(wishart.log_prob(torch.ones(2, 1, 1)).size(), torch.Size((2,)))

    def test_normal_shape_scalar_params(self):
        normal = Normal(0, 1)
        self.assertEqual(normal._batch_shape, torch.Size())
        self.assertEqual(normal._event_shape, torch.Size())
        self.assertEqual(normal.sample().size(), torch.Size())
        self.assertEqual(normal.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, normal.log_prob, self.scalar_sample)
        self.assertEqual(normal.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(normal.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_normal_shape_tensor_params(self):
        normal = Normal(torch.tensor([0., 0.]), torch.tensor([1., 1.]))
        self.assertEqual(normal._batch_shape, torch.Size((2,)))
        self.assertEqual(normal._event_shape, torch.Size(()))
        self.assertEqual(normal.sample().size(), torch.Size((2,)))
        self.assertEqual(normal.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(normal.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, normal.log_prob, self.tensor_sample_2)
        self.assertEqual(normal.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_uniform_shape_scalar_params(self):
        uniform = Uniform(0, 1)
        self.assertEqual(uniform._batch_shape, torch.Size())
        self.assertEqual(uniform._event_shape, torch.Size())
        self.assertEqual(uniform.sample().size(), torch.Size())
        self.assertEqual(uniform.sample(torch.Size((3, 2))).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, uniform.log_prob, self.scalar_sample)
        self.assertEqual(uniform.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(uniform.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_uniform_shape_tensor_params(self):
        uniform = Uniform(torch.tensor([0., 0.]), torch.tensor([1., 1.]))
        self.assertEqual(uniform._batch_shape, torch.Size((2,)))
        self.assertEqual(uniform._event_shape, torch.Size(()))
        self.assertEqual(uniform.sample().size(), torch.Size((2,)))
        self.assertEqual(uniform.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2)))
        self.assertEqual(uniform.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, uniform.log_prob, self.tensor_sample_2)
        self.assertEqual(uniform.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_exponential_shape_scalar_param(self):
        expon = Exponential(1.)
        self.assertEqual(expon._batch_shape, torch.Size())
        self.assertEqual(expon._event_shape, torch.Size())
        self.assertEqual(expon.sample().size(), torch.Size())
        self.assertEqual(expon.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, expon.log_prob, self.scalar_sample)
        self.assertEqual(expon.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(expon.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_exponential_shape_tensor_param(self):
        expon = Exponential(torch.tensor([1., 1.]))
        self.assertEqual(expon._batch_shape, torch.Size((2,)))
        self.assertEqual(expon._event_shape, torch.Size(()))
        self.assertEqual(expon.sample().size(), torch.Size((2,)))
        self.assertEqual(expon.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(expon.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, expon.log_prob, self.tensor_sample_2)
        self.assertEqual(expon.log_prob(torch.ones(2, 2)).size(), torch.Size((2, 2)))

    def test_laplace_shape_scalar_params(self):
        laplace = Laplace(0, 1)
        self.assertEqual(laplace._batch_shape, torch.Size())
        self.assertEqual(laplace._event_shape, torch.Size())
        self.assertEqual(laplace.sample().size(), torch.Size())
        self.assertEqual(laplace.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, laplace.log_prob, self.scalar_sample)
        self.assertEqual(laplace.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(laplace.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_laplace_shape_tensor_params(self):
        laplace = Laplace(torch.tensor([0., 0.]), torch.tensor([1., 1.]))
        self.assertEqual(laplace._batch_shape, torch.Size((2,)))
        self.assertEqual(laplace._event_shape, torch.Size(()))
        self.assertEqual(laplace.sample().size(), torch.Size((2,)))
        self.assertEqual(laplace.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(laplace.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, laplace.log_prob, self.tensor_sample_2)
        self.assertEqual(laplace.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_continuous_bernoulli_shape_scalar_params(self):
        continuous_bernoulli = ContinuousBernoulli(0.3)
        self.assertEqual(continuous_bernoulli._batch_shape, torch.Size())
        self.assertEqual(continuous_bernoulli._event_shape, torch.Size())
        self.assertEqual(continuous_bernoulli.sample().size(), torch.Size())
        self.assertEqual(continuous_bernoulli.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, continuous_bernoulli.log_prob, self.scalar_sample)
        self.assertEqual(continuous_bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(continuous_bernoulli.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_continuous_bernoulli_shape_tensor_params(self):
        continuous_bernoulli = ContinuousBernoulli(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(continuous_bernoulli._batch_shape, torch.Size((3, 2)))
        self.assertEqual(continuous_bernoulli._event_shape, torch.Size(()))
        self.assertEqual(continuous_bernoulli.sample().size(), torch.Size((3, 2)))
        self.assertEqual(continuous_bernoulli.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(continuous_bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, continuous_bernoulli.log_prob, self.tensor_sample_2)
        self.assertEqual(continuous_bernoulli.log_prob(torch.ones(3, 1, 1)).size(), torch.Size((3, 3, 2)))


@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestKL(DistributionsTestCase):

    def setUp(self):
        super().setUp()

        class Binomial30(Binomial):
            def __init__(self, probs):
                super().__init__(30, probs)

        # These are pairs of distributions with 4 x 4 parameters as specified.
        # The first of the pair e.g. bernoulli[0] varies column-wise and the second
        # e.g. bernoulli[1] varies row-wise; that way we test all param pairs.
        bernoulli = pairwise(Bernoulli, [0.1, 0.2, 0.6, 0.9])
        binomial30 = pairwise(Binomial30, [0.1, 0.2, 0.6, 0.9])
        binomial_vectorized_count = (Binomial(torch.tensor([3, 4]), torch.tensor([0.4, 0.6])),
                                     Binomial(torch.tensor([3, 4]), torch.tensor([0.5, 0.8])))
        beta = pairwise(Beta, [1.0, 2.5, 1.0, 2.5], [1.5, 1.5, 3.5, 3.5])
        categorical = pairwise(Categorical, [[0.4, 0.3, 0.3],
                                             [0.2, 0.7, 0.1],
                                             [0.33, 0.33, 0.34],
                                             [0.2, 0.2, 0.6]])
        cauchy = pairwise(Cauchy, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
        chi2 = pairwise(Chi2, [1.0, 2.0, 2.5, 5.0])
        dirichlet = pairwise(Dirichlet, [[0.1, 0.2, 0.7],
                                         [0.5, 0.4, 0.1],
                                         [0.33, 0.33, 0.34],
                                         [0.2, 0.2, 0.4]])
        exponential = pairwise(Exponential, [1.0, 2.5, 5.0, 10.0])
        gamma = pairwise(Gamma, [1.0, 2.5, 1.0, 2.5], [1.5, 1.5, 3.5, 3.5])
        gumbel = pairwise(Gumbel, [-2.0, 4.0, -3.0, 6.0], [1.0, 2.5, 1.0, 2.5])
        halfnormal = pairwise(HalfNormal, [1.0, 2.0, 1.0, 2.0])
        laplace = pairwise(Laplace, [-2.0, 4.0, -3.0, 6.0], [1.0, 2.5, 1.0, 2.5])
        lognormal = pairwise(LogNormal, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
        normal = pairwise(Normal, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
        independent = (Independent(normal[0], 1), Independent(normal[1], 1))
        onehotcategorical = pairwise(OneHotCategorical, [[0.4, 0.3, 0.3],
                                                         [0.2, 0.7, 0.1],
                                                         [0.33, 0.33, 0.34],
                                                         [0.2, 0.2, 0.6]])
        pareto = (Pareto(torch.tensor([2.5, 4.0, 2.5, 4.0]).expand(4, 4),
                         torch.tensor([2.25, 3.75, 2.25, 3.75]).expand(4, 4)),
                  Pareto(torch.tensor([2.25, 3.75, 2.25, 3.8]).expand(4, 4),
                         torch.tensor([2.25, 3.75, 2.25, 3.75]).expand(4, 4)))
        poisson = pairwise(Poisson, [0.3, 1.0, 5.0, 10.0])
        uniform_within_unit = pairwise(Uniform, [0.1, 0.9, 0.2, 0.75], [0.15, 0.95, 0.25, 0.8])
        uniform_positive = pairwise(Uniform, [1, 1.5, 2, 4], [1.2, 2.0, 3, 7])
        uniform_real = pairwise(Uniform, [-2., -1, 0, 2], [-1., 1, 1, 4])
        uniform_pareto = pairwise(Uniform, [6.5, 7.5, 6.5, 8.5], [7.5, 8.5, 9.5, 9.5])
        continuous_bernoulli = pairwise(ContinuousBernoulli, [0.1, 0.2, 0.5, 0.9])

        # These tests should pass with precision = 0.01, but that makes tests very expensive.
        # Instead, we test with precision = 0.1 and only test with higher precision locally
        # when adding a new KL implementation.
        # The following pairs are not tested due to very high variance of the monte carlo
        # estimator; their implementations have been reviewed with extra care:
        # - (pareto, normal)
        self.precision = 0.1  # Set this to 0.01 when testing a new KL implementation.
        self.max_samples = int(1e07)  # Increase this when testing at smaller precision.
        self.samples_per_batch = int(1e04)
        self.finite_examples = [
            (bernoulli, bernoulli),
            (bernoulli, poisson),
            (beta, beta),
            (beta, chi2),
            (beta, exponential),
            (beta, gamma),
            (beta, normal),
            (binomial30, binomial30),
            (binomial_vectorized_count, binomial_vectorized_count),
            (categorical, categorical),
            (cauchy, cauchy),
            (chi2, chi2),
            (chi2, exponential),
            (chi2, gamma),
            (chi2, normal),
            (dirichlet, dirichlet),
            (exponential, chi2),
            (exponential, exponential),
            (exponential, gamma),
            (exponential, gumbel),
            (exponential, normal),
            (gamma, chi2),
            (gamma, exponential),
            (gamma, gamma),
            (gamma, gumbel),
            (gamma, normal),
            (gumbel, gumbel),
            (gumbel, normal),
            (halfnormal, halfnormal),
            (independent, independent),
            (laplace, laplace),
            (lognormal, lognormal),
            (laplace, normal),
            (normal, gumbel),
            (normal, laplace),
            (normal, normal),
            (onehotcategorical, onehotcategorical),
            (pareto, chi2),
            (pareto, pareto),
            (pareto, exponential),
            (pareto, gamma),
            (poisson, poisson),
            (uniform_within_unit, beta),
            (uniform_positive, chi2),
            (uniform_positive, exponential),
            (uniform_positive, gamma),
            (uniform_real, gumbel),
            (uniform_real, normal),
            (uniform_pareto, pareto),
            (continuous_bernoulli, continuous_bernoulli),
            (continuous_bernoulli, exponential),
            (continuous_bernoulli, normal),
            (beta, continuous_bernoulli)
        ]

        self.infinite_examples = [
            (Bernoulli(0), Bernoulli(1)),
            (Bernoulli(1), Bernoulli(0)),
            (Categorical(torch.tensor([0.9, 0.1])), Categorical(torch.tensor([1., 0.]))),
            (Categorical(torch.tensor([[0.9, 0.1], [.9, .1]])), Categorical(torch.tensor([1., 0.]))),
            (Beta(1, 2), Uniform(0.25, 1)),
            (Beta(1, 2), Uniform(0, 0.75)),
            (Beta(1, 2), Uniform(0.25, 0.75)),
            (Beta(1, 2), Pareto(1, 2)),
            (Binomial(31, 0.7), Binomial(30, 0.3)),
            (Binomial(torch.tensor([3, 4]), torch.tensor([0.4, 0.6])),
             Binomial(torch.tensor([2, 3]), torch.tensor([0.5, 0.8]))),
            (Chi2(1), Beta(2, 3)),
            (Chi2(1), Pareto(2, 3)),
            (Chi2(1), Uniform(-2, 3)),
            (Exponential(1), Beta(2, 3)),
            (Exponential(1), Pareto(2, 3)),
            (Exponential(1), Uniform(-2, 3)),
            (Gamma(1, 2), Beta(3, 4)),
            (Gamma(1, 2), Pareto(3, 4)),
            (Gamma(1, 2), Uniform(-3, 4)),
            (Gumbel(-1, 2), Beta(3, 4)),
            (Gumbel(-1, 2), Chi2(3)),
            (Gumbel(-1, 2), Exponential(3)),
            (Gumbel(-1, 2), Gamma(3, 4)),
            (Gumbel(-1, 2), Pareto(3, 4)),
            (Gumbel(-1, 2), Uniform(-3, 4)),
            (Laplace(-1, 2), Beta(3, 4)),
            (Laplace(-1, 2), Chi2(3)),
            (Laplace(-1, 2), Exponential(3)),
            (Laplace(-1, 2), Gamma(3, 4)),
            (Laplace(-1, 2), Pareto(3, 4)),
            (Laplace(-1, 2), Uniform(-3, 4)),
            (Normal(-1, 2), Beta(3, 4)),
            (Normal(-1, 2), Chi2(3)),
            (Normal(-1, 2), Exponential(3)),
            (Normal(-1, 2), Gamma(3, 4)),
            (Normal(-1, 2), Pareto(3, 4)),
            (Normal(-1, 2), Uniform(-3, 4)),
            (Pareto(2, 1), Chi2(3)),
            (Pareto(2, 1), Exponential(3)),
            (Pareto(2, 1), Gamma(3, 4)),
            (Pareto(1, 2), Normal(-3, 4)),
            (Pareto(1, 2), Pareto(3, 4)),
            (Poisson(2), Bernoulli(0.5)),
            (Poisson(2.3), Binomial(10, 0.2)),
            (Uniform(-1, 1), Beta(2, 2)),
            (Uniform(0, 2), Beta(3, 4)),
            (Uniform(-1, 2), Beta(3, 4)),
            (Uniform(-1, 2), Chi2(3)),
            (Uniform(-1, 2), Exponential(3)),
            (Uniform(-1, 2), Gamma(3, 4)),
            (Uniform(-1, 2), Pareto(3, 4)),
            (ContinuousBernoulli(0.25), Uniform(0.25, 1)),
            (ContinuousBernoulli(0.25), Uniform(0, 0.75)),
            (ContinuousBernoulli(0.25), Uniform(0.25, 0.75)),
            (ContinuousBernoulli(0.25), Pareto(1, 2)),
            (Exponential(1), ContinuousBernoulli(0.75)),
            (Gamma(1, 2), ContinuousBernoulli(0.75)),
            (Gumbel(-1, 2), ContinuousBernoulli(0.75)),
            (Laplace(-1, 2), ContinuousBernoulli(0.75)),
            (Normal(-1, 2), ContinuousBernoulli(0.75)),
            (Uniform(-1, 1), ContinuousBernoulli(0.75)),
            (Uniform(0, 2), ContinuousBernoulli(0.75)),
            (Uniform(-1, 2), ContinuousBernoulli(0.75))
        ]

    def test_kl_monte_carlo(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for (p, _), (_, q) in self.finite_examples:
            actual = kl_divergence(p, q)
            numerator = 0
            denominator = 0
            while denominator < self.max_samples:
                x = p.sample(sample_shape=(self.samples_per_batch,))
                numerator += (p.log_prob(x) - q.log_prob(x)).sum(0)
                denominator += x.size(0)
                expected = numerator / denominator
                error = torch.abs(expected - actual) / (1 + expected)
                if error[error == error].max() < self.precision:
                    break
            self.assertLess(error[error == error].max(), self.precision, '\n'.join([
                f'Incorrect KL({type(p).__name__}, {type(q).__name__}).',
                f'Expected ({denominator} Monte Carlo samples): {expected}',
                f'Actual (analytic): {actual}',
            ]))

    # Multivariate normal has a separate Monte Carlo based test due to the requirement of random generation of
    # positive (semi) definite matrices. n is set to 5, but can be increased during testing.
    def test_kl_multivariate_normal(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        n = 5  # Number of tests for multivariate_normal
        for i in range(0, n):
            loc = [torch.randn(4) for _ in range(0, 2)]
            scale_tril = [transform_to(constraints.lower_cholesky)(torch.randn(4, 4)) for _ in range(0, 2)]
            p = MultivariateNormal(loc=loc[0], scale_tril=scale_tril[0])
            q = MultivariateNormal(loc=loc[1], scale_tril=scale_tril[1])
            actual = kl_divergence(p, q)
            numerator = 0
            denominator = 0
            while denominator < self.max_samples:
                x = p.sample(sample_shape=(self.samples_per_batch,))
                numerator += (p.log_prob(x) - q.log_prob(x)).sum(0)
                denominator += x.size(0)
                expected = numerator / denominator
                error = torch.abs(expected - actual) / (1 + expected)
                if error[error == error].max() < self.precision:
                    break
            self.assertLess(error[error == error].max(), self.precision, '\n'.join([
                f'Incorrect KL(MultivariateNormal, MultivariateNormal) instance {i + 1}/{n}',
                f'Expected ({denominator} Monte Carlo sample): {expected}',
                f'Actual (analytic): {actual}',
            ]))

    def test_kl_multivariate_normal_batched(self):
        b = 7  # Number of batches
        loc = [torch.randn(b, 3) for _ in range(0, 2)]
        scale_tril = [transform_to(constraints.lower_cholesky)(torch.randn(b, 3, 3)) for _ in range(0, 2)]
        expected_kl = torch.stack([
            kl_divergence(MultivariateNormal(loc[0][i], scale_tril=scale_tril[0][i]),
                          MultivariateNormal(loc[1][i], scale_tril=scale_tril[1][i])) for i in range(0, b)])
        actual_kl = kl_divergence(MultivariateNormal(loc[0], scale_tril=scale_tril[0]),
                                  MultivariateNormal(loc[1], scale_tril=scale_tril[1]))
        self.assertEqual(expected_kl, actual_kl)

    def test_kl_multivariate_normal_batched_broadcasted(self):
        b = 7  # Number of batches
        loc = [torch.randn(b, 3) for _ in range(0, 2)]
        scale_tril = [transform_to(constraints.lower_cholesky)(torch.randn(b, 3, 3)),
                      transform_to(constraints.lower_cholesky)(torch.randn(3, 3))]
        expected_kl = torch.stack([
            kl_divergence(MultivariateNormal(loc[0][i], scale_tril=scale_tril[0][i]),
                          MultivariateNormal(loc[1][i], scale_tril=scale_tril[1])) for i in range(0, b)])
        actual_kl = kl_divergence(MultivariateNormal(loc[0], scale_tril=scale_tril[0]),
                                  MultivariateNormal(loc[1], scale_tril=scale_tril[1]))
        self.assertEqual(expected_kl, actual_kl)

    def test_kl_lowrank_multivariate_normal(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        n = 5  # Number of tests for lowrank_multivariate_normal
        for i in range(0, n):
            loc = [torch.randn(4) for _ in range(0, 2)]
            cov_factor = [torch.randn(4, 3) for _ in range(0, 2)]
            cov_diag = [transform_to(constraints.positive)(torch.randn(4)) for _ in range(0, 2)]
            covariance_matrix = [cov_factor[i].matmul(cov_factor[i].t()) +
                                 cov_diag[i].diag() for i in range(0, 2)]
            p = LowRankMultivariateNormal(loc[0], cov_factor[0], cov_diag[0])
            q = LowRankMultivariateNormal(loc[1], cov_factor[1], cov_diag[1])
            p_full = MultivariateNormal(loc[0], covariance_matrix[0])
            q_full = MultivariateNormal(loc[1], covariance_matrix[1])
            expected = kl_divergence(p_full, q_full)

            actual_lowrank_lowrank = kl_divergence(p, q)
            actual_lowrank_full = kl_divergence(p, q_full)
            actual_full_lowrank = kl_divergence(p_full, q)

            error_lowrank_lowrank = torch.abs(actual_lowrank_lowrank - expected).max()
            self.assertLess(error_lowrank_lowrank, self.precision, '\n'.join([
                f'Incorrect KL(LowRankMultivariateNormal, LowRankMultivariateNormal) instance {i + 1}/{n}',
                f'Expected (from KL MultivariateNormal): {expected}',
                f'Actual (analytic): {actual_lowrank_lowrank}',
            ]))

            error_lowrank_full = torch.abs(actual_lowrank_full - expected).max()
            self.assertLess(error_lowrank_full, self.precision, '\n'.join([
                f'Incorrect KL(LowRankMultivariateNormal, MultivariateNormal) instance {i + 1}/{n}',
                f'Expected (from KL MultivariateNormal): {expected}',
                f'Actual (analytic): {actual_lowrank_full}',
            ]))

            error_full_lowrank = torch.abs(actual_full_lowrank - expected).max()
            self.assertLess(error_full_lowrank, self.precision, '\n'.join([
                f'Incorrect KL(MultivariateNormal, LowRankMultivariateNormal) instance {i + 1}/{n}',
                f'Expected (from KL MultivariateNormal): {expected}',
                f'Actual (analytic): {actual_full_lowrank}',
            ]))

    def test_kl_lowrank_multivariate_normal_batched(self):
        b = 7  # Number of batches
        loc = [torch.randn(b, 3) for _ in range(0, 2)]
        cov_factor = [torch.randn(b, 3, 2) for _ in range(0, 2)]
        cov_diag = [transform_to(constraints.positive)(torch.randn(b, 3)) for _ in range(0, 2)]
        expected_kl = torch.stack([
            kl_divergence(LowRankMultivariateNormal(loc[0][i], cov_factor[0][i], cov_diag[0][i]),
                          LowRankMultivariateNormal(loc[1][i], cov_factor[1][i], cov_diag[1][i]))
            for i in range(0, b)])
        actual_kl = kl_divergence(LowRankMultivariateNormal(loc[0], cov_factor[0], cov_diag[0]),
                                  LowRankMultivariateNormal(loc[1], cov_factor[1], cov_diag[1]))
        self.assertEqual(expected_kl, actual_kl)

    def test_kl_exponential_family(self):
        for (p, _), (_, q) in self.finite_examples:
            if type(p) == type(q) and issubclass(type(p), ExponentialFamily):
                actual = kl_divergence(p, q)
                expected = _kl_expfamily_expfamily(p, q)
                self.assertEqual(actual, expected, msg='\n'.join([
                    f'Incorrect KL({type(p).__name__}, {type(q).__name__}).',
                    f'Expected (using Bregman Divergence) {expected}',
                    f'Actual (analytic) {actual}',
                    f'max error = {torch.abs(actual - expected).max()}'
                ]))

    def test_kl_infinite(self):
        for p, q in self.infinite_examples:
            self.assertTrue((kl_divergence(p, q) == inf).all(),
                            f'Incorrect KL({type(p).__name__}, {type(q).__name__})')

    def test_kl_edgecases(self):
        self.assertEqual(kl_divergence(Bernoulli(0), Bernoulli(0)), 0)
        self.assertEqual(kl_divergence(Bernoulli(1), Bernoulli(1)), 0)
        self.assertEqual(kl_divergence(Categorical(torch.tensor([0., 1.])), Categorical(torch.tensor([0., 1.]))), 0)

    def test_kl_shape(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    kl = kl_divergence(dist, dist)
                except NotImplementedError:
                    continue
                expected_shape = dist.batch_shape if dist.batch_shape else torch.Size()
                self.assertEqual(kl.shape, expected_shape, msg='\n'.join([
                    f'{Dist.__name__} example {i + 1}/{len(params)}',
                    f'Expected {expected_shape}',
                    f'Actual {kl.shape}',
                ]))

    def test_kl_transformed(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/34859
        scale = torch.ones(2, 3)
        loc = torch.zeros(2, 3)
        normal = Normal(loc=loc, scale=scale)
        diag_normal = Independent(normal, reinterpreted_batch_ndims=1)
        trans_dist = TransformedDistribution(diag_normal, AffineTransform(loc=0., scale=2.))
        self.assertEqual(kl_divergence(diag_normal, diag_normal).shape, (2,))
        self.assertEqual(kl_divergence(trans_dist, trans_dist).shape, (2,))

    def test_entropy_monte_carlo(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    actual = dist.entropy()
                except NotImplementedError:
                    continue
                x = dist.sample(sample_shape=(60000,))
                expected = -dist.log_prob(x).mean(0)
                ignore = (expected == inf) | (expected == -inf)
                expected[ignore] = actual[ignore]
                self.assertEqual(actual, expected, atol=0.2, rtol=0, msg='\n'.join([
                    f'{Dist.__name__} example {i + 1}/{len(params)}, incorrect .entropy().',
                    f'Expected (monte carlo) {expected}',
                    f'Actual (analytic) {actual}',
                    f'max error = {torch.abs(actual - expected).max()}',
                ]))

    def test_entropy_exponential_family(self):
        for Dist, params in EXAMPLES:
            if not issubclass(Dist, ExponentialFamily):
                continue
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    actual = dist.entropy()
                except NotImplementedError:
                    continue
                try:
                    expected = ExponentialFamily.entropy(dist)
                except NotImplementedError:
                    continue
                self.assertEqual(actual, expected, msg='\n'.join([
                    f'{Dist.__name__} example {i + 1}/{len(params)}, incorrect .entropy().',
                    f'Expected (Bregman Divergence) {expected}',
                    f'Actual (analytic) {actual}',
                    f'max error = {torch.abs(actual - expected).max()}'
                ]))


class TestConstraints(DistributionsTestCase):
    def test_params_constraints(self):
        normalize_probs_dists = (
            Categorical,
            Multinomial,
            OneHotCategorical,
            OneHotCategoricalStraightThrough,
            RelaxedOneHotCategorical
        )

        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                for name, value in param.items():
                    if isinstance(value, numbers.Number):
                        value = torch.tensor([value])
                    if Dist in normalize_probs_dists and name == 'probs':
                        # These distributions accept positive probs, but elsewhere we
                        # use a stricter constraint to the simplex.
                        value = value / value.sum(-1, True)
                    try:
                        constraint = dist.arg_constraints[name]
                    except KeyError:
                        continue  # ignore optional parameters

                    # Check param shape is compatible with distribution shape.
                    self.assertGreaterEqual(value.dim(), constraint.event_dim)
                    value_batch_shape = value.shape[:value.dim() - constraint.event_dim]
                    torch.broadcast_shapes(dist.batch_shape, value_batch_shape)

                    if is_dependent(constraint):
                        continue

                    message = f'{Dist.__name__} example {i + 1}/{len(params)} parameter {name} = {value}'
                    self.assertTrue(constraint.check(value).all(), msg=message)

    def test_support_constraints(self):
        for Dist, params in EXAMPLES:
            self.assertIsInstance(Dist.support, Constraint)
            for i, param in enumerate(params):
                dist = Dist(**param)
                value = dist.sample()
                constraint = dist.support
                message = f'{Dist.__name__} example {i + 1}/{len(params)} sample = {value}'
                self.assertEqual(constraint.event_dim, len(dist.event_shape), msg=message)
                ok = constraint.check(value)
                self.assertEqual(ok.shape, dist.batch_shape, msg=message)
                self.assertTrue(ok.all(), msg=message)


@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestNumericalStability(DistributionsTestCase):
    def _test_pdf_score(self,
                        dist_class,
                        x,
                        expected_value,
                        probs=None,
                        logits=None,
                        expected_gradient=None,
                        atol=1e-5):
        if probs is not None:
            p = probs.detach().requires_grad_()
            dist = dist_class(p)
        else:
            p = logits.detach().requires_grad_()
            dist = dist_class(logits=p)
        log_pdf = dist.log_prob(x)
        log_pdf.sum().backward()
        self.assertEqual(log_pdf,
                         expected_value,
                         atol=atol,
                         rtol=0,
                         msg='Incorrect value for tensor type: {}. Expected = {}, Actual = {}'
                         .format(type(x), expected_value, log_pdf))
        if expected_gradient is not None:
            self.assertEqual(p.grad,
                             expected_gradient,
                             atol=atol,
                             rtol=0,
                             msg='Incorrect gradient for tensor type: {}. Expected = {}, Actual = {}'
                             .format(type(x), expected_gradient, p.grad))

    def test_bernoulli_gradient(self):
        for tensor_type in [torch.FloatTensor, torch.DoubleTensor]:
            self._test_pdf_score(dist_class=Bernoulli,
                                 probs=tensor_type([0]),
                                 x=tensor_type([0]),
                                 expected_value=tensor_type([0]),
                                 expected_gradient=tensor_type([0]))

            self._test_pdf_score(dist_class=Bernoulli,
                                 probs=tensor_type([0]),
                                 x=tensor_type([1]),
                                 expected_value=tensor_type([torch.finfo(tensor_type([]).dtype).eps]).log(),
                                 expected_gradient=tensor_type([0]))

            self._test_pdf_score(dist_class=Bernoulli,
                                 probs=tensor_type([1e-4]),
                                 x=tensor_type([1]),
                                 expected_value=tensor_type([math.log(1e-4)]),
                                 expected_gradient=tensor_type([10000]))

            # Lower precision due to:
            # >>> 1 / (1 - torch.FloatTensor([0.9999]))
            # 9998.3408
            # [torch.FloatTensor of size 1]
            self._test_pdf_score(dist_class=Bernoulli,
                                 probs=tensor_type([1 - 1e-4]),
                                 x=tensor_type([0]),
                                 expected_value=tensor_type([math.log(1e-4)]),
                                 expected_gradient=tensor_type([-10000]),
                                 atol=2)

            self._test_pdf_score(dist_class=Bernoulli,
                                 logits=tensor_type([math.log(9999)]),
                                 x=tensor_type([0]),
                                 expected_value=tensor_type([math.log(1e-4)]),
                                 expected_gradient=tensor_type([-1]),
                                 atol=1e-3)

    def test_bernoulli_with_logits_underflow(self):
        for tensor_type, lim in ([(torch.FloatTensor, -1e38),
                                  (torch.DoubleTensor, -1e308)]):
            self._test_pdf_score(dist_class=Bernoulli,
                                 logits=tensor_type([lim]),
                                 x=tensor_type([0]),
                                 expected_value=tensor_type([0]),
                                 expected_gradient=tensor_type([0]))

    def test_bernoulli_with_logits_overflow(self):
        for tensor_type, lim in ([(torch.FloatTensor, 1e38),
                                  (torch.DoubleTensor, 1e308)]):
            self._test_pdf_score(dist_class=Bernoulli,
                                 logits=tensor_type([lim]),
                                 x=tensor_type([1]),
                                 expected_value=tensor_type([0]),
                                 expected_gradient=tensor_type([0]))

    def test_categorical_log_prob(self):
        for dtype in ([torch.float, torch.double]):
            p = torch.tensor([0, 1], dtype=dtype, requires_grad=True)
            categorical = OneHotCategorical(p)
            log_pdf = categorical.log_prob(torch.tensor([0, 1], dtype=dtype))
            self.assertEqual(log_pdf.item(), 0)

    def test_categorical_log_prob_with_logits(self):
        for dtype in ([torch.float, torch.double]):
            p = torch.tensor([-inf, 0], dtype=dtype, requires_grad=True)
            categorical = OneHotCategorical(logits=p)
            log_pdf_prob_1 = categorical.log_prob(torch.tensor([0, 1], dtype=dtype))
            self.assertEqual(log_pdf_prob_1.item(), 0)
            log_pdf_prob_0 = categorical.log_prob(torch.tensor([1, 0], dtype=dtype))
            self.assertEqual(log_pdf_prob_0.item(), -inf)

    def test_multinomial_log_prob(self):
        for dtype in ([torch.float, torch.double]):
            p = torch.tensor([0, 1], dtype=dtype, requires_grad=True)
            s = torch.tensor([0, 10], dtype=dtype)
            multinomial = Multinomial(10, p)
            log_pdf = multinomial.log_prob(s)
            self.assertEqual(log_pdf.item(), 0)

    def test_multinomial_log_prob_with_logits(self):
        for dtype in ([torch.float, torch.double]):
            p = torch.tensor([-inf, 0], dtype=dtype, requires_grad=True)
            multinomial = Multinomial(10, logits=p)
            log_pdf_prob_1 = multinomial.log_prob(torch.tensor([0, 10], dtype=dtype))
            self.assertEqual(log_pdf_prob_1.item(), 0)
            log_pdf_prob_0 = multinomial.log_prob(torch.tensor([10, 0], dtype=dtype))
            self.assertEqual(log_pdf_prob_0.item(), -inf)

    def test_continuous_bernoulli_gradient(self):

        def expec_val(x, probs=None, logits=None):
            assert not (probs is None and logits is None)
            if logits is not None:
                probs = 1. / (1. + math.exp(-logits))
            bern_log_lik = x * math.log(probs) + (1. - x) * math.log1p(-probs)
            if probs < 0.499 or probs > 0.501:  # using default values of lims here
                log_norm_const = math.log(
                    math.fabs(math.atanh(1. - 2. * probs))) - math.log(math.fabs(1. - 2. * probs)) + math.log(2.)
            else:
                aux = math.pow(probs - 0.5, 2)
                log_norm_const = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * aux) * aux
            log_lik = bern_log_lik + log_norm_const
            return log_lik

        def expec_grad(x, probs=None, logits=None):
            assert not (probs is None and logits is None)
            if logits is not None:
                probs = 1. / (1. + math.exp(-logits))
            grad_bern_log_lik = x / probs - (1. - x) / (1. - probs)
            if probs < 0.499 or probs > 0.501:  # using default values of lims here
                grad_log_c = 2. * probs - 4. * (probs - 1.) * probs * math.atanh(1. - 2. * probs) - 1.
                grad_log_c /= 2. * (probs - 1.) * probs * (2. * probs - 1.) * math.atanh(1. - 2. * probs)
            else:
                grad_log_c = 8. / 3. * (probs - 0.5) + 416. / 45. * math.pow(probs - 0.5, 3)
            grad = grad_bern_log_lik + grad_log_c
            if logits is not None:
                grad *= 1. / (1. + math.exp(logits)) - 1. / math.pow(1. + math.exp(logits), 2)
            return grad

        for tensor_type in [torch.FloatTensor, torch.DoubleTensor]:
            self._test_pdf_score(dist_class=ContinuousBernoulli,
                                 probs=tensor_type([0.1]),
                                 x=tensor_type([0.1]),
                                 expected_value=tensor_type([expec_val(0.1, probs=0.1)]),
                                 expected_gradient=tensor_type([expec_grad(0.1, probs=0.1)]))

            self._test_pdf_score(dist_class=ContinuousBernoulli,
                                 probs=tensor_type([0.1]),
                                 x=tensor_type([1.]),
                                 expected_value=tensor_type([expec_val(1., probs=0.1)]),
                                 expected_gradient=tensor_type([expec_grad(1., probs=0.1)]))

            self._test_pdf_score(dist_class=ContinuousBernoulli,
                                 probs=tensor_type([0.4999]),
                                 x=tensor_type([0.9]),
                                 expected_value=tensor_type([expec_val(0.9, probs=0.4999)]),
                                 expected_gradient=tensor_type([expec_grad(0.9, probs=0.4999)]))

            self._test_pdf_score(dist_class=ContinuousBernoulli,
                                 probs=tensor_type([1e-4]),
                                 x=tensor_type([1]),
                                 expected_value=tensor_type([expec_val(1, probs=1e-4)]),
                                 expected_gradient=tensor_type(tensor_type([expec_grad(1, probs=1e-4)])),
                                 atol=1e-3)

            self._test_pdf_score(dist_class=ContinuousBernoulli,
                                 probs=tensor_type([1 - 1e-4]),
                                 x=tensor_type([0.1]),
                                 expected_value=tensor_type([expec_val(0.1, probs=1 - 1e-4)]),
                                 expected_gradient=tensor_type([expec_grad(0.1, probs=1 - 1e-4)]),
                                 atol=2)

            self._test_pdf_score(dist_class=ContinuousBernoulli,
                                 logits=tensor_type([math.log(9999)]),
                                 x=tensor_type([0]),
                                 expected_value=tensor_type([expec_val(0, logits=math.log(9999))]),
                                 expected_gradient=tensor_type([expec_grad(0, logits=math.log(9999))]),
                                 atol=1e-3)

            self._test_pdf_score(dist_class=ContinuousBernoulli,
                                 logits=tensor_type([0.001]),
                                 x=tensor_type([0.5]),
                                 expected_value=tensor_type([expec_val(0.5, logits=0.001)]),
                                 expected_gradient=tensor_type([expec_grad(0.5, logits=0.001)]))

    def test_continuous_bernoulli_with_logits_underflow(self):
        for tensor_type, lim, expected in ([(torch.FloatTensor, -1e38, 2.76898),
                                            (torch.DoubleTensor, -1e308, 3.58473)]):
            self._test_pdf_score(dist_class=ContinuousBernoulli,
                                 logits=tensor_type([lim]),
                                 x=tensor_type([0]),
                                 expected_value=tensor_type([expected]),
                                 expected_gradient=tensor_type([0.]))

    def test_continuous_bernoulli_with_logits_overflow(self):
        for tensor_type, lim, expected in ([(torch.FloatTensor, 1e38, 2.76898),
                                            (torch.DoubleTensor, 1e308, 3.58473)]):
            self._test_pdf_score(dist_class=ContinuousBernoulli,
                                 logits=tensor_type([lim]),
                                 x=tensor_type([1]),
                                 expected_value=tensor_type([expected]),
                                 expected_gradient=tensor_type([0.]))


# TODO: make this a pytest parameterized test
class TestLazyLogitsInitialization(DistributionsTestCase):
    def setUp(self):
        super().setUp()
        # ContinuousBernoulli is not tested because log_prob is not computed simply
        # from 'logits', but 'probs' is also needed
        self.examples = [e for e in EXAMPLES if e.Dist in
                         (Categorical, OneHotCategorical, Bernoulli, Binomial, Multinomial)]

    def test_lazy_logits_initialization(self):
        for Dist, params in self.examples:
            param = params[0].copy()
            if 'probs' not in param:
                continue
            probs = param.pop('probs')
            param['logits'] = probs_to_logits(probs)
            dist = Dist(**param)
            # Create new instance to generate a valid sample
            dist.log_prob(Dist(**param).sample())
            message = f'Failed for {Dist.__name__} example 0/{len(params)}'
            self.assertNotIn('probs', dist.__dict__, msg=message)
            try:
                dist.enumerate_support()
            except NotImplementedError:
                pass
            self.assertNotIn('probs', dist.__dict__, msg=message)
            batch_shape, event_shape = dist.batch_shape, dist.event_shape
            self.assertNotIn('probs', dist.__dict__, msg=message)

    def test_lazy_probs_initialization(self):
        for Dist, params in self.examples:
            param = params[0].copy()
            if 'probs' not in param:
                continue
            dist = Dist(**param)
            dist.sample()
            message = f'Failed for {Dist.__name__} example 0/{len(params)}'
            self.assertNotIn('logits', dist.__dict__, msg=message)
            try:
                dist.enumerate_support()
            except NotImplementedError:
                pass
            self.assertNotIn('logits', dist.__dict__, msg=message)
            batch_shape, event_shape = dist.batch_shape, dist.event_shape
            self.assertNotIn('logits', dist.__dict__, msg=message)


@unittest.skipIf(not TEST_NUMPY, "NumPy not found")
class TestAgainstScipy(DistributionsTestCase):
    def setUp(self):
        super().setUp()
        positive_var = torch.randn(20).exp()
        positive_var2 = torch.randn(20).exp()
        random_var = torch.randn(20)
        simplex_tensor = softmax(torch.randn(20), dim=-1)
        cov_tensor = torch.randn(20, 20)
        cov_tensor = cov_tensor @ cov_tensor.mT
        self.distribution_pairs = [
            (
                Bernoulli(simplex_tensor),
                scipy.stats.bernoulli(simplex_tensor)
            ),
            (
                Beta(positive_var, positive_var2),
                scipy.stats.beta(positive_var, positive_var2)
            ),
            (
                Binomial(10, simplex_tensor),
                scipy.stats.binom(10 * np.ones(simplex_tensor.shape), simplex_tensor.numpy())
            ),
            (
                Cauchy(random_var, positive_var),
                scipy.stats.cauchy(loc=random_var, scale=positive_var)
            ),
            (
                Dirichlet(positive_var),
                scipy.stats.dirichlet(positive_var)
            ),
            (
                Exponential(positive_var),
                scipy.stats.expon(scale=positive_var.reciprocal())
            ),
            (
                FisherSnedecor(positive_var, 4 + positive_var2),  # var for df2<=4 is undefined
                scipy.stats.f(positive_var, 4 + positive_var2)
            ),
            (
                Gamma(positive_var, positive_var2),
                scipy.stats.gamma(positive_var, scale=positive_var2.reciprocal())
            ),
            (
                Geometric(simplex_tensor),
                scipy.stats.geom(simplex_tensor, loc=-1)
            ),
            (
                Gumbel(random_var, positive_var2),
                scipy.stats.gumbel_r(random_var, positive_var2)
            ),
            (
                HalfCauchy(positive_var),
                scipy.stats.halfcauchy(scale=positive_var)
            ),
            (
                HalfNormal(positive_var2),
                scipy.stats.halfnorm(scale=positive_var2)
            ),
            (
                Laplace(random_var, positive_var2),
                scipy.stats.laplace(random_var, positive_var2)
            ),
            (
                # Tests fail 1e-5 threshold if scale > 3
                LogNormal(random_var, positive_var.clamp(max=3)),
                scipy.stats.lognorm(s=positive_var.clamp(max=3), scale=random_var.exp())
            ),
            (
                LowRankMultivariateNormal(random_var, torch.zeros(20, 1), positive_var2),
                scipy.stats.multivariate_normal(random_var, torch.diag(positive_var2))
            ),
            (
                Multinomial(10, simplex_tensor),
                scipy.stats.multinomial(10, simplex_tensor)
            ),
            (
                MultivariateNormal(random_var, torch.diag(positive_var2)),
                scipy.stats.multivariate_normal(random_var, torch.diag(positive_var2))
            ),
            (
                MultivariateNormal(random_var, cov_tensor),
                scipy.stats.multivariate_normal(random_var, cov_tensor)
            ),
            (
                Normal(random_var, positive_var2),
                scipy.stats.norm(random_var, positive_var2)
            ),
            (
                OneHotCategorical(simplex_tensor),
                scipy.stats.multinomial(1, simplex_tensor)
            ),
            (
                Pareto(positive_var, 2 + positive_var2),
                scipy.stats.pareto(2 + positive_var2, scale=positive_var)
            ),
            (
                Poisson(positive_var),
                scipy.stats.poisson(positive_var)
            ),
            (
                StudentT(2 + positive_var, random_var, positive_var2),
                scipy.stats.t(2 + positive_var, random_var, positive_var2)
            ),
            (
                Uniform(random_var, random_var + positive_var),
                scipy.stats.uniform(random_var, positive_var)
            ),
            (
                VonMises(random_var, positive_var),
                scipy.stats.vonmises(positive_var, loc=random_var)
            ),
            (
                Weibull(positive_var[0], positive_var2[0]),  # scipy var for Weibull only supports scalars
                scipy.stats.weibull_min(c=positive_var2[0], scale=positive_var[0])
            ),
            (
                # scipy var for Wishart only supports scalars
                # SciPy allowed ndim -1 < df < ndim for Wishar distribution after version 1.7.0
                Wishart(
                    (20 if version.parse(scipy.__version__) < version.parse("1.7.0") else 19) + positive_var[0],
                    cov_tensor,
                ),
                scipy.stats.wishart(
                    (20 if version.parse(scipy.__version__) < version.parse("1.7.0") else 19) + positive_var[0].item(),
                    cov_tensor,
                ),
            ),
        ]

    def test_mean(self):
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            if isinstance(pytorch_dist, (Cauchy, HalfCauchy)):
                # Cauchy, HalfCauchy distributions' mean is nan, skipping check
                continue
            elif isinstance(pytorch_dist, (LowRankMultivariateNormal, MultivariateNormal)):
                self.assertEqual(pytorch_dist.mean, scipy_dist.mean, msg=pytorch_dist)
            else:
                self.assertEqual(pytorch_dist.mean, scipy_dist.mean(), msg=pytorch_dist)

    def test_variance_stddev(self):
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            if isinstance(pytorch_dist, (Cauchy, HalfCauchy, VonMises)):
                # Cauchy, HalfCauchy distributions' standard deviation is nan, skipping check
                # VonMises variance is circular and scipy doesn't produce a correct result
                continue
            elif isinstance(pytorch_dist, (Multinomial, OneHotCategorical)):
                self.assertEqual(pytorch_dist.variance, np.diag(scipy_dist.cov()), msg=pytorch_dist)
                self.assertEqual(pytorch_dist.stddev, np.diag(scipy_dist.cov()) ** 0.5, msg=pytorch_dist)
            elif isinstance(pytorch_dist, (LowRankMultivariateNormal, MultivariateNormal)):
                self.assertEqual(pytorch_dist.variance, np.diag(scipy_dist.cov), msg=pytorch_dist)
                self.assertEqual(pytorch_dist.stddev, np.diag(scipy_dist.cov) ** 0.5, msg=pytorch_dist)
            else:
                self.assertEqual(pytorch_dist.variance, scipy_dist.var(), msg=pytorch_dist)
                self.assertEqual(pytorch_dist.stddev, scipy_dist.var() ** 0.5, msg=pytorch_dist)

    def test_cdf(self):
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            samples = pytorch_dist.sample((5,))
            try:
                cdf = pytorch_dist.cdf(samples)
            except NotImplementedError:
                continue
            self.assertEqual(cdf, scipy_dist.cdf(samples), msg=pytorch_dist)

    def test_icdf(self):
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            samples = torch.rand((5,) + pytorch_dist.batch_shape)
            try:
                icdf = pytorch_dist.icdf(samples)
            except NotImplementedError:
                continue
            self.assertEqual(icdf, scipy_dist.ppf(samples), msg=pytorch_dist)


class TestFunctors(DistributionsTestCase):
    def test_cat_transform(self):
        x1 = -1 * torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        x2 = (torch.arange(1, 101, dtype=torch.float).view(-1, 100) - 1) / 100
        x3 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        t1, t2, t3 = ExpTransform(), AffineTransform(1, 100), identity_transform
        dim = 0
        x = torch.cat([x1, x2, x3], dim=dim)
        t = CatTransform([t1, t2, t3], dim=dim)
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.cat([t1.domain.check(x1),
                                        t2.domain.check(x2),
                                        t3.domain.check(x3)], dim=dim)
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.cat([t1(x1), t2(x2), t3(x3)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y2 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y3 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y = torch.cat([y1, y2, y3], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.cat([t1.codomain.check(y1),
                                        t2.codomain.check(y2),
                                        t3.codomain.check(y3)], dim=dim)
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(y)
        expected_inv = torch.cat([t1.inv(y1), t2.inv(y2), t3.inv(y3)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.cat([t1.log_abs_det_jacobian(x1, y1),
                                  t2.log_abs_det_jacobian(x2, y2),
                                  t3.log_abs_det_jacobian(x3, y3)], dim=dim)
        self.assertEqual(actual_jac, expected_jac)

    def test_cat_transform_non_uniform(self):
        x1 = -1 * torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        x2 = torch.cat([(torch.arange(1, 101, dtype=torch.float).view(-1, 100) - 1) / 100,
                        torch.arange(1, 101, dtype=torch.float).view(-1, 100)])
        t1 = ExpTransform()
        t2 = CatTransform([AffineTransform(1, 100), identity_transform], dim=0)
        dim = 0
        x = torch.cat([x1, x2], dim=dim)
        t = CatTransform([t1, t2], dim=dim, lengths=[1, 2])
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.cat([t1.domain.check(x1),
                                        t2.domain.check(x2)], dim=dim)
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.cat([t1(x1), t2(x2)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y2 = torch.cat([torch.arange(1, 101, dtype=torch.float).view(-1, 100),
                        torch.arange(1, 101, dtype=torch.float).view(-1, 100)])
        y = torch.cat([y1, y2], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.cat([t1.codomain.check(y1),
                                        t2.codomain.check(y2)], dim=dim)
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(y)
        expected_inv = torch.cat([t1.inv(y1), t2.inv(y2)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.cat([t1.log_abs_det_jacobian(x1, y1),
                                  t2.log_abs_det_jacobian(x2, y2)], dim=dim)
        self.assertEqual(actual_jac, expected_jac)

    def test_cat_event_dim(self):
        t1 = AffineTransform(0, 2 * torch.ones(2), event_dim=1)
        t2 = AffineTransform(0, 2 * torch.ones(2), event_dim=1)
        dim = 1
        bs = 16
        x1 = torch.randn(bs, 2)
        x2 = torch.randn(bs, 2)
        x = torch.cat([x1, x2], dim=1)
        t = CatTransform([t1, t2], dim=dim, lengths=[2, 2])
        y1 = t1(x1)
        y2 = t2(x2)
        y = t(x)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = sum([t1.log_abs_det_jacobian(x1, y1),
                            t2.log_abs_det_jacobian(x2, y2)])

    def test_stack_transform(self):
        x1 = -1 * torch.arange(1, 101, dtype=torch.float)
        x2 = (torch.arange(1, 101, dtype=torch.float) - 1) / 100
        x3 = torch.arange(1, 101, dtype=torch.float)
        t1, t2, t3 = ExpTransform(), AffineTransform(1, 100), identity_transform
        dim = 0
        x = torch.stack([x1, x2, x3], dim=dim)
        t = StackTransform([t1, t2, t3], dim=dim)
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.stack([t1.domain.check(x1),
                                          t2.domain.check(x2),
                                          t3.domain.check(x3)], dim=dim)
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.stack([t1(x1), t2(x2), t3(x3)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float)
        y2 = torch.arange(1, 101, dtype=torch.float)
        y3 = torch.arange(1, 101, dtype=torch.float)
        y = torch.stack([y1, y2, y3], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.stack([t1.codomain.check(y1),
                                          t2.codomain.check(y2),
                                          t3.codomain.check(y3)], dim=dim)
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(x)
        expected_inv = torch.stack([t1.inv(x1), t2.inv(x2), t3.inv(x3)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.stack([t1.log_abs_det_jacobian(x1, y1),
                                    t2.log_abs_det_jacobian(x2, y2),
                                    t3.log_abs_det_jacobian(x3, y3)], dim=dim)
        self.assertEqual(actual_jac, expected_jac)


class TestValidation(DistributionsTestCase):
    def test_valid(self):
        for Dist, params in EXAMPLES:
            for param in params:
                Dist(validate_args=True, **param)

    def test_invalid_log_probs_arg(self):
        # Check that validation errors are indeed disabled,
        # but they might raise another error
        for Dist, params in EXAMPLES:
            if Dist == TransformedDistribution:
                # TransformedDistribution has a distribution instance
                # as the argument, so we cannot do much about that
                continue
            for i, param in enumerate(params):
                d_nonval = Dist(validate_args=False, **param)
                d_val = Dist(validate_args=True, **param)
                for v in torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]):
                    # samples with incorrect shape must throw ValueError only
                    try:
                        log_prob = d_val.log_prob(v)
                    except ValueError:
                        pass
                    # get sample of correct shape
                    val = torch.full(d_val.batch_shape + d_val.event_shape, v)
                    # check samples with incorrect support
                    try:
                        log_prob = d_val.log_prob(val)
                    except ValueError as e:
                        if e.args and 'must be within the support' in e.args[0]:
                            try:
                                log_prob = d_nonval.log_prob(val)
                            except RuntimeError:
                                pass

                # check correct samples are ok
                valid_value = d_val.sample()
                d_val.log_prob(valid_value)
                # check invalid values raise ValueError
                if valid_value.dtype == torch.long:
                    valid_value = valid_value.float()
                invalid_value = torch.full_like(valid_value, math.nan)
                try:
                    with self.assertRaisesRegex(
                        ValueError,
                        "Expected value argument .* to be within the support .*",
                    ):
                        d_val.log_prob(invalid_value)
                except AssertionError as e:
                    fail_string = "Support ValueError not raised for {} example {}/{}"
                    raise AssertionError(
                        fail_string.format(Dist.__name__, i + 1, len(params))
                    ) from e

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_invalid(self):
        for Dist, params in BAD_EXAMPLES:
            for i, param in enumerate(params):
                try:
                    with self.assertRaises(ValueError):
                        Dist(validate_args=True, **param)
                except AssertionError as e:
                    fail_string = "ValueError not raised for {} example {}/{}"
                    raise AssertionError(
                        fail_string.format(Dist.__name__, i + 1, len(params))
                    ) from e

    def test_warning_unimplemented_constraints(self):
        class Delta(Distribution):
            def __init__(self, validate_args=True):
                super().__init__(validate_args=validate_args)

            def sample(self, sample_shape=torch.Size()):
                return torch.tensor(0.).expand(sample_shape)

            def log_prob(self, value):
                if self._validate_args:
                    self._validate_sample(value)
                value[value != 0.] = -float('inf')
                value[value == 0.] = 0.
                return value

        with self.assertWarns(UserWarning):
            d = Delta()
        sample = d.sample((2,))
        with self.assertWarns(UserWarning):
            d.log_prob(sample)


class TestJit(DistributionsTestCase):
    def _examples(self):
        for Dist, params in EXAMPLES:
            for param in params:
                keys = param.keys()
                values = tuple(param[key] for key in keys)
                if not all(isinstance(x, torch.Tensor) for x in values):
                    continue
                sample = Dist(**param).sample()
                yield Dist, keys, values, sample

    def _perturb_tensor(self, value, constraint):
        if isinstance(constraint, constraints._IntegerGreaterThan):
            return value + 1
        if isinstance(constraint, (constraints._PositiveDefinite, constraints._PositiveSemidefinite)):
            return value + torch.eye(value.shape[-1])
        if value.dtype in [torch.float, torch.double]:
            transform = transform_to(constraint)
            delta = value.new(value.shape).normal_()
            return transform(transform.inv(value) + delta)
        if value.dtype == torch.long:
            result = value.clone()
            result[value == 0] = 1
            result[value == 1] = 0
            return result
        raise NotImplementedError

    def _perturb(self, Dist, keys, values, sample):
        with torch.no_grad():
            if Dist is Uniform:
                param = dict(zip(keys, values))
                param['low'] = param['low'] - torch.rand(param['low'].shape)
                param['high'] = param['high'] + torch.rand(param['high'].shape)
                values = [param[key] for key in keys]
            else:
                values = [self._perturb_tensor(value, Dist.arg_constraints.get(key, constraints.real))
                          for key, value in zip(keys, values)]
            param = dict(zip(keys, values))
            sample = Dist(**param).sample()
            return values, sample

    def test_sample(self):
        for Dist, keys, values, sample in self._examples():

            def f(*values):
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.sample()

            traced_f = torch.jit.trace(f, values, check_trace=False)

            # FIXME Schema not found for node
            xfail = [
                Cauchy,  # aten::cauchy(Double(2,1), float, float, Generator)
                HalfCauchy,  # aten::cauchy(Double(2, 1), float, float, Generator)
                VonMises  # Variance is not Euclidean
            ]
            if Dist in xfail:
                continue

            with torch.random.fork_rng():
                sample = f(*values)
            traced_sample = traced_f(*values)
            self.assertEqual(sample, traced_sample)

            # FIXME no nondeterministic nodes found in trace
            xfail = [Beta, Dirichlet]
            if Dist not in xfail:
                self.assertTrue(any(n.isNondeterministic() for n in traced_f.graph.nodes()))

    def test_rsample(self):
        for Dist, keys, values, sample in self._examples():
            if not Dist.has_rsample:
                continue

            def f(*values):
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.rsample()

            traced_f = torch.jit.trace(f, values, check_trace=False)

            # FIXME Schema not found for node
            xfail = [
                Cauchy,  # aten::cauchy(Double(2,1), float, float, Generator)
                HalfCauchy,  # aten::cauchy(Double(2, 1), float, float, Generator)
            ]
            if Dist in xfail:
                continue

            with torch.random.fork_rng():
                sample = f(*values)
            traced_sample = traced_f(*values)
            self.assertEqual(sample, traced_sample)

            # FIXME no nondeterministic nodes found in trace
            xfail = [Beta, Dirichlet]
            if Dist not in xfail:
                self.assertTrue(any(n.isNondeterministic() for n in traced_f.graph.nodes()))

    def test_log_prob(self):
        for Dist, keys, values, sample in self._examples():
            # FIXME traced functions produce incorrect results
            xfail = [LowRankMultivariateNormal, MultivariateNormal]
            if Dist in xfail:
                continue

            def f(sample, *values):
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.log_prob(sample)

            traced_f = torch.jit.trace(f, (sample,) + values)

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(sample, *values)
            actual = traced_f(sample, *values)
            self.assertEqual(expected, actual,
                             msg=f'{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}')

    def test_enumerate_support(self):
        for Dist, keys, values, sample in self._examples():
            # FIXME traced functions produce incorrect results
            xfail = [Binomial]
            if Dist in xfail:
                continue

            def f(*values):
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.enumerate_support()

            try:
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                continue

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(*values)
            actual = traced_f(*values)
            self.assertEqual(expected, actual,
                             msg=f'{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}')

    def test_mean(self):
        for Dist, keys, values, sample in self._examples():

            def f(*values):
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.mean

            try:
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                continue

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(*values)
            actual = traced_f(*values)
            expected[expected == float('inf')] = 0.
            actual[actual == float('inf')] = 0.
            self.assertEqual(expected, actual,
                             msg=f'{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}')

    def test_variance(self):
        for Dist, keys, values, sample in self._examples():
            if Dist in [Cauchy, HalfCauchy]:
                continue  # infinite variance

            def f(*values):
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.variance

            try:
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                continue

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(*values).clone()
            actual = traced_f(*values).clone()
            expected[expected == float('inf')] = 0.
            actual[actual == float('inf')] = 0.
            self.assertEqual(expected, actual,
                             msg=f'{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}')

    def test_entropy(self):
        for Dist, keys, values, sample in self._examples():
            # FIXME traced functions produce incorrect results
            xfail = [LowRankMultivariateNormal, MultivariateNormal]
            if Dist in xfail:
                continue

            def f(*values):
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.entropy()

            try:
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                continue

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(*values)
            actual = traced_f(*values)
            self.assertEqual(expected, actual,
                             msg=f'{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}')

    def test_cdf(self):
        for Dist, keys, values, sample in self._examples():

            def f(sample, *values):
                param = dict(zip(keys, values))
                dist = Dist(**param)
                cdf = dist.cdf(sample)
                return dist.icdf(cdf)

            try:
                traced_f = torch.jit.trace(f, (sample,) + values)
            except NotImplementedError:
                continue

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(sample, *values)
            actual = traced_f(sample, *values)
            self.assertEqual(expected, actual,
                             msg=f'{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}')


if __name__ == '__main__' and torch._C.has_lapack:
    run_tests()
