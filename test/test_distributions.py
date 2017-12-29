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
"""

import math
import unittest
from collections import namedtuple
from itertools import product

import torch
from common import TestCase, run_tests, set_rng_seed
from torch.autograd import Variable, gradcheck
from torch.distributions import (Bernoulli, Beta, Categorical, Cauchy,
                                 Dirichlet, Exponential, Gamma, Laplace,
                                 Normal, OneHotCategorical, Uniform)

TEST_NUMPY = True
try:
    import numpy as np
    import scipy.stats
    import scipy.special
except ImportError:
    TEST_NUMPY = False


# Register all distributions for generic tests.
Example = namedtuple('Example', ['Dist', 'params'])
EXAMPLES = [
    Example(Bernoulli, [
        {'probs': Variable(torch.Tensor([0.7, 0.2, 0.4]), requires_grad=True)},
        {'probs': Variable(torch.Tensor([0.3]), requires_grad=True)},
        {'probs': 0.3},
    ]),
    Example(Beta, [
        {
            'alpha': Variable(torch.exp(torch.randn(2, 3)), requires_grad=True),
            'beta': Variable(torch.exp(torch.randn(2, 3)), requires_grad=True),
        },
        {
            'alpha': Variable(torch.exp(torch.randn(4)), requires_grad=True),
            'beta': Variable(torch.exp(torch.randn(4)), requires_grad=True),
        },
    ]),
    Example(Categorical, [
        {'probs': Variable(torch.Tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]), requires_grad=True)},
        {'probs': Variable(torch.Tensor([[1.0, 0.0], [0.0, 1.0]]), requires_grad=True)},
    ]),
    Example(OneHotCategorical, [
        {'probs': Variable(torch.Tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]), requires_grad=True)},
        {'probs': Variable(torch.Tensor([[1.0, 0.0], [0.0, 1.0]]), requires_grad=True)},
    ]),
    Example(Cauchy, [
        {'loc': 0.0, 'scale': 1.0},
        {'loc': Variable(torch.Tensor([0.0])), 'scale': 1.0},
        {'loc': Variable(torch.Tensor([[0.0], [0.0]])),
         'scale': Variable(torch.Tensor([[1.0], [1.0]]))}
    ]),
    Example(Gamma, [
        {
            'alpha': Variable(torch.exp(torch.randn(2, 3)), requires_grad=True),
            'beta': Variable(torch.exp(torch.randn(2, 3)), requires_grad=True),
        },
        {
            'alpha': Variable(torch.exp(torch.randn(1)), requires_grad=True),
            'beta': Variable(torch.exp(torch.randn(1)), requires_grad=True),
        },
    ]),
    Example(Dirichlet, [
        {'alpha': Variable(torch.exp(torch.randn(2, 3)), requires_grad=True)},
        {'alpha': Variable(torch.exp(torch.randn(4)), requires_grad=True)},
    ]),
    Example(Exponential, [
        {'rate': Variable(torch.randn(5, 5).abs(), requires_grad=True)},
        {'rate': Variable(torch.randn(1).abs(), requires_grad=True)},
    ]),
    Example(Normal, [
        {
            'mean': Variable(torch.randn(5, 5), requires_grad=True),
            'std': Variable(torch.randn(5, 5).abs(), requires_grad=True),
        },
        {
            'mean': Variable(torch.randn(1), requires_grad=True),
            'std': Variable(torch.randn(1), requires_grad=True),
        },
        {
            'mean': torch.Tensor([1.0, 0.0]),
            'std': torch.Tensor([1e-5, 1e-5]),
        },
    ]),
    Example(Uniform, [
        {
            'low': Variable(torch.zeros(5, 5), requires_grad=True),
            'high': Variable(torch.ones(5, 5), requires_grad=True),
        },
        {
            'low': Variable(torch.zeros(1), requires_grad=True),
            'high': Variable(torch.ones(1), requires_grad=True),
        },
        {
            'low': torch.Tensor([1.0, 1.0]),
            'high': torch.Tensor([2.0, 3.0]),
        },
    ]),
    Example(Laplace, [
        {
            'loc': Variable(torch.randn(5, 5), requires_grad=True),
            'scale': Variable(torch.randn(5, 5).abs(), requires_grad=True),
        },
        {
            'loc': Variable(torch.randn(1), requires_grad=True),
            'scale': Variable(torch.randn(1), requires_grad=True),
        },
        {
            'loc': torch.Tensor([1.0, 0.0]),
            'scale': torch.Tensor([1e-5, 1e-5]),
        },
    ]),
]


class TestDistributions(TestCase):
    def _gradcheck_log_prob(self, dist_ctor, ctor_params):
        # performs gradient checks on log_prob
        distribution = dist_ctor(*ctor_params)
        s = distribution.sample()

        expected_shape = distribution.batch_shape + distribution.event_shape
        if not expected_shape:
            expected_shape = torch.Size((1,))  # Work around lack of scalars.
        self.assertEqual(s.size(), expected_shape)

        def apply_fn(*params):
            return dist_ctor(*params).log_prob(s)

        gradcheck(apply_fn, ctor_params, raise_exception=True)

    def _check_log_prob(self, dist, asset_fn):
        # checks that the log_prob matches a reference function
        s = dist.sample()
        log_probs = dist.log_prob(s)
        for i, (val, log_prob) in enumerate(zip(s.data.view(-1), log_probs.data.view(-1))):
            asset_fn(i, val, log_prob)

    def _check_sampler_sampler(self, torch_dist, ref_dist, message, multivariate=False,
                               num_samples=10000, failure_rate=1e-3):
        # Checks that the .sample() method matches a reference function.
        torch_samples = torch_dist.sample_n(num_samples).squeeze()
        if isinstance(torch_samples, Variable):
            torch_samples = torch_samples.data
        torch_samples = torch_samples.cpu().numpy()
        ref_samples = ref_dist.rvs(num_samples)
        if multivariate:
            # Project onto a random axis.
            axis = np.random.normal(size=torch_samples.shape[-1])
            axis /= np.linalg.norm(axis)
            torch_samples = np.dot(torch_samples, axis)
            ref_samples = np.dot(ref_samples, axis)
        samples = [(x, +1) for x in torch_samples] + [(x, -1) for x in ref_samples]
        samples.sort()
        samples = np.array(samples)[:, 1]

        # Aggragate into bins filled with roughly zero-mean unit-variance RVs.
        num_bins = 10
        samples_per_bin = len(samples) // num_bins
        bins = samples.reshape((num_bins, samples_per_bin)).mean(axis=1)
        stddev = samples_per_bin ** -0.5
        threshold = stddev * scipy.special.erfinv(1 - 2 * failure_rate / num_bins)
        message = '{}.sample() is biased:\n{}'.format(message, bins)
        for bias in bins:
            self.assertLess(-threshold, bias, message)
            self.assertLess(bias, threshold, message)

    def _check_enumerate_support(self, dist, examples):
        for param, expected in examples:
            param = torch.Tensor(param)
            expected = torch.Tensor(expected)
            actual = dist(param).enumerate_support()
            self.assertEqual(actual, expected)
            param = Variable(param)
            expected = Variable(expected)
            actual = dist(param).enumerate_support()
            self.assertEqual(actual, expected)

    def test_bernoulli(self):
        p = Variable(torch.Tensor([0.7, 0.2, 0.4]), requires_grad=True)
        r = Variable(torch.Tensor([0.3]), requires_grad=True)
        s = 0.3
        self.assertEqual(Bernoulli(p).sample_n(8).size(), (8, 3))
        self.assertEqual(Bernoulli(r).sample_n(8).size(), (8, 1))
        self.assertEqual(Bernoulli(r).sample().size(), (1,))
        self.assertEqual(Bernoulli(r).sample((3, 2)).size(), (3, 2, 1))
        self.assertEqual(Bernoulli(s).sample().size(), (1,))
        self._gradcheck_log_prob(Bernoulli, (p,))

        def ref_log_prob(idx, val, log_prob):
            prob = p.data[idx]
            self.assertEqual(log_prob, math.log(prob if val else 1 - prob))

        self._check_log_prob(Bernoulli(p), ref_log_prob)
        self.assertRaises(NotImplementedError, Bernoulli(r).rsample)

    def test_bernoulli_enumerate_support(self):
        examples = [
            ([0.1], [[0], [1]]),
            ([0.1, 0.9], [[0, 0], [1, 1]]),
            ([[0.1, 0.2], [0.3, 0.4]], [[[0, 0], [0, 0]], [[1, 1], [1, 1]]]),
        ]
        self._check_enumerate_support(Bernoulli, examples)

    def test_bernoulli_3d(self):
        p = Variable(torch.Tensor(2, 3, 5).fill_(0.5), requires_grad=True)
        self.assertEqual(Bernoulli(p).sample().size(), (2, 3, 5))
        self.assertEqual(Bernoulli(p).sample(sample_shape=(2, 5)).size(),
                         (2, 5, 2, 3, 5))
        self.assertEqual(Bernoulli(p).sample_n(2).size(), (2, 2, 3, 5))

    def test_categorical_1d(self):
        p = Variable(torch.Tensor([0.1, 0.2, 0.3]), requires_grad=True)
        # TODO: this should return a 0-dim tensor once we have Scalar support
        self.assertEqual(Categorical(p).sample().size(), (1,))
        self.assertEqual(Categorical(p).sample((2, 2)).size(), (2, 2))
        self.assertEqual(Categorical(p).sample_n(1).size(), (1,))
        self._gradcheck_log_prob(Categorical, (p,))
        self.assertRaises(NotImplementedError, Categorical(p).rsample)

    def test_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = Variable(torch.Tensor(probabilities), requires_grad=True)
        s = Variable(torch.Tensor(probabilities_1), requires_grad=True)
        self.assertEqual(Categorical(p).sample().size(), (2,))
        self.assertEqual(Categorical(p).sample(sample_shape=(3, 4)).size(), (3, 4, 2))
        self.assertEqual(Categorical(p).sample_n(6).size(), (6, 2))
        self._gradcheck_log_prob(Categorical, (p,))

        # sample check for extreme value of probs
        set_rng_seed(0)
        self.assertEqual(Categorical(s).sample(sample_shape=(2,)).data,
                         torch.Tensor([[0, 1], [0, 1]]))

        def ref_log_prob(idx, val, log_prob):
            sample_prob = p.data[idx][val] / p.data[idx].sum()
            self.assertEqual(log_prob, math.log(sample_prob))

        self._check_log_prob(Categorical(p), ref_log_prob)

    def test_categorical_enumerate_support(self):
        examples = [
            ([0.1, 0.2, 0.7], [0, 1, 2]),
            ([[0.1, 0.9], [0.3, 0.7]], [[0, 0], [1, 1]]),
        ]
        self._check_enumerate_support(Categorical, examples)

    def test_one_hot_categorical_1d(self):
        p = Variable(torch.Tensor([0.1, 0.2, 0.3]), requires_grad=True)
        self.assertEqual(OneHotCategorical(p).sample().size(), (3,))
        self.assertEqual(OneHotCategorical(p).sample((2, 2)).size(), (2, 2, 3))
        self.assertEqual(OneHotCategorical(p).sample_n(1).size(), (1, 3))
        self._gradcheck_log_prob(OneHotCategorical, (p,))
        self.assertRaises(NotImplementedError, OneHotCategorical(p).rsample)

    def test_one_hot_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = Variable(torch.Tensor(probabilities), requires_grad=True)
        s = Variable(torch.Tensor(probabilities_1), requires_grad=True)
        self.assertEqual(OneHotCategorical(p).sample().size(), (2, 3))
        self.assertEqual(OneHotCategorical(p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3))
        self.assertEqual(OneHotCategorical(p).sample_n(6).size(), (6, 2, 3))
        self._gradcheck_log_prob(OneHotCategorical, (p,))

        dist = OneHotCategorical(p)
        x = dist.sample()
        self.assertEqual(dist.log_prob(x), Categorical(p).log_prob(x.max(-1)[1]))

    def test_one_hot_categorical_enumerate_support(self):
        examples = [
            ([0.1, 0.2, 0.7], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ([[0.1, 0.9], [0.3, 0.7]], [[[1, 0], [1, 0]], [[0, 1], [0, 1]]]),
        ]
        self._check_enumerate_support(OneHotCategorical, examples)

    def test_uniform(self):
        low = Variable(torch.zeros(5, 5), requires_grad=True)
        high = Variable(torch.ones(5, 5) * 3, requires_grad=True)
        low_1d = Variable(torch.zeros(1), requires_grad=True)
        high_1d = Variable(torch.ones(1) * 3, requires_grad=True)
        self.assertEqual(Uniform(low, high).sample().size(), (5, 5))
        self.assertEqual(Uniform(low, high).sample_n(7).size(), (7, 5, 5))
        self.assertEqual(Uniform(low_1d, high_1d).sample().size(), (1,))
        self.assertEqual(Uniform(low_1d, high_1d).sample_n(1).size(), (1, 1))
        self.assertEqual(Uniform(0.0, 1.0).sample_n(1).size(), (1,))

        # Check log_prob computation when value outside range
        uniform = Uniform(low_1d, high_1d)
        above_high = Variable(torch.Tensor([4.0]))
        below_low = Variable(torch.Tensor([-1.0]))
        self.assertEqual(uniform.log_prob(above_high).data[0], -float('inf'), allow_inf=True)
        self.assertEqual(uniform.log_prob(below_low).data[0], -float('inf'), allow_inf=True)

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

    def test_cauchy(self):
        loc = Variable(torch.zeros(5, 5), requires_grad=True)
        scale = Variable(torch.ones(5, 5), requires_grad=True)
        loc_1d = Variable(torch.zeros(1), requires_grad=True)
        scale_1d = Variable(torch.ones(1), requires_grad=True)
        self.assertEqual(Cauchy(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Cauchy(loc, scale).sample_n(7).size(), (7, 5, 5))
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample_n(1).size(), (1, 1))
        self.assertEqual(Cauchy(0.0, 1.0).sample_n(1).size(), (1,))

        set_rng_seed(1)
        self._gradcheck_log_prob(Uniform, (loc, scale))
        self._gradcheck_log_prob(Uniform, (loc, 1.0))
        self._gradcheck_log_prob(Uniform, (0.0, scale))

        state = torch.get_rng_state()
        eps = loc.new(loc.size()).cauchy_()
        torch.set_rng_state(state)
        c = Cauchy(loc, scale).rsample()
        c.backward(torch.ones_like(c))
        self.assertEqual(loc.grad, torch.ones_like(scale))
        self.assertEqual(scale.grad, eps)
        loc.grad.zero_()
        scale.grad.zero_()

    def test_normal(self):
        mean = Variable(torch.randn(5, 5), requires_grad=True)
        std = Variable(torch.randn(5, 5).abs(), requires_grad=True)
        mean_1d = Variable(torch.randn(1), requires_grad=True)
        std_1d = Variable(torch.randn(1), requires_grad=True)
        mean_delta = torch.Tensor([1.0, 0.0])
        std_delta = torch.Tensor([1e-5, 1e-5])
        self.assertEqual(Normal(mean, std).sample().size(), (5, 5))
        self.assertEqual(Normal(mean, std).sample_n(7).size(), (7, 5, 5))
        self.assertEqual(Normal(mean_1d, std_1d).sample_n(1).size(), (1, 1))
        self.assertEqual(Normal(mean_1d, std_1d).sample().size(), (1,))
        self.assertEqual(Normal(0.2, .6).sample_n(1).size(), (1,))
        self.assertEqual(Normal(-0.7, 50.0).sample_n(1).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(Normal(mean_delta, std_delta).sample(sample_shape=(1, 2)),
                         torch.Tensor([[[1.0, 0.0], [1.0, 0.0]]]),
                         prec=1e-4)

        self._gradcheck_log_prob(Normal, (mean, std))
        self._gradcheck_log_prob(Normal, (mean, 1.0))
        self._gradcheck_log_prob(Normal, (0.0, std))

        state = torch.get_rng_state()
        eps = torch.normal(torch.zeros_like(mean), torch.ones_like(std))
        torch.set_rng_state(state)
        z = Normal(mean, std).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(mean.grad, torch.ones_like(mean))
        self.assertEqual(std.grad, eps)
        mean.grad.zero_()
        std.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            m = mean.data.view(-1)[idx]
            s = std.data.view(-1)[idx]
            expected = (math.exp(-(x - m) ** 2 / (2 * s ** 2)) /
                        math.sqrt(2 * math.pi * s ** 2))
            self.assertAlmostEqual(log_prob, math.log(expected), places=3)

        self._check_log_prob(Normal(mean, std), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_normal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for mean, std in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Normal(mean, std),
                                        scipy.stats.norm(loc=mean, scale=std),
                                        'Normal(mean={}, std={})'.format(mean, std))

    def test_exponential(self):
        rate = Variable(torch.randn(5, 5).abs(), requires_grad=True)
        rate_1d = Variable(torch.randn(1).abs(), requires_grad=True)
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
            m = rate.data.view(-1)[idx]
            expected = math.log(m) - m * x
            self.assertAlmostEqual(log_prob, expected, places=3)

        self._check_log_prob(Exponential(rate), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_exponential_sample(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for rate in [1e-5, 1.0, 10.]:
            self._check_sampler_sampler(Exponential(rate),
                                        scipy.stats.expon(scale=1. / rate),
                                        'Exponential(rate={})'.format(rate))

    def test_laplace(self):
        loc = Variable(torch.randn(5, 5), requires_grad=True)
        scale = Variable(torch.randn(5, 5).abs(), requires_grad=True)
        loc_1d = Variable(torch.randn(1), requires_grad=True)
        scale_1d = Variable(torch.randn(1), requires_grad=True)
        loc_delta = torch.Tensor([1.0, 0.0])
        scale_delta = torch.Tensor([1e-5, 1e-5])
        self.assertEqual(Laplace(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Laplace(loc, scale).sample_n(7).size(), (7, 5, 5))
        self.assertEqual(Laplace(loc_1d, scale_1d).sample_n(1).size(), (1, 1))
        self.assertEqual(Laplace(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Laplace(0.2, .6).sample_n(1).size(), (1,))
        self.assertEqual(Laplace(-0.7, 50.0).sample_n(1).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(0)
        self.assertEqual(Laplace(loc_delta, scale_delta).sample(sample_shape=(1, 2)),
                         torch.Tensor([[[1.0, 0.0], [1.0, 0.0]]]),
                         prec=1e-4)

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
            m = loc.data.view(-1)[idx]
            s = scale.data.view(-1)[idx]
            expected = (-math.log(2 * s) - abs(x - m) / s)
            self.assertAlmostEqual(log_prob, expected, places=3)

        self._check_log_prob(Laplace(loc, scale), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_laplace_sample(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for loc, scale in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Laplace(loc, scale),
                                        scipy.stats.laplace(loc=loc, scale=scale),
                                        'Laplace(loc={}, scale={})'.format(loc, scale))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_gamma_shape(self):
        alpha = Variable(torch.exp(torch.randn(2, 3)), requires_grad=True)
        beta = Variable(torch.exp(torch.randn(2, 3)), requires_grad=True)
        alpha_1d = Variable(torch.exp(torch.randn(1)), requires_grad=True)
        beta_1d = Variable(torch.exp(torch.randn(1)), requires_grad=True)
        self.assertEqual(Gamma(alpha, beta).sample().size(), (2, 3))
        self.assertEqual(Gamma(alpha, beta).sample_n(5).size(), (5, 2, 3))
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample_n(1).size(), (1, 1))
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample().size(), (1,))
        self.assertEqual(Gamma(0.5, 0.5).sample().size(), (1,))
        self.assertEqual(Gamma(0.5, 0.5).sample_n(1).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            a = alpha.data.view(-1)[idx]
            b = beta.data.view(-1)[idx]
            expected = scipy.stats.gamma.logpdf(x, a, scale=1 / b)
            self.assertAlmostEqual(log_prob, expected, places=3)

        self._check_log_prob(Gamma(alpha, beta), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_gamma_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for alpha, beta in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Gamma(alpha, beta),
                                        scipy.stats.gamma(alpha, scale=1.0 / beta),
                                        'Gamma(alpha={}, beta={})'.format(alpha, beta))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_gamma_sample_grad(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        num_samples = 100
        for alpha in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            alphas = Variable(torch.FloatTensor([alpha] * num_samples), requires_grad=True)
            betas = Variable(torch.ones(num_samples).type_as(alphas))
            x = Gamma(alphas, betas).rsample()
            x.sum().backward()
            x, ind = x.data.sort()
            x = x.numpy()
            actual_grad = alphas.grad.data[ind].numpy()
            # Compare with expected gradient dx/dalpha along constant cdf(x,alpha).
            cdf = scipy.stats.gamma.cdf
            pdf = scipy.stats.gamma.pdf
            eps = 0.01 * alpha / (1.0 + alpha ** 0.5)
            cdf_alpha = (cdf(x, alpha + eps) - cdf(x, alpha - eps)) / (2 * eps)
            cdf_x = pdf(x, alpha)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(np.max(rel_error), 0.0005,
                            '\n'.join(['Bad gradients for Gamma({}, 1)'.format(alpha),
                                       'x {}'.format(x),
                                       'expected {}'.format(expected_grad),
                                       'actual {}'.format(actual_grad),
                                       'rel error {}'.format(rel_error),
                                       'max error {}'.format(rel_error.max()),
                                       'at alpha={}, x={}'.format(alpha, x[rel_error.argmax()])]))

    def test_dirichlet_shape(self):
        alpha = Variable(torch.exp(torch.randn(2, 3)), requires_grad=True)
        alpha_1d = Variable(torch.exp(torch.randn(4)), requires_grad=True)
        self.assertEqual(Dirichlet(alpha).sample().size(), (2, 3))
        self.assertEqual(Dirichlet(alpha).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Dirichlet(alpha_1d).sample().size(), (4,))
        self.assertEqual(Dirichlet(alpha_1d).sample((1,)).size(), (1, 4))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_dirichlet_log_prob(self):
        num_samples = 10
        alpha = torch.exp(torch.randn(5))
        dist = Dirichlet(alpha)
        x = dist.sample((num_samples,))
        actual_log_prob = dist.log_prob(x)
        for i in range(num_samples):
            expected_log_prob = scipy.stats.dirichlet.logpdf(x[i].numpy(), alpha.numpy())
            self.assertAlmostEqual(actual_log_prob[i], expected_log_prob, places=3)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_dirichlet_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        alpha = torch.exp(torch.randn(3))
        self._check_sampler_sampler(Dirichlet(alpha),
                                    scipy.stats.dirichlet(alpha.numpy()),
                                    'Dirichlet(alpha={})'.format(list(alpha)),
                                    multivariate=True)

    def test_beta_shape(self):
        alpha = Variable(torch.exp(torch.randn(2, 3)), requires_grad=True)
        beta = Variable(torch.exp(torch.randn(2, 3)), requires_grad=True)
        alpha_1d = Variable(torch.exp(torch.randn(4)), requires_grad=True)
        beta_1d = Variable(torch.exp(torch.randn(4)), requires_grad=True)
        self.assertEqual(Beta(alpha, beta).sample().size(), (2, 3))
        self.assertEqual(Beta(alpha, beta).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Beta(alpha_1d, beta_1d).sample().size(), (4,))
        self.assertEqual(Beta(alpha_1d, beta_1d).sample((1,)).size(), (1, 4))
        self.assertEqual(Beta(0.1, 0.3).sample().size(), (1,))
        self.assertEqual(Beta(0.1, 0.3).sample((5,)).size(), (5,))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_beta_log_prob(self):
        for _ in range(100):
            alpha = np.exp(np.random.normal())
            beta = np.exp(np.random.normal())
            dist = Beta(alpha, beta)
            x = dist.sample()
            actual_log_prob = dist.log_prob(x).sum()
            expected_log_prob = scipy.stats.beta.logpdf(x, alpha, beta)[0]
            self.assertAlmostEqual(actual_log_prob, expected_log_prob, places=3, allow_inf=True)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_beta_sample(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for alpha, beta in product([0.1, 1.0, 10.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Beta(alpha, beta),
                                        scipy.stats.beta(alpha, beta),
                                        'Beta(alpha={}, beta={})'.format(alpha, beta))
        # Check that small alphas do not cause NANs.
        for Tensor in [torch.FloatTensor, torch.DoubleTensor]:
            x = Beta(Tensor([1e-6]), Tensor([1e-6])).sample()[0]
            self.assertTrue(np.isfinite(x) and x > 0, 'Invalid Beta.sample(): {}'.format(x))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_beta_sample_grad(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        num_samples = 20
        for alpha, beta in product([1e-2, 1e0, 1e2], [1e-2, 1e0, 1e2]):
            alphas = Variable(torch.Tensor([alpha] * num_samples), requires_grad=True)
            betas = Variable(torch.Tensor([beta] * num_samples))
            x = Beta(alphas, betas).rsample()
            x.sum().backward()
            x, ind = x.data.sort()
            x = x.numpy()
            actual_grad = alphas.grad.data[ind].numpy()
            # Compare with expected gradient dx/dalpha along constant cdf(x,alpha,beta).
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            eps = 0.02 * alpha / (1.0 + np.sqrt(alpha))
            cdf_alpha = (cdf(x, alpha + eps, beta) - cdf(x, alpha - eps, beta)) / (2 * eps)
            cdf_x = pdf(x, alpha, beta)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-100)
            self.assertLess(np.max(rel_error), 0.01,
                            '\n'.join(['Bad gradients for Beta({}, {})'.format(alpha, beta),
                                       'x {}'.format(x),
                                       'expected {}'.format(expected_grad),
                                       'actual {}'.format(actual_grad),
                                       'rel error {}'.format(rel_error),
                                       'max error {}'.format(rel_error.max())]))

    def test_valid_parameter_broadcasting(self):
        # Test correct broadcasting of parameter sizes for distributions that have multiple
        # parameters.
        # example type (distribution instance, expected sample shape)
        valid_examples = [
            (Normal(mean=torch.Tensor([0, 0]), std=1),
             (2,)),
            (Normal(mean=0, std=torch.Tensor([1, 1])),
             (2,)),
            (Normal(mean=torch.Tensor([0, 0]), std=torch.Tensor([1])),
             (2,)),
            (Normal(mean=torch.Tensor([0, 0]), std=torch.Tensor([[1], [1]])),
             (2, 2)),
            (Normal(mean=torch.Tensor([0, 0]), std=torch.Tensor([[1]])),
             (1, 2)),
            (Normal(mean=torch.Tensor([0]), std=torch.Tensor([[1]])),
             (1, 1)),
            (Gamma(alpha=torch.Tensor([1, 1]), beta=1),
             (2,)),
            (Gamma(alpha=1, beta=torch.Tensor([1, 1])),
             (2,)),
            (Gamma(alpha=torch.Tensor([1, 1]), beta=torch.Tensor([[1], [1], [1]])),
             (3, 2)),
            (Gamma(alpha=torch.Tensor([1, 1]), beta=torch.Tensor([[1], [1]])),
             (2, 2)),
            (Gamma(alpha=torch.Tensor([1, 1]), beta=torch.Tensor([[1]])),
             (1, 2)),
            (Gamma(alpha=torch.Tensor([1]), beta=torch.Tensor([[1]])),
             (1, 1)),
            (Laplace(loc=torch.Tensor([0, 0]), scale=1),
             (2,)),
            (Laplace(loc=0, scale=torch.Tensor([1, 1])),
             (2,)),
            (Laplace(loc=torch.Tensor([0, 0]), scale=torch.Tensor([1])),
             (2,)),
            (Laplace(loc=torch.Tensor([0, 0]), scale=torch.Tensor([[1], [1]])),
             (2, 2)),
            (Laplace(loc=torch.Tensor([0, 0]), scale=torch.Tensor([[1]])),
             (1, 2)),
            (Laplace(loc=torch.Tensor([0]), scale=torch.Tensor([[1]])),
             (1, 1)),
        ]

        for dist, expected_size in valid_examples:
            dist_sample_size = dist.sample().size()
            self.assertEqual(dist_sample_size, expected_size,
                             'actual size: {} != expected size: {}'.format(dist_sample_size, expected_size))

    def test_invalid_parameter_broadcasting(self):
        # invalid broadcasting cases; should throw error
        # example type (distribution class, distribution params)
        invalid_examples = [
            (Normal, {
                'mean': torch.Tensor([[0, 0]]),
                'std': torch.Tensor([1, 1, 1, 1])
            }),
            (Normal, {
                'mean': torch.Tensor([[[0, 0, 0], [0, 0, 0]]]),
                'std': torch.Tensor([1, 1])
            }),
            (Gamma, {
                'alpha': torch.Tensor([0, 0]),
                'beta': torch.Tensor([1, 1, 1])
            }),
            (Laplace, {
                'loc': torch.Tensor([0, 0]),
                'scale': torch.Tensor([1, 1, 1])
            })
        ]

        for dist, kwargs in invalid_examples:
            self.assertRaises(RuntimeError, dist, **kwargs)


class TestDistributionShapes(TestCase):
    def setUp(self):
        super(TestCase, self).setUp()
        self.scalar_sample = 1
        self.tensor_sample_1 = torch.ones(3, 2)
        self.tensor_sample_2 = torch.ones(3, 2, 3)

    def test_entropy_shape(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                actual_shape = dist.entropy().size()
                expected_shape = dist._batch_shape
                if not expected_shape:
                    expected_shape = torch.Size((1,))  # TODO Remove this once scalars are supported.
                message = '{} example {}/{}, shape mismatch. expected {}, actual {}'.format(
                    Dist.__name__, i, len(params), expected_shape, actual_shape)
                self.assertEqual(actual_shape, expected_shape, message=message)

    def test_bernoulli_shape_scalar_params(self):
        bernoulli = Bernoulli(0.3)
        self.assertEqual(bernoulli._batch_shape, torch.Size())
        self.assertEqual(bernoulli._event_shape, torch.Size())
        self.assertEqual(bernoulli.sample().size(), torch.Size((1,)))
        self.assertEqual(bernoulli.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, bernoulli.log_prob, self.scalar_sample)
        self.assertEqual(bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(bernoulli.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_bernoulli_shape_tensor_params(self):
        bernoulli = Bernoulli(torch.Tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(bernoulli._batch_shape, torch.Size((3, 2)))
        self.assertEqual(bernoulli._event_shape, torch.Size(()))
        self.assertEqual(bernoulli.sample().size(), torch.Size((3, 2)))
        self.assertEqual(bernoulli.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, bernoulli.log_prob, self.tensor_sample_2)

    def test_beta_shape_scalar_params(self):
        dist = Beta(0.1, 0.1)
        self.assertEqual(dist._batch_shape, torch.Size())
        self.assertEqual(dist._event_shape, torch.Size())
        self.assertEqual(dist.sample().size(), torch.Size((1,)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, dist.log_prob, self.scalar_sample)
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_beta_shape_tensor_params(self):
        dist = Beta(torch.Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                    torch.Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
        self.assertEqual(dist._batch_shape, torch.Size((3, 2)))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)

    def test_categorical_shape(self):
        dist = Categorical(torch.Tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size((3,)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)
        self.assertEqual(dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_one_hot_categorical_shape(self):
        dist = OneHotCategorical(torch.Tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        self.assertEqual(dist.log_prob(dist.enumerate_support()).size(), torch.Size((2, 3)))

    def test_cauchy_shape_scalar_params(self):
        cauchy = Cauchy(0, 1)
        self.assertEqual(cauchy._batch_shape, torch.Size())
        self.assertEqual(cauchy._event_shape, torch.Size())
        self.assertEqual(cauchy.sample().size(), torch.Size((1,)))
        self.assertEqual(cauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, cauchy.log_prob, self.scalar_sample)
        self.assertEqual(cauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(cauchy.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_cauchy_shape_tensor_params(self):
        cauchy = Cauchy(torch.Tensor([0, 0]), torch.Tensor([1, 1]))
        self.assertEqual(cauchy._batch_shape, torch.Size((2,)))
        self.assertEqual(cauchy._event_shape, torch.Size(()))
        self.assertEqual(cauchy.sample().size(), torch.Size((2,)))
        self.assertEqual(cauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2)))
        self.assertEqual(cauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, cauchy.log_prob, self.tensor_sample_2)

    def test_dirichlet_shape(self):
        dist = Dirichlet(torch.Tensor([[0.6, 0.3], [1.6, 1.3], [2.6, 2.3]]))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((5, 4)).size(), torch.Size((5, 4, 3, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)

    def test_gamma_shape_scalar_params(self):
        gamma = Gamma(1, 1)
        self.assertEqual(gamma._batch_shape, torch.Size())
        self.assertEqual(gamma._event_shape, torch.Size())
        self.assertEqual(gamma.sample().size(), torch.Size((1,)))
        self.assertEqual(gamma.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, gamma.log_prob, self.scalar_sample)
        self.assertEqual(gamma.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(gamma.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_gamma_shape_tensor_params(self):
        gamma = Gamma(torch.Tensor([1, 1]), torch.Tensor([1, 1]))
        self.assertEqual(gamma._batch_shape, torch.Size((2,)))
        self.assertEqual(gamma._event_shape, torch.Size(()))
        self.assertEqual(gamma.sample().size(), torch.Size((2,)))
        self.assertEqual(gamma.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(gamma.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, gamma.log_prob, self.tensor_sample_2)

    def test_normal_shape_scalar_params(self):
        normal = Normal(0, 1)
        self.assertEqual(normal._batch_shape, torch.Size())
        self.assertEqual(normal._event_shape, torch.Size())
        self.assertEqual(normal.sample().size(), torch.Size((1,)))
        self.assertEqual(normal.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, normal.log_prob, self.scalar_sample)
        self.assertEqual(normal.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(normal.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_normal_shape_tensor_params(self):
        normal = Normal(torch.Tensor([0, 0]), torch.Tensor([1, 1]))
        self.assertEqual(normal._batch_shape, torch.Size((2,)))
        self.assertEqual(normal._event_shape, torch.Size(()))
        self.assertEqual(normal.sample().size(), torch.Size((2,)))
        self.assertEqual(normal.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(normal.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, normal.log_prob, self.tensor_sample_2)

    def test_uniform_shape_scalar_params(self):
        uniform = Uniform(0, 1)
        self.assertEqual(uniform._batch_shape, torch.Size())
        self.assertEqual(uniform._event_shape, torch.Size())
        self.assertEqual(uniform.sample().size(), torch.Size((1,)))
        self.assertEqual(uniform.sample(torch.Size((3, 2))).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, uniform.log_prob, self.scalar_sample)
        self.assertEqual(uniform.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(uniform.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_uniform_shape_tensor_params(self):
        uniform = Uniform(torch.Tensor([0, 0]), torch.Tensor([1, 1]))
        self.assertEqual(uniform._batch_shape, torch.Size((2,)))
        self.assertEqual(uniform._event_shape, torch.Size(()))
        self.assertEqual(uniform.sample().size(), torch.Size((2,)))
        self.assertEqual(uniform.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2)))
        self.assertEqual(uniform.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, uniform.log_prob, self.tensor_sample_2)

    def test_exponential_shape_scalar_param(self):
        expon = Exponential(1.)
        self.assertEqual(expon._batch_shape, torch.Size())
        self.assertEqual(expon._event_shape, torch.Size())
        self.assertEqual(expon.sample().size(), torch.Size((1,)))
        self.assertEqual(expon.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, expon.log_prob, self.scalar_sample)
        self.assertEqual(expon.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(expon.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_exponential_shape_tensor_param(self):
        expon = Exponential(torch.Tensor([1, 1]))
        self.assertEqual(expon._batch_shape, torch.Size((2,)))
        self.assertEqual(expon._event_shape, torch.Size(()))
        self.assertEqual(expon.sample().size(), torch.Size((2,)))
        self.assertEqual(expon.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(expon.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, expon.log_prob, self.tensor_sample_2)

    def test_laplace_shape_scalar_params(self):
        laplace = Laplace(0, 1)
        self.assertEqual(laplace._batch_shape, torch.Size())
        self.assertEqual(laplace._event_shape, torch.Size())
        self.assertEqual(laplace.sample().size(), torch.Size((1,)))
        self.assertEqual(laplace.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, laplace.log_prob, self.scalar_sample)
        self.assertEqual(laplace.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(laplace.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

    def test_laplace_shape_tensor_params(self):
        laplace = Laplace(torch.Tensor([0, 0]), torch.Tensor([1, 1]))
        self.assertEqual(laplace._batch_shape, torch.Size((2,)))
        self.assertEqual(laplace._event_shape, torch.Size(()))
        self.assertEqual(laplace.sample().size(), torch.Size((2,)))
        self.assertEqual(laplace.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(laplace.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, laplace.log_prob, self.tensor_sample_2)


if __name__ == '__main__':
    run_tests()
