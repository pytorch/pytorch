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
from torch.autograd import Variable, grad, gradcheck
from torch.distributions import (LogNormal, Normal, TransformedDistribution,
                                 Categorical, OneHotCategorical, Multinomial, HalfNormal)
from torch.distributions.transforms import *
from torch.distributions.constraints import Constraint, is_dependent
from torch.distributions.utils import _finfo, probs_to_logits, logits_to_probs

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
    Example(LogNormal, [
        {
            'loc': Variable(torch.randn(5, 5), requires_grad=True),
            'scale': Variable(torch.randn(5, 5).abs(), requires_grad=True),
        },
        {
            'loc': Variable(torch.randn(1), requires_grad=True),
            'scale': Variable(torch.randn(1).abs(), requires_grad=True),
        },
        {
            'loc': torch.Tensor([1.0, 0.0]),
            'scale': torch.Tensor([1e-5, 1e-5]),
        },
    ]),
    Example(HalfNormal, [
        {'scale': Variable(torch.randn(5, 5).abs(), requires_grad=True)},
        {'scale': Variable(torch.randn(1).abs(), requires_grad=True)},
        {'scale': torch.Tensor([1e-5, 1e-5])},
    ]),
]


def unwrap(value):
    if isinstance(value, Variable):
        return value.data
    return value


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

    def test_enumerate_support_type(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    self.assertTrue(type(unwrap(dist.sample())) is type(unwrap(dist.enumerate_support())),
                                    msg=('{} example {}/{}, return type mismatch between ' +
                                         'sample and enumerate_support.').format(Dist.__name__, i, len(params)))
                except NotImplementedError:
                    pass

    def test_lognormal(self):
        mean = Variable(torch.randn(5, 5), requires_grad=True)
        std = Variable(torch.randn(5, 5).abs(), requires_grad=True)
        mean_1d = Variable(torch.randn(1), requires_grad=True)
        std_1d = Variable(torch.randn(1), requires_grad=True)
        mean_delta = torch.Tensor([1.0, 0.0])
        std_delta = torch.Tensor([1e-5, 1e-5])
        self.assertEqual(LogNormal(mean, std).sample().size(), (5, 5))
        self.assertEqual(LogNormal(mean, std).sample_n(7).size(), (7, 5, 5))
        self.assertEqual(LogNormal(mean_1d, std_1d).sample_n(1).size(), (1, 1))
        self.assertEqual(LogNormal(mean_1d, std_1d).sample().size(), (1,))
        self.assertEqual(LogNormal(0.2, .6).sample_n(1).size(), (1,))
        self.assertEqual(LogNormal(-0.7, 50.0).sample_n(1).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(LogNormal(mean_delta, std_delta).sample(sample_shape=(1, 2)),
                         torch.Tensor([[[math.exp(1), 1.0], [math.exp(1), 1.0]]]),
                         prec=1e-4)

        self._gradcheck_log_prob(LogNormal, (mean, std))
        self._gradcheck_log_prob(LogNormal, (mean, 1.0))
        self._gradcheck_log_prob(LogNormal, (0.0, std))

        def ref_log_prob(idx, x, log_prob):
            m = mean.data.view(-1)[idx]
            s = std.data.view(-1)[idx]
            expected = scipy.stats.lognorm(s=s, scale=math.exp(m)).logpdf(x)
            self.assertAlmostEqual(log_prob, expected, places=3)

        self._check_log_prob(LogNormal(mean, std), ref_log_prob)

    def test_entropy_monte_carlo(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    actual = dist.entropy()
                except NotImplementedError:
                    continue
                x = dist.sample(sample_shape=(20000,))
                expected = -dist.log_prob(x).mean(0)
                if isinstance(actual, Variable):
                    actual = actual.data
                    expected = expected.data
                ignore = (expected == float('inf'))
                expected[ignore] = actual[ignore]
                self.assertEqual(actual, expected, prec=0.2, message='\n'.join([
                    '{} example {}/{}, incorrect .entropy().'.format(Dist.__name__, i, len(params)),
                    'Expected (monte carlo) {}'.format(expected),
                    'Actual (analytic) {}'.format(actual),
                    'max error = {}'.format(torch.abs(actual - expected).max()),
                ]))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lognormal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for mean, std in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(LogNormal(mean, std),
                                        scipy.stats.lognorm(scale=math.exp(mean), s=std),
                                        'LogNormal(loc={}, scale={})'.format(mean, std))

    def test_halfnormal(self):
        std = Variable(torch.randn(5, 5).abs(), requires_grad=True)
        std_1d = Variable(torch.randn(1), requires_grad=True)
        std_delta = torch.Tensor([1e-5, 1e-5])
        self.assertEqual(HalfNormal(std).sample().size(), (5, 5))
        self.assertEqual(HalfNormal(std).sample_n(7).size(), (7, 5, 5))
        self.assertEqual(HalfNormal(std_1d).sample_n(1).size(), (1, 1))
        self.assertEqual(HalfNormal(std_1d).sample().size(), (1,))
        self.assertEqual(HalfNormal(.6).sample_n(1).size(), (1,))
        self.assertEqual(HalfNormal(50.0).sample_n(1).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(HalfNormal(std_delta).sample(sample_shape=(1, 2)),
                         torch.Tensor([[[0, 0], [0, 0]]]),
                         prec=1e-4)

        self._gradcheck_log_prob(HalfNormal, (std,))

        def ref_log_prob(idx, x, log_prob):
            s = std.data.view(-1)[idx]
            expected = scipy.stats.halfnorm(scale=s).logpdf(x)
            self.assertAlmostEqual(log_prob, expected, places=3)

        self._check_log_prob(HalfNormal(std), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_halfnormal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for std in [0.1, 1.0, 10.0]:
            self._check_sampler_sampler(HalfNormal(std),
                                        scipy.stats.halfnorm(scale=std),
                                        'HalfNormal(scale={})'.format(std))

    def test_normal_no_transforms(self):
        mean = Variable(torch.randn(5, 5), requires_grad=True)
        std = Variable(torch.randn(5, 5).abs(), requires_grad=True)
        mean_1d = Variable(torch.randn(1), requires_grad=True)
        std_1d = Variable(torch.randn(1), requires_grad=True)
        mean_delta = torch.Tensor([1.0, 0.0])
        std_delta = torch.Tensor([1e-5, 1e-5])
        dist = TransformedDistribution(Normal(mean, std))
        self.assertEqual(dist.sample().size(), (5, 5))
        self.assertEqual(dist.sample_n(7).size(), (7, 5, 5))
        dist = TransformedDistribution(Normal(mean_1d, std_1d))
        self.assertEqual(dist.sample_n(1).size(), (1, 1))
        self.assertEqual(dist.sample().size(), (1,))
        self.assertEqual(TransformedDistribution(Normal(0.2, .6)).sample_n(1).size(), (1,))
        self.assertEqual(TransformedDistribution(Normal(-0.7, 50.0)).sample_n(1).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(TransformedDistribution(Normal(mean_delta, std_delta)).sample(sample_shape=(1, 2)),
                         torch.Tensor([[[1.0, 0.0], [1.0, 0.0]]]),
                         prec=1e-4)

        self._gradcheck_log_prob(lambda x, y: TransformedDistribution(Normal(x, y)), (mean, std))
        self._gradcheck_log_prob(lambda x, y: TransformedDistribution(Normal(x, y)), (mean, 1.0))
        self._gradcheck_log_prob(lambda x, y: TransformedDistribution(Normal(x, y)), (0.0, std))

        def ref_log_prob(idx, x, log_prob):
            m = mean.data.view(-1)[idx]
            s = std.data.view(-1)[idx]
            expected = scipy.stats.norm(loc=m, scale=s).logpdf(x)
            self.assertAlmostEqual(log_prob, expected, places=3)

        self._check_log_prob(TransformedDistribution(Normal(mean, std)), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_normal_no_transforms_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for mean, std in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(TransformedDistribution(Normal(mean, std)),
                                        scipy.stats.norm(loc=mean, scale=std),
                                        'TransformedDistribution(Normal(loc={}, scale={}))'.format(mean, std))


class TestConstraints(TestCase):
    def test_params_contains(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                for name, value in param.items():
                    if not (torch.is_tensor(value) or isinstance(value, Variable)):
                        value = torch.Tensor([value])
                    if Dist in (Categorical, OneHotCategorical, Multinomial) and name == 'probs':
                        # These distributions accept positive probs, but elsewhere we
                        # use a stricter constraint to the simplex.
                        value = value / value.sum(-1, True)
                    try:
                        constraint = dist.params[name]
                    except KeyError:
                        continue  # ignore optional parameters
                    if is_dependent(constraint):
                        continue
                    message = '{} example {}/{} parameter {} = {}'.format(
                        Dist.__name__, i, len(params), name, value)
                    self.assertTrue(constraint.check(value).all(), msg=message)

    def test_support_contains(self):
        for Dist, params in EXAMPLES:
            self.assertIsInstance(Dist.support, Constraint)
            for i, param in enumerate(params):
                dist = Dist(**param)
                value = dist.sample()
                constraint = dist.support
                message = '{} example {}/{} sample = {}'.format(
                    Dist.__name__, i, len(params), value)
                self.assertTrue(constraint.check(value).all(), msg=message)


if __name__ == '__main__':
    run_tests()
