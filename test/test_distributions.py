from common import TestCase, run_tests
import math
import torch
import unittest
from itertools import product
from torch.autograd import Variable, gradcheck
from torch.distributions import Bernoulli, Categorical, Normal, Gamma

TEST_NUMPY = True
try:
    import numpy as np
    import scipy.stats
    import scipy.special
except ImportError:
    TEST_NUMPY = False


class TestDistributions(TestCase):
    def _set_rng_seed(self, seed=0):
        torch.manual_seed(seed)
        if TEST_NUMPY:
            np.random.seed(seed)

    def _gradcheck_log_prob(self, dist_ctor, ctor_params):
        # performs gradient checks on log_prob
        distribution = dist_ctor(*ctor_params)
        s = distribution.sample()

        self.assertEqual(s.size(), distribution.log_prob(s).size())

        def apply_fn(*params):
            return dist_ctor(*params).log_prob(s)

        gradcheck(apply_fn, ctor_params, raise_exception=True)

    def _check_log_prob(self, dist, asset_fn):
        # checks that the log_prob matches a reference function
        s = dist.sample()
        log_probs = dist.log_prob(s)
        for i, (val, log_prob) in enumerate(zip(s.data.view(-1), log_probs.data.view(-1))):
            asset_fn(i, val, log_prob)

    def _check_sampler_sampler(self, torch_dist, ref_dist, message,
                               num_samples=10000, failure_rate=1e-3):
        # Checks that the .sample() method matches a reference function.
        torch_samples = torch_dist.sample_n(num_samples).squeeze()
        if isinstance(torch_samples, Variable):
            torch_samples = torch_samples.data
        torch_samples = torch_samples.cpu().numpy()
        ref_samples = ref_dist.rvs(num_samples)
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

    def test_bernoulli(self):
        p = Variable(torch.Tensor([0.7, 0.2, 0.4]), requires_grad=True)
        r = Variable(torch.Tensor([0.3]), requires_grad=True)
        self.assertEqual(Bernoulli(p).sample_n(8).size(), (8, 3))
        self.assertEqual(Bernoulli(r).sample_n(8).size(), (8, 1))
        self.assertEqual(Bernoulli(r).sample().size(), (1,))
        self._gradcheck_log_prob(Bernoulli, (p,))

        def ref_log_prob(idx, val, log_prob):
            prob = p.data[idx]
            self.assertEqual(log_prob, math.log(prob if val else 1 - prob))

        self._check_log_prob(Bernoulli(p), ref_log_prob)

    def test_bernoulli_3d(self):
        p = Variable(torch.Tensor(2, 3, 5).fill_(0.5), requires_grad=True)
        self.assertEqual(Bernoulli(p).sample().size(), (2, 3, 5))
        self.assertEqual(Bernoulli(p).sample_n(2).size(), (2, 2, 3, 5))

    def test_multinomial_1d(self):
        p = Variable(torch.Tensor([0.1, 0.2, 0.3]), requires_grad=True)
        # TODO: this should return a 0-dim tensor once we have Scalar support
        self.assertEqual(Categorical(p).sample().size(), (1,))
        self.assertEqual(Categorical(p).sample_n(1).size(), (1, 1))
        self._gradcheck_log_prob(Categorical, (p,))

    def test_multinomial_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        p = Variable(torch.Tensor(probabilities), requires_grad=True)
        self.assertEqual(Categorical(p).sample().size(), (2,))
        self.assertEqual(Categorical(p).sample_n(6).size(), (6, 2))
        self._gradcheck_log_prob(Categorical, (p,))

        def ref_log_prob(idx, val, log_prob):
            sample_prob = p.data[idx][val] / p.data[idx].sum()
            self.assertEqual(log_prob, math.log(sample_prob))

        self._check_log_prob(Categorical(p), ref_log_prob)

    def test_normal(self):
        mean = Variable(torch.randn(5, 5), requires_grad=True)
        std = Variable(torch.randn(5, 5).abs(), requires_grad=True)
        mean_1d = Variable(torch.randn(1), requires_grad=True)
        std_1d = Variable(torch.randn(1), requires_grad=True)
        self.assertEqual(Normal(mean, std).sample().size(), (5, 5))
        self.assertEqual(Normal(mean, std).sample_n(7).size(), (7, 5, 5))
        self.assertEqual(Normal(mean_1d, std_1d).sample_n(1).size(), (1, 1))
        self.assertEqual(Normal(mean_1d, std_1d).sample().size(), (1,))
        self.assertEqual(Normal(0.2, .6).sample_n(1).size(), (1, 1))
        self.assertEqual(Normal(-0.7, 50.0).sample_n(1).size(), (1, 1))

        self._gradcheck_log_prob(Normal, (mean, std))
        self._gradcheck_log_prob(Normal, (mean, 1.0))
        self._gradcheck_log_prob(Normal, (0.0, std))

        def ref_log_prob(idx, x, log_prob):
            m = mean.data.view(-1)[idx]
            s = std.data.view(-1)[idx]
            expected = (math.exp(-(x - m) ** 2 / (2 * s ** 2)) /
                        math.sqrt(2 * math.pi * s ** 2))
            self.assertAlmostEqual(log_prob, math.log(expected), places=3)

        self._check_log_prob(Normal(mean, std), ref_log_prob)

    # This is a randomized test.
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_normal_sample(self):
        self._set_rng_seed()
        for mean, std in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Normal(mean, std),
                                        scipy.stats.norm(loc=mean, scale=std),
                                        'Normal(mean={}, std={})'.format(mean, std))

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
        self.assertEqual(Gamma(0.5, 0.5).sample_n(1).size(), (1, 1))

        def ref_log_prob(idx, x, log_prob):
            a = alpha.data.view(-1)[idx]
            b = beta.data.view(-1)[idx]
            expected = scipy.stats.gamma.logpdf(x, a, scale=1 / b)
            self.assertAlmostEqual(log_prob, expected, places=3)

        self._check_log_prob(Gamma(alpha, beta), ref_log_prob)

    # This is a randomized test.
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_gamma_sample(self):
        self._set_rng_seed()
        for alpha, beta in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(Gamma(alpha, beta),
                                        scipy.stats.gamma(alpha, scale=1.0 / beta),
                                        'Gamma(alpha={}, beta={})'.format(alpha, beta))

    # This is a randomized test.
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_gamma_sample_grad(self):
        self._set_rng_seed(1)
        num_samples = 100
        for alpha in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            alphas = Variable(torch.Tensor([alpha] * num_samples), requires_grad=True)
            betas = Variable(torch.ones(num_samples))
            x = Gamma(alphas, betas).sample()
            x.sum().backward()
            x, ind = x.data.sort()
            x = x.numpy()
            actual_grad = alphas.grad.data[ind].numpy()
            # Compare with expected gradient dx/dalpha along constant cdf(x,alpha).
            cdf = scipy.stats.gamma.cdf
            pdf = scipy.stats.gamma.pdf
            eps = 0.02 * alpha if alpha < 100 else 0.02 * alpha ** 0.5
            cdf_alpha = (cdf(x, alpha + eps) - cdf(x, alpha - eps)) / (2 * eps)
            cdf_x = pdf(x, alpha)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-100)
            self.assertLess(np.max(rel_error), 0.005,
                            '\n'.join(['Bad gradients for Gamma({}, 1)'.format(alpha),
                                       'x {}'.format(x),
                                       'expected {}'.format(expected_grad),
                                       'actual {}'.format(actual_grad),
                                       'rel error {}'.format(rel_error),
                                       'max error {}'.format(rel_error.max())]))


if __name__ == '__main__':
    run_tests()
