from common import TestCase, run_tests
import math
import torch
import unittest
from itertools import product
from torch.autograd import Variable, gradcheck
from torch.distributions import Bernoulli, Categorical, Normal, Gamma, Distribution

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

        def call_rsample():
            return Bernoulli(r).rsample()
        self.assertRaises(NotImplementedError, call_rsample)

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

        def call_rsample():
            return Categorical(p).rsample()
        self.assertRaises(NotImplementedError, call_rsample)

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
        self._set_rng_seed(0)
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
        self._set_rng_seed(1)
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
        self.assertEqual(Gamma(0.5, 0.5).sample_n(1).size(), (1,))

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
            x = Gamma(alphas, betas).rsample()
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
            })
        ]

        for dist, kwargs in invalid_examples:
            self.assertRaises(RuntimeError, dist, **kwargs)


class TestDistributionShapes(TestCase):
    def setUp(self):
        self.scalar_sample = 1
        self.tensor_sample_1 = torch.ones(3, 2)
        self.tensor_sample_2 = torch.ones(3, 2, 3)

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

    def test_categorical_shape(self):
        categorical = Categorical(torch.Tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(categorical._batch_shape, torch.Size((3,)))
        self.assertEqual(categorical._event_shape, torch.Size(()))
        self.assertEqual(categorical.sample().size(), torch.Size((3,)))
        self.assertEqual(categorical.sample((3, 2)).size(), torch.Size((3, 2, 3,)))
        self.assertRaises(ValueError, categorical.log_prob, self.tensor_sample_1)
        self.assertEqual(categorical.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3)))

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


if __name__ == '__main__':
    run_tests()
