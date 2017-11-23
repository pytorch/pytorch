from common import TestCase, run_tests
import math
import numpy as np
import scipy.stats
import torch
from torch.autograd import Variable, gradcheck
from torch.distributions import Bernoulli, Categorical, Normal, Gamma


def _test_univariate_sampler(torch_dist, ref_dist, num_samples=1000, plot=False):
    # Kolmogorov-Smirnov test of density.
    torch_samples = torch_dist.sample_n(num_samples).squeeze().cpu().numpy()
    ref_samples = ref_dist.rvs(num_samples)
    torch_samples.sort()
    ref_samples.sort()
    # TODO(fritzo) Check density statistics.
    if plot:
        from matplotlib import pyplot
        pyplot.plot(torch_samples, ref_samples)
        pyplot.title(type(torch_dist).__name__)
        pyplot.xlabel('PyTorch')
        pyplot.ylabel('Reference')
        pyplot.tight_layout()
        pyplot.show()


class TestDistributions(TestCase):
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

    def test_normal_sample(self):
        for mean in [-1.0, 0.0, 1.0]:
            for std in [0.1, 1.0, 10.0]:
                _test_univariate_sampler(Normal(mean, std), scipy.stats.norm(loc=mean, scale=std))

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

    # FIXME this fails due to bad numerics
    def test_gamma_sample(self):
        for alpha in [0.1, 1.0, 5.0]:
            for beta in [0.1, 1.0, 10.0]:
                _test_univariate_sampler(Gamma(alpha, beta),
                                         scipy.stats.gamma(alpha, scale=1 / beta))


if __name__ == '__main__':
    run_tests()
