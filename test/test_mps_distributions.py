# Owner(s): ["module: mps"]

import math
import random
import sys
import unittest
from itertools import product

import numpy as np
import torch
import torch.backends.mps
from torch.distributions import Dirichlet
from torch.distributions.dirichlet import _Dirichlet_backward
from torch.testing._internal.common_utils import (
    NoTest,
    TEST_SCIPY,
    TestCase,
    run_tests,
    set_rng_seed
)


if not torch.backends.mps.is_available():
    print("MPS not available, skipping tests", file=sys.stderr)
    TestCase = NoTest


class TestMPSDistributions(TestCase):
    def setUp(self):
        super().setUp()
        torch.distributions.Distribution.set_default_validate_args(True)

    def test_dirichlet_shape(self):
        alpha = torch.randn(2, 3, device="mps").exp().requires_grad_()
        alpha_1d = torch.randn(4, device="mps").exp().requires_grad_()
        self.assertEqual(Dirichlet(alpha).sample().size(), (2, 3))
        self.assertEqual(Dirichlet(alpha).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Dirichlet(alpha_1d).sample().size(), (4,))
        self.assertEqual(Dirichlet(alpha_1d).sample((1,)).size(), (1, 4))

    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_dirichlet_log_prob_zero(self):
        import scipy.stats

        alpha = torch.tensor([1, 2], device="mps")
        x = torch.tensor([0, 1], device="mps")
        actual_log_prob = Dirichlet(alpha).log_prob(x).cpu()
        expected_log_prob = scipy.stats.dirichlet.logpdf(x.cpu().numpy(), alpha.cpu().numpy())
        self.assertEqual(actual_log_prob, expected_log_prob, atol=1e-3, rtol=0)

    def test_dirichlet_mode(self):
        concentrations_and_modes = [
            ([2, 2, 1], [0.5, 0.5, 0.0]),
            ([3, 2, 1], [2 / 3, 1 / 3, 0]),
            ([0.5, 0.2, 0.2], [1.0, 0.0, 0.0]),
            ([1, 1, 1], [math.nan, math.nan, math.nan]),
        ]
        for concentration, mode in concentrations_and_modes:
            dist = Dirichlet(torch.tensor(concentration, device="mps"))
            self.assertEqual(dist.mode, torch.tensor(mode, device="mps"))

    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_dirichlet_sample(self):
        import scipy.special
        import scipy.stats

        set_rng_seed(0)
        num_samples = 10000
        alpha = torch.exp(torch.randn(3, device="mps"))
        torch_samples = Dirichlet(alpha).sample((num_samples,)).cpu().numpy()
        ref_samples = scipy.stats.dirichlet(alpha.cpu().numpy()).rvs(num_samples).astype(np.float64)

        axis = np.random.normal(size=(1,) + torch_samples.shape[1:])
        axis /= np.linalg.norm(axis)
        torch_samples = (axis * torch_samples).reshape(num_samples, -1).sum(-1)
        ref_samples = (axis * ref_samples).reshape(num_samples, -1).sum(-1)
        samples = [(x, 1) for x in torch_samples] + [(x, -1) for x in ref_samples]
        random.shuffle(samples)
        samples.sort(key=lambda x: x[0])
        samples = np.array(samples)[:, 1]

        num_bins = 10
        bins = samples.reshape((num_bins, len(samples) // num_bins)).mean(axis=1)
        threshold = (len(samples) // num_bins) ** -0.5 * scipy.special.erfinv(1 - 2e-4)
        for bias in bins:
            self.assertLess(-threshold, bias, f"Dirichlet.sample() is biased:\n{bins}")
            self.assertLess(bias, threshold, f"Dirichlet.sample() is biased:\n{bins}")

    def test_dirichlet_simplex(self):
        alpha = torch.tensor([[0.3, 1.2, 4.5], [2.0, 0.7, 0.4]], device="mps")
        sample = Dirichlet(alpha).sample((8,))

        self.assertEqual(sample.shape, (8, 2, 3))
        self.assertGreaterEqual(sample.min().item(), 0.0)
        self.assertEqual(sample.sum(dim=-1), torch.ones(8, 2, device="mps"), atol=1e-5, rtol=0)

    def test_dirichlet_distribution_shape(self):
        dist = Dirichlet(torch.tensor([[0.6, 0.3], [1.6, 1.3], [2.6, 2.3]], device="mps"))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((5, 4)).size(), torch.Size((5, 4, 3, 2)))

        simplex_sample = torch.ones(3, 2, device="mps")
        simplex_sample = simplex_sample / simplex_sample.sum(-1, keepdim=True)
        self.assertEqual(dist.log_prob(simplex_sample).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, torch.ones(3, 2, 3, device="mps"))

        simplex_sample = torch.ones(3, 1, 2, device="mps")
        simplex_sample = simplex_sample / simplex_sample.sum(-1).unsqueeze(-1)
        self.assertEqual(dist.log_prob(simplex_sample).size(), torch.Size((3, 3)))

    def test_dirichlet_cpu_mps_consistency(self):
        set_rng_seed(0)
        alpha_cpu = torch.tensor([[0.3, 1.2, 4.5], [2.0, 0.7, 0.4]], device="cpu")
        alpha_mps = alpha_cpu.to("mps")

        cpu_samples = Dirichlet(alpha_cpu).sample((4096,))
        mps_samples = Dirichlet(alpha_mps).sample((4096,)).cpu()

        self.assertEqual(cpu_samples.shape, mps_samples.shape)
        self.assertEqual(cpu_samples.sum(dim=-1), torch.ones(4096, 2), atol=1e-6, rtol=0)
        self.assertEqual(mps_samples.sum(dim=-1), torch.ones(4096, 2), atol=1e-6, rtol=0)

        self.assertEqual(mps_samples.mean(0), cpu_samples.mean(0), atol=0.02, rtol=0)
        self.assertEqual(mps_samples.var(0), cpu_samples.var(0), atol=0.01, rtol=0)

    def test_sample_dirichlet(self):
        dtype = torch.float32
        n_samples = 1000

        for k in [3, 5, 10]:
            alpha = torch.ones((n_samples, k), device="mps", dtype=dtype)
            samples = torch._sample_dirichlet(alpha)

            self.assertEqual(samples.shape, (n_samples, k))
            self.assertTrue((samples > 0).all(), f"Dirichlet samples should be positive for k={k}")

            sums = samples.sum(dim=-1)
            self.assertTrue(
                torch.allclose(sums, torch.ones_like(sums), atol=1e-4),
                f"Dirichlet samples should sum to 1 for k={k}",
            )

            expected_mean = 1.0 / k
            mean = samples.float().mean(dim=0)
            for i in range(k):
                self.assertAlmostEqual(
                    mean[i].item(),
                    expected_mean,
                    delta=0.05,
                    msg=f"Mean[{i}] should be close to {expected_mean} for k={k}",
                )

        empty_alpha = torch.empty(0, 3, device="mps", dtype=dtype)
        empty_samples = torch._sample_dirichlet(empty_alpha)
        self.assertEqual(empty_samples.numel(), 0)

    def test_dirichlet_grad_matches_cpu(self):
        test_cases = [
            (
                torch.tensor([1e-4, 1e-3, 0.1], dtype=torch.float32),
                torch.tensor([0.1, 1.0, 10.0], dtype=torch.float32),
                torch.tensor([0.2, 3.0, 20.0], dtype=torch.float32),
            ),
            (
                torch.tensor([0.9, 0.99, 0.999], dtype=torch.float32),
                torch.tensor([0.5, 2.0, 8.0], dtype=torch.float32),
                torch.tensor([1.0, 5.0, 20.0], dtype=torch.float32),
            ),
            (
                torch.tensor([0.45, 0.5, 0.55], dtype=torch.float32),
                torch.tensor([8.0, 20.0, 50.0], dtype=torch.float32),
                torch.tensor([16.0, 40.0, 100.0], dtype=torch.float32),
            ),
        ]

        for x_cpu, alpha_cpu, total_cpu in test_cases:
            expected = torch._dirichlet_grad(x_cpu, alpha_cpu, total_cpu)
            actual = torch._dirichlet_grad(x_cpu.to("mps"), alpha_cpu.to("mps"), total_cpu.to("mps")).cpu()
            self.assertEqual(actual, expected, atol=1e-5, rtol=1e-5)

    def test_dirichlet_multivariate_mps(self):
        alpha_crit = 0.25 * (5.0**0.5 - 1.0)
        num_samples = 100000
        for shift in [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10]:
            alpha = torch.tensor([alpha_crit + shift], dtype=torch.float, device="mps", requires_grad=True)
            alpha_vec = torch.cat([alpha, alpha, alpha.new_tensor([1.0])])
            z = Dirichlet(alpha_vec.expand(num_samples, 3)).rsample()
            mean_z3 = 1.0 / (2.0 * alpha + 1.0)
            loss = torch.pow(z[:, 2] - mean_z3, 2.0).mean()
            actual_grad = torch.autograd.grad(loss, [alpha])[0]
            num = 1.0 - 2.0 * alpha - 4.0 * alpha**2
            den = (1.0 + alpha) ** 2 * (1.0 + 2.0 * alpha) ** 3
            expected_grad = num / den
            self.assertEqual(actual_grad, expected_grad, atol=0.002, rtol=0)


    def test_sample_dirichlet(self):
        """Test _sample_dirichlet on MPS matches expected statistical properties."""
        n_samples = 1000

        for k in [3, 5, 10]:
            alpha = torch.ones((n_samples, k), device='mps', dtype=torch.float32)
            samples = torch._sample_dirichlet(alpha)

            self.assertEqual(samples.shape, (n_samples, k))

            self.assertTrue((samples > 0).all(),
                            f"Dirichlet samples should be positive for k={k}")

            # Samples should sum to 1 along last dimension
            sums = samples.sum(dim=-1)
            self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4),
                            f"Dirichlet samples should sum to 1 for k={k}")

            # Check mean is close to 1/k for uniform alpha
            expected_mean = 1.0 / k
            mean = samples.float().mean(dim=0)
            for i in range(k):
                self.assertAlmostEqual(mean[i].item(), expected_mean, delta=0.05,
                                       msg=f"Mean[{i}] should be close to {expected_mean} for k={k}")

        # Test empty tensor
        empty_alpha = torch.empty(0, 3, device='mps', dtype=torch.float32)
        empty_samples = torch._sample_dirichlet(empty_alpha)
        self.assertEqual(empty_samples.numel(), 0)

    def test_dirichlet_tangent_field_mps(self):
        interior_x = torch.tensor(
            [[0.2, 0.3, 0.5], [0.25, 0.25, 0.5], [0.4, 0.35, 0.25], [0.6, 0.2, 0.2]],
            dtype=torch.float,
            device="mps",
        ).repeat(5, 1)
        num_samples = interior_x.size(0)
        alpha_grid = [0.5, 1.0, 2.0]

        def compute_v(x, alpha):
            eye = torch.eye(3, 3, device="mps")
            return torch.stack([_Dirichlet_backward(x, alpha, eye[i].expand_as(x))[:, 0] for i in range(3)], dim=-1)

        for a1, a2, a3 in product(alpha_grid, alpha_grid, alpha_grid):
            alpha = torch.tensor([a1, a2, a3], dtype=torch.float, device="mps", requires_grad=True).expand(
                num_samples, 3
            )
            x = interior_x.clone().requires_grad_()
            dlogp_da = torch.autograd.grad(
                [Dirichlet(alpha).log_prob(x.detach()).sum()],
                [alpha],
                retain_graph=True,
            )[0][:, 0]
            dlogp_dx = torch.autograd.grad(
                [Dirichlet(alpha.detach()).log_prob(x).sum()], [x], retain_graph=True
            )[0]
            v = compute_v(x, alpha)
            dx = torch.tensor([[2.0, -1.0, -1.0], [0.0, 1.0, -1.0]], device="mps")
            dx /= dx.norm(2, -1, True)
            eps = 1e-2 * x.min(-1, True)[0]
            dv0 = (compute_v(x + eps * dx[0], alpha) - compute_v(x - eps * dx[0], alpha)) / (2 * eps)
            dv1 = (compute_v(x + eps * dx[1], alpha) - compute_v(x - eps * dx[1], alpha)) / (2 * eps)
            div_v = (dv0 * dx[0] + dv1 * dx[1]).sum(-1)
            error = dlogp_da + (dlogp_dx * v).sum(-1) + div_v
            self.assertLess(torch.abs(error).max().item(), 0.006)

    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_dirichlet_on_diagonal_mps(self):
        import scipy.stats

        set_rng_seed(0)
        num_samples = 20
        grid = [1e-1, 1e0, 1e1]
        for a0, a1, a2 in product(grid, grid, grid):
            alphas = torch.tensor(
                [[a0, a1, a2]] * num_samples,
                dtype=torch.float,
                device="mps",
                requires_grad=True,
            )
            x = Dirichlet(alphas).rsample()[:, 0]
            x.sum().backward()
            x, ind = x.sort()
            x_np = x.detach().cpu().numpy()
            actual_grad = alphas.grad[ind].cpu().numpy()[:, 0]
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            alpha, beta = a0, a1 + a2
            eps = 0.01 * alpha / (1.0 + np.sqrt(alpha))
            cdf_alpha = (cdf(x_np, alpha + eps, beta) - cdf(x_np, alpha - eps, beta)) / (2 * eps)
            cdf_x = pdf(x_np, alpha, beta)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(
                np.max(rel_error),
                0.01,
                "\n".join(
                    [
                        f"Bad gradient dx[0]/dalpha[0] for Dirichlet([{a0}, {a1}, {a2}])",
                        f"x {x_np}",
                        f"expected {expected_grad}",
                        f"actual {actual_grad}",
                        f"rel error {rel_error}",
                        f"max error {rel_error.max()}",
                        f"at x={x_np[rel_error.argmax()]}",
                    ]
                ),
            )


if __name__ == "__main__":
    run_tests()
