# Owner(s): ["module: optimizer"]

from __future__ import annotations

import unittest

import torch
from torch import Tensor
from torch.optim import Muon
from torch.optim._muon import DEFAULT_GRAM_NS_COEFFICIENTS
from torch.testing._internal.common_utils import (
    load_tests,
    run_tests,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127


class TestMuonStandardNS(TestCase):
    """Tests for Muon with use_gram_newton_schulz=False (standard Newton-Schulz)."""

    def test_basic_step(self) -> None:
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(8, 4))
        param.grad = torch.randn_like(param)

        opt = Muon([param], use_gram_newton_schulz=False)
        initial = param.clone()
        opt.step()

        self.assertFalse(torch.equal(param, initial))
        self.assertTrue(param.isfinite().all())

    def test_square_matrix(self) -> None:
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(8, 8))
        param.grad = torch.randn_like(param)

        opt = Muon([param], use_gram_newton_schulz=False)
        initial = param.clone()
        opt.step()

        self.assertFalse(torch.equal(param, initial))
        self.assertTrue(param.isfinite().all())

    def test_wide_matrix(self) -> None:
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(4, 16))
        param.grad = torch.randn_like(param)

        opt = Muon([param], use_gram_newton_schulz=False)
        initial = param.clone()
        opt.step()

        self.assertFalse(torch.equal(param, initial))
        self.assertTrue(param.isfinite().all())

    def test_multiple_steps(self) -> None:
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(8, 4))

        opt = Muon([param], use_gram_newton_schulz=False)
        for _ in range(10):
            param.grad = torch.randn_like(param)
            opt.step()

        self.assertTrue(param.isfinite().all())

    def test_multiple_params(self) -> None:
        torch.manual_seed(42)
        p1 = torch.nn.Parameter(torch.randn(8, 4))
        p2 = torch.nn.Parameter(torch.randn(4, 16))
        p1.grad = torch.randn_like(p1)
        p2.grad = torch.randn_like(p2)

        opt = Muon([p1, p2], use_gram_newton_schulz=False)
        opt.step()

        self.assertTrue(p1.isfinite().all())
        self.assertTrue(p2.isfinite().all())

    def test_weight_decay(self) -> None:
        """Weight decay should shrink parameters toward zero."""
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.ones(8, 4))
        param.grad = torch.zeros_like(param)

        # With lr=0 and zero grad, weight decay scale is (1 - 0 * wd) = 1.0
        # so param should be unchanged.
        opt = Muon([param], lr=0.0, weight_decay=0.1, use_gram_newton_schulz=False)
        opt.step()
        self.assertEqual(param, torch.ones(8, 4))

        # With nonzero lr, weight decay should shrink params.
        opt2 = Muon([param], lr=0.01, weight_decay=0.5, use_gram_newton_schulz=False)
        param.grad = torch.zeros_like(param)
        initial = param.clone()
        opt2.step()
        self.assertTrue((param.abs() <= initial.abs()).all())

    def test_nesterov_toggle(self) -> None:
        """nesterov=True vs False should produce different results."""
        torch.manual_seed(42)
        p1 = torch.nn.Parameter(torch.randn(8, 4))
        p2 = torch.nn.Parameter(p1.clone().detach())

        grad = torch.randn(8, 4)
        p1.grad = grad.clone()
        p2.grad = grad.clone()

        opt1 = Muon([p1], nesterov=True, use_gram_newton_schulz=False)
        opt2 = Muon([p2], nesterov=False, use_gram_newton_schulz=False)

        # Need two steps for nesterov to diverge (first step momentum buf is zero).
        opt1.step()
        opt2.step()
        p1.grad = torch.randn(8, 4)
        p2.grad = p1.grad.clone()
        opt1.step()
        opt2.step()

        self.assertFalse(torch.equal(p1, p2))


class TestMuonVanillaRegression(TestCase):
    """Frozen-value regression test: vanilla Muon (use_gram_newton_schulz=False)
    must produce bit-identical output to the implementation that existed at the
    parent of D104311008. Values were captured at that parent commit by running
    the same setup against the pre-diff Muon. Do not regenerate these literals
    without an explicit BC review — a mismatch means the supposedly bit-compatible
    vanilla path has silently diverged.
    """

    # torch.manual_seed(42); p = nn.Parameter(torch.randn(8, 4));
    # opt = Muon([p], lr=0.02, weight_decay=0.1, momentum=0.95, nesterov=True)
    # for _ in range(5): p.grad = torch.randn_like(p); opt.step()
    EXPECTED_PARAM_NESTEROV_TRUE: list[list[float]] = [
        [1.938483476638794, 1.463821530342102, 0.8684228658676147, -2.107022762298584],
        [0.6464439034461975, -1.2333816289901733, -0.05520281195640564, -1.594115972518921],
        [-0.7017418742179871, 1.5892770290374756, -0.3630223572254181, -1.3906430006027222],
        [-0.6950568556785583, -0.6230637431144714, -0.7075115442276001, 0.7674455046653748],
        [1.6756398677825928, -0.14238590002059937, -0.4740765392780304, 0.41970568895339966],
        [-0.75552898645401, 1.045496940612793, 0.8197356462478638, 1.681230068206787],
        [1.2763956785202026, 1.2807981967926025, 0.6196531653404236, 1.2518298625946045],
        [-0.20052097737789154, 0.02047128602862358, -0.25045672059059143, 0.7741371989250183],
    ]

    # Same setup but nesterov=False, weight_decay=0.0
    EXPECTED_PARAM_NESTEROV_FALSE: list[list[float]] = [
        [1.9527949094772339, 1.4754483699798584, 0.8651961088180542, -2.1447465419769287],
        [0.6431010365486145, -1.2468913793563843, -0.0508497953414917, -1.615515112876892],
        [-0.7089078426361084, 1.601587176322937, -0.370036244392395, -1.3938430547714233],
        [-0.7046897411346436, -0.6338974237442017, -0.6964432597160339, 0.7840660214424133],
        [1.7094919681549072, -0.1214023157954216, -0.4710087776184082, 0.4312562346458435],
        [-0.7728015780448914, 1.0400207042694092, 0.8233118653297424, 1.6969393491744995],
        [1.2878977060317993, 1.301850438117981, 0.6351599097251892, 1.2659329175949097],
        [-0.2075868397951126, 0.010008741170167923, -0.25453078746795654, 0.7897554636001587],
    ]

    # The momentum buffer only depends on (grads, momentum), not nesterov, so
    # both scenarios yield the same buf for the same seed.
    EXPECTED_BUF: list[list[float]] = [
        [-0.11098092794418335, 0.08274538069963455, 0.05440763011574745, 0.022694384679198265],
        [0.06464089453220367, 0.04171084985136986, 0.05522793158888817, -0.024408867582678795],
        [-0.10506078600883484, 0.16549666225910187, -0.036471910774707794, 0.048419296741485596],
        [-0.11045779287815094, 0.2032998502254486, -0.13876627385616302, 0.06581886857748032],
        [-0.14690940082073212, 0.005613995250314474, -0.0447036549448967, 0.0763719230890274],
        [0.0020838298369199038, -0.01441915426403284, -0.09339231252670288, -0.025113143026828766],
        [-0.0352008081972599, -0.03741401806473732, -0.16004668176174164, 0.17050980031490326],
        [-0.09506358951330185, 0.007350748870521784, 0.01989450491964817, 0.16971878707408905],
    ]

    def _run_vanilla(
        self, *, nesterov: bool, weight_decay: float
    ) -> tuple[Tensor, Tensor]:
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.randn(8, 4))
        opt = Muon(
            [p],
            lr=0.02,
            weight_decay=weight_decay,
            momentum=0.95,
            nesterov=nesterov,
            use_gram_newton_schulz=False,
        )
        for _ in range(5):
            p.grad = torch.randn_like(p)
            opt.step()
        return p.detach().clone(), opt.state[p]["momentum_buffer"].clone()

    def test_vanilla_nesterov_true_matches_pre_diff_baseline(self) -> None:
        param, buf = self._run_vanilla(nesterov=True, weight_decay=0.1)
        self.assertEqual(
            param,
            torch.tensor(self.EXPECTED_PARAM_NESTEROV_TRUE),
            atol=0,
            rtol=0,
        )
        self.assertEqual(buf, torch.tensor(self.EXPECTED_BUF), atol=0, rtol=0)

    def test_vanilla_nesterov_false_matches_pre_diff_baseline(self) -> None:
        param, buf = self._run_vanilla(nesterov=False, weight_decay=0.0)
        self.assertEqual(
            param,
            torch.tensor(self.EXPECTED_PARAM_NESTEROV_FALSE),
            atol=0,
            rtol=0,
        )
        self.assertEqual(buf, torch.tensor(self.EXPECTED_BUF), atol=0, rtol=0)


class TestMuonGramNS(TestCase):
    """Tests for Muon with use_gram_newton_schulz=True (Gram Newton-Schulz)."""

    def test_basic_step(self) -> None:
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(8, 16))
        param.grad = torch.randn_like(param)

        opt = Muon([param], use_gram_newton_schulz=True)
        initial = param.clone()
        opt.step()

        self.assertFalse(torch.equal(param, initial))
        self.assertTrue(param.isfinite().all())

    def test_tall_matrix(self) -> None:
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(16, 4))
        param.grad = torch.randn_like(param)

        opt = Muon([param], use_gram_newton_schulz=True)
        initial = param.clone()
        opt.step()

        self.assertFalse(torch.equal(param, initial))
        self.assertTrue(param.isfinite().all())

    def test_wide_matrix(self) -> None:
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(4, 16))
        param.grad = torch.randn_like(param)

        opt = Muon([param], use_gram_newton_schulz=True)
        initial = param.clone()
        opt.step()

        self.assertFalse(torch.equal(param, initial))
        self.assertTrue(param.isfinite().all())

    def test_square_matrix_falls_back_to_standard(self) -> None:
        """Square matrix should automatically fall back to standard NS."""
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(8, 8))
        param.grad = torch.randn_like(param)

        opt = Muon([param], use_gram_newton_schulz=True)
        initial = param.clone()
        opt.step()

        self.assertFalse(torch.equal(param, initial))
        self.assertTrue(param.isfinite().all())

    def test_multiple_steps(self) -> None:
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(8, 16))

        opt = Muon([param], use_gram_newton_schulz=True)
        for _ in range(10):
            param.grad = torch.randn_like(param)
            opt.step()

        self.assertTrue(param.isfinite().all())

    def test_multiple_params(self) -> None:
        torch.manual_seed(42)
        p1 = torch.nn.Parameter(torch.randn(8, 16))
        p2 = torch.nn.Parameter(torch.randn(4, 32))
        p1.grad = torch.randn_like(p1)
        p2.grad = torch.randn_like(p2)

        opt = Muon([p1, p2], use_gram_newton_schulz=True)
        opt.step()

        self.assertTrue(p1.isfinite().all())
        self.assertTrue(p2.isfinite().all())

    def test_default_is_vanilla(self) -> None:
        """With use_gram_newton_schulz left at its default (False), the optimizer
        should match the explicit-False configuration bit-for-bit."""
        torch.manual_seed(42)
        p_default = torch.nn.Parameter(torch.randn(8, 16))
        p_explicit = torch.nn.Parameter(p_default.clone().detach())

        grad = torch.randn(8, 16)
        p_default.grad = grad.clone()
        p_explicit.grad = grad.clone()

        opt_default = Muon([p_default])
        opt_explicit = Muon([p_explicit], use_gram_newton_schulz=False)
        opt_default.step()
        opt_explicit.step()

        self.assertEqual(p_default, p_explicit)

    def test_gram_vs_standard_differ(self) -> None:
        """Gram NS and standard NS should produce different updates for rectangular matrices."""
        torch.manual_seed(42)
        p1 = torch.nn.Parameter(torch.randn(8, 16))
        p2 = torch.nn.Parameter(p1.clone().detach())

        grad = torch.randn(8, 16)
        p1.grad = grad.clone()
        p2.grad = grad.clone()

        opt_gram = Muon([p1], use_gram_newton_schulz=True)
        opt_std = Muon([p2], use_gram_newton_schulz=False)
        opt_gram.step()
        opt_std.step()

        self.assertFalse(torch.equal(p1, p2))


class TestMuonGramConfig(TestCase):
    """Tests for the gram_newton_schulz_config dict parameter."""

    def test_unknown_key_raises(self) -> None:
        param = torch.nn.Parameter(torch.randn(8, 16))
        with self.assertRaises(ValueError):
            Muon(
                [param],
                use_gram_newton_schulz=True,
                gram_newton_schulz_config={"bogus_key": 1},
            )

    def test_config_with_gram_disabled_raises(self) -> None:
        """gram_newton_schulz_config requires use_gram_newton_schulz=True so
        callers cannot accidentally configure CUDA graph (or other gram-only
        knobs) on the vanilla path."""
        param = torch.nn.Parameter(torch.randn(8, 16))
        with self.assertRaises(ValueError):
            Muon(
                [param],
                use_gram_newton_schulz=False,
                gram_newton_schulz_config={"use_cuda_graph": True},
            )
        with self.assertRaises(ValueError):
            Muon(
                [param],
                use_gram_newton_schulz=False,
                gram_newton_schulz_config={},
            )

    def test_vanilla_default_does_not_use_cuda_graph(self) -> None:
        """With use_gram_newton_schulz=False (default), the optimizer's stored
        use_cuda_graph is False."""
        param = torch.nn.Parameter(torch.randn(8, 16))
        opt = Muon([param])
        self.assertFalse(opt.param_groups[0]["use_cuda_graph"])
        self.assertFalse(opt.param_groups[0]["use_gram_newton_schulz"])

    def test_none_config_uses_defaults(self) -> None:
        """None config should set gram defaults (vanilla Muon constants, [], False)."""
        param = torch.nn.Parameter(torch.randn(8, 16))
        opt = Muon([param], use_gram_newton_schulz=True)
        group = opt.param_groups[0]
        self.assertEqual(
            group["gram_ns_coefficients"],
            [list(c) for c in DEFAULT_GRAM_NS_COEFFICIENTS],
        )
        self.assertEqual(group["gram_ns_reset_iterations"], [])
        self.assertFalse(group["use_cuda_graph"])

    def test_partial_override(self) -> None:
        """Setting only one key in the dict should leave others at defaults."""
        param = torch.nn.Parameter(torch.randn(8, 16))
        custom_coeffs = [[1.0, -1.0, 0.5], [2.0, -2.0, 1.0]]
        opt = Muon(
            [param],
            use_gram_newton_schulz=True,
            gram_newton_schulz_config={"gram_ns_coefficients": custom_coeffs},
        )
        group = opt.param_groups[0]
        self.assertEqual(group["gram_ns_coefficients"], custom_coeffs)
        self.assertEqual(group["gram_ns_reset_iterations"], [])
        self.assertFalse(group["use_cuda_graph"])

    def test_full_override(self) -> None:
        param = torch.nn.Parameter(torch.randn(8, 16))
        custom_coeffs = [[1.0, -1.0, 0.5]]
        custom_resets = [0]
        opt = Muon(
            [param],
            use_gram_newton_schulz=True,
            gram_newton_schulz_config={
                "gram_ns_coefficients": custom_coeffs,
                "gram_ns_reset_iterations": custom_resets,
                "use_cuda_graph": True,
            },
        )
        group = opt.param_groups[0]
        self.assertEqual(group["gram_ns_coefficients"], custom_coeffs)
        self.assertEqual(group["gram_ns_reset_iterations"], custom_resets)
        self.assertTrue(group["use_cuda_graph"])


class TestMuonConvergence(TestCase):
    """End-to-end functional test: Muon should drive loss down on a toy task."""

    def _train_toy_regression(
        self,
        muon_kwargs: dict,
        steps: int = 200,
    ) -> tuple[float, float]:
        """Train a 2-layer linear model on a learnable regression target.

        The target is a fixed linear function of the input, so the model can
        actually fit it. Returns (initial_loss, final_loss).
        """
        torch.manual_seed(0)
        # Both weights are rectangular 2D, so the gram NS path actually runs.
        linear1 = torch.nn.Linear(8, 16, bias=False)
        linear2 = torch.nn.Linear(16, 4, bias=False)
        params = list(linear1.parameters()) + list(linear2.parameters())

        opt = Muon(params, lr=0.02, weight_decay=0.0, **muon_kwargs)

        x = torch.randn(64, 8)
        true_w1 = torch.randn(8, 16)
        true_w2 = torch.randn(16, 4)
        target = (x @ true_w1) @ true_w2

        def forward_loss() -> Tensor:
            y = linear2(linear1(x))
            return ((y - target) ** 2).mean()

        with torch.no_grad():
            initial_loss = forward_loss().item()
        for _ in range(steps):
            opt.zero_grad()
            loss = forward_loss()
            loss.backward()
            opt.step()
        with torch.no_grad():
            final_loss = forward_loss().item()
        return initial_loss, final_loss

    def test_gram_muon_reduces_loss(self) -> None:
        """Loss should drop by at least 90% on a learnable regression target."""
        initial, final = self._train_toy_regression({"use_gram_newton_schulz": True})
        self.assertLess(
            final,
            initial * 0.1,
            f"Gram Muon should reduce loss materially: "
            f"initial={initial:.4f}, final={final:.4f}",
        )

    def test_vanilla_muon_reduces_loss(self) -> None:
        """Sanity check: vanilla path also converges on the same task."""
        initial, final = self._train_toy_regression({"use_gram_newton_schulz": False})
        self.assertLess(
            final,
            initial * 0.1,
            f"Vanilla Muon should reduce loss materially: "
            f"initial={initial:.4f}, final={final:.4f}",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestMuonCudaGraph(TestCase):
    """Tests for CUDA graph support in the Muon optimizer."""

    def test_cuda_graph_matches_eager_gram_ns(self) -> None:
        """CUDA graph Gram NS should produce the same result as eager Gram NS."""
        torch.manual_seed(42)
        p_eager = torch.nn.Parameter(torch.randn(8, 16, device="cuda"))
        p_graph = torch.nn.Parameter(p_eager.clone().detach())

        grad = torch.randn(8, 16, device="cuda")
        p_eager.grad = grad.clone()
        p_graph.grad = grad.clone()

        opt_eager = Muon([p_eager], use_gram_newton_schulz=True)
        opt_graph = Muon(
            [p_graph],
            use_gram_newton_schulz=True,
            gram_newton_schulz_config={"use_cuda_graph": True},
        )
        opt_eager.step()
        opt_graph.step()

        self.assertEqual(p_eager, p_graph, atol=1e-5, rtol=0)

    def test_cuda_graph_multiple_steps(self) -> None:
        """CUDA graph should produce correct results across multiple steps."""
        torch.manual_seed(42)
        p_eager = torch.nn.Parameter(torch.randn(8, 16, device="cuda"))
        p_graph = torch.nn.Parameter(p_eager.clone().detach())

        opt_eager = Muon([p_eager], use_gram_newton_schulz=True)
        opt_graph = Muon(
            [p_graph],
            use_gram_newton_schulz=True,
            gram_newton_schulz_config={"use_cuda_graph": True},
        )

        for _ in range(5):
            grad = torch.randn(8, 16, device="cuda")
            p_eager.grad = grad.clone()
            p_graph.grad = grad.clone()
            opt_eager.step()
            opt_graph.step()

        self.assertEqual(p_eager, p_graph, atol=1e-4, rtol=0)

    def test_cuda_graph_multiple_params_same_shape(self) -> None:
        """Multiple params with same shape should correctly share the CUDA graph."""
        torch.manual_seed(42)
        p1_eager = torch.nn.Parameter(torch.randn(8, 16, device="cuda"))
        p2_eager = torch.nn.Parameter(torch.randn(8, 16, device="cuda"))
        p1_graph = torch.nn.Parameter(p1_eager.clone().detach())
        p2_graph = torch.nn.Parameter(p2_eager.clone().detach())

        g1 = torch.randn(8, 16, device="cuda")
        g2 = torch.randn(8, 16, device="cuda")
        p1_eager.grad = g1.clone()
        p2_eager.grad = g2.clone()
        p1_graph.grad = g1.clone()
        p2_graph.grad = g2.clone()

        opt_eager = Muon([p1_eager, p2_eager], use_gram_newton_schulz=True)
        opt_graph = Muon(
            [p1_graph, p2_graph],
            use_gram_newton_schulz=True,
            gram_newton_schulz_config={"use_cuda_graph": True},
        )
        opt_eager.step()
        opt_graph.step()

        self.assertEqual(p1_eager, p1_graph, atol=1e-5, rtol=0)
        self.assertEqual(p2_eager, p2_graph, atol=1e-5, rtol=0)

    def test_cuda_graph_different_configs_same_shape(self) -> None:
        """Same-shape params in different groups with different configs should
        use separate CUDA graphs and produce correct results."""
        torch.manual_seed(42)
        p1_eager = torch.nn.Parameter(torch.randn(8, 16, device="cuda"))
        p1_graph = torch.nn.Parameter(p1_eager.clone().detach())
        p2_eager = torch.nn.Parameter(torch.randn(8, 16, device="cuda"))
        p2_graph = torch.nn.Parameter(p2_eager.clone().detach())

        g1 = torch.randn(8, 16, device="cuda")
        g2 = torch.randn(8, 16, device="cuda")
        p1_eager.grad = g1.clone()
        p2_eager.grad = g2.clone()
        p1_graph.grad = g1.clone()
        p2_graph.grad = g2.clone()

        # Top-level use_gram_newton_schulz=True is required so that
        # gram_newton_schulz_config is accepted; the second param group still
        # overrides it back to False.
        opt_eager = Muon(
            [
                {"params": [p1_eager], "use_gram_newton_schulz": True},
                {"params": [p2_eager], "use_gram_newton_schulz": False},
            ],
            use_gram_newton_schulz=True,
        )
        opt_graph = Muon(
            [
                {"params": [p1_graph], "use_gram_newton_schulz": True},
                {"params": [p2_graph], "use_gram_newton_schulz": False},
            ],
            use_gram_newton_schulz=True,
            gram_newton_schulz_config={"use_cuda_graph": True},
        )
        opt_eager.step()
        opt_graph.step()

        self.assertEqual(p1_eager, p1_graph, atol=1e-5, rtol=0)
        self.assertEqual(p2_eager, p2_graph, atol=1e-5, rtol=0)
        self.assertFalse(torch.equal(p1_graph, p2_graph))

    def test_cuda_graph_tall_matrix(self) -> None:
        """CUDA graph should work correctly for tall matrices."""
        torch.manual_seed(42)
        p_eager = torch.nn.Parameter(torch.randn(32, 8, device="cuda"))
        p_graph = torch.nn.Parameter(p_eager.clone().detach())

        grad = torch.randn(32, 8, device="cuda")
        p_eager.grad = grad.clone()
        p_graph.grad = grad.clone()

        opt_eager = Muon([p_eager], use_gram_newton_schulz=True)
        opt_graph = Muon(
            [p_graph],
            use_gram_newton_schulz=True,
            gram_newton_schulz_config={"use_cuda_graph": True},
        )
        opt_eager.step()
        opt_graph.step()

        self.assertEqual(p_eager, p_graph, atol=1e-5, rtol=0)


if __name__ == "__main__":
    run_tests()
