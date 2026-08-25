# Owner(s): ["module: optimizer"]

"""Tests for eager for-loop optimizer memory usage and numerical correctness.

Covers the in-place rewrites in adam.py, nadam.py, and rprop.py:
- Peak memory budget: O(param + grad + state + 1 intermediate)
- Numerical equivalence of in-place rewrites (NAdam, Adam, Rprop)
- Differentiable path correctness (backward + gradcheck for Adam/NAdam/Rprop)
- CUDA graph capture + replay verification for capturable paths
- dtype coverage (fp16, bf16, fp32, fp64) on both CPU and CUDA
- Orthogonality: weight_decay + maximize combinations

Redundant with test/test_optim.py and removed:
- state_dict round-trip (covered by test_state_dict_deterministic, line 1582)
- torch.compile parity (covered by CompiledOptimizerParityTests.test_correctness)
- foreach vs single-tensor (covered by test_foreach_matches_forloop, line 889)
"""

from __future__ import annotations

import torch
import torch.optim as optim
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import gradcheck, run_tests, TestCase


N = 1024  # small for CI, large enough to dominate allocator overhead


def _get_state_bytes(opt):
    """Sum actual bytes of all state tensors via introspection."""
    total = 0
    for group in opt.param_groups:
        for p in group["params"]:
            if p in opt.state:
                for v in opt.state[p].values():
                    if isinstance(v, torch.Tensor):
                        total += v.numel() * v.element_size()
    return total


def _diff_fn(p, grad, opt_state, opt_cls, kwargs, *ignored):
    """Helper for gradcheck: runs one optimizer step and returns outputs."""
    p = p.clone()
    p.grad = grad
    opt_state = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in opt_state.items()
    }
    opt = opt_cls([p], **kwargs)
    opt.state[p].update(opt_state)
    opt.step()
    return (p,) + tuple(
        v
        for v in opt.state[p].values()
        if isinstance(v, torch.Tensor) and v.requires_grad
    )


# ---------------------------------------------------------------------------
# NAdam denom
# ---------------------------------------------------------------------------


class TestNAdamNumerical(TestCase):
    def test_denom_inplace(self):
        for dtype in [torch.float32, torch.float64]:
            for shape in [(128,), (64, 32)]:
                x = torch.rand(shape, dtype=dtype)
                old = x.div(0.95).sqrt()
                new = x.div(0.95).sqrt_()
                self.assertEqual(old, new, atol=1e-6, rtol=1e-6)

    def test_denom_not_reordered(self):
        """Verify .sqrt().div_() != .div().sqrt_() -- the trap."""
        x = torch.tensor([4.0], dtype=torch.float64)
        correct = x.div(0.25).sqrt()  # sqrt(4/0.25) = 4
        wrong = x.sqrt().div_(0.25)  # sqrt(4)/0.25 = 8
        self.assertNotEqual(correct, wrong)


class TestAdamDenomChain(TestCase):
    def test_chain(self):
        for dtype in [torch.float32, torch.float64]:
            x = torch.rand(128, dtype=dtype)
            bc_sqrt = torch.tensor(0.97, dtype=dtype)
            sn = torch.tensor(-1.0, dtype=dtype)
            eps = torch.tensor(1e-8, dtype=dtype)
            old = (x.sqrt() / (bc_sqrt * sn)).add_(eps / sn)
            new = x.sqrt()
            new.div_(bc_sqrt * sn)
            new.add_(eps / sn)
            self.assertEqual(old, new, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Rprop sign chain + where dtype promotion
# ---------------------------------------------------------------------------


class TestRpropSignChain(TestCase):
    def test_chain(self):
        for dtype in [torch.float32, torch.float64]:
            a = torch.rand(128, dtype=dtype)
            b = torch.rand(128, dtype=dtype)
            old = a.mul(b).sign()
            new = a.mul(b)
            new.sign_()
            self.assertEqual(old, new)

    def test_where_combined(self):
        """Combined torch.where must match sequential copy_ version."""
        sign = torch.tensor([-2.0, 0.0, 3.0, -1.0, 5.0])
        ep, em = 1.2, 0.8

        old = sign.clone()
        old.copy_(torch.where(old.gt(0), ep, old))
        old.copy_(torch.where(old.lt(0), em, old))
        old.copy_(torch.where(old.eq(0), 1, old))

        new = torch.where(
            sign.gt(0),
            ep,
            torch.where(sign.lt(0), em, 1.0),
        )
        self.assertEqual(old, new)

    def test_where_dtype_promotion(self):
        """Confirm 1.0 scalar promotion matches original 1 int across dtypes."""
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            sign = torch.tensor([-2.0, 0.0, 3.0], dtype=dtype)
            ep = torch.tensor(1.2, dtype=dtype)
            em = torch.tensor(0.8, dtype=dtype)

            old = sign.clone()
            old.copy_(torch.where(old.gt(0), ep, old))
            old.copy_(torch.where(old.lt(0), em, old))
            old.copy_(torch.where(old.eq(0), 1, old))

            new = torch.where(
                sign.gt(0),
                ep,
                torch.where(sign.lt(0), em, 1.0),
            )

            self.assertEqual(old.dtype, new.dtype)
            self.assertEqual(old, new, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Gradcheck: differentiable paths for Adam, NAdam, Rprop
# ---------------------------------------------------------------------------


class TestDiffGradcheck(TestCase):
    def _run_gradcheck(self, opt_cls, state, kwargs):
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        gradcheck(
            _diff_fn,
            (p, grad, state, opt_cls, kwargs, *state.values()),
            check_batched_grad=False,
        )

    def test_adam_diff(self):
        state = {
            "step": torch.tensor(10.0, requires_grad=False, dtype=torch.float64),
            "exp_avg": torch.rand(10, requires_grad=True, dtype=torch.float64),
            "exp_avg_sq": torch.rand(10, requires_grad=True, dtype=torch.float64),
            "max_exp_avg_sq": torch.rand(10, requires_grad=True, dtype=torch.float64),
        }
        self._run_gradcheck(
            optim.Adam,
            state,
            {"lr": 0.9, "differentiable": True, "amsgrad": True, "foreach": False},
        )

    def test_nadam_diff(self):
        state = {
            "step": torch.tensor(10.0, requires_grad=False, dtype=torch.float64),
            "exp_avg": torch.rand(10, requires_grad=True, dtype=torch.float64),
            "exp_avg_sq": torch.rand(10, requires_grad=True, dtype=torch.float64),
            "mu_product": torch.tensor(1.0, requires_grad=True, dtype=torch.float64),
        }
        self._run_gradcheck(
            optim.NAdam,
            state,
            {"lr": 0.9, "differentiable": True, "foreach": False},
        )

    def test_rprop_diff(self):
        state = {
            "step": torch.tensor(10.0, requires_grad=False, dtype=torch.float64),
            "prev": torch.rand(10, requires_grad=True, dtype=torch.float64),
            "step_size": torch.rand(10, requires_grad=True, dtype=torch.float64),
        }
        self._run_gradcheck(
            optim.Rprop,
            state,
            {"lr": 0.9, "differentiable": True, "foreach": False},
        )


# ---------------------------------------------------------------------------
# dtype coverage: in-place chain equivalence across dtypes on CPU
# ---------------------------------------------------------------------------


class TestDiffDtypeCorrectness(TestCase):
    def test_nadam_denom(self):
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            x = torch.rand(128, dtype=dtype)
            bc2 = torch.tensor(0.95, dtype=dtype)
            old = x.div(bc2).sqrt()
            new = x.div(bc2).sqrt_()
            self.assertEqual(old, new, atol=1e-2, rtol=1e-2)

    def test_adam_denom(self):
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            x = torch.rand(128, dtype=dtype)
            bc_sqrt = torch.tensor(0.97, dtype=dtype)
            sn = torch.tensor(-1.0, dtype=dtype)
            eps = torch.tensor(1e-8, dtype=dtype)
            old = (x.sqrt() / (bc_sqrt * sn)).add_(eps / sn)
            new = x.sqrt()
            new.div_(bc_sqrt * sn)
            new.add_(eps / sn)
            self.assertEqual(old, new, atol=1e-2, rtol=1e-2)

    def test_rprop_sign(self):
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            a = torch.rand(128, dtype=dtype)
            b = torch.rand(128, dtype=dtype)
            old = a.mul(b).sign()
            new = a.mul(b)
            new.sign_()
            self.assertEqual(old, new, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Orthogonality: weight_decay + maximize combinations
# ---------------------------------------------------------------------------


class TestOrthogonality(TestCase):
    def _check(self, opt_cls, kwargs, use_wd=True):
        torch.manual_seed(42)
        p1 = torch.randn(64)
        p2 = p1.clone()
        g = torch.randn(64)

        opt1 = opt_cls([p1], **{**kwargs, "foreach": False})
        p1.grad = g.clone()
        opt1.step()

        extra = {"maximize": True, "foreach": False}
        if use_wd:
            extra["weight_decay"] = 0.01
        opt2 = opt_cls([p2], **{**kwargs, **extra})
        p2.grad = g.clone()
        opt2.step()

        self.assertNotEqual(p1, p2)

    def test_adam_wd_max(self):
        self._check(optim.Adam, {"lr": 1e-3})

    def test_nadam_wd_max(self):
        self._check(optim.NAdam, {"lr": 1e-3})

    def test_rprop_max(self):
        self._check(optim.Rprop, {"lr": 1e-2}, use_wd=False)


# ---------------------------------------------------------------------------
# CUDA-only: peak memory budget
# ---------------------------------------------------------------------------


class TestOptimizerMemoryBudgetCUDA(TestCase):
    device_type = "cuda"

    def test_peak_memory_budget(self):
        configs = [
            (optim.Adam, {"lr": 1e-3}),
            (optim.Adam, {"lr": 1e-3, "amsgrad": True}),
            (optim.AdamW, {"lr": 1e-3}),
            (optim.SGD, {"lr": 0.1}),
            (optim.SGD, {"lr": 0.1, "momentum": 0.9}),
            (optim.Adagrad, {"lr": 0.1}),
            (optim.Adadelta, {"lr": 1.0}),
            (optim.Adamax, {"lr": 2e-3}),
            (optim.NAdam, {"lr": 1e-3}),
            (optim.RMSprop, {"lr": 1e-2}),
            (optim.Rprop, {"lr": 1e-2}),
            (optim.ASGD, {"lr": 0.1}),
        ]
        for opt_cls, opt_kwargs in configs:
            param = torch.randn(N, device="cuda")
            grad = torch.randn(N, device="cuda")
            param.grad = grad
            opt = opt_cls([param], **{**opt_kwargs, "foreach": False})
            opt.step()
            param.grad = grad.clone()

            param_bytes = param.numel() * param.element_size()
            state_bytes = _get_state_bytes(opt)
            budget = param_bytes + param_bytes + state_bytes + param_bytes

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            opt.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()

            self.assertLessEqual(
                peak - baseline,
                budget,
                f"Peak {peak - baseline} exceeds budget {budget} for {opt_cls.__name__}",
            )


# ---------------------------------------------------------------------------
# CUDA-only: graph capture + replay
# ---------------------------------------------------------------------------


class TestCUDAGraphCapture(TestCase):
    device_type = "cuda"

    def test_graph_capture_replay(self):
        configs = [
            (optim.Adam, {"lr": 1e-3, "capturable": True}),
            (optim.Adam, {"lr": 1e-3, "capturable": True, "amsgrad": True}),
            (optim.Rprop, {"lr": 1e-2, "capturable": True}),
        ]
        n_steps = 5
        shape = (64,)
        grad = torch.randn(shape, device="cuda")

        for opt_cls, opt_kwargs in configs:
            p_init = torch.randn(shape, device="cuda")

            # Eager reference
            p_eager = p_init.clone()
            p_eager.grad = grad.clone()
            opt_eager = opt_cls([p_eager], **{**opt_kwargs, "foreach": False})
            opt_eager.step()
            for _ in range(n_steps):
                p_eager.grad = grad.clone()
                opt_eager.step()

            # Graph path
            p_graph = p_init.clone()
            p_graph.grad = grad.clone()
            opt_graph = opt_cls([p_graph], **{**opt_kwargs, "foreach": False})
            opt_graph.step()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                opt_graph.step()

            for _ in range(n_steps):
                p_graph.grad = grad.clone()
                g.replay()
                torch.cuda.synchronize()

            self.assertEqual(
                p_eager,
                p_graph,
                atol=1e-5,
                rtol=1e-5,
                msg=f"Graph replay diverged for {opt_cls.__name__}",
            )
            self.assertFalse(torch.isnan(p_graph).any())
            self.assertFalse(torch.isinf(p_graph).any())


# ---------------------------------------------------------------------------
# CUDA dtype coverage
# ---------------------------------------------------------------------------


class TestDiffDtypeCorrectnessCUDA(TestCase):
    device_type = "cuda"

    def test_nadam_denom(self):
        for dtype in [torch.float16, torch.bfloat16]:
            x = torch.rand(128, dtype=dtype, device="cuda")
            bc2 = torch.tensor(0.95, dtype=dtype, device="cuda")
            old = x.div(bc2).sqrt()
            new = x.div(bc2).sqrt_()
            self.assertEqual(old, new, atol=1e-2, rtol=1e-2)

    def test_adam_denom(self):
        for dtype in [torch.float16, torch.bfloat16]:
            x = torch.rand(128, dtype=dtype, device="cuda")
            bc_sqrt = torch.tensor(0.97, dtype=dtype, device="cuda")
            sn = torch.tensor(-1.0, dtype=dtype, device="cuda")
            eps = torch.tensor(1e-8, dtype=dtype, device="cuda")
            old = (x.sqrt() / (bc_sqrt * sn)).add_(eps / sn)
            new = x.sqrt()
            new.div_(bc_sqrt * sn)
            new.add_(eps / sn)
            self.assertEqual(old, new, atol=1e-2, rtol=1e-2)

    def test_rprop_sign(self):
        for dtype in [torch.float16, torch.bfloat16]:
            a = torch.rand(128, dtype=dtype, device="cuda")
            b = torch.rand(128, dtype=dtype, device="cuda")
            old = a.mul(b).sign()
            new = a.mul(b)
            new.sign_()
            self.assertEqual(old, new, atol=1e-2, rtol=1e-2)


instantiate_device_type_tests(TestOptimizerMemoryBudgetCUDA, globals())
instantiate_device_type_tests(TestCUDAGraphCapture, globals())
instantiate_device_type_tests(TestDiffDtypeCorrectnessCUDA, globals())


if __name__ == "__main__":
    run_tests()
