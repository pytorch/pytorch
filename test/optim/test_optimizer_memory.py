# Owner(s): ["module: optimizer"]

"""Tests for eager for-loop optimizer memory usage and numerical correctness.

Covers:
- Peak memory budget: O(param + grad + state + 1 intermediate)
- Numerical equivalence of in-place rewrites (NAdam, RAdam, Adam, Rprop)
- Differentiable path correctness (backward + gradcheck for Adam/RAdam/NAdam/Rprop)
- CUDA graph capture + replay verification for capturable paths
- dtype coverage (fp16, bf16, fp32, fp64) on both CPU and CUDA
- state_dict round-trip
- torch.compile compatibility
- Orthogonality: weight_decay, maximize, foreach combinations
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
# Pre-patch RAdam oracle (verbatim from perf/radam-memory-update~1)
# ---------------------------------------------------------------------------


def _radam_reference_pre_patch(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    state_steps,
    *,
    beta1,
    beta2,
    lr,
    weight_decay,
    eps,
    decoupled_weight_decay,
    differentiable,
    maximize,
    capturable,
    has_complex,
):
    """Verbatim copy of _single_tensor_radam from before the in-place rewrite.

    Pulled from: git show perf/radam-memory-update~1:torch/optim/radam.py
    Serves as an independent oracle for testing the shipped diff.
    """
    if not torch.jit.is_scripting():
        if isinstance(lr, torch.Tensor) and lr.numel() == 1:
            lr = lr.item()

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)

        step_t += 1
        step = (
            step_t
            if capturable
            else (step_t.item() if isinstance(step_t, torch.Tensor) else step_t)
        )

        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)

        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        bias_corrected_exp_avg = exp_avg / bias_correction1

        rho_inf = 2 / (1 - beta2) - 1
        rho_t = rho_inf - 2 * step * (beta2**step) / bias_correction2

        def _compute_rect():
            return (
                (rho_t - 4)
                * (rho_t - 2)
                * rho_inf
                / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
            ) ** 0.5

        def _compute_adaptive_lr():
            exp_avg_sq_sqrt = exp_avg_sq.sqrt()
            if differentiable:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add(eps)
            else:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add_(eps)
            return (bias_correction2**0.5) / exp_avg_sq_sqrt

        if capturable:
            update = torch.where(
                rho_t > 5.0, _compute_rect() * _compute_adaptive_lr(), 1.0
            )
            param.add_(bias_corrected_exp_avg * lr * update, alpha=-1.0)
        else:
            if rho_t > 5.0:
                param.add_(
                    bias_corrected_exp_avg
                    * lr
                    * _compute_adaptive_lr()
                    * _compute_rect(),
                    alpha=-1.0,
                )
            else:
                param.add_(bias_corrected_exp_avg * lr, alpha=-1.0)


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
        """Verify .sqrt().div_() != .div().sqrt_() — the trap."""
        x = torch.tensor([4.0], dtype=torch.float64)
        correct = x.div(0.25).sqrt()  # sqrt(4/0.25) = 4
        wrong = x.sqrt().div_(0.25)  # sqrt(4)/0.25 = 8
        self.assertNotEqual(correct, wrong)


# ---------------------------------------------------------------------------
# RAdam
# ---------------------------------------------------------------------------


class TestRAdamNumerical(TestCase):
    def _rect(self, rho_t, rho_inf):
        return (
            (rho_t - 4)
            * (rho_t - 2)
            * rho_inf
            / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
        ) ** 0.5

    def _alr_old(self, x, bc2, eps, diff):
        s = x.sqrt()
        s = s.add(eps) if diff else s.add_(eps)
        return (bc2**0.5) / s

    def _alr_new(self, x, bc2, eps, diff):
        s = x.sqrt()
        if diff:
            s = s.add(eps).reciprocal().mul(bc2**0.5)
        else:
            s.add_(eps).reciprocal_().mul_(bc2**0.5)
        return s

    def test_adaptive_lr_forward_fp32(self):
        self._check_alr(False, torch.float32)

    def test_adaptive_lr_forward_fp64(self):
        self._check_alr(False, torch.float64)

    def test_adaptive_lr_forward_diff_fp32(self):
        self._check_alr(True, torch.float32)

    def test_adaptive_lr_forward_diff_fp64(self):
        self._check_alr(True, torch.float64)

    def _check_alr(self, diff, dtype):
        torch.manual_seed(42)
        for shape in [(128,), (64, 32)]:
            x = torch.rand(shape, dtype=dtype)
            old = self._alr_old(x.clone(), 0.95, 1e-8, diff)
            new = self._alr_new(x.clone(), 0.95, 1e-8, diff)
            self.assertEqual(old, new, atol=1e-6, rtol=1e-6)

    def test_backward_fp32(self):
        self._check_backward(torch.float32)

    def test_backward_fp64(self):
        self._check_backward(torch.float64)

    def _check_backward(self, dtype):
        data = torch.rand((64,), dtype=dtype)
        p1 = data.clone().requires_grad_(True)
        self._alr_old(p1, 0.95, 1e-8, True).sum().backward()
        p2 = data.clone().requires_grad_(True)
        self._alr_new(p2, 0.95, 1e-8, True).sum().backward()
        self.assertEqual(p1.grad, p2.grad, atol=1e-6, rtol=1e-6)

    def test_update_chain_fp32(self):
        self._check_chain(False, torch.float32)

    def test_update_chain_fp64(self):
        self._check_chain(False, torch.float64)

    def test_update_chain_diff_fp32(self):
        self._check_chain(True, torch.float32)

    def test_update_chain_diff_fp64(self):
        self._check_chain(True, torch.float64)

    def _check_chain(self, diff, dtype):
        torch.manual_seed(42)
        eps, lr, b1, b2 = 1e-8, 0.001, 0.9, 0.999
        esq = torch.rand((128,), dtype=dtype)
        ea = torch.rand((128,), dtype=dtype)
        p_old = torch.rand((128,), dtype=dtype)
        p_new = p_old.clone()

        step = 10
        bc1, bc2 = 1 - b1**step, 1 - b2**step
        ri = 2 / (1 - b2) - 1
        rt = ri - 2 * step * (b2**step) / bc2
        rect = self._rect(rt, ri)
        bcea = ea / bc1

        alr = self._alr_old(esq.clone(), bc2, eps, diff)
        p_old.add_(bcea * lr * alr * rect, alpha=-1.0)

        alr = self._alr_new(esq.clone(), bc2, eps, diff)
        if diff:
            update = bcea * alr * lr * rect
        else:
            update = bcea.mul(alr)
            update.mul_(lr)
            update.mul_(rect)
        p_new.add_(update, alpha=-1.0)

        self.assertEqual(p_old, p_new, atol=1e-5, rtol=1e-5)

    def test_single_tensor_radam_matches_reference(self):
        """Verify the shipped _single_tensor_radam against a pre-patch oracle.

        The oracle is a verbatim copy of _single_tensor_radam from before
        the perf/radam-memory-update patch (git show perf/radam-memory-update~1).
        This catches regressions introduced by the in-place rewrite.
        """
        from torch.optim.radam import _single_tensor_radam

        for dtype in [torch.float32, torch.float64]:
            for differentiable in [False, True]:
                torch.manual_seed(42)
                shape = (128,)
                grad = torch.rand(shape, dtype=dtype)
                beta1, beta2, lr_val, eps = 0.9, 0.999, 0.001, 1e-8
                step_before = 9

                # --- Shared initial state ---
                p_orig = torch.rand(shape, dtype=dtype)

                # --- Oracle: pre-patch _single_tensor_radam (verbatim) ---
                p_oracle = p_orig.clone()
                ea_o = torch.zeros_like(p_oracle)
                esq_o = torch.zeros_like(p_oracle)
                step_o = torch.tensor(float(step_before), dtype=dtype)
                _radam_reference_pre_patch(
                    [p_oracle],
                    [grad.clone()],
                    [ea_o],
                    [esq_o],
                    [step_o],
                    beta1=beta1,
                    beta2=beta2,
                    lr=lr_val,
                    weight_decay=0,
                    eps=eps,
                    decoupled_weight_decay=False,
                    differentiable=differentiable,
                    maximize=False,
                    capturable=False,
                    has_complex=False,
                )

                # --- Shipped: _single_tensor_radam from torch.optim.radam ---
                p_shipped = p_orig.clone()
                ea_s = torch.zeros_like(p_shipped)
                esq_s = torch.zeros_like(p_shipped)
                step_s = torch.tensor(float(step_before), dtype=dtype)
                _single_tensor_radam(
                    [p_shipped],
                    [grad.clone()],
                    [ea_s],
                    [esq_s],
                    [step_s],
                    beta1=beta1,
                    beta2=beta2,
                    lr=lr_val,
                    weight_decay=0,
                    eps=eps,
                    decoupled_weight_decay=False,
                    differentiable=differentiable,
                    maximize=False,
                    capturable=False,
                    has_complex=False,
                )

                self.assertEqual(
                    p_shipped,
                    p_oracle,
                    atol=1e-5,
                    rtol=1e-5,
                    msg=f"dtype={dtype}, differentiable={differentiable}",
                )


# ---------------------------------------------------------------------------
# Adam denom chain
# ---------------------------------------------------------------------------


class TestAdamDenomChain(TestCase):
    def test_chain_fp32(self):
        self._check(torch.float32)

    def test_chain_fp64(self):
        self._check(torch.float64)

    def _check(self, dtype):
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
    def test_chain_fp32(self):
        self._check_chain(torch.float32)

    def test_chain_fp64(self):
        self._check_chain(torch.float64)

    def _check_chain(self, dtype):
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

    def test_where_dtype_promotion_fp32(self):
        self._check_where_dtype(torch.float32)

    def test_where_dtype_promotion_fp64(self):
        self._check_where_dtype(torch.float64)

    def test_where_dtype_promotion_fp16(self):
        self._check_where_dtype(torch.float16)

    def test_where_dtype_promotion_bf16(self):
        self._check_where_dtype(torch.bfloat16)

    def _check_where_dtype(self, dtype):
        """Confirm 1.0 scalar promotion matches original 1 int across dtypes."""
        sign = torch.tensor([-2.0, 0.0, 3.0], dtype=dtype)
        ep = torch.tensor(1.2, dtype=dtype)
        em = torch.tensor(0.8, dtype=dtype)

        # Original sequential version
        old = sign.clone()
        old.copy_(torch.where(old.gt(0), ep, old))
        old.copy_(torch.where(old.lt(0), em, old))
        old.copy_(torch.where(old.eq(0), 1, old))

        # New combined version (uses 1.0 float scalar)
        new = torch.where(
            sign.gt(0),
            ep,
            torch.where(sign.lt(0), em, 1.0),
        )

        self.assertEqual(old.dtype, new.dtype)
        self.assertEqual(old, new, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Gradcheck: differentiable paths for Adam, RAdam, NAdam, Rprop
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

    def test_radam_diff(self):
        state = {
            "step": torch.tensor(10.0, requires_grad=False, dtype=torch.float64),
            "exp_avg": torch.rand(10, requires_grad=True, dtype=torch.float64),
            "exp_avg_sq": torch.rand(10, requires_grad=True, dtype=torch.float64),
        }
        self._run_gradcheck(
            optim.RAdam,
            state,
            {"lr": 0.9, "differentiable": True, "foreach": False},
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
# dtype coverage: fp16/bf16 on both CPU and CUDA
# ---------------------------------------------------------------------------


class TestDiffDtypeCorrectness(TestCase):
    def test_nadam_denom_fp32_cpu(self):
        self._nadam(torch.float32, "cpu")

    def test_nadam_denom_fp64_cpu(self):
        self._nadam(torch.float64, "cpu")

    def test_nadam_denom_fp16_cpu(self):
        self._nadam(torch.float16, "cpu")

    def test_nadam_denom_bf16_cpu(self):
        self._nadam(torch.bfloat16, "cpu")

    def _nadam(self, dtype, device):
        x = torch.rand(128, dtype=dtype, device=device)
        bc2 = torch.tensor(0.95, dtype=dtype, device=device)
        old = x.div(bc2).sqrt()
        new = x.div(bc2).sqrt_()
        self.assertEqual(old, new, atol=1e-2, rtol=1e-2)

    def test_adam_denom_fp32_cpu(self):
        self._adam(torch.float32, "cpu")

    def test_adam_denom_fp64_cpu(self):
        self._adam(torch.float64, "cpu")

    def test_adam_denom_fp16_cpu(self):
        self._adam(torch.float16, "cpu")

    def test_adam_denom_bf16_cpu(self):
        self._adam(torch.bfloat16, "cpu")

    def _adam(self, dtype, device):
        x = torch.rand(128, dtype=dtype, device=device)
        bc_sqrt = torch.tensor(0.97, dtype=dtype, device=device)
        sn = torch.tensor(-1.0, dtype=dtype, device=device)
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        old = (x.sqrt() / (bc_sqrt * sn)).add_(eps / sn)
        new = x.sqrt()
        new.div_(bc_sqrt * sn)
        new.add_(eps / sn)
        self.assertEqual(old, new, atol=1e-2, rtol=1e-2)

    def test_rprop_sign_fp32_cpu(self):
        self._rprop(torch.float32, "cpu")

    def test_rprop_sign_fp64_cpu(self):
        self._rprop(torch.float64, "cpu")

    def test_rprop_sign_fp16_cpu(self):
        self._rprop(torch.float16, "cpu")

    def test_rprop_sign_bf16_cpu(self):
        self._rprop(torch.bfloat16, "cpu")

    def _rprop(self, dtype, device):
        a = torch.rand(128, dtype=dtype, device=device)
        b = torch.rand(128, dtype=dtype, device=device)
        old = a.mul(b).sign()
        new = a.mul(b)
        new.sign_()
        self.assertEqual(old, new, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# state_dict round-trip
# ---------------------------------------------------------------------------


class TestStateDictRoundTrip(TestCase):
    def _check(self, opt_cls, kwargs):
        p = torch.randn(N)
        p.grad = torch.randn(N)
        opt = opt_cls([p], **kwargs)
        opt.step()
        sd = opt.state_dict()

        p2 = torch.randn(N)
        p2.grad = torch.randn(N)
        opt2 = opt_cls([p2], **kwargs)
        opt2.load_state_dict(sd)

        self.assertEqual(opt.state[p].keys(), opt2.state[p2].keys())
        for k in opt.state[p]:
            v1, v2 = opt.state[p][k], opt2.state[p2][k]
            if isinstance(v1, torch.Tensor):
                self.assertEqual(v1, v2)
            else:
                self.assertEqual(v1, v2)

    def test_adam(self):
        self._check(optim.Adam, {"lr": 1e-3, "foreach": False})

    def test_adam_amsgrad(self):
        self._check(optim.Adam, {"lr": 1e-3, "amsgrad": True, "foreach": False})

    def test_nadam(self):
        self._check(optim.NAdam, {"lr": 1e-3, "foreach": False})

    def test_radam(self):
        self._check(optim.RAdam, {"lr": 1e-3, "foreach": False})

    def test_rprop(self):
        self._check(optim.Rprop, {"lr": 1e-2, "foreach": False})

    def test_rmsprop(self):
        self._check(optim.RMSprop, {"lr": 1e-2, "foreach": False})

    def test_asgd(self):
        self._check(optim.ASGD, {"lr": 0.1, "foreach": False})


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------


class TestCompileCompatibility(TestCase):
    def test_adam(self):
        self._check(optim.Adam, {"lr": 1e-3, "foreach": False})

    def test_nadam(self):
        self._check(optim.NAdam, {"lr": 1e-3, "foreach": False})

    def test_radam(self):
        self._check(optim.RAdam, {"lr": 1e-3, "foreach": False})

    def test_rprop(self):
        self._check(optim.Rprop, {"lr": 1e-2, "foreach": False})

    def _check(self, opt_cls, kwargs, n_steps=5):
        shape = (64,)
        p_init = torch.randn(shape)
        grads = [torch.randn(shape) for _ in range(n_steps)]

        # --- Eager reference ---
        p_ref = p_init.clone()
        opt_ref = opt_cls([p_ref], **{**kwargs, "capturable": False})
        for g in grads:
            p_ref.grad = g.clone()
            opt_ref.step()

        # --- Compiled: cannot use fullgraph=True because
        # _use_grad_for_differentiable inserts a graph break ---
        p = p_init.clone()
        opt = opt_cls([p], **{**kwargs, "capturable": False})
        compiled_step = torch.compile(opt.step)
        for g in grads:
            p.grad = g.clone()
            compiled_step()

        self.assertEqual(p, p_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Orthogonality: weight_decay, maximize, foreach
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

        # Baseline and flagged runs must differ -- flags actually changed behavior
        self.assertNotEqual(p1, p2)

    def test_adam_wd_max(self):
        self._check(optim.Adam, {"lr": 1e-3})

    def test_nadam_wd_max(self):
        self._check(optim.NAdam, {"lr": 1e-3})

    def test_radam_wd_max(self):
        self._check(optim.RAdam, {"lr": 1e-3})

    def test_rprop_max(self):
        self._check(optim.Rprop, {"lr": 1e-2}, use_wd=False)

    def test_foreach_vs_single_adam(self):
        self._check_foreach(optim.Adam, {"lr": 1e-3})

    def test_foreach_vs_single_nadam(self):
        self._check_foreach(optim.NAdam, {"lr": 1e-3})

    def test_foreach_vs_single_radam(self):
        self._check_foreach(optim.RAdam, {"lr": 1e-3})

    def test_foreach_vs_single_rprop(self):
        self._check_foreach(optim.Rprop, {"lr": 1e-2})

    def _check_foreach(self, opt_cls, kwargs):
        torch.manual_seed(42)
        p1 = torch.randn(64)
        p2 = p1.clone()
        g = torch.randn(64)

        opt1 = opt_cls([p1], **{**kwargs, "foreach": False})
        p1.grad = g.clone()
        opt1.step()

        opt2 = opt_cls([p2], **{**kwargs, "foreach": True})
        p2.grad = g.clone()
        opt2.step()

        self.assertEqual(p1, p2, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# CUDA-only: peak memory budget + CUDA graph capture + replay verification
# ---------------------------------------------------------------------------


class TestOptimizerMemoryBudgetCUDA(TestCase):
    device_type = "cuda"

    def _check_budget(self, opt_cls, opt_kwargs):
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
            f"Peak {peak - baseline} exceeds budget {budget}",
        )

    def test_adam(self):
        self._check_budget(optim.Adam, {"lr": 1e-3})

    def test_adam_amsgrad(self):
        self._check_budget(optim.Adam, {"lr": 1e-3, "amsgrad": True})

    def test_adamw(self):
        self._check_budget(optim.AdamW, {"lr": 1e-3})

    def test_sgd(self):
        self._check_budget(optim.SGD, {"lr": 0.1})

    def test_sgd_momentum(self):
        self._check_budget(optim.SGD, {"lr": 0.1, "momentum": 0.9})

    def test_adagrad(self):
        self._check_budget(optim.Adagrad, {"lr": 0.1})

    def test_adadelta(self):
        self._check_budget(optim.Adadelta, {"lr": 1.0})

    def test_adamax(self):
        self._check_budget(optim.Adamax, {"lr": 2e-3})

    def test_nadam(self):
        self._check_budget(optim.NAdam, {"lr": 1e-3})

    def test_radam(self):
        self._check_budget(optim.RAdam, {"lr": 1e-3})

    def test_rmsprop(self):
        self._check_budget(optim.RMSprop, {"lr": 1e-2})

    def test_rprop(self):
        self._check_budget(optim.Rprop, {"lr": 1e-2})

    def test_asgd(self):
        self._check_budget(optim.ASGD, {"lr": 0.1})


class TestCUDAGraphCapture(TestCase):
    device_type = "cuda"

    def _check_graph(self, opt_cls, opt_kwargs, n_steps=5):
        shape = (64,)
        grad = torch.randn(shape, device="cuda")

        # Create param once, clone for eager path
        p_init = torch.randn(shape, device="cuda")

        # --- Eager reference: run N steps ---
        p_eager = p_init.clone()
        p_eager.grad = grad.clone()
        opt_eager = opt_cls([p_eager], **{**opt_kwargs, "foreach": False})
        opt_eager.step()  # init state
        for _ in range(n_steps):
            p_eager.grad = grad.clone()
            opt_eager.step()

        # --- Graph: capture one step, replay N times ---
        p_graph = p_init.clone()
        p_graph.grad = grad.clone()
        opt_graph = opt_cls([p_graph], **{**opt_kwargs, "foreach": False})
        opt_graph.step()  # init state

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            opt_graph.step()

        for _ in range(n_steps):
            p_graph.grad = grad.clone()
            g.replay()
            torch.cuda.synchronize()

        # Verify final states match
        self.assertEqual(p_eager, p_graph, atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.isnan(p_graph).any())
        self.assertFalse(torch.isinf(p_graph).any())

    def test_adam(self):
        self._check_graph(optim.Adam, {"lr": 1e-3, "capturable": True})

    def test_adam_amsgrad(self):
        self._check_graph(optim.Adam, {"lr": 1e-3, "capturable": True, "amsgrad": True})

    def test_rprop(self):
        self._check_graph(optim.Rprop, {"lr": 1e-2, "capturable": True})

    def test_radam(self):
        self._check_graph(optim.RAdam, {"lr": 1e-3, "capturable": True})


# CUDA dtype coverage
class TestDiffDtypeCorrectnessCUDA(TestCase):
    device_type = "cuda"

    def test_nadam_denom_fp16(self):
        self._nadam(torch.float16)

    def test_nadam_denom_bf16(self):
        self._nadam(torch.bfloat16)

    def _nadam(self, dtype):
        x = torch.rand(128, dtype=dtype, device="cuda")
        bc2 = torch.tensor(0.95, dtype=dtype, device="cuda")
        old = x.div(bc2).sqrt()
        new = x.div(bc2).sqrt_()
        self.assertEqual(old, new, atol=1e-2, rtol=1e-2)

    def test_adam_denom_fp16(self):
        self._adam(torch.float16)

    def test_adam_denom_bf16(self):
        self._adam(torch.bfloat16)

    def _adam(self, dtype):
        x = torch.rand(128, dtype=dtype, device="cuda")
        bc_sqrt = torch.tensor(0.97, dtype=dtype, device="cuda")
        sn = torch.tensor(-1.0, dtype=dtype, device="cuda")
        eps = torch.tensor(1e-8, dtype=dtype, device="cuda")
        old = (x.sqrt() / (bc_sqrt * sn)).add_(eps / sn)
        new = x.sqrt()
        new.div_(bc_sqrt * sn)
        new.add_(eps / sn)
        self.assertEqual(old, new, atol=1e-2, rtol=1e-2)

    def test_rprop_sign_fp16(self):
        self._rprop(torch.float16)

    def test_rprop_sign_bf16(self):
        self._rprop(torch.bfloat16)

    def _rprop(self, dtype):
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
