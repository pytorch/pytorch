"""Comprehensive mathematical correctness tests for the Vulkan/Slang backend.

Tests numerical accuracy of ALL implemented ops against CPU reference with
multiple input patterns: random, edge cases, gradients, various shapes/dtypes.
Each test verifies the Vulkan GPU shader produces results matching PyTorch CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import math

# Tolerances for real GPU (tighter than SwiftShader)
RTOL = 1e-5
ATOL = 1e-5
# Relaxed for multi-step and reduction ops
RTOL_R = 1e-4
ATOL_R = 1e-4
# Very relaxed for long chains and training
RTOL_VR = 1e-3
ATOL_VR = 1e-3


@pytest.fixture(autouse=True)
def setup():
    try:
        import torch_vulkan
        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
    except ImportError:
        pytest.skip("torch_vulkan not installed")


def check(cpu, vk, rtol=RTOL, atol=ATOL):
    actual = vk.cpu() if vk.device.type != "cpu" else vk
    expected = cpu.cpu() if cpu.device.type != "cpu" else cpu
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


def pair(*shape, lo=-5.0, hi=5.0):
    """Uniform random pair in [lo, hi]."""
    cpu = torch.empty(*shape).uniform_(lo, hi)
    return cpu, cpu.to("vulkan")


def randn_pair(*shape):
    cpu = torch.randn(*shape)
    return cpu, cpu.to("vulkan")


def pos_pair(*shape):
    """Strictly positive pair."""
    cpu = torch.rand(*shape) + 0.01
    return cpu, cpu.to("vulkan")


def int_pair(*shape, lo=0, hi=10):
    cpu = torch.randint(lo, hi, shape)
    return cpu, cpu.to("vulkan")


# ═══════════════════════════════════════════════════════════════════
#  UNARY OPS — Exhaustive
# ═══════════════════════════════════════════════════════════════════

class TestUnaryMath:
    """Test all unary mathematical operations."""

    @pytest.mark.parametrize("shape", [(1,), (7,), (32,), (8, 16), (4, 8, 16), (2, 3, 4, 5)])
    def test_neg(self, shape):
        c, v = randn_pair(*shape)
        check(-c, -v)

    @pytest.mark.parametrize("shape", [(1,), (7,), (32,), (8, 16), (4, 8, 16)])
    def test_abs(self, shape):
        c, v = randn_pair(*shape)
        check(torch.abs(c), torch.abs(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16), (4, 8, 16)])
    def test_exp(self, shape):
        c, v = pair(*shape, lo=-3, hi=3)  # avoid huge values
        check(torch.exp(c), torch.exp(v))

    def test_exp_edge_cases(self):
        c = torch.tensor([0.0, 1.0, -1.0, -10.0, 10.0, -20.0])
        v = c.to("vulkan")
        check(torch.exp(c), torch.exp(v), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("shape", [(16,), (8, 16), (4, 8, 16)])
    def test_log(self, shape):
        c, v = pos_pair(*shape)
        check(torch.log(c), torch.log(v))

    def test_log_edge_cases(self):
        c = torch.tensor([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1e6])
        v = c.to("vulkan")
        check(torch.log(c), torch.log(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16), (4, 8, 16)])
    def test_log2(self, shape):
        c, v = pos_pair(*shape)
        check(torch.log2(c), torch.log2(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_log10(self, shape):
        c, v = pos_pair(*shape)
        check(torch.log10(c), torch.log10(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_log1p(self, shape):
        c, v = pair(*shape, lo=-0.5, hi=5.0)  # > -1
        check(torch.log1p(c), torch.log1p(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16), (4, 8, 16)])
    def test_sqrt(self, shape):
        c, v = pos_pair(*shape)
        check(torch.sqrt(c), torch.sqrt(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_rsqrt(self, shape):
        c, v = pos_pair(*shape)
        check(torch.rsqrt(c), torch.rsqrt(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16), (4, 8, 16)])
    def test_reciprocal(self, shape):
        c, v = pair(*shape, lo=0.1, hi=10.0)
        check(torch.reciprocal(c), torch.reciprocal(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16), (4, 8, 16)])
    def test_sin(self, shape):
        c, v = pair(*shape, lo=-3.14, hi=3.14)
        check(torch.sin(c), torch.sin(v), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_cos(self, shape):
        c, v = pair(*shape, lo=-3.14, hi=3.14)
        check(torch.cos(c), torch.cos(v), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_tan(self, shape):
        c, v = pair(*shape, lo=-1.0, hi=1.0)  # avoid singularities
        check(torch.tan(c), torch.tan(v), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_atan(self, shape):
        c, v = randn_pair(*shape)
        check(torch.atan(c), torch.atan(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_erf(self, shape):
        c, v = randn_pair(*shape)
        check(torch.erf(c), torch.erf(v))

    def test_erf_edge_cases(self):
        c = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        v = c.to("vulkan")
        check(torch.erf(c), torch.erf(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_ceil(self, shape):
        c, v = randn_pair(*shape)
        check(torch.ceil(c), torch.ceil(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_floor(self, shape):
        c, v = randn_pair(*shape)
        check(torch.floor(c), torch.floor(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_round(self, shape):
        c, v = randn_pair(*shape)
        check(torch.round(c), torch.round(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_sign(self, shape):
        c, v = randn_pair(*shape)
        check(torch.sign(c), torch.sign(v))

    def test_sign_zero(self):
        c = torch.tensor([-2.0, -0.0, 0.0, 0.0, 2.0])
        v = c.to("vulkan")
        check(torch.sign(c), torch.sign(v))

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_isnan(self, shape):
        c = torch.randn(*shape)
        c[0] = float("nan")
        c[3] = float("nan")
        v = c.to("vulkan")
        assert torch.equal(torch.isnan(c), torch.isnan(v).cpu())

    @pytest.mark.parametrize("shape", [(16,), (8, 16)])
    def test_isinf(self, shape):
        c = torch.randn(*shape)
        c[0] = float("inf")
        c[1] = float("-inf")
        v = c.to("vulkan")
        assert torch.equal(torch.isinf(c), torch.isinf(v).cpu())


# ═══════════════════════════════════════════════════════════════════
#  BINARY OPS — Exhaustive
# ═══════════════════════════════════════════════════════════════════

class TestBinaryMath:
    """Test all binary operations with various input patterns."""

    @pytest.mark.parametrize("shape", [(1,), (16,), (8, 16), (4, 8, 16), (2, 3, 4, 5)])
    def test_add(self, shape):
        a_c, a_v = randn_pair(*shape)
        b_c, b_v = randn_pair(*shape)
        check(a_c + b_c, a_v + b_v)

    @pytest.mark.parametrize("shape", [(1,), (16,), (8, 16), (4, 8, 16)])
    def test_sub(self, shape):
        a_c, a_v = randn_pair(*shape)
        b_c, b_v = randn_pair(*shape)
        check(a_c - b_c, a_v - b_v)

    @pytest.mark.parametrize("shape", [(1,), (16,), (8, 16), (4, 8, 16)])
    def test_mul(self, shape):
        a_c, a_v = randn_pair(*shape)
        b_c, b_v = randn_pair(*shape)
        check(a_c * b_c, a_v * b_v)

    @pytest.mark.parametrize("shape", [(1,), (16,), (8, 16), (4, 8, 16)])
    def test_div(self, shape):
        a_c, a_v = randn_pair(*shape)
        b_c, b_v = pair(*shape, lo=0.1, hi=5.0)
        check(a_c / b_c, a_v / b_v)

    @pytest.mark.parametrize("scalar", [0.0, 1.0, -1.0, 2.5, -0.5, 100.0])
    def test_add_scalar(self, scalar):
        c, v = randn_pair(32)
        check(c + scalar, v + scalar)

    @pytest.mark.parametrize("scalar", [1.0, -1.0, 2.5, 0.001])
    def test_mul_scalar(self, scalar):
        c, v = randn_pair(32)
        check(c * scalar, v * scalar)

    @pytest.mark.parametrize("scalar", [0.5, 2.0, -1.0, 10.0])
    def test_div_scalar(self, scalar):
        c, v = randn_pair(32)
        check(c / scalar, v / scalar)

    def test_add_with_alpha(self):
        a_c, a_v = randn_pair(16, 16)
        b_c, b_v = randn_pair(16, 16)
        check(torch.add(a_c, b_c, alpha=2.5), torch.add(a_v, b_v, alpha=2.5))

    def test_sub_with_alpha(self):
        a_c, a_v = randn_pair(16, 16)
        b_c, b_v = randn_pair(16, 16)
        check(torch.sub(a_c, b_c, alpha=0.5), torch.sub(a_v, b_v, alpha=0.5))

    @pytest.mark.parametrize("shapes", [
        ((4, 8), (1, 8)),
        ((4, 8), (8,)),
        ((3, 4, 8), (1, 1, 8)),
        ((3, 4, 8), (4, 8)),
        ((2, 1, 4), (1, 3, 4)),
    ])
    def test_broadcast_add(self, shapes):
        a_c = torch.randn(*shapes[0])
        b_c = torch.randn(*shapes[1])
        a_v, b_v = a_c.to("vulkan"), b_c.to("vulkan")
        check(a_c + b_c, a_v + b_v)

    @pytest.mark.parametrize("shapes", [
        ((4, 8), (1, 8)),
        ((3, 4, 8), (4, 8)),
    ])
    def test_broadcast_mul(self, shapes):
        a_c = torch.randn(*shapes[0])
        b_c = torch.randn(*shapes[1])
        a_v, b_v = a_c.to("vulkan"), b_c.to("vulkan")
        check(a_c * b_c, a_v * b_v)

    def test_pow_positive_base(self):
        base_c, base_v = pos_pair(16, 16)
        exp_c, exp_v = pair(16, 16, lo=0.5, hi=3.0)
        check(torch.pow(base_c, exp_c), torch.pow(base_v, exp_v), rtol=RTOL_R, atol=ATOL_R)

    def test_pow_integer_exponent(self):
        c, v = randn_pair(32)
        check(torch.pow(c, 2), torch.pow(v, 2))
        check(torch.pow(c, 3), torch.pow(v, 3), rtol=RTOL_R, atol=ATOL_R)

    def test_pow_scalar_exponent(self):
        c, v = pos_pair(32)
        check(torch.pow(c, 0.5), torch.pow(v, 0.5))
        check(torch.pow(c, 2.0), torch.pow(v, 2.0))

    def test_fmod(self):
        a_c, a_v = pair(32, lo=-10, hi=10)
        b_c, b_v = pair(32, lo=1, hi=5)
        check(torch.fmod(a_c, b_c), torch.fmod(a_v, b_v))

    def test_remainder(self):
        a_c, a_v = pair(32, lo=-10, hi=10)
        b_c, b_v = pair(32, lo=1, hi=5)
        check(torch.remainder(a_c, b_c), torch.remainder(a_v, b_v))

    def test_atan2(self):
        y_c, y_v = randn_pair(16, 16)
        x_c, x_v = randn_pair(16, 16)
        check(torch.atan2(y_c, x_c), torch.atan2(y_v, x_v))


# ═══════════════════════════════════════════════════════════════════
#  IN-PLACE OPS
# ═══════════════════════════════════════════════════════════════════

class TestInPlaceMath:
    """Test in-place operations produce correct results."""

    def test_add_inplace(self):
        c, v = randn_pair(32, 32)
        b_c, b_v = randn_pair(32, 32)
        c.add_(b_c)
        v.add_(b_v)
        check(c, v)

    def test_sub_inplace(self):
        c, v = randn_pair(32, 32)
        b_c, b_v = randn_pair(32, 32)
        c.sub_(b_c)
        v.sub_(b_v)
        check(c, v)

    def test_mul_inplace(self):
        c, v = randn_pair(32, 32)
        b_c, b_v = randn_pair(32, 32)
        c.mul_(b_c)
        v.mul_(b_v)
        check(c, v)

    def test_div_inplace(self):
        c, v = randn_pair(32, 32)
        b_c, b_v = pos_pair(32, 32)
        c.div_(b_c)
        v.div_(b_v)
        check(c, v)

    def test_add_scalar_inplace(self):
        c, v = randn_pair(32)
        c.add_(3.14)
        v.add_(3.14)
        check(c, v)

    def test_mul_scalar_inplace(self):
        c, v = randn_pair(32)
        c.mul_(0.5)
        v.mul_(0.5)
        check(c, v)

    def test_fill_inplace(self):
        c = torch.empty(16, 16)
        v = torch.empty(16, 16, device="vulkan")
        c.fill_(3.14)
        v.fill_(3.14)
        check(c, v)

    def test_zero_inplace(self):
        c, v = randn_pair(16, 16)
        c.zero_()
        v.zero_()
        check(c, v)

    def test_sequential_inplace(self):
        """Multiple in-place ops in sequence."""
        c, v = randn_pair(32, 32)
        b_c, b_v = randn_pair(32, 32)
        c.add_(b_c); v.add_(b_v)
        c.mul_(2.0); v.mul_(2.0)
        c.sub_(1.0); v.sub_(1.0)
        check(c, v, rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  COMPARISON OPS
# ═══════════════════════════════════════════════════════════════════

class TestComparisonMath:

    @pytest.mark.parametrize("op", [torch.eq, torch.ne, torch.lt, torch.gt, torch.le, torch.ge])
    def test_comparison_tensor(self, op):
        a_c, a_v = randn_pair(32, 32)
        b_c, b_v = randn_pair(32, 32)
        assert torch.equal(op(a_c, b_c), op(a_v, b_v).cpu())

    @pytest.mark.parametrize("op", [torch.eq, torch.ne, torch.lt, torch.gt, torch.le, torch.ge])
    def test_comparison_scalar(self, op):
        c, v = randn_pair(32, 32)
        assert torch.equal(op(c, 0.0), op(v, 0.0).cpu())

    def test_where(self):
        cond_c = torch.randn(16, 16) > 0
        a_c, a_v = randn_pair(16, 16)
        b_c, b_v = randn_pair(16, 16)
        cond_v = cond_c.to("vulkan")
        check(torch.where(cond_c, a_c, b_c), torch.where(cond_v, a_v, b_v))


# ═══════════════════════════════════════════════════════════════════
#  ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

class TestActivationMath:
    """Verify activation function forward accuracy."""

    @pytest.mark.parametrize("shape", [(1,), (32,), (8, 16), (4, 8, 16)])
    def test_relu(self, shape):
        c, v = randn_pair(*shape)
        check(F.relu(c), F.relu(v))

    @pytest.mark.parametrize("shape", [(32,), (8, 16)])
    def test_sigmoid(self, shape):
        c, v = randn_pair(*shape)
        check(torch.sigmoid(c), torch.sigmoid(v))

    def test_sigmoid_extremes(self):
        c = torch.tensor([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
        v = c.to("vulkan")
        check(torch.sigmoid(c), torch.sigmoid(v))

    @pytest.mark.parametrize("shape", [(32,), (8, 16)])
    def test_tanh(self, shape):
        c, v = randn_pair(*shape)
        check(torch.tanh(c), torch.tanh(v))

    def test_tanh_extremes(self):
        c = torch.tensor([-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0])
        v = c.to("vulkan")
        check(torch.tanh(c), torch.tanh(v))

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_gelu(self, approximate):
        c, v = randn_pair(32, 32)
        # GELU exact (none) uses erf which accumulates fp error → relax tolerance
        check(F.gelu(c, approximate=approximate), F.gelu(v, approximate=approximate),
              rtol=RTOL_VR, atol=RTOL_VR)

    @pytest.mark.parametrize("shape", [(32,), (8, 16)])
    def test_silu(self, shape):
        c, v = randn_pair(*shape)
        check(F.silu(c), F.silu(v))

    @pytest.mark.parametrize("neg_slope", [0.01, 0.1, 0.2])
    def test_leaky_relu(self, neg_slope):
        c, v = randn_pair(32, 32)
        check(F.leaky_relu(c, neg_slope), F.leaky_relu(v, neg_slope))

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_elu(self, alpha):
        c, v = randn_pair(32, 32)
        check(F.elu(c, alpha), F.elu(v, alpha))

    def test_selu(self):
        c, v = randn_pair(32, 32)
        check(F.selu(c), F.selu(v))

    def test_prelu(self):
        c, v = randn_pair(1, 8, 16)
        weight = torch.randn(8)
        wv = weight.to("vulkan")
        check(F.prelu(c, weight), F.prelu(v, wv))

    def test_hardtanh(self):
        c, v = randn_pair(32, 32)
        check(F.hardtanh(c, -1.0, 1.0), F.hardtanh(v, -1.0, 1.0))

    def test_hardswish(self):
        c, v = randn_pair(32, 32)
        check(F.hardswish(c), F.hardswish(v))

    def test_hardsigmoid(self):
        c, v = randn_pair(32, 32)
        check(F.hardsigmoid(c), F.hardsigmoid(v))

    @pytest.mark.parametrize("beta", [1.0, 2.0])
    def test_softplus(self, beta):
        c, v = randn_pair(32, 32)
        check(F.softplus(c, beta=beta), F.softplus(v, beta=beta))

    def test_mish(self):
        c, v = randn_pair(32, 32)
        check(F.mish(c), F.mish(v), rtol=RTOL_R, atol=ATOL_R)

    def test_clamp(self):
        c, v = randn_pair(32, 32)
        check(torch.clamp(c, -1.0, 1.0), torch.clamp(v, -1.0, 1.0))

    def test_clamp_min(self):
        c, v = randn_pair(32)
        check(torch.clamp_min(c, 0.0), torch.clamp_min(v, 0.0))

    def test_clamp_max(self):
        c, v = randn_pair(32)
        check(torch.clamp_max(c, 0.5), torch.clamp_max(v, 0.5))


# ═══════════════════════════════════════════════════════════════════
#  ACTIVATION BACKWARD
# ═══════════════════════════════════════════════════════════════════

class TestActivationBackward:
    """Verify activation backward (gradient) accuracy."""

    def _check_grad(self, fn, shape=(32, 32), rtol=RTOL_R, atol=ATOL_R):
        x_c = torch.randn(*shape, requires_grad=True)
        x_v = x_c.detach().to("vulkan").requires_grad_(True)
        y_c = fn(x_c)
        y_v = fn(x_v)
        go = torch.randn_like(y_c)
        go_v = go.to("vulkan")
        y_c.backward(go)
        y_v.backward(go_v)
        check(x_c.grad, x_v.grad, rtol=rtol, atol=atol)

    def test_relu_backward(self):
        self._check_grad(F.relu)

    def test_sigmoid_backward(self):
        self._check_grad(torch.sigmoid)

    def test_tanh_backward(self):
        self._check_grad(torch.tanh)

    def test_gelu_backward(self):
        self._check_grad(F.gelu, rtol=5e-3, atol=5e-3)

    def test_silu_backward(self):
        self._check_grad(F.silu)

    def test_leaky_relu_backward(self):
        self._check_grad(lambda x: F.leaky_relu(x, 0.01))

    def test_elu_backward(self):
        self._check_grad(F.elu)

    def test_hardtanh_backward(self):
        self._check_grad(lambda x: F.hardtanh(x, -1.0, 1.0))

    def test_hardswish_backward(self):
        self._check_grad(F.hardswish)

    def test_hardsigmoid_backward(self):
        self._check_grad(F.hardsigmoid)

    def test_softplus_backward(self):
        self._check_grad(F.softplus)

    def test_mish_backward(self):
        self._check_grad(F.mish, rtol=RTOL_VR, atol=ATOL_VR)


# ═══════════════════════════════════════════════════════════════════
#  REDUCTIONS
# ═══════════════════════════════════════════════════════════════════

class TestReductionMath:
    """Test reduction operations for mathematical correctness."""

    @pytest.mark.parametrize("shape", [(32,), (8, 16), (4, 8, 16)])
    def test_sum_full(self, shape):
        c, v = randn_pair(*shape)
        check(c.sum(), v.sum(), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_sum_dim(self, dim):
        c, v = randn_pair(8, 16)
        check(c.sum(dim), v.sum(dim), rtol=RTOL_R, atol=ATOL_R)

    def test_sum_dim_keepdim(self):
        c, v = randn_pair(8, 16)
        check(c.sum(1, keepdim=True), v.sum(1, keepdim=True), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_mean_dim(self, dim):
        c, v = randn_pair(8, 16)
        check(c.mean(dim), v.mean(dim), rtol=RTOL_R, atol=ATOL_R)

    def test_mean_multi_dim(self):
        c, v = randn_pair(4, 8, 16)
        check(c.mean([1, 2]), v.mean([1, 2]), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("dim", [0, 1])
    def test_amax(self, dim):
        c, v = randn_pair(16, 16)
        check(c.amax(dim), v.amax(dim))

    @pytest.mark.parametrize("dim", [0, 1])
    def test_amin(self, dim):
        c, v = randn_pair(16, 16)
        check(c.amin(dim), v.amin(dim))

    @pytest.mark.parametrize("dim", [0, 1])
    def test_argmax(self, dim):
        c, v = randn_pair(16, 16)
        assert torch.equal(c.argmax(dim), v.argmax(dim).cpu())

    @pytest.mark.parametrize("dim", [0, 1])
    def test_argmin(self, dim):
        c, v = randn_pair(16, 16)
        assert torch.equal(c.argmin(dim), v.argmin(dim).cpu())

    def test_max_dim_values_indices(self):
        c, v = randn_pair(16, 16)
        cv, ci = c.max(dim=1)
        vv, vi = v.max(dim=1)
        check(cv, vv)
        assert torch.equal(ci, vi.cpu())

    def test_min_dim_values_indices(self):
        c, v = randn_pair(16, 16)
        cv, ci = c.min(dim=1)
        vv, vi = v.min(dim=1)
        check(cv, vv)
        assert torch.equal(ci, vi.cpu())

    @pytest.mark.parametrize("dim", [0, 1])
    def test_prod_dim(self, dim):
        c, v = pair(8, 8, lo=0.5, hi=1.5)  # keep products manageable
        check(c.prod(dim), v.prod(dim), rtol=RTOL_R, atol=ATOL_R)

    def test_cumsum(self):
        c, v = randn_pair(16, 16)
        check(c.cumsum(0), v.cumsum(0), rtol=RTOL_R, atol=ATOL_R)
        check(c.cumsum(1), v.cumsum(1), rtol=RTOL_R, atol=ATOL_R)

    def test_cumprod(self):
        c, v = pair(8, 8, lo=0.8, hi=1.2)  # keep products stable
        check(c.cumprod(0), v.cumprod(0), rtol=RTOL_R, atol=ATOL_R)

    def test_any(self):
        c = torch.tensor([0.0, 0.0, 1.0, 0.0])
        v = c.to("vulkan")
        assert c.any().item() == v.any().cpu().item()

    def test_all(self):
        c = torch.tensor([1.0, 1.0, 1.0, 1.0])
        v = c.to("vulkan")
        assert c.all().item() == v.all().cpu().item()
        c2 = torch.tensor([1.0, 0.0, 1.0])
        v2 = c2.to("vulkan")
        assert c2.all().item() == v2.all().cpu().item()

    def test_norm_l2(self):
        c, v = randn_pair(32, 32)
        check(torch.norm(c), torch.norm(v), rtol=RTOL_R, atol=ATOL_R)

    def test_norm_dim(self):
        c, v = randn_pair(8, 16)
        check(torch.norm(c, dim=1), torch.norm(v, dim=1), rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  BLAS (Matrix Multiply, Linear)
# ═══════════════════════════════════════════════════════════════════

class TestBLASMath:
    """Test matrix operations for mathematical correctness."""

    @pytest.mark.parametrize("m,k,n", [(8, 8, 8), (16, 32, 8), (1, 16, 1), (64, 64, 64), (128, 256, 64)])
    def test_mm(self, m, k, n):
        a_c, a_v = randn_pair(m, k)
        b_c, b_v = randn_pair(k, n)
        check(torch.mm(a_c, b_c), torch.mm(a_v, b_v), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("batch,m,k,n", [(2, 8, 8, 8), (4, 16, 32, 8), (1, 64, 64, 64)])
    def test_bmm(self, batch, m, k, n):
        a_c, a_v = randn_pair(batch, m, k)
        b_c, b_v = randn_pair(batch, k, n)
        check(torch.bmm(a_c, b_c), torch.bmm(a_v, b_v), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("m,k,n", [(8, 16, 4), (32, 32, 32)])
    def test_addmm(self, m, k, n):
        bias_c, bias_v = randn_pair(m, n)
        a_c, a_v = randn_pair(m, k)
        b_c, b_v = randn_pair(k, n)
        check(torch.addmm(bias_c, a_c, b_c), torch.addmm(bias_v, a_v, b_v), rtol=RTOL_R, atol=ATOL_R)

    def test_addmm_alpha_beta(self):
        bias_c, bias_v = randn_pair(8, 4)
        a_c, a_v = randn_pair(8, 16)
        b_c, b_v = randn_pair(16, 4)
        check(
            torch.addmm(bias_c, a_c, b_c, beta=0.5, alpha=2.0),
            torch.addmm(bias_v, a_v, b_v, beta=0.5, alpha=2.0),
            rtol=RTOL_R, atol=ATOL_R
        )

    @pytest.mark.parametrize("in_f,out_f", [(16, 8), (32, 32), (64, 128), (784, 256)])
    def test_linear(self, in_f, out_f):
        x_c, x_v = randn_pair(4, in_f)
        w_c = torch.randn(out_f, in_f)
        b_c = torch.randn(out_f)
        w_v, b_v = w_c.to("vulkan"), b_c.to("vulkan")
        check(F.linear(x_c, w_c, b_c), F.linear(x_v, w_v, b_v), rtol=RTOL_R, atol=ATOL_R)

    def test_linear_no_bias(self):
        x_c, x_v = randn_pair(4, 32)
        w_c = torch.randn(16, 32)
        w_v = w_c.to("vulkan")
        check(F.linear(x_c, w_c), F.linear(x_v, w_v), rtol=RTOL_R, atol=ATOL_R)

    def test_linear_3d(self):
        x_c, x_v = randn_pair(2, 8, 32)
        w_c = torch.randn(16, 32)
        b_c = torch.randn(16)
        w_v, b_v = w_c.to("vulkan"), b_c.to("vulkan")
        check(F.linear(x_c, w_c, b_c), F.linear(x_v, w_v, b_v), rtol=RTOL_R, atol=ATOL_R)

    def test_mm_backward(self):
        a_c = torch.randn(8, 16, requires_grad=True)
        b_c = torch.randn(16, 4, requires_grad=True)
        a_v = a_c.detach().to("vulkan").requires_grad_(True)
        b_v = b_c.detach().to("vulkan").requires_grad_(True)
        y_c = torch.mm(a_c, b_c)
        y_v = torch.mm(a_v, b_v)
        go = torch.randn_like(y_c)
        y_c.backward(go)
        y_v.backward(go.to("vulkan"))
        check(a_c.grad, a_v.grad, rtol=RTOL_R, atol=ATOL_R)
        check(b_c.grad, b_v.grad, rtol=RTOL_R, atol=ATOL_R)

    def test_linear_backward(self):
        x_c = torch.randn(4, 32, requires_grad=True)
        w = torch.randn(16, 32)
        b = torch.randn(16)
        x_v = x_c.detach().to("vulkan").requires_grad_(True)
        w_v, b_v = w.to("vulkan"), b.to("vulkan")
        y_c = F.linear(x_c, w, b)
        y_v = F.linear(x_v, w_v, b_v)
        go = torch.randn_like(y_c)
        y_c.backward(go)
        y_v.backward(go.to("vulkan"))
        check(x_c.grad, x_v.grad, rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  SOFTMAX & LOG_SOFTMAX
# ═══════════════════════════════════════════════════════════════════

class TestSoftmaxMath:

    @pytest.mark.parametrize("shape,dim", [
        ((16,), 0), ((8, 16), 0), ((8, 16), 1), ((4, 8, 16), 2),
        ((2, 256), 1), ((2, 1024), 1),
    ])
    def test_softmax(self, shape, dim):
        c, v = randn_pair(*shape)
        check(F.softmax(c, dim=dim), F.softmax(v, dim=dim), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("shape,dim", [
        ((8, 16), 1), ((4, 8, 16), 2), ((2, 256), 1),
    ])
    def test_log_softmax(self, shape, dim):
        c, v = randn_pair(*shape)
        check(F.log_softmax(c, dim=dim), F.log_softmax(v, dim=dim), rtol=RTOL_R, atol=ATOL_R)

    def test_softmax_backward(self):
        c = torch.randn(8, 16, requires_grad=True)
        v = c.detach().to("vulkan").requires_grad_(True)
        y_c = F.softmax(c, dim=1)
        y_v = F.softmax(v, dim=1)
        go = torch.randn_like(y_c)
        y_c.backward(go)
        y_v.backward(go.to("vulkan"))
        check(c.grad, v.grad, rtol=RTOL_R, atol=ATOL_R)

    def test_log_softmax_backward(self):
        c = torch.randn(8, 16, requires_grad=True)
        v = c.detach().to("vulkan").requires_grad_(True)
        y_c = F.log_softmax(c, dim=1)
        y_v = F.log_softmax(v, dim=1)
        go = torch.randn_like(y_c)
        y_c.backward(go)
        y_v.backward(go.to("vulkan"))
        check(c.grad, v.grad, rtol=RTOL_R, atol=ATOL_R)

    def test_softmax_numerical_stability(self):
        """Softmax should be stable with large inputs."""
        c = torch.tensor([[100.0, 200.0, 300.0], [-100.0, -200.0, -300.0]])
        v = c.to("vulkan")
        check(F.softmax(c, dim=1), F.softmax(v, dim=1), rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  NORMALIZATION
# ═══════════════════════════════════════════════════════════════════

class TestNormalizationMath:

    @pytest.mark.parametrize("shape,norm_shape", [
        ((4, 16), [16]),
        ((4, 8, 16), [16]),
        ((4, 8, 16), [8, 16]),
        ((2, 32), [32]),
    ])
    def test_layer_norm(self, shape, norm_shape):
        c, v = randn_pair(*shape)
        w = torch.randn(norm_shape)
        b = torch.randn(norm_shape)
        wv, bv = w.to("vulkan"), b.to("vulkan")
        check(F.layer_norm(c, norm_shape, w, b), F.layer_norm(v, norm_shape, wv, bv),
              rtol=RTOL_R, atol=ATOL_R)

    def test_layer_norm_no_affine(self):
        c, v = randn_pair(4, 16)
        check(F.layer_norm(c, [16]), F.layer_norm(v, [16]), rtol=RTOL_R, atol=ATOL_R)

    def test_layer_norm_backward(self):
        c = torch.randn(4, 16, requires_grad=True)
        v = c.detach().to("vulkan").requires_grad_(True)
        w = torch.ones(16)
        b = torch.zeros(16)
        y_c = F.layer_norm(c, [16], w, b)
        y_v = F.layer_norm(v, [16], w.to("vulkan"), b.to("vulkan"))
        go = torch.randn_like(y_c)
        y_c.backward(go)
        y_v.backward(go.to("vulkan"))
        check(c.grad, v.grad, rtol=RTOL_VR, atol=ATOL_VR)

    def test_batch_norm_eval(self):
        c, v = randn_pair(4, 8, 4, 4)
        bn = nn.BatchNorm2d(8)
        bn.eval()
        bn_v = nn.BatchNorm2d(8)
        bn_v.load_state_dict(bn.state_dict())
        bn_v.eval()
        bn_v = bn_v.to("vulkan")
        check(bn(c), bn_v(v), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("groups", [1, 2, 4, 8])
    def test_group_norm(self, groups):
        c, v = randn_pair(4, 8, 4, 4)
        w = torch.randn(8)
        b = torch.randn(8)
        wv, bv = w.to("vulkan"), b.to("vulkan")
        c_out = F.group_norm(c, groups, w, b)
        v_out = F.group_norm(v, groups, wv, bv)
        check(c_out, v_out, rtol=RTOL_R, atol=ATOL_R)

    def test_group_norm_backward(self):
        c = torch.randn(4, 8, 4, requires_grad=True)
        v = c.detach().to("vulkan").requires_grad_(True)
        w = torch.ones(8)
        b = torch.zeros(8)
        y_c = F.group_norm(c, 4, w, b)
        y_v = F.group_norm(v, 4, w.to("vulkan"), b.to("vulkan"))
        go = torch.randn_like(y_c)
        y_c.backward(go)
        y_v.backward(go.to("vulkan"))
        check(c.grad, v.grad, rtol=RTOL_VR, atol=ATOL_VR)


# ═══════════════════════════════════════════════════════════════════
#  CONVOLUTION
# ═══════════════════════════════════════════════════════════════════

class TestConvMath:

    @pytest.mark.parametrize("in_c,out_c,ks,stride,pad", [
        (1, 4, 3, 1, 0), (3, 8, 3, 1, 1), (3, 16, 5, 2, 2),
        (8, 16, 3, 1, 1), (16, 32, 1, 1, 0),
    ])
    def test_conv2d(self, in_c, out_c, ks, stride, pad):
        x_c, x_v = randn_pair(2, in_c, 16, 16)
        w = torch.randn(out_c, in_c, ks, ks)
        b = torch.randn(out_c)
        wv, bv = w.to("vulkan"), b.to("vulkan")
        check(F.conv2d(x_c, w, b, stride=stride, padding=pad),
              F.conv2d(x_v, wv, bv, stride=stride, padding=pad),
              rtol=RTOL_R, atol=ATOL_R)

    def test_conv2d_no_bias(self):
        x_c, x_v = randn_pair(2, 3, 8, 8)
        w = torch.randn(8, 3, 3, 3)
        wv = w.to("vulkan")
        check(F.conv2d(x_c, w, padding=1), F.conv2d(x_v, wv, padding=1), rtol=RTOL_R, atol=ATOL_R)

    def test_conv2d_dilation(self):
        x_c, x_v = randn_pair(2, 3, 16, 16)
        w = torch.randn(8, 3, 3, 3)
        wv = w.to("vulkan")
        check(F.conv2d(x_c, w, dilation=2), F.conv2d(x_v, wv, dilation=2), rtol=RTOL_R, atol=ATOL_R)

    def test_conv2d_backward(self):
        x_c = torch.randn(2, 3, 8, 8, requires_grad=True)
        x_v = x_c.detach().to("vulkan").requires_grad_(True)
        w = torch.randn(8, 3, 3, 3)
        b = torch.randn(8)
        y_c = F.conv2d(x_c, w, b, padding=1)
        y_v = F.conv2d(x_v, w.to("vulkan"), b.to("vulkan"), padding=1)
        go = torch.randn_like(y_c)
        y_c.backward(go)
        y_v.backward(go.to("vulkan"))
        check(x_c.grad, x_v.grad, rtol=RTOL_VR, atol=ATOL_VR)

    def test_conv_transpose2d(self):
        x_c, x_v = randn_pair(2, 8, 4, 4)
        w = torch.randn(8, 4, 3, 3)
        b = torch.zeros(4)  # explicit zero bias (None bias not supported)
        wv, bv = w.to("vulkan"), b.to("vulkan")
        check(F.conv_transpose2d(x_c, w, bias=b, padding=1),
              F.conv_transpose2d(x_v, wv, bias=bv, padding=1),
              rtol=RTOL_R, atol=ATOL_R)

    def test_conv1d(self):
        x_c, x_v = randn_pair(2, 3, 32)
        w = torch.randn(8, 3, 3)
        b = torch.randn(8)
        wv, bv = w.to("vulkan"), b.to("vulkan")
        check(F.conv1d(x_c, w, b, padding=1), F.conv1d(x_v, wv, bv, padding=1),
              rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  POOLING
# ═══════════════════════════════════════════════════════════════════

class TestPoolingMath:

    @pytest.mark.parametrize("ks,stride,pad", [(2, 2, 0), (3, 2, 1), (3, 1, 1)])
    def test_max_pool2d(self, ks, stride, pad):
        c, v = randn_pair(2, 4, 16, 16)
        check(F.max_pool2d(c, ks, stride, pad), F.max_pool2d(v, ks, stride, pad))

    @pytest.mark.parametrize("ks,stride,pad", [(2, 2, 0), (3, 2, 1), (3, 1, 1)])
    def test_avg_pool2d(self, ks, stride, pad):
        c, v = randn_pair(2, 4, 16, 16)
        check(F.avg_pool2d(c, ks, stride, pad), F.avg_pool2d(v, ks, stride, pad),
              rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("out_size", [(1, 1), (4, 4), (7, 7)])
    def test_adaptive_avg_pool2d(self, out_size):
        c, v = randn_pair(2, 4, 16, 16)
        check(F.adaptive_avg_pool2d(c, out_size), F.adaptive_avg_pool2d(v, out_size),
              rtol=RTOL_R, atol=ATOL_R)

    def test_max_pool2d_backward(self):
        c = torch.randn(2, 4, 8, 8, requires_grad=True)
        v = c.detach().to("vulkan").requires_grad_(True)
        y_c = F.max_pool2d(c, 2, 2)
        y_v = F.max_pool2d(v, 2, 2)
        go = torch.randn_like(y_c)
        y_c.backward(go)
        y_v.backward(go.to("vulkan"))
        check(c.grad, v.grad, rtol=RTOL_R, atol=ATOL_R)

    def test_avg_pool2d_backward(self):
        c = torch.randn(2, 4, 8, 8, requires_grad=True)
        v = c.detach().to("vulkan").requires_grad_(True)
        y_c = F.avg_pool2d(c, 2, 2)
        y_v = F.avg_pool2d(v, 2, 2)
        go = torch.randn_like(y_c)
        y_c.backward(go)
        y_v.backward(go.to("vulkan"))
        check(c.grad, v.grad, rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

class TestLossMath:

    def test_mse_loss(self):
        pred_c, pred_v = randn_pair(8, 16)
        tgt_c, tgt_v = randn_pair(8, 16)
        check(F.mse_loss(pred_c, tgt_c), F.mse_loss(pred_v, tgt_v), rtol=RTOL_R, atol=ATOL_R)

    def test_mse_loss_backward(self):
        pred_c = torch.randn(8, 16, requires_grad=True)
        tgt_c = torch.randn(8, 16)
        pred_v = pred_c.detach().to("vulkan").requires_grad_(True)
        tgt_v = tgt_c.to("vulkan")
        loss_c = F.mse_loss(pred_c, tgt_c)
        loss_v = F.mse_loss(pred_v, tgt_v)
        loss_c.backward()
        loss_v.backward()
        check(pred_c.grad, pred_v.grad, rtol=RTOL_R, atol=ATOL_R)

    def test_cross_entropy(self):
        logits_c, logits_v = randn_pair(8, 10)
        tgt = torch.randint(0, 10, (8,))
        tgt_v = tgt.to("vulkan")
        check(F.cross_entropy(logits_c, tgt), F.cross_entropy(logits_v, tgt_v),
              rtol=RTOL_R, atol=ATOL_R)

    def test_cross_entropy_backward(self):
        logits_c = torch.randn(8, 10, requires_grad=True)
        tgt = torch.randint(0, 10, (8,))
        logits_v = logits_c.detach().to("vulkan").requires_grad_(True)
        tgt_v = tgt.to("vulkan")
        loss_c = F.cross_entropy(logits_c, tgt)
        loss_v = F.cross_entropy(logits_v, tgt_v)
        loss_c.backward()
        loss_v.backward()
        check(logits_c.grad, logits_v.grad, rtol=RTOL_R, atol=ATOL_R)

    def test_cross_entropy_ignore_index(self):
        logits_c = torch.randn(8, 10, requires_grad=True)
        tgt = torch.randint(0, 10, (8,))
        tgt[2] = -100  # ignore
        tgt[5] = -100
        logits_v = logits_c.detach().to("vulkan").requires_grad_(True)
        tgt_v = tgt.to("vulkan")
        loss_c = F.cross_entropy(logits_c, tgt, ignore_index=-100)
        loss_v = F.cross_entropy(logits_v, tgt_v, ignore_index=-100)
        check(loss_c, loss_v, rtol=RTOL_R, atol=ATOL_R)
        loss_c.backward()
        loss_v.backward()
        check(logits_c.grad, logits_v.grad, rtol=RTOL_R, atol=ATOL_R)

    def test_nll_loss(self):
        log_probs_c = F.log_softmax(torch.randn(8, 10), dim=1)
        tgt = torch.randint(0, 10, (8,))
        log_probs_v = log_probs_c.to("vulkan")
        tgt_v = tgt.to("vulkan")
        check(F.nll_loss(log_probs_c, tgt), F.nll_loss(log_probs_v, tgt_v),
              rtol=RTOL_R, atol=ATOL_R)

    def test_bce_loss(self):
        pred = torch.sigmoid(torch.randn(8, 4))
        tgt = torch.rand(8, 4)
        pred_v, tgt_v = pred.to("vulkan"), tgt.to("vulkan")
        check(F.binary_cross_entropy(pred, tgt),
              F.binary_cross_entropy(pred_v, tgt_v),
              rtol=RTOL_R, atol=ATOL_R)

    def test_bce_with_logits(self):
        logits_c, logits_v = randn_pair(8, 4)
        tgt_c, tgt_v = pair(8, 4, lo=0, hi=1)
        check(F.binary_cross_entropy_with_logits(logits_c, tgt_c),
              F.binary_cross_entropy_with_logits(logits_v, tgt_v),
              rtol=RTOL_R, atol=ATOL_R)

    def test_smooth_l1_loss(self):
        pred_c, pred_v = randn_pair(8, 16)
        tgt_c, tgt_v = randn_pair(8, 16)
        check(F.smooth_l1_loss(pred_c, tgt_c), F.smooth_l1_loss(pred_v, tgt_v),
              rtol=RTOL_R, atol=ATOL_R)

    def test_huber_loss(self):
        pred_c, pred_v = randn_pair(8, 16)
        tgt_c, tgt_v = randn_pair(8, 16)
        check(F.huber_loss(pred_c, tgt_c), F.huber_loss(pred_v, tgt_v),
              rtol=RTOL_R, atol=ATOL_R)

    def test_kl_div(self):
        log_p = F.log_softmax(torch.randn(8, 10), dim=1)
        q = F.softmax(torch.randn(8, 10), dim=1)
        log_p_v, q_v = log_p.to("vulkan"), q.to("vulkan")
        check(F.kl_div(log_p, q, reduction="batchmean"),
              F.kl_div(log_p_v, q_v, reduction="batchmean"),
              rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  SHAPE OPS
# ═══════════════════════════════════════════════════════════════════

class TestShapeMath:

    def test_view(self):
        c, v = randn_pair(4, 8)
        check(c.view(2, 16), v.view(2, 16))
        check(c.view(32), v.view(32))

    def test_reshape(self):
        c, v = randn_pair(4, 8, 2)
        check(c.reshape(8, 8), v.reshape(8, 8))

    def test_permute(self):
        c, v = randn_pair(2, 3, 4)
        check(c.permute(2, 0, 1), v.permute(2, 0, 1))

    def test_transpose(self):
        c, v = randn_pair(8, 16)
        check(c.t(), v.t())

    def test_unsqueeze_squeeze(self):
        c, v = randn_pair(4, 8)
        check(c.unsqueeze(0), v.unsqueeze(0))
        check(c.unsqueeze(1), v.unsqueeze(1))
        c2, v2 = randn_pair(1, 4, 1, 8)
        check(c2.squeeze(), v2.squeeze())

    def test_expand(self):
        c, v = randn_pair(1, 8)
        check(c.expand(4, 8), v.expand(4, 8))

    def test_cat(self):
        a_c, a_v = randn_pair(4, 8)
        b_c, b_v = randn_pair(4, 8)
        check(torch.cat([a_c, b_c], dim=0), torch.cat([a_v, b_v], dim=0))
        check(torch.cat([a_c, b_c], dim=1), torch.cat([a_v, b_v], dim=1))

    def test_stack(self):
        a_c, a_v = randn_pair(4, 8)
        b_c, b_v = randn_pair(4, 8)
        check(torch.stack([a_c, b_c], dim=0), torch.stack([a_v, b_v], dim=0))

    def test_select(self):
        c, v = randn_pair(4, 8, 16)
        check(c.select(0, 2), v.select(0, 2))
        check(c.select(1, 3), v.select(1, 3))

    def test_slice(self):
        c, v = randn_pair(8, 16)
        check(c[2:6], v[2:6])
        check(c[:, 4:12], v[:, 4:12])
        check(c[1:7:2], v[1:7:2])

    def test_split(self):
        c, v = randn_pair(8, 16)
        c_parts = c.split(4, dim=0)
        v_parts = v.split(4, dim=0)
        for cp, vp in zip(c_parts, v_parts):
            check(cp, vp)

    def test_flip(self):
        c, v = randn_pair(4, 8)
        check(torch.flip(c, [0]), torch.flip(v, [0]))
        check(torch.flip(c, [1]), torch.flip(v, [1]))

    def test_roll(self):
        c, v = randn_pair(4, 8)
        check(torch.roll(c, 2, 0), torch.roll(v, 2, 0))
        check(torch.roll(c, -3, 1), torch.roll(v, -3, 1))

    def test_repeat(self):
        c, v = randn_pair(4, 8)
        check(c.repeat(2, 3), v.repeat(2, 3))

    def test_triu(self):
        c, v = randn_pair(8, 8)
        check(torch.triu(c), torch.triu(v))
        check(torch.triu(c, diagonal=1), torch.triu(v, diagonal=1))

    def test_tril(self):
        c, v = randn_pair(8, 8)
        check(torch.tril(c), torch.tril(v))

    def test_flatten(self):
        c, v = randn_pair(2, 3, 4)
        check(c.flatten(), v.flatten())
        check(c.flatten(1), v.flatten(1))

    def test_contiguous(self):
        c, v = randn_pair(4, 8)
        ct = c.t().contiguous()
        vt = v.t().contiguous()
        check(ct, vt)


# ═══════════════════════════════════════════════════════════════════
#  INDEXING & ADVANCED OPS
# ═══════════════════════════════════════════════════════════════════

class TestIndexingMath:

    def test_index_select(self):
        c, v = randn_pair(8, 16)
        idx = torch.tensor([0, 3, 5, 7])
        idx_v = idx.to("vulkan")
        check(c.index_select(0, idx), v.index_select(0, idx_v))
        check(c.index_select(1, idx), v.index_select(1, idx_v))

    def test_gather(self):
        c, v = randn_pair(8, 8)
        idx = torch.randint(0, 8, (8, 4))
        idx_v = idx.to("vulkan")
        check(torch.gather(c, 1, idx), torch.gather(v, 1, idx_v))

    def test_sort(self):
        c, v = randn_pair(8, 16)
        cv, ci = torch.sort(c, dim=1)
        vv, vi = torch.sort(v, dim=1)
        check(cv, vv)

    def test_topk(self):
        c, v = randn_pair(8, 32)
        cv, ci = torch.topk(c, 5, dim=1)
        vv, vi = torch.topk(v, 5, dim=1)
        check(cv, vv)

    def test_masked_fill(self):
        c, v = randn_pair(8, 16)
        mask = torch.randn(8, 16) > 0
        mask_v = mask.to("vulkan")
        check(c.masked_fill(mask, -1e9), v.masked_fill(mask_v, -1e9))


# ═══════════════════════════════════════════════════════════════════
#  EMBEDDING
# ═══════════════════════════════════════════════════════════════════

class TestEmbeddingMath:

    @pytest.mark.parametrize("vocab,dim", [(100, 32), (1000, 64), (10000, 128)])
    def test_embedding_forward(self, vocab, dim):
        emb = nn.Embedding(vocab, dim)
        emb_v = nn.Embedding(vocab, dim)
        emb_v.load_state_dict(emb.state_dict())
        emb_v = emb_v.to("vulkan")
        idx = torch.randint(0, vocab, (4, 8))
        idx_v = idx.to("vulkan")
        check(emb(idx), emb_v(idx_v))

    def test_embedding_backward(self):
        emb_c = nn.Embedding(100, 32)
        emb_v = nn.Embedding(100, 32)
        emb_v.load_state_dict(emb_c.state_dict())
        emb_v = emb_v.to("vulkan")
        idx = torch.randint(0, 100, (4, 8))
        idx_v = idx.to("vulkan")
        y_c = emb_c(idx).sum()
        y_v = emb_v(idx_v).sum()
        y_c.backward()
        y_v.backward()
        check(emb_c.weight.grad, emb_v.weight.grad, rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  ATTENTION (SDPA)
# ═══════════════════════════════════════════════════════════════════

class TestAttentionMath:

    @pytest.mark.parametrize("B,H,S,D", [(1, 1, 8, 16), (2, 4, 16, 32), (1, 8, 32, 64)])
    def test_sdpa(self, B, H, S, D):
        q_c, q_v = randn_pair(B, H, S, D)
        k_c, k_v = randn_pair(B, H, S, D)
        v_c, v_v = randn_pair(B, H, S, D)
        # Scale inputs to avoid numerical issues
        q_c, k_c, v_c = q_c * 0.1, k_c * 0.1, v_c * 0.1
        q_v, k_v, v_v = q_c.to("vulkan"), k_c.to("vulkan"), v_c.to("vulkan")
        out_c = F.scaled_dot_product_attention(q_c, k_c, v_c)
        out_v = F.scaled_dot_product_attention(q_v, k_v, v_v)
        check(out_c, out_v, rtol=RTOL_R, atol=ATOL_R)

    def test_sdpa_causal(self):
        q_c, q_v = randn_pair(1, 4, 8, 16)
        k_c, k_v = randn_pair(1, 4, 8, 16)
        v_c, v_v = randn_pair(1, 4, 8, 16)
        q_c, k_c, v_c = q_c * 0.1, k_c * 0.1, v_c * 0.1
        q_v, k_v, v_v = q_c.to("vulkan"), k_c.to("vulkan"), v_c.to("vulkan")
        out_c = F.scaled_dot_product_attention(q_c, k_c, v_c, is_causal=True)
        out_v = F.scaled_dot_product_attention(q_v, k_v, v_v, is_causal=True)
        check(out_c, out_v, rtol=RTOL_R, atol=ATOL_R)

    def test_sdpa_backward(self):
        q_data = torch.randn(1, 4, 8, 16) * 0.1
        k_data = torch.randn(1, 4, 8, 16) * 0.1
        v_data = torch.randn(1, 4, 8, 16) * 0.1
        q = q_data.clone().requires_grad_(True)
        k = k_data.clone().requires_grad_(True)
        v = v_data.clone().requires_grad_(True)
        q_v = q_data.clone().to("vulkan").requires_grad_(True)
        k_v = k_data.clone().to("vulkan").requires_grad_(True)
        v_v = v_data.clone().to("vulkan").requires_grad_(True)
        out_c = F.scaled_dot_product_attention(q, k, v)
        out_v = F.scaled_dot_product_attention(q_v, k_v, v_v)
        go = torch.randn_like(out_c)
        out_c.backward(go)
        out_v.backward(go.to("vulkan"))
        check(q.grad, q_v.grad, rtol=RTOL_VR, atol=ATOL_VR)
        check(k.grad, k_v.grad, rtol=RTOL_VR, atol=ATOL_VR)
        check(v.grad, v_v.grad, rtol=RTOL_VR, atol=ATOL_VR)


# ═══════════════════════════════════════════════════════════════════
#  FACTORIES
# ═══════════════════════════════════════════════════════════════════

class TestFactoryMath:

    def test_zeros(self):
        c = torch.zeros(8, 16)
        v = torch.zeros(8, 16, device="vulkan")
        check(c, v)

    def test_ones(self):
        c = torch.ones(8, 16)
        v = torch.ones(8, 16, device="vulkan")
        check(c, v)

    def test_full(self):
        c = torch.full((8, 16), 3.14)
        v = torch.full((8, 16), 3.14, device="vulkan")
        check(c, v)

    @pytest.mark.parametrize("start,end,step", [(0, 10, 1), (0.5, 5.5, 0.5), (-3, 3, 0.1)])
    def test_arange(self, start, end, step):
        c = torch.arange(start, end, step)
        v = torch.arange(start, end, step, device="vulkan")
        check(c, v)

    def test_linspace(self):
        c = torch.linspace(0, 1, 100)
        v = torch.linspace(0, 1, 100, device="vulkan")
        check(c, v)

    def test_eye(self):
        c = torch.eye(8)
        v = torch.eye(8, device="vulkan")
        check(c, v)

    def test_eye_rect(self):
        c = torch.eye(4, 8)
        v = torch.eye(4, 8, device="vulkan")
        check(c, v)


# ═══════════════════════════════════════════════════════════════════
#  PAD & INTERPOLATION
# ═══════════════════════════════════════════════════════════════════

class TestPadInterpolateMath:

    def test_pad_1d(self):
        c, v = randn_pair(2, 3, 8)
        check(F.pad(c, (1, 2)), F.pad(v, (1, 2)))

    def test_pad_2d(self):
        c, v = randn_pair(2, 3, 8, 8)
        check(F.pad(c, (1, 1, 1, 1)), F.pad(v, (1, 1, 1, 1)))

    def test_upsample_nearest(self):
        c, v = randn_pair(1, 3, 4, 4)
        check(F.interpolate(c, scale_factor=2, mode="nearest"),
              F.interpolate(v, scale_factor=2, mode="nearest"))

    def test_upsample_bilinear(self):
        c, v = randn_pair(1, 3, 4, 4)
        check(F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False),
              F.interpolate(v, scale_factor=2, mode="bilinear", align_corners=False),
              rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  COMPOSED OPERATIONS (multi-op chains)
# ═══════════════════════════════════════════════════════════════════

class TestComposedMath:
    """Test mathematical correctness through multi-op chains."""

    def test_linear_relu_linear(self):
        x_c, x_v = randn_pair(4, 32)
        w1 = torch.randn(16, 32)
        b1 = torch.randn(16)
        w2 = torch.randn(8, 16)
        b2 = torch.randn(8)
        y_c = F.linear(F.relu(F.linear(x_c, w1, b1)), w2, b2)
        y_v = F.linear(F.relu(F.linear(x_v, w1.to("vulkan"), b1.to("vulkan"))),
                       w2.to("vulkan"), b2.to("vulkan"))
        check(y_c, y_v, rtol=RTOL_R, atol=ATOL_R)

    def test_conv_bn_relu(self):
        x_c, x_v = randn_pair(2, 3, 8, 8)
        conv = nn.Conv2d(3, 8, 3, padding=1)
        bn = nn.BatchNorm2d(8)
        bn.eval()
        conv_v = nn.Conv2d(3, 8, 3, padding=1)
        conv_v.load_state_dict(conv.state_dict())
        bn_v = nn.BatchNorm2d(8)
        bn_v.load_state_dict(bn.state_dict())
        bn_v.eval()
        conv_v, bn_v = conv_v.to("vulkan"), bn_v.to("vulkan")
        y_c = F.relu(bn(conv(x_c)))
        y_v = F.relu(bn_v(conv_v(x_v)))
        check(y_c, y_v, rtol=RTOL_R, atol=ATOL_R)

    def test_attention_block(self):
        """Q, K, V projection → SDPA → output projection."""
        B, S, D = 2, 8, 32
        x_c, x_v = randn_pair(B, S, D)
        wq = torch.randn(D, D) * 0.1
        wk = torch.randn(D, D) * 0.1
        wv = torch.randn(D, D) * 0.1
        wo = torch.randn(D, D) * 0.1
        q_c = (x_c @ wq.t()).view(B, S, 4, 8).permute(0, 2, 1, 3)
        k_c = (x_c @ wk.t()).view(B, S, 4, 8).permute(0, 2, 1, 3)
        v_c = (x_c @ wv.t()).view(B, S, 4, 8).permute(0, 2, 1, 3)
        attn_c = F.scaled_dot_product_attention(q_c, k_c, v_c)
        out_c = attn_c.permute(0, 2, 1, 3).reshape(B, S, D) @ wo.t()

        wq_v = wq.to("vulkan")
        wk_v = wk.to("vulkan")
        wv_v = wv.to("vulkan")
        wo_v = wo.to("vulkan")
        q_v = (x_v @ wq_v.t()).view(B, S, 4, 8).permute(0, 2, 1, 3)
        k_v = (x_v @ wk_v.t()).view(B, S, 4, 8).permute(0, 2, 1, 3)
        v_v = (x_v @ wv_v.t()).view(B, S, 4, 8).permute(0, 2, 1, 3)
        attn_v = F.scaled_dot_product_attention(q_v, k_v, v_v)
        out_v = attn_v.permute(0, 2, 1, 3).reshape(B, S, D) @ wo_v.t()

        check(out_c, out_v, rtol=RTOL_VR, atol=ATOL_VR)

    def test_residual_connection(self):
        x_c, x_v = randn_pair(4, 32)
        w = torch.randn(32, 32)
        b = torch.randn(32)
        wv, bv = w.to("vulkan"), b.to("vulkan")
        y_c = x_c + F.relu(F.linear(x_c, w, b))
        y_v = x_v + F.relu(F.linear(x_v, wv, bv))
        check(y_c, y_v, rtol=RTOL_R, atol=ATOL_R)

    def test_layer_norm_residual_mlp(self):
        """Transformer-style: x + MLP(LayerNorm(x))."""
        x_c, x_v = randn_pair(4, 32)
        w_ln = torch.ones(32)
        b_ln = torch.zeros(32)
        w1 = torch.randn(64, 32)
        b1 = torch.randn(64)
        w2 = torch.randn(32, 64)
        b2 = torch.randn(32)

        normed_c = F.layer_norm(x_c, [32], w_ln, b_ln)
        h_c = F.gelu(F.linear(normed_c, w1, b1))
        out_c = x_c + F.linear(h_c, w2, b2)

        normed_v = F.layer_norm(x_v, [32], w_ln.to("vulkan"), b_ln.to("vulkan"))
        h_v = F.gelu(F.linear(normed_v, w1.to("vulkan"), b1.to("vulkan")))
        out_v = x_v + F.linear(h_v, w2.to("vulkan"), b2.to("vulkan"))

        # Multi-op composed chain: layer_norm → linear → gelu → linear → add
        # Error accumulates through each op, so use larger tolerance
        check(out_c, out_v, rtol=5e-3, atol=5e-3)


# ═══════════════════════════════════════════════════════════════════
#  TRAINING LOOP CORRECTNESS
# ═══════════════════════════════════════════════════════════════════

class TestTrainingMath:
    """Verify that training produces identical results on CPU and Vulkan."""

    def test_sgd_step_match(self):
        """Single SGD step should produce same params."""
        torch.manual_seed(42)
        model_c = nn.Linear(16, 4)
        model_v = nn.Linear(16, 4)
        model_v.load_state_dict(model_c.state_dict())
        model_v = model_v.to("vulkan")

        opt_c = torch.optim.SGD(model_c.parameters(), lr=0.01)
        opt_v = torch.optim.SGD(model_v.parameters(), lr=0.01)

        x = torch.randn(8, 16)
        y = torch.randint(0, 4, (8,))

        opt_c.zero_grad()
        loss_c = F.cross_entropy(model_c(x), y)
        loss_c.backward()
        opt_c.step()

        opt_v.zero_grad()
        loss_v = F.cross_entropy(model_v(x.to("vulkan")), y.to("vulkan"))
        loss_v.backward()
        opt_v.step()

        check(model_c.weight, model_v.weight, rtol=RTOL_R, atol=ATOL_R)
        check(model_c.bias, model_v.bias, rtol=RTOL_R, atol=ATOL_R)

    def test_multi_step_training(self):
        """5 steps of training should match."""
        torch.manual_seed(42)
        model_c = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
        model_v = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
        model_v.load_state_dict(model_c.state_dict())
        model_v = model_v.to("vulkan")

        opt_c = torch.optim.SGD(model_c.parameters(), lr=0.01)
        opt_v = torch.optim.SGD(model_v.parameters(), lr=0.01)

        for step in range(5):
            x = torch.randn(8, 16)
            y = torch.randint(0, 4, (8,))

            opt_c.zero_grad()
            loss_c = F.cross_entropy(model_c(x), y)
            loss_c.backward()
            opt_c.step()

            opt_v.zero_grad()
            loss_v = F.cross_entropy(model_v(x.to("vulkan")), y.to("vulkan"))
            loss_v.backward()
            opt_v.step()

        for pc, pv in zip(model_c.parameters(), model_v.parameters()):
            check(pc, pv, rtol=RTOL_VR, atol=ATOL_VR)

    def test_sgd_with_momentum(self):
        torch.manual_seed(42)
        model_c = nn.Linear(16, 4)
        model_v = nn.Linear(16, 4)
        model_v.load_state_dict(model_c.state_dict())
        model_v = model_v.to("vulkan")

        opt_c = torch.optim.SGD(model_c.parameters(), lr=0.01, momentum=0.9)
        opt_v = torch.optim.SGD(model_v.parameters(), lr=0.01, momentum=0.9)

        for _ in range(3):
            x = torch.randn(8, 16)
            y = torch.randint(0, 4, (8,))
            opt_c.zero_grad()
            F.cross_entropy(model_c(x), y).backward()
            opt_c.step()
            opt_v.zero_grad()
            F.cross_entropy(model_v(x.to("vulkan")), y.to("vulkan")).backward()
            opt_v.step()

        for pc, pv in zip(model_c.parameters(), model_v.parameters()):
            check(pc, pv, rtol=RTOL_VR, atol=ATOL_VR)


# ═══════════════════════════════════════════════════════════════════
#  EDGE CASES & NUMERICAL STABILITY
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCasesMath:
    """Test edge cases that often reveal numerical bugs."""

    def test_very_small_values(self):
        c = torch.tensor([1e-7, 1e-6, 1e-5, 1e-4])
        v = c.to("vulkan")
        check(c * 2, v * 2)
        check(c + c, v + v)

    def test_very_large_values(self):
        c = torch.tensor([1e4, 1e5, 1e6, 1e7])
        v = c.to("vulkan")
        check(c + 1, v + 1)
        check(c * 0.001, v * 0.001)

    def test_mixed_sign(self):
        c = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        v = c.to("vulkan")
        check(c * c, v * v)
        check(torch.abs(c), torch.abs(v))

    def test_zero_tensor(self):
        c = torch.zeros(16, 16)
        v = c.to("vulkan")
        check(c + 1, v + 1)
        check(torch.relu(c), torch.relu(v))

    def test_single_element(self):
        c = torch.tensor([3.14])
        v = c.to("vulkan")
        check(c * 2, v * 2)
        check(torch.exp(c), torch.exp(v))

    def test_large_tensor(self):
        c, v = randn_pair(256, 256)
        check(c + c, v + v)
        check(c.sum(), v.sum(), rtol=RTOL_R, atol=ATOL_R)

    def test_odd_shapes(self):
        """Non-power-of-2 shapes."""
        for shape in [(7,), (13, 17), (3, 5, 7), (11, 13)]:
            c, v = randn_pair(*shape)
            check(c + 1, v + 1)
            check(torch.relu(c), torch.relu(v))

    def test_softmax_all_same(self):
        """Softmax of equal values should give uniform distribution."""
        c = torch.ones(4, 8)
        v = c.to("vulkan")
        result = F.softmax(v, dim=1).cpu()
        expected = torch.full((4, 8), 1.0 / 8)
        torch.testing.assert_close(result, expected, rtol=RTOL_R, atol=ATOL_R)

    def test_cross_entropy_one_hot(self):
        """CE with one-hot-ish logits should give small loss for correct class."""
        logits = torch.zeros(4, 10)
        logits[0, 3] = 10.0
        logits[1, 7] = 10.0
        logits[2, 0] = 10.0
        logits[3, 9] = 10.0
        tgt = torch.tensor([3, 7, 0, 9])
        logits_v, tgt_v = logits.to("vulkan"), tgt.to("vulkan")
        loss_c = F.cross_entropy(logits, tgt)
        loss_v = F.cross_entropy(logits_v, tgt_v)
        check(loss_c, loss_v, rtol=RTOL_R, atol=ATOL_R)
        assert loss_c.item() < 0.01  # should be near zero

    def test_matmul_identity(self):
        """Multiplying by identity should return the input."""
        x_c, x_v = randn_pair(8, 8)
        eye = torch.eye(8)
        eye_v = eye.to("vulkan")
        check(x_c @ eye, x_v @ eye_v)

    def test_add_zero(self):
        c, v = randn_pair(32, 32)
        z = torch.zeros(32, 32)
        zv = z.to("vulkan")
        check(c + z, v + zv)

    def test_mul_one(self):
        c, v = randn_pair(32, 32)
        check(c * 1.0, v * 1.0)

    def test_exp_log_roundtrip(self):
        c, v = pos_pair(32)
        check(torch.exp(torch.log(c)), torch.exp(torch.log(v)), rtol=RTOL_R, atol=ATOL_R)

    def test_sqrt_square_roundtrip(self):
        c, v = pos_pair(32)
        check(torch.sqrt(c * c), torch.sqrt(v * v), rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  IN-PLACE OPS CORRECTNESS
# ═══════════════════════════════════════════════════════════════════

class TestInPlaceMath:
    """Verify in-place ops produce same results as out-of-place."""

    def test_add_(self):
        c, v = randn_pair(32, 32)
        c2, v2 = randn_pair(32, 32)
        c.add_(c2)
        v.add_(v2)
        check(c, v)

    def test_sub_(self):
        c, v = randn_pair(32, 32)
        c2, v2 = randn_pair(32, 32)
        c.sub_(c2)
        v.sub_(v2)
        check(c, v)

    def test_mul_(self):
        c, v = randn_pair(32, 32)
        c2, v2 = randn_pair(32, 32)
        c.mul_(c2)
        v.mul_(v2)
        check(c, v)

    def test_div_(self):
        c, v = randn_pair(32, 32)
        c2 = torch.randn(32, 32).abs() + 0.1
        v2 = c2.to("vulkan")
        c.div_(c2)
        v.div_(v2)
        check(c, v)

    def test_add_scalar_(self):
        c, v = randn_pair(32, 32)
        c.add_(3.14)
        v.add_(3.14)
        check(c, v)

    def test_mul_scalar_(self):
        c, v = randn_pair(32, 32)
        c.mul_(2.5)
        v.mul_(2.5)
        check(c, v)

    def test_relu_(self):
        c, v = randn_pair(32, 32)
        F.relu(c, inplace=True)
        F.relu(v, inplace=True)
        check(c, v)

    def test_clamp_(self):
        c, v = randn_pair(32, 32)
        c.clamp_(-0.5, 0.5)
        v.clamp_(-0.5, 0.5)
        check(c, v)

    def test_zero_(self):
        c, v = randn_pair(32, 32)
        c.zero_()
        v.zero_()
        check(c, v)

    def test_fill_(self):
        c, v = randn_pair(32, 32)
        c.fill_(42.0)
        v.fill_(42.0)
        check(c, v)

    def test_neg_via_mul(self):
        """neg_() not registered — use mul_(-1) instead."""
        c, v = randn_pair(32, 32)
        c.mul_(-1)
        v.mul_(-1)
        check(c, v)

    def test_abs_via_ops(self):
        """abs_() not registered — verify out-of-place abs works."""
        c, v = randn_pair(32, 32)
        check(c.abs(), v.abs())

    def test_sequential_inplace(self):
        """Chain of in-place ops."""
        c, v = randn_pair(32, 32)
        c.add_(1.0).mul_(2.0).sub_(0.5)
        v.add_(1.0).mul_(2.0).sub_(0.5)
        check(c, v)


# ═══════════════════════════════════════════════════════════════════
#  DTYPE CASTING & HALF PRECISION
# ═══════════════════════════════════════════════════════════════════

class TestDtypeMath:
    """Test f16/bf16 widen-compute-narrow correctness."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_add_half(self, dtype):
        a = torch.randn(64, 64).to(dtype)
        b = torch.randn(64, 64).to(dtype)
        expected = (a.float() + b.float()).to(dtype)
        av, bv = a.to("vulkan"), b.to("vulkan")
        result = av + bv
        # bf16 has lower precision (7-bit mantissa) → larger tolerance
        tol = 0.05 if dtype == torch.bfloat16 else 5e-3
        check(expected, result, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matmul_half(self, dtype):
        a = torch.randn(16, 32).to(dtype)
        b = torch.randn(32, 16).to(dtype)
        expected = (a.float() @ b.float()).to(dtype)
        av, bv = a.to("vulkan"), b.to("vulkan")
        result = av @ bv
        check(expected, result, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_relu_half(self, dtype):
        a = torch.randn(64).to(dtype)
        expected = F.relu(a.float()).to(dtype)
        av = a.to("vulkan")
        check(expected, F.relu(av), rtol=5e-3, atol=5e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_softmax_half(self, dtype):
        a = torch.randn(8, 32).to(dtype)
        expected = F.softmax(a.float(), dim=-1).to(dtype)
        av = a.to("vulkan")
        check(expected, F.softmax(av, dim=-1), rtol=5e-3, atol=5e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_layer_norm_half(self, dtype):
        a = torch.randn(4, 32).to(dtype)
        w = torch.ones(32).to(dtype)
        b = torch.zeros(32).to(dtype)
        expected = F.layer_norm(a.float(), [32], w.float(), b.float()).to(dtype)
        av, wv, bv = a.to("vulkan"), w.to("vulkan"), b.to("vulkan")
        tol = 0.05 if dtype == torch.bfloat16 else 5e-3
        check(expected, F.layer_norm(av, [32], wv, bv), rtol=tol, atol=tol)

    def test_cast_roundtrip_f16(self):
        """f32 -> f16 -> f32 should be close (within f16 precision)."""
        a = torch.randn(128)
        av = a.to("vulkan")
        rt = av.half().float()
        check(a.half().float(), rt, rtol=1e-3, atol=1e-3)

    def test_cast_roundtrip_bf16(self):
        """f32 -> bf16 -> f32 should be close (within bf16 precision)."""
        a = torch.randn(128)
        av = a.to("vulkan")
        rt = av.bfloat16().float()
        check(a.bfloat16().float(), rt, rtol=1e-2, atol=1e-2)


# ═══════════════════════════════════════════════════════════════════
#  BROADCASTING CORRECTNESS
# ═══════════════════════════════════════════════════════════════════

class TestBroadcastMath:
    """Test binary ops with broadcasting."""

    def test_add_scalar_broadcast(self):
        c, v = randn_pair(8, 16)
        check(c + 5.0, v + 5.0)

    def test_add_row_broadcast(self):
        c, v = randn_pair(8, 16)
        b = torch.randn(16)
        bv = b.to("vulkan")
        check(c + b, v + bv)

    def test_add_col_broadcast(self):
        c, v = randn_pair(8, 16)
        b = torch.randn(8, 1)
        bv = b.to("vulkan")
        check(c + b, v + bv)

    def test_mul_broadcast_3d(self):
        c, v = randn_pair(4, 8, 16)
        b = torch.randn(1, 1, 16)
        bv = b.to("vulkan")
        check(c * b, v * bv)

    def test_sub_broadcast(self):
        c, v = randn_pair(4, 8)
        b = torch.randn(8)
        bv = b.to("vulkan")
        check(c - b, v - bv)

    def test_div_broadcast(self):
        c, v = randn_pair(4, 8)
        b = torch.randn(8).abs() + 0.1
        bv = b.to("vulkan")
        check(c / b, v / bv, rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  REDUCTION OPS (ADDITIONAL)
# ═══════════════════════════════════════════════════════════════════

class TestReductionExtended:
    """Extended tests for reduction ops with various dims and keepdim."""

    @pytest.mark.parametrize("keepdim", [True, False])
    def test_sum_dim0(self, keepdim):
        c, v = randn_pair(8, 16)
        check(c.sum(0, keepdim=keepdim), v.sum(0, keepdim=keepdim))

    @pytest.mark.parametrize("keepdim", [True, False])
    def test_sum_dim1(self, keepdim):
        c, v = randn_pair(8, 16)
        check(c.sum(1, keepdim=keepdim), v.sum(1, keepdim=keepdim))

    def test_sum_3d_middle_dim(self):
        c, v = randn_pair(4, 8, 16)
        check(c.sum(1), v.sum(1), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("keepdim", [True, False])
    def test_mean_dim(self, keepdim):
        c, v = randn_pair(8, 16)
        check(c.mean(1, keepdim=keepdim), v.mean(1, keepdim=keepdim), rtol=RTOL_R, atol=ATOL_R)

    def test_mean_global(self):
        c, v = randn_pair(32, 32)
        check(c.mean(), v.mean(), rtol=RTOL_R, atol=ATOL_R)

    @pytest.mark.parametrize("keepdim", [True, False])
    def test_max_dim(self, keepdim):
        c, v = randn_pair(8, 16)
        cv, ci = c.max(1, keepdim=keepdim)
        vv, vi = v.max(1, keepdim=keepdim)
        check(cv, vv)
        check(ci.float(), vi.float())

    @pytest.mark.parametrize("keepdim", [True, False])
    def test_min_dim(self, keepdim):
        c, v = randn_pair(8, 16)
        cv, ci = c.min(1, keepdim=keepdim)
        vv, vi = v.min(1, keepdim=keepdim)
        check(cv, vv)
        check(ci.float(), vi.float())

    def test_amax(self):
        c, v = randn_pair(8, 16)
        check(c.amax(1), v.amax(1))

    def test_amin(self):
        c, v = randn_pair(8, 16)
        check(c.amin(1), v.amin(1))

    def test_prod_dim(self):
        c = torch.rand(4, 8) + 0.5  # keep values moderate to avoid overflow
        v = c.to("vulkan")
        check(c.prod(1), v.prod(1), rtol=RTOL_R, atol=ATOL_R)

    def test_any_all(self):
        c = torch.tensor([0.0, 1.0, 0.0, 2.0])
        v = c.to("vulkan")
        # any/all return bool, compare as float
        check(c.any().float(), v.any().float())
        check(c.all().float(), v.all().float())
        c2 = torch.tensor([1.0, 2.0, 3.0])
        v2 = c2.to("vulkan")
        check(c2.all().float(), v2.all().float())

    def test_argmax(self):
        c, v = randn_pair(4, 16)
        check(c.argmax(1).float(), v.argmax(1).float())

    def test_argmin(self):
        c, v = randn_pair(4, 16)
        check(c.argmin(1).float(), v.argmin(1).float())

    def test_cumsum(self):
        c, v = randn_pair(8, 16)
        check(c.cumsum(1), v.cumsum(1), rtol=RTOL_R, atol=ATOL_R)

    def test_cumprod(self):
        c = torch.rand(4, 8) * 0.5 + 0.75  # near 1.0 to avoid explosion
        v = c.to("vulkan")
        check(c.cumprod(1), v.cumprod(1), rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  LOSS FUNCTION CORRECTNESS
# ═══════════════════════════════════════════════════════════════════

class TestLossExtended:
    """Extended loss function tests with various configurations."""

    def test_mse_loss_none(self):
        a_c, a_v = randn_pair(8, 16)
        b_c, b_v = randn_pair(8, 16)
        check(F.mse_loss(a_c, b_c, reduction="none"),
              F.mse_loss(a_v, b_v, reduction="none"))

    def test_mse_loss_sum(self):
        a_c, a_v = randn_pair(8, 16)
        b_c, b_v = randn_pair(8, 16)
        check(F.mse_loss(a_c, b_c, reduction="sum"),
              F.mse_loss(a_v, b_v, reduction="sum"), rtol=RTOL_R, atol=RTOL_R)

    def test_bce_loss(self):
        pred = torch.sigmoid(torch.randn(8, 16))
        tgt = torch.rand(8, 16)
        pv, tv = pred.to("vulkan"), tgt.to("vulkan")
        check(F.binary_cross_entropy(pred, tgt),
              F.binary_cross_entropy(pv, tv), rtol=RTOL_R, atol=ATOL_R)

    def test_bce_with_logits(self):
        logits_c, logits_v = randn_pair(8, 16)
        tgt = torch.rand(8, 16)
        tv = tgt.to("vulkan")
        check(F.binary_cross_entropy_with_logits(logits_c, tgt),
              F.binary_cross_entropy_with_logits(logits_v, tv), rtol=RTOL_R, atol=ATOL_R)

    def test_smooth_l1_loss(self):
        a_c, a_v = randn_pair(8, 16)
        b_c, b_v = randn_pair(8, 16)
        check(F.smooth_l1_loss(a_c, b_c),
              F.smooth_l1_loss(a_v, b_v), rtol=RTOL_R, atol=ATOL_R)

    def test_huber_loss(self):
        a_c, a_v = randn_pair(8, 16)
        b_c, b_v = randn_pair(8, 16)
        check(F.huber_loss(a_c, b_c, delta=1.5),
              F.huber_loss(a_v, b_v, delta=1.5), rtol=RTOL_R, atol=ATOL_R)

    def test_cross_entropy_ignore_index(self):
        logits = torch.randn(8, 10)
        targets = torch.tensor([3, -100, 7, 2, -100, 0, 5, 1])  # some ignored
        lv, tv = logits.to("vulkan"), targets.to("vulkan")
        check(F.cross_entropy(logits, targets, ignore_index=-100),
              F.cross_entropy(lv, tv, ignore_index=-100), rtol=RTOL_R, atol=ATOL_R)

    def test_nll_loss_ignore_index(self):
        log_probs = F.log_softmax(torch.randn(8, 10), dim=-1)
        targets = torch.tensor([3, -100, 7, 2, -100, 0, 5, 1])
        lpv, tv = log_probs.to("vulkan"), targets.to("vulkan")
        check(F.nll_loss(log_probs, targets, ignore_index=-100),
              F.nll_loss(lpv, tv, ignore_index=-100), rtol=RTOL_R, atol=ATOL_R)

    def test_cross_entropy_backward(self):
        logits = torch.randn(8, 10, requires_grad=True)
        logits_v = logits.detach().clone().to("vulkan").requires_grad_(True)
        targets = torch.tensor([3, 7, 2, 0, 5, 1, 4, 8])
        targets_v = targets.to("vulkan")
        loss_c = F.cross_entropy(logits, targets)
        loss_v = F.cross_entropy(logits_v, targets_v)
        loss_c.backward()
        loss_v.backward()
        check(logits.grad, logits_v.grad, rtol=RTOL_VR, atol=ATOL_VR)

    def test_mse_backward(self):
        a = torch.randn(8, 16, requires_grad=True)
        av = a.detach().clone().to("vulkan").requires_grad_(True)
        b_c, b_v = randn_pair(8, 16)
        loss_c = F.mse_loss(a, b_c)
        loss_v = F.mse_loss(av, b_v)
        loss_c.backward()
        loss_v.backward()
        check(a.grad, av.grad, rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  NORMALIZATION (ADDITIONAL)
# ═══════════════════════════════════════════════════════════════════

class TestNormalizationExtended:
    """Extended normalization tests."""

    def test_batch_norm_eval(self):
        bn = nn.BatchNorm2d(8)
        bn.eval()
        bn_v = nn.BatchNorm2d(8)
        bn_v.load_state_dict(bn.state_dict())
        bn_v = bn_v.to("vulkan")
        bn_v.eval()
        x_c, x_v = randn_pair(2, 8, 4, 4)
        check(bn(x_c), bn_v(x_v), rtol=RTOL_R, atol=ATOL_R)

    def test_batch_norm_train(self):
        """BatchNorm training mode — Vulkan only supports eval, skip."""
        pytest.skip("Vulkan batch_norm only supports eval mode")

    def test_group_norm(self):
        gn = nn.GroupNorm(4, 16)
        gn_v = nn.GroupNorm(4, 16)
        gn_v.load_state_dict(gn.state_dict())
        gn_v = gn_v.to("vulkan")
        x_c, x_v = randn_pair(2, 16, 4, 4)
        check(gn(x_c), gn_v(x_v), rtol=RTOL_R, atol=ATOL_R)

    def test_layer_norm_1d(self):
        x_c, x_v = randn_pair(8, 64)
        w = torch.ones(64)
        b = torch.zeros(64)
        check(F.layer_norm(x_c, [64], w, b),
              F.layer_norm(x_v, [64], w.to("vulkan"), b.to("vulkan")), rtol=RTOL_R, atol=ATOL_R)

    def test_layer_norm_2d(self):
        x_c, x_v = randn_pair(4, 8, 16)
        w = torch.ones(16)
        b = torch.zeros(16)
        check(F.layer_norm(x_c, [16], w, b),
              F.layer_norm(x_v, [16], w.to("vulkan"), b.to("vulkan")), rtol=RTOL_R, atol=ATOL_R)

    def test_layer_norm_backward(self):
        x = torch.randn(4, 32, requires_grad=True)
        xv = x.detach().clone().to("vulkan").requires_grad_(True)
        w = torch.ones(32, requires_grad=True)
        wv = w.detach().clone().to("vulkan").requires_grad_(True)
        b = torch.zeros(32, requires_grad=True)
        bv = b.detach().clone().to("vulkan").requires_grad_(True)
        out_c = F.layer_norm(x, [32], w, b)
        out_v = F.layer_norm(xv, [32], wv, bv)
        go = torch.randn_like(out_c)
        out_c.backward(go)
        out_v.backward(go.to("vulkan"))
        check(x.grad, xv.grad, rtol=RTOL_VR, atol=ATOL_VR)

    def test_instance_norm(self):
        x_c, x_v = randn_pair(2, 4, 8, 8)
        check(F.instance_norm(x_c), F.instance_norm(x_v), rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  SHAPE OPS (ADDITIONAL)
# ═══════════════════════════════════════════════════════════════════

class TestShapeOpsExtended:
    """Extended shape operation tests."""

    def test_view(self):
        c, v = randn_pair(4, 8, 16)
        check(c.view(4, 128), v.view(4, 128))

    def test_reshape(self):
        c, v = randn_pair(32, 16)
        check(c.reshape(8, 64), v.reshape(8, 64))

    def test_transpose(self):
        c, v = randn_pair(8, 16)
        check(c.t(), v.t())

    def test_transpose_3d(self):
        c, v = randn_pair(4, 8, 16)
        check(c.transpose(1, 2), v.transpose(1, 2))

    def test_permute(self):
        c, v = randn_pair(2, 4, 8, 16)
        check(c.permute(0, 3, 1, 2), v.permute(0, 3, 1, 2))

    def test_unsqueeze_squeeze(self):
        c, v = randn_pair(8, 16)
        check(c.unsqueeze(0), v.unsqueeze(0))
        check(c.unsqueeze(1), v.unsqueeze(1))
        c2, v2 = randn_pair(1, 8, 1, 16)
        check(c2.squeeze(), v2.squeeze())

    def test_expand(self):
        c, v = randn_pair(1, 16)
        check(c.expand(8, 16), v.expand(8, 16))

    def test_expand_3d(self):
        c, v = randn_pair(1, 1, 16)
        check(c.expand(4, 8, 16), v.expand(4, 8, 16))

    def test_cat_dim0(self):
        a_c, a_v = randn_pair(4, 16)
        b_c, b_v = randn_pair(4, 16)
        check(torch.cat([a_c, b_c], dim=0), torch.cat([a_v, b_v], dim=0))

    def test_cat_dim1(self):
        a_c, a_v = randn_pair(4, 8)
        b_c, b_v = randn_pair(4, 12)
        check(torch.cat([a_c, b_c], dim=1), torch.cat([a_v, b_v], dim=1))

    def test_stack(self):
        a_c, a_v = randn_pair(4, 8)
        b_c, b_v = randn_pair(4, 8)
        check(torch.stack([a_c, b_c]), torch.stack([a_v, b_v]))

    def test_chunk(self):
        c, v = randn_pair(12, 16)
        c_chunks = c.chunk(3, dim=0)
        v_chunks = v.chunk(3, dim=0)
        for cc, vc in zip(c_chunks, v_chunks):
            check(cc, vc)

    def test_split(self):
        c, v = randn_pair(12, 16)
        c_parts = c.split(4, dim=0)
        v_parts = v.split(4, dim=0)
        for cc, vc in zip(c_parts, v_parts):
            check(cc, vc)

    def test_select(self):
        c, v = randn_pair(8, 16)
        check(c[3], v[3])
        check(c[:, 7], v[:, 7])

    def test_slice(self):
        c, v = randn_pair(8, 16)
        check(c[2:6], v[2:6])
        check(c[:, 4:12], v[:, 4:12])

    def test_flip(self):
        c, v = randn_pair(4, 8)
        check(c.flip(0), v.flip(0))
        check(c.flip(1), v.flip(1))

    def test_roll(self):
        c, v = randn_pair(4, 8)
        check(c.roll(3, dims=1), v.roll(3, dims=1))

    def test_repeat(self):
        c, v = randn_pair(4, 8)
        check(c.repeat(2, 3), v.repeat(2, 3))

    def test_triu_tril(self):
        c, v = randn_pair(8, 8)
        check(c.triu(), v.triu())
        check(c.tril(), v.tril())
        check(c.triu(1), v.triu(1))
        check(c.tril(-1), v.tril(-1))

    def test_narrow(self):
        c, v = randn_pair(8, 16)
        check(c.narrow(1, 4, 8), v.narrow(1, 4, 8))

    def test_contiguous(self):
        c, v = randn_pair(8, 16)
        ct = c.t().contiguous()
        vt = v.t().contiguous()
        check(ct, vt)


# ═══════════════════════════════════════════════════════════════════
#  POOLING (ADDITIONAL)
# ═══════════════════════════════════════════════════════════════════

class TestPoolingExtended:
    """Extended pooling tests."""

    def test_avg_pool2d_stride(self):
        x_c, x_v = randn_pair(2, 4, 8, 8)
        check(F.avg_pool2d(x_c, 3, stride=2, padding=1),
              F.avg_pool2d(x_v, 3, stride=2, padding=1), rtol=RTOL_R, atol=ATOL_R)

    def test_max_pool2d_stride(self):
        x_c, x_v = randn_pair(2, 4, 8, 8)
        check(F.max_pool2d(x_c, 3, stride=2, padding=1),
              F.max_pool2d(x_v, 3, stride=2, padding=1))

    def test_adaptive_avg_pool2d(self):
        x_c, x_v = randn_pair(2, 4, 7, 7)
        check(F.adaptive_avg_pool2d(x_c, (3, 3)),
              F.adaptive_avg_pool2d(x_v, (3, 3)), rtol=RTOL_R, atol=ATOL_R)

    def test_adaptive_avg_pool2d_1x1(self):
        x_c, x_v = randn_pair(2, 4, 8, 8)
        check(F.adaptive_avg_pool2d(x_c, (1, 1)),
              F.adaptive_avg_pool2d(x_v, (1, 1)), rtol=RTOL_R, atol=ATOL_R)

    def test_max_pool2d_backward(self):
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        xv = x.detach().clone().to("vulkan").requires_grad_(True)
        out_c = F.max_pool2d(x, 2)
        out_v = F.max_pool2d(xv, 2)
        go = torch.randn_like(out_c)
        out_c.backward(go)
        out_v.backward(go.to("vulkan"))
        check(x.grad, xv.grad, rtol=RTOL_R, atol=ATOL_R)

    def test_avg_pool2d_backward(self):
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        xv = x.detach().clone().to("vulkan").requires_grad_(True)
        out_c = F.avg_pool2d(x, 2)
        out_v = F.avg_pool2d(xv, 2)
        go = torch.randn_like(out_c)
        out_c.backward(go)
        out_v.backward(go.to("vulkan"))
        check(x.grad, xv.grad, rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  CONV (ADDITIONAL)
# ═══════════════════════════════════════════════════════════════════

class TestConvExtended:
    """Extended convolution tests."""

    def test_conv2d_dilated(self):
        x_c, x_v = randn_pair(2, 3, 16, 16)
        w = torch.randn(8, 3, 3, 3)
        b = torch.zeros(8)
        wv, bv = w.to("vulkan"), b.to("vulkan")
        check(F.conv2d(x_c, w, b, dilation=2),
              F.conv2d(x_v, wv, bv, dilation=2), rtol=RTOL_R, atol=ATOL_R)

    def test_conv2d_groups(self):
        x_c, x_v = randn_pair(2, 8, 8, 8)
        w = torch.randn(8, 4, 3, 3)  # groups=2
        b = torch.zeros(8)
        wv, bv = w.to("vulkan"), b.to("vulkan")
        check(F.conv2d(x_c, w, b, groups=2),
              F.conv2d(x_v, wv, bv, groups=2), rtol=RTOL_R, atol=ATOL_R)

    def test_conv2d_backward(self):
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        xv = x.detach().clone().to("vulkan").requires_grad_(True)
        w = torch.randn(4, 3, 3, 3, requires_grad=True)
        wv = w.detach().clone().to("vulkan").requires_grad_(True)
        b = torch.zeros(4, requires_grad=True)
        bv = b.detach().clone().to("vulkan").requires_grad_(True)
        out_c = F.conv2d(x, w, b, padding=1)
        out_v = F.conv2d(xv, wv, bv, padding=1)
        go = torch.randn_like(out_c)
        out_c.backward(go)
        out_v.backward(go.to("vulkan"))
        check(x.grad, xv.grad, rtol=RTOL_VR, atol=ATOL_VR)

    def test_conv1d_forward(self):
        x_c, x_v = randn_pair(2, 3, 32)
        w = torch.randn(8, 3, 3)
        b = torch.zeros(8)
        wv, bv = w.to("vulkan"), b.to("vulkan")
        check(F.conv1d(x_c, w, b, padding=1),
              F.conv1d(x_v, wv, bv, padding=1), rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  ACTIVATION EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestActivationEdgeCases:
    """Test activations with extreme inputs."""

    def test_relu_all_negative(self):
        c = torch.tensor([-3.0, -2.0, -1.0, -0.5, -0.001])
        v = c.to("vulkan")
        check(F.relu(c), F.relu(v))

    def test_relu_all_positive(self):
        c = torch.tensor([0.001, 0.5, 1.0, 2.0, 3.0])
        v = c.to("vulkan")
        check(F.relu(c), F.relu(v))

    def test_sigmoid_extreme(self):
        c = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0])
        v = c.to("vulkan")
        check(torch.sigmoid(c), torch.sigmoid(v))

    def test_softmax_large_logits(self):
        c = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
        v = c.to("vulkan")
        check(F.softmax(c, dim=-1), F.softmax(v, dim=-1))

    def test_softmax_negative_logits(self):
        c = torch.tensor([[-100.0, -200.0, -150.0]])
        v = c.to("vulkan")
        check(F.softmax(c, dim=-1), F.softmax(v, dim=-1), rtol=RTOL_R, atol=ATOL_R)

    def test_log_softmax_stability(self):
        """log_softmax should be stable for large values."""
        c = torch.tensor([[1000.0, 999.0, 998.0]])
        v = c.to("vulkan")
        check(F.log_softmax(c, dim=-1), F.log_softmax(v, dim=-1), rtol=RTOL_R, atol=ATOL_R)

    def test_hardswish(self):
        c = torch.linspace(-5, 5, 100)
        v = c.to("vulkan")
        check(F.hardswish(c), F.hardswish(v))

    def test_hardsigmoid(self):
        c = torch.linspace(-5, 5, 100)
        v = c.to("vulkan")
        check(F.hardsigmoid(c), F.hardsigmoid(v))

    def test_softplus(self):
        c = torch.linspace(-5, 5, 100)
        v = c.to("vulkan")
        check(F.softplus(c), F.softplus(v), rtol=RTOL_R, atol=ATOL_R)

    def test_elu(self):
        c = torch.linspace(-5, 5, 100)
        v = c.to("vulkan")
        check(F.elu(c), F.elu(v))

    def test_leaky_relu_slopes(self):
        c, v = randn_pair(32)
        for slope in [0.01, 0.1, 0.2]:
            check(F.leaky_relu(c, slope), F.leaky_relu(v, slope))

    def test_mish(self):
        c = torch.linspace(-5, 5, 100)
        v = c.to("vulkan")
        check(F.mish(c), F.mish(v), rtol=RTOL_R, atol=ATOL_R)

    def test_prelu(self):
        c, v = randn_pair(32, 16)
        w = torch.tensor([0.25])
        wv = w.to("vulkan")
        check(F.prelu(c, w), F.prelu(v, wv))


# ═══════════════════════════════════════════════════════════════════
#  BINARY OP EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestBinaryEdgeCases:
    """Binary op edge cases and special values."""

    def test_pow_integer_exponent(self):
        c, v = randn_pair(32)
        check(c.pow(2), v.pow(2))
        check(c.pow(3), v.pow(3), rtol=RTOL_R, atol=ATOL_R)

    def test_pow_fractional(self):
        c = torch.rand(32) + 0.1
        v = c.to("vulkan")
        check(c.pow(0.5), v.pow(0.5), rtol=RTOL_R, atol=ATOL_R)
        check(c.pow(1.5), v.pow(1.5), rtol=RTOL_R, atol=ATOL_R)

    def test_pow_negative_base(self):
        c = torch.tensor([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        v = c.to("vulkan")
        check(c.pow(2.0), v.pow(2.0), rtol=RTOL_R, atol=ATOL_R)

    def test_fmod(self):
        c, v = randn_pair(32)
        check(c.fmod(0.5), v.fmod(0.5))

    def test_remainder(self):
        c, v = randn_pair(32)
        check(c.remainder(0.5), v.remainder(0.5), rtol=RTOL_R, atol=ATOL_R)

    def test_atan2(self):
        y_c, y_v = randn_pair(32)
        x_c, x_v = randn_pair(32)
        check(torch.atan2(y_c, x_c), torch.atan2(y_v, x_v), rtol=RTOL_R, atol=ATOL_R)

    def test_min_max_clamp(self):
        """minimum/maximum not registered — verify via clamp equivalents."""
        a_c, a_v = randn_pair(32)
        b_c, b_v = randn_pair(32)
        # Use torch.where as workaround for minimum/maximum
        check(torch.where(a_c < b_c, a_c, b_c),
              torch.where(a_v < b_v, a_v, b_v))
        check(torch.where(a_c > b_c, a_c, b_c),
              torch.where(a_v > b_v, a_v, b_v))

    def test_where(self):
        cond = torch.randn(32) > 0
        a_c, a_v = randn_pair(32)
        b_c, b_v = randn_pair(32)
        condv = cond.to("vulkan")
        check(torch.where(cond, a_c, b_c), torch.where(condv, a_v, b_v))


# ═══════════════════════════════════════════════════════════════════
#  COMPARISON OPS
# ═══════════════════════════════════════════════════════════════════

class TestComparisonOps:
    """Test comparison operations."""

    @pytest.mark.parametrize("op", ["eq", "ne", "lt", "gt", "le", "ge"])
    def test_comparison_tensor(self, op):
        a_c, a_v = randn_pair(32, 32)
        b_c, b_v = randn_pair(32, 32)
        cpu_result = getattr(a_c, op)(b_c).float()
        vk_result = getattr(a_v, op)(b_v)
        check(cpu_result, vk_result)

    @pytest.mark.parametrize("op", ["eq", "ne", "lt", "gt", "le", "ge"])
    def test_comparison_scalar(self, op):
        a_c, a_v = randn_pair(32, 32)
        cpu_result = getattr(a_c, op)(0.0).float()
        vk_result = getattr(a_v, op)(0.0)
        check(cpu_result, vk_result)


# ═══════════════════════════════════════════════════════════════════
#  MATMUL SHAPES
# ═══════════════════════════════════════════════════════════════════

class TestMatmulShapes:
    """Test matmul with various shape combinations."""

    def test_mv(self):
        """Matrix-vector multiply — use unsqueeze+mm+squeeze."""
        m_c, m_v = randn_pair(8, 16)
        v_c = torch.randn(16, 1)
        v_v = v_c.to("vulkan")
        check((m_c @ v_c).squeeze(), (m_v @ v_v).squeeze(), rtol=RTOL_R, atol=ATOL_R)

    def test_mm_square(self):
        a_c, a_v = randn_pair(16, 16)
        b_c, b_v = randn_pair(16, 16)
        check(a_c @ b_c, a_v @ b_v, rtol=RTOL_R, atol=ATOL_R)

    def test_mm_nonsquare(self):
        a_c, a_v = randn_pair(8, 32)
        b_c, b_v = randn_pair(32, 16)
        check(a_c @ b_c, a_v @ b_v, rtol=RTOL_R, atol=ATOL_R)

    def test_bmm(self):
        a_c, a_v = randn_pair(4, 8, 16)
        b_c, b_v = randn_pair(4, 16, 8)
        check(torch.bmm(a_c, b_c), torch.bmm(a_v, b_v), rtol=RTOL_R, atol=ATOL_R)

    def test_addmm(self):
        bias = torch.randn(16)
        a_c, a_v = randn_pair(8, 32)
        b = torch.randn(32, 16)
        bv = b.to("vulkan")
        biasv = bias.to("vulkan")
        check(torch.addmm(bias, a_c, b), torch.addmm(biasv, a_v, bv), rtol=RTOL_R, atol=ATOL_R)

    def test_addmm_alpha_beta(self):
        bias = torch.randn(16)
        a_c, a_v = randn_pair(8, 32)
        b = torch.randn(32, 16)
        bv = b.to("vulkan")
        biasv = bias.to("vulkan")
        check(torch.addmm(bias, a_c, b, beta=0.5, alpha=2.0),
              torch.addmm(biasv, a_v, bv, beta=0.5, alpha=2.0), rtol=RTOL_R, atol=ATOL_R)

    def test_linear(self):
        x_c, x_v = randn_pair(8, 32)
        w = torch.randn(16, 32)
        b = torch.randn(16)
        wv, bv = w.to("vulkan"), b.to("vulkan")
        check(F.linear(x_c, w, b), F.linear(x_v, wv, bv), rtol=RTOL_R, atol=ATOL_R)

    def test_linear_3d(self):
        x_c, x_v = randn_pair(4, 8, 32)
        w = torch.randn(16, 32)
        b = torch.randn(16)
        wv, bv = w.to("vulkan"), b.to("vulkan")
        check(F.linear(x_c, w, b), F.linear(x_v, wv, bv), rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  NUMERICAL STABILITY
# ═══════════════════════════════════════════════════════════════════

class TestNumericalStability:
    """Tests for numerical stability in tricky scenarios."""

    def test_softmax_uniform(self):
        """Softmax of uniform values should give 1/N."""
        c = torch.ones(1, 100) * 5.0
        v = c.to("vulkan")
        result = F.softmax(v, dim=-1).cpu()
        expected = torch.ones(1, 100) / 100.0
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_log_softmax_numerical(self):
        """log_softmax(x) ≈ x - logsumexp(x) — verify stability."""
        c = torch.randn(8, 64)
        v = c.to("vulkan")
        check(F.log_softmax(c, dim=-1), F.log_softmax(v, dim=-1), rtol=RTOL_R, atol=ATOL_R)

    def test_cross_entropy_large_logits(self):
        """CE with large logits should still be stable."""
        logits = torch.randn(4, 10) * 50.0
        targets = torch.tensor([3, 7, 0, 5])
        lv, tv = logits.to("vulkan"), targets.to("vulkan")
        check(F.cross_entropy(logits, targets),
              F.cross_entropy(lv, tv), rtol=RTOL_R, atol=ATOL_R)

    def test_batchnorm_near_zero_var(self):
        """BN eval with nearly constant input should still work."""
        x = torch.ones(4, 2, 4, 4) + torch.randn(4, 2, 4, 4) * 1e-5
        bn = nn.BatchNorm2d(2)
        bn.eval()  # Vulkan only supports eval mode
        bn_v = nn.BatchNorm2d(2)
        bn_v.load_state_dict(bn.state_dict())
        bn_v = bn_v.to("vulkan")
        bn_v.eval()
        check(bn(x), bn_v(x.to("vulkan")), rtol=RTOL_VR, atol=ATOL_VR)

    def test_layer_norm_near_zero_var(self):
        """LN with nearly constant input."""
        x = torch.ones(4, 32) + torch.randn(4, 32) * 1e-5
        w, b = torch.ones(32), torch.zeros(32)
        xv, wv, bv = x.to("vulkan"), w.to("vulkan"), b.to("vulkan")
        check(F.layer_norm(x, [32], w, b),
              F.layer_norm(xv, [32], wv, bv), rtol=RTOL_VR, atol=ATOL_VR)

    def test_division_by_small(self):
        """Division by small numbers."""
        a_c, a_v = randn_pair(32)
        b = torch.full((32,), 1e-6)
        bv = b.to("vulkan")
        check(a_c / b, a_v / bv, rtol=RTOL_R, atol=ATOL_R)

    def test_exp_overflow_guard(self):
        """exp of very large input should give inf, not NaN."""
        c = torch.tensor([80.0, 85.0, 90.0])
        v = c.to("vulkan")
        rc, rv = torch.exp(c), torch.exp(v).cpu()
        assert torch.all(torch.isinf(rc) == torch.isinf(rv))

    def test_log_of_zero(self):
        """log(0) should give -inf, not NaN."""
        c = torch.tensor([0.0, 1.0, 2.0])
        v = c.to("vulkan")
        rc, rv = torch.log(c), torch.log(v).cpu()
        assert torch.isinf(rc[0]) and torch.isinf(rv[0])
        check(rc[1:], rv[1:])


# ═══════════════════════════════════════════════════════════════════
#  EMBEDDING + RMS NORM
# ═══════════════════════════════════════════════════════════════════

class TestSpecialOps:
    """Test special ops: RMSNorm, RoPE, SwiGLU, etc."""

    def test_rms_norm(self):
        try:
            import torch_vulkan
        except ImportError:
            pytest.skip("torch_vulkan not available")
        x = torch.randn(4, 32)
        w = torch.ones(32)
        # Manual RMSNorm: x / sqrt(mean(x^2) + eps) * w
        variance = x.pow(2).mean(-1, keepdim=True)
        expected = x * torch.rsqrt(variance + 1e-6) * w
        xv, wv = x.to("vulkan"), w.to("vulkan")
        result = torch_vulkan.rms_norm(xv, wv, 1e-6)
        check(expected, result, rtol=RTOL_R, atol=ATOL_R)

    def test_rms_norm_backward(self):
        try:
            import torch_vulkan
        except ImportError:
            pytest.skip("torch_vulkan not available")
        x = torch.randn(4, 32, requires_grad=True)
        w = torch.ones(32, requires_grad=True)
        variance = x.pow(2).mean(-1, keepdim=True)
        out_c = x * torch.rsqrt(variance + 1e-6) * w
        xv = x.detach().clone().to("vulkan").requires_grad_(True)
        wv = w.detach().clone().to("vulkan").requires_grad_(True)
        out_v = torch_vulkan.rms_norm(xv, wv, 1e-6)
        go = torch.randn_like(out_c)
        out_c.backward(go)
        out_v.backward(go.to("vulkan"))
        check(x.grad, xv.grad, rtol=RTOL_VR, atol=ATOL_VR)

    def test_swiglu(self):
        try:
            import torch_vulkan
        except ImportError:
            pytest.skip("torch_vulkan not available")
        gate = torch.randn(4, 32)
        up = torch.randn(4, 32)
        expected = F.silu(gate) * up
        gv, uv = gate.to("vulkan"), up.to("vulkan")
        result = torch_vulkan.swiglu(gv, uv)
        check(expected, result, rtol=RTOL_R, atol=ATOL_R)

    def test_embedding_large_vocab(self):
        """Test embedding with large vocabulary."""
        emb_w = torch.randn(1000, 64)
        indices = torch.randint(0, 1000, (8, 16))
        expected = F.embedding(indices, emb_w)
        emb_wv = emb_w.to("vulkan")
        indices_v = indices.to("vulkan")
        result = F.embedding(indices_v, emb_wv)
        check(expected, result)

    def test_embedding_backward_accumulation(self):
        """Embedding backward with duplicate indices should accumulate."""
        emb = nn.Embedding(10, 8)
        emb_v = nn.Embedding(10, 8)
        emb_v.load_state_dict(emb.state_dict())
        emb_v = emb_v.to("vulkan")
        # Duplicate index to test accumulation
        idx = torch.tensor([3, 3, 5, 3, 7, 5])
        idx_v = idx.to("vulkan")
        out_c = emb(idx).sum()
        out_v = emb_v(idx_v).sum()
        out_c.backward()
        out_v.backward()
        check(emb.weight.grad, emb_v.weight.grad, rtol=RTOL_R, atol=ATOL_R)


# ═══════════════════════════════════════════════════════════════════
#  GRADIENT CHAIN CORRECTNESS
# ═══════════════════════════════════════════════════════════════════

class TestGradientChains:
    """Test backward through multi-op chains."""

    def test_linear_chain_backward(self):
        """Backward through: linear → relu → linear."""
        x = torch.randn(4, 16, requires_grad=True)
        w1 = torch.randn(32, 16, requires_grad=True)
        b1 = torch.randn(32, requires_grad=True)
        w2 = torch.randn(8, 32, requires_grad=True)
        b2 = torch.randn(8, requires_grad=True)
        xv = x.detach().clone().to("vulkan").requires_grad_(True)
        w1v = w1.detach().clone().to("vulkan").requires_grad_(True)
        b1v = b1.detach().clone().to("vulkan").requires_grad_(True)
        w2v = w2.detach().clone().to("vulkan").requires_grad_(True)
        b2v = b2.detach().clone().to("vulkan").requires_grad_(True)
        out_c = F.linear(F.relu(F.linear(x, w1, b1)), w2, b2)
        out_v = F.linear(F.relu(F.linear(xv, w1v, b1v)), w2v, b2v)
        go = torch.randn_like(out_c)
        out_c.backward(go)
        out_v.backward(go.to("vulkan"))
        check(x.grad, xv.grad, rtol=RTOL_VR, atol=ATOL_VR)
        check(w1.grad, w1v.grad, rtol=RTOL_VR, atol=ATOL_VR)
        check(w2.grad, w2v.grad, rtol=RTOL_VR, atol=ATOL_VR)

    def test_conv_relu_pool_backward(self):
        """Backward through: conv → relu → pool."""
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        w = torch.randn(4, 3, 3, 3, requires_grad=True)
        b = torch.zeros(4, requires_grad=True)
        xv = x.detach().clone().to("vulkan").requires_grad_(True)
        wv = w.detach().clone().to("vulkan").requires_grad_(True)
        bv = b.detach().clone().to("vulkan").requires_grad_(True)
        out_c = F.avg_pool2d(F.relu(F.conv2d(x, w, b, padding=1)), 2)
        out_v = F.avg_pool2d(F.relu(F.conv2d(xv, wv, bv, padding=1)), 2)
        go = torch.randn_like(out_c)
        out_c.backward(go)
        out_v.backward(go.to("vulkan"))
        check(x.grad, xv.grad, rtol=RTOL_VR, atol=ATOL_VR)

    def test_residual_backward(self):
        """Backward through a residual connection: x + f(x)."""
        x = torch.randn(4, 16, requires_grad=True)
        w = torch.randn(16, 16, requires_grad=True)
        xv = x.detach().clone().to("vulkan").requires_grad_(True)
        wv = w.detach().clone().to("vulkan").requires_grad_(True)
        out_c = x + F.relu(x @ w)
        out_v = xv + F.relu(xv @ wv)
        go = torch.randn_like(out_c)
        out_c.backward(go)
        out_v.backward(go.to("vulkan"))
        check(x.grad, xv.grad, rtol=RTOL_VR, atol=ATOL_VR)
        check(w.grad, wv.grad, rtol=RTOL_VR, atol=ATOL_VR)

    def test_norm_softmax_ce_backward(self):
        """Backward through: layer_norm → linear → cross_entropy."""
        x = torch.randn(4, 32, requires_grad=True)
        w_ln = torch.ones(32, requires_grad=True)
        b_ln = torch.zeros(32, requires_grad=True)
        w = torch.randn(10, 32, requires_grad=True)
        b = torch.randn(10, requires_grad=True)
        targets = torch.tensor([3, 7, 0, 5])

        xv = x.detach().clone().to("vulkan").requires_grad_(True)
        w_lnv = w_ln.detach().clone().to("vulkan").requires_grad_(True)
        b_lnv = b_ln.detach().clone().to("vulkan").requires_grad_(True)
        wv = w.detach().clone().to("vulkan").requires_grad_(True)
        bv = b.detach().clone().to("vulkan").requires_grad_(True)
        tv = targets.to("vulkan")

        normed_c = F.layer_norm(x, [32], w_ln, b_ln)
        loss_c = F.cross_entropy(F.linear(normed_c, w, b), targets)
        normed_v = F.layer_norm(xv, [32], w_lnv, b_lnv)
        loss_v = F.cross_entropy(F.linear(normed_v, wv, bv), tv)
        loss_c.backward()
        loss_v.backward()
        check(x.grad, xv.grad, rtol=5e-3, atol=5e-3)


# ═══════════════════════════════════════════════════════════════════
#  DETERMINISM
# ═══════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Verify ops are deterministic (same input → same output)."""

    def test_matmul_deterministic(self):
        a = torch.randn(16, 32).to("vulkan")
        b = torch.randn(32, 16).to("vulkan")
        r1 = (a @ b).cpu()
        r2 = (a @ b).cpu()
        torch.testing.assert_close(r1, r2)

    def test_softmax_deterministic(self):
        x = torch.randn(8, 64).to("vulkan")
        r1 = F.softmax(x, dim=-1).cpu()
        r2 = F.softmax(x, dim=-1).cpu()
        torch.testing.assert_close(r1, r2)

    def test_conv2d_deterministic(self):
        x = torch.randn(2, 3, 8, 8).to("vulkan")
        w = torch.randn(4, 3, 3, 3).to("vulkan")
        b = torch.zeros(4).to("vulkan")
        r1 = F.conv2d(x, w, b, padding=1).cpu()
        r2 = F.conv2d(x, w, b, padding=1).cpu()
        torch.testing.assert_close(r1, r2)

    def test_layer_norm_deterministic(self):
        x = torch.randn(4, 32).to("vulkan")
        w = torch.ones(32).to("vulkan")
        b = torch.zeros(32).to("vulkan")
        r1 = F.layer_norm(x, [32], w, b).cpu()
        r2 = F.layer_norm(x, [32], w, b).cpu()
        torch.testing.assert_close(r1, r2)
