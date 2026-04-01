"""Correctness verification: compare Vulkan outputs against CPU reference.

Tests numerical accuracy across all op categories with tight tolerances.
Each test runs the same operation on CPU and Vulkan, then compares results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import math

# Tight tolerances — SwiftShader should be numerically close to CPU
RTOL = 1e-4
ATOL = 1e-5
# Relaxed for reductions and multi-step ops (accumulation error)
RTOL_RELAXED = 1e-3
ATOL_RELAXED = 1e-3


@pytest.fixture(autouse=True)
def setup():
    try:
        import torch_vulkan
        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
    except ImportError:
        pytest.skip("torch_vulkan not installed")


def check(cpu_result, vulkan_result, rtol=RTOL, atol=ATOL):
    """Compare Vulkan result against CPU reference."""
    actual = vulkan_result.cpu() if vulkan_result.device.type != "cpu" else vulkan_result
    expected = cpu_result.cpu() if cpu_result.device.type != "cpu" else cpu_result
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def randn_pair(*shape):
    """Create matching CPU and Vulkan tensors."""
    cpu = torch.randn(*shape)
    vk = cpu.to("vulkan")
    return cpu, vk


def rand_pair(*shape):
    """Create matching positive CPU and Vulkan tensors."""
    cpu = torch.rand(*shape) + 0.1  # avoid zero
    vk = cpu.to("vulkan")
    return cpu, vk


# ═══════════════════════════════════════════════════════════════════
# Binary ops
# ═══════════════════════════════════════════════════════════════════

class TestBinaryOpsCorrectness:
    """Verify binary ops match CPU exactly."""

    @pytest.mark.parametrize("op", [torch.add, torch.sub, torch.mul, torch.div])
    def test_binary_elementwise(self, op):
        a_cpu, a_vk = randn_pair(32, 32)
        b_cpu, b_vk = randn_pair(32, 32)
        if op == torch.div:
            b_cpu = b_cpu.abs() + 0.1
            b_vk = b_cpu.to("vulkan")
        check(op(a_cpu, b_cpu), op(a_vk, b_vk))

    @pytest.mark.parametrize("op", [torch.add, torch.sub, torch.mul, torch.div])
    def test_binary_scalar(self, op):
        a_cpu, a_vk = randn_pair(16, 16)
        scalar = 2.5
        if op in (torch.add, torch.sub):
            check(op(a_cpu, scalar), op(a_vk, scalar))
        else:
            check(op(a_cpu, scalar), op(a_vk, scalar))

    def test_binary_broadcast(self):
        """Broadcast: (4,8) + (1,8) and (4,8) + (8,)."""
        a_cpu, a_vk = randn_pair(4, 8)
        b_cpu = torch.randn(1, 8)
        b_vk = b_cpu.to("vulkan")
        check(a_cpu + b_cpu, a_vk + b_vk)

        c_cpu = torch.randn(8)
        c_vk = c_cpu.to("vulkan")
        check(a_cpu + c_cpu, a_vk + c_vk)

    def test_pow_positive_base(self):
        base_cpu, base_vk = rand_pair(16, 16)
        exp_cpu = torch.randn(16, 16)
        exp_vk = exp_cpu.to("vulkan")
        check(torch.pow(base_cpu, exp_cpu), torch.pow(base_vk, exp_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_pow_negative_base_integer_exp(self):
        """pow with negative base and integer exponent."""
        base = torch.tensor([-2.0, -3.0, -1.0, 2.0])
        exp = torch.tensor([2.0, 3.0, 4.0, 3.0])
        check(torch.pow(base, exp), torch.pow(base.to("vulkan"), exp.to("vulkan")),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# Unary ops
# ═══════════════════════════════════════════════════════════════════

class TestUnaryOpsCorrectness:
    """Verify unary ops match CPU."""

    @pytest.mark.parametrize("op", [
        torch.neg, torch.abs, torch.exp, torch.sqrt, torch.rsqrt,
        torch.ceil, torch.floor, torch.round, torch.sign,
        torch.reciprocal, torch.sin, torch.cos,
    ])
    def test_unary_op(self, op):
        if op in (torch.sqrt, torch.rsqrt, torch.reciprocal):
            cpu, vk = rand_pair(32, 32)
        elif op == torch.exp:
            # Avoid overflow
            cpu = torch.randn(32, 32).clamp(-5, 5)
            vk = cpu.to("vulkan")
        else:
            cpu, vk = randn_pair(32, 32)
        # sin/cos have slightly lower GPU precision
        tol = (RTOL_RELAXED, ATOL_RELAXED) if op in (torch.sin, torch.cos) else (RTOL, ATOL)
        check(op(cpu), op(vk), rtol=tol[0], atol=tol[1])

    def test_log_positive(self):
        cpu, vk = rand_pair(32, 32)
        check(torch.log(cpu), torch.log(vk))

    def test_erf(self):
        cpu, vk = randn_pair(32, 32)
        check(torch.erf(cpu), torch.erf(vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# Activations
# ═══════════════════════════════════════════════════════════════════

class TestActivationsCorrectness:
    """Verify activation functions match CPU."""

    def test_relu(self):
        cpu, vk = randn_pair(32, 32)
        check(F.relu(cpu), F.relu(vk))

    def test_sigmoid(self):
        cpu, vk = randn_pair(32, 32)
        check(torch.sigmoid(cpu), torch.sigmoid(vk))

    def test_tanh(self):
        cpu, vk = randn_pair(32, 32)
        check(torch.tanh(cpu), torch.tanh(vk))

    def test_gelu(self):
        cpu, vk = randn_pair(32, 32)
        check(F.gelu(cpu), F.gelu(vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_silu(self):
        cpu, vk = randn_pair(32, 32)
        check(F.silu(cpu), F.silu(vk))

    def test_leaky_relu(self):
        cpu, vk = randn_pair(32, 32)
        check(F.leaky_relu(cpu, 0.01), F.leaky_relu(vk, 0.01))

    def test_elu(self):
        cpu, vk = randn_pair(32, 32)
        check(F.elu(cpu), F.elu(vk))

    def test_selu(self):
        cpu, vk = randn_pair(32, 32)
        check(F.selu(cpu), F.selu(vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_prelu(self):
        cpu, vk = randn_pair(1, 8, 4, 4)
        weight = torch.randn(8)
        m_cpu = nn.PReLU(8)
        m_cpu.weight.data.copy_(weight)
        m_vk = nn.PReLU(8)
        m_vk.weight.data.copy_(weight)
        m_vk = m_vk.to("vulkan")
        check(m_cpu(cpu), m_vk(vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# BLAS / Linear algebra
# ═══════════════════════════════════════════════════════════════════

class TestBLASCorrectness:
    """Verify mm, bmm, addmm, linear match CPU."""

    def test_mm(self):
        a_cpu, a_vk = randn_pair(16, 32)
        b_cpu, b_vk = randn_pair(32, 8)
        check(torch.mm(a_cpu, b_cpu), torch.mm(a_vk, b_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_mm_large(self):
        """Larger mm to stress GPU compute."""
        a_cpu, a_vk = randn_pair(64, 128)
        b_cpu, b_vk = randn_pair(128, 64)
        check(torch.mm(a_cpu, b_cpu), torch.mm(a_vk, b_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_bmm(self):
        a_cpu, a_vk = randn_pair(4, 8, 16)
        b_cpu, b_vk = randn_pair(4, 16, 8)
        check(torch.bmm(a_cpu, b_cpu), torch.bmm(a_vk, b_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_addmm(self):
        bias_cpu = torch.randn(8)
        a_cpu, a_vk = randn_pair(16, 32)
        b_cpu, b_vk = randn_pair(32, 8)
        bias_vk = bias_cpu.to("vulkan")
        check(torch.addmm(bias_cpu, a_cpu, b_cpu),
              torch.addmm(bias_vk, a_vk, b_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_linear(self):
        torch.manual_seed(42)
        m_cpu = nn.Linear(32, 16)
        m_vk = nn.Linear(32, 16)
        m_vk.load_state_dict(m_cpu.state_dict())
        m_vk = m_vk.to("vulkan")
        x_cpu, x_vk = randn_pair(4, 32)
        check(m_cpu(x_cpu), m_vk(x_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_linear_batched(self):
        """3D input to linear (batched)."""
        torch.manual_seed(42)
        m_cpu = nn.Linear(16, 8)
        m_vk = nn.Linear(16, 8)
        m_vk.load_state_dict(m_cpu.state_dict())
        m_vk = m_vk.to("vulkan")
        x_cpu, x_vk = randn_pair(2, 4, 16)
        check(m_cpu(x_cpu), m_vk(x_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# Reductions
# ═══════════════════════════════════════════════════════════════════

class TestReductionsCorrectness:
    """Verify reductions match CPU."""

    def test_sum_full(self):
        cpu, vk = randn_pair(16, 16)
        check(cpu.sum(), vk.sum(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_sum_dim(self):
        cpu, vk = randn_pair(8, 16)
        check(cpu.sum(dim=0), vk.sum(dim=0), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(cpu.sum(dim=1), vk.sum(dim=1), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_sum_keepdim(self):
        cpu, vk = randn_pair(4, 8)
        check(cpu.sum(dim=1, keepdim=True), vk.sum(dim=1, keepdim=True),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_mean_full(self):
        cpu, vk = randn_pair(16, 16)
        check(cpu.mean(), vk.mean(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_mean_dim(self):
        cpu, vk = randn_pair(4, 8, 16)
        check(cpu.mean(dim=1), vk.mean(dim=1), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(cpu.mean(dim=2), vk.mean(dim=2), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_mean_multi_dim(self):
        cpu, vk = randn_pair(4, 8, 16)
        check(cpu.mean(dim=[1, 2]), vk.mean(dim=[1, 2]), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_amax(self):
        cpu, vk = randn_pair(8, 16)
        check(cpu.amax(), vk.amax())
        check(cpu.amax(dim=0), vk.amax(dim=0))
        check(cpu.amax(dim=1), vk.amax(dim=1))

    def test_amin(self):
        cpu, vk = randn_pair(8, 16)
        check(cpu.amin(), vk.amin())
        check(cpu.amin(dim=0), vk.amin(dim=0))

    def test_argmax(self):
        cpu, vk = randn_pair(4, 8)
        assert cpu.argmax().item() == vk.argmax().cpu().item()
        check(cpu.argmax(dim=1), vk.argmax(dim=1))

    def test_argmin(self):
        cpu, vk = randn_pair(4, 8)
        assert cpu.argmin().item() == vk.argmin().cpu().item()

    def test_prod_dim(self):
        # Small values to avoid overflow
        cpu = torch.rand(4, 4) * 0.5 + 0.5
        vk = cpu.to("vulkan")
        check(cpu.prod(dim=0), vk.prod(dim=0), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# Comparison ops
# ═══════════════════════════════════════════════════════════════════

class TestComparisonCorrectness:
    """Verify comparison ops produce identical boolean results."""

    @pytest.mark.parametrize("op", [torch.eq, torch.ne, torch.lt, torch.gt, torch.le, torch.ge])
    def test_comparison_tensor(self, op):
        a_cpu, a_vk = randn_pair(16, 16)
        b_cpu, b_vk = randn_pair(16, 16)
        cpu_r = op(a_cpu, b_cpu)
        vk_r = op(a_vk, b_vk).cpu()
        assert (cpu_r == vk_r).all(), f"{op.__name__} mismatch"

    @pytest.mark.parametrize("op", [torch.eq, torch.ne, torch.lt, torch.gt, torch.le, torch.ge])
    def test_comparison_scalar(self, op):
        a_cpu, a_vk = randn_pair(16, 16)
        cpu_r = op(a_cpu, 0.0)
        vk_r = op(a_vk, 0.0).cpu()
        assert (cpu_r == vk_r).all(), f"{op.__name__} scalar mismatch"

    def test_where(self):
        cond = torch.randn(8, 8) > 0
        a_cpu, a_vk = randn_pair(8, 8)
        b_cpu, b_vk = randn_pair(8, 8)
        check(torch.where(cond, a_cpu, b_cpu),
              torch.where(cond.to("vulkan"), a_vk, b_vk))


# ═══════════════════════════════════════════════════════════════════
# Shape ops
# ═══════════════════════════════════════════════════════════════════

class TestShapeOpsCorrectness:
    """Verify shape ops preserve data."""

    def test_view_reshape(self):
        cpu, vk = randn_pair(4, 8)
        check(cpu.view(2, 16), vk.view(2, 16))
        check(cpu.reshape(32), vk.reshape(32))

    def test_permute_transpose(self):
        cpu, vk = randn_pair(2, 3, 4)
        check(cpu.permute(2, 0, 1), vk.permute(2, 0, 1))
        check(cpu.transpose(0, 2), vk.transpose(0, 2))

    def test_unsqueeze_squeeze(self):
        cpu, vk = randn_pair(4, 8)
        r_cpu = cpu.unsqueeze(0).squeeze(0)
        r_vk = vk.unsqueeze(0).squeeze(0)
        check(r_cpu, r_vk)

    def test_expand(self):
        cpu, vk = randn_pair(1, 8)
        check(cpu.expand(4, 8), vk.expand(4, 8))

    def test_cat(self):
        a_cpu, a_vk = randn_pair(4, 8)
        b_cpu, b_vk = randn_pair(4, 8)
        check(torch.cat([a_cpu, b_cpu], dim=0), torch.cat([a_vk, b_vk], dim=0))
        check(torch.cat([a_cpu, b_cpu], dim=1), torch.cat([a_vk, b_vk], dim=1))

    def test_stack(self):
        a_cpu, a_vk = randn_pair(4, 8)
        b_cpu, b_vk = randn_pair(4, 8)
        check(torch.stack([a_cpu, b_cpu]), torch.stack([a_vk, b_vk]))

    def test_select_slice(self):
        cpu, vk = randn_pair(4, 8, 16)
        check(cpu[1], vk[1])
        check(cpu[:, 2:5], vk[:, 2:5])
        check(cpu[:, :, ::2], vk[:, :, ::2])

    def test_split(self):
        cpu, vk = randn_pair(8, 16)
        cpu_splits = torch.split(cpu, 4, dim=0)
        vk_splits = torch.split(vk, 4, dim=0)
        for c, v in zip(cpu_splits, vk_splits):
            check(c, v)

    def test_flip(self):
        cpu, vk = randn_pair(4, 8)
        check(torch.flip(cpu, [0]), torch.flip(vk, [0]))
        check(torch.flip(cpu, [0, 1]), torch.flip(vk, [0, 1]))

    def test_roll(self):
        cpu, vk = randn_pair(4, 8)
        check(torch.roll(cpu, 2, 0), torch.roll(vk, 2, 0))
        check(torch.roll(cpu, -3, 1), torch.roll(vk, -3, 1))

    def test_repeat(self):
        cpu, vk = randn_pair(2, 4)
        check(cpu.repeat(3, 2), vk.repeat(3, 2))

    def test_triu_tril(self):
        cpu, vk = randn_pair(4, 4)
        check(torch.triu(cpu), torch.triu(vk))
        check(torch.tril(cpu), torch.tril(vk))
        check(torch.triu(cpu, 1), torch.triu(vk, 1))
        check(torch.tril(cpu, -1), torch.tril(vk, -1))


# ═══════════════════════════════════════════════════════════════════
# Normalization layers
# ═══════════════════════════════════════════════════════════════════

class TestNormalizationCorrectness:
    """Verify normalization layers match CPU."""

    def test_layer_norm(self):
        torch.manual_seed(42)
        cpu, vk = randn_pair(2, 8, 32)
        ln_cpu = nn.LayerNorm(32)
        ln_vk = nn.LayerNorm(32)
        ln_vk.load_state_dict(ln_cpu.state_dict())
        ln_vk = ln_vk.to("vulkan")
        check(ln_cpu(cpu), ln_vk(vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_batch_norm(self):
        torch.manual_seed(42)
        cpu, vk = randn_pair(4, 8, 4, 4)
        bn_cpu = nn.BatchNorm2d(8).eval()
        bn_vk = nn.BatchNorm2d(8).eval()
        bn_vk.load_state_dict(bn_cpu.state_dict())
        bn_vk = bn_vk.to("vulkan")
        check(bn_cpu(cpu), bn_vk(vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_group_norm(self):
        torch.manual_seed(42)
        cpu, vk = randn_pair(2, 8, 4, 4)
        gn_cpu = nn.GroupNorm(4, 8)
        gn_vk = nn.GroupNorm(4, 8)
        gn_vk.load_state_dict(gn_cpu.state_dict())
        gn_vk = gn_vk.to("vulkan")
        check(gn_cpu(cpu), gn_vk(vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_softmax(self):
        cpu, vk = randn_pair(4, 16)
        check(F.softmax(cpu, dim=-1), F.softmax(vk, dim=-1), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_log_softmax(self):
        cpu, vk = randn_pair(4, 16)
        check(F.log_softmax(cpu, dim=-1), F.log_softmax(vk, dim=-1), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# Pooling
# ═══════════════════════════════════════════════════════════════════

class TestPoolingCorrectness:
    """Verify pooling layers match CPU."""

    def test_max_pool2d(self):
        cpu, vk = randn_pair(1, 4, 8, 8)
        check(F.max_pool2d(cpu, 2), F.max_pool2d(vk, 2))

    def test_max_pool2d_stride_padding(self):
        cpu, vk = randn_pair(2, 8, 16, 16)
        check(F.max_pool2d(cpu, 3, stride=2, padding=1),
              F.max_pool2d(vk, 3, stride=2, padding=1))

    def test_avg_pool2d(self):
        cpu, vk = randn_pair(1, 4, 8, 8)
        check(F.avg_pool2d(cpu, 2), F.avg_pool2d(vk, 2), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_avg_pool2d_count_include_pad(self):
        cpu, vk = randn_pair(1, 4, 8, 8)
        check(F.avg_pool2d(cpu, 3, padding=1, count_include_pad=False),
              F.avg_pool2d(vk, 3, padding=1, count_include_pad=False),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_adaptive_avg_pool2d(self):
        cpu, vk = randn_pair(2, 8, 16, 16)
        check(F.adaptive_avg_pool2d(cpu, (4, 4)),
              F.adaptive_avg_pool2d(vk, (4, 4)), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_adaptive_avg_pool2d_1x1(self):
        cpu, vk = randn_pair(2, 8, 16, 16)
        check(F.adaptive_avg_pool2d(cpu, 1),
              F.adaptive_avg_pool2d(vk, 1), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# Convolution
# ═══════════════════════════════════════════════════════════════════

class TestConvCorrectness:
    """Verify conv2d matches CPU."""

    def test_conv2d_basic(self):
        torch.manual_seed(42)
        m_cpu = nn.Conv2d(3, 16, 3, padding=1)
        m_vk = nn.Conv2d(3, 16, 3, padding=1)
        m_vk.load_state_dict(m_cpu.state_dict())
        m_vk = m_vk.to("vulkan")
        x_cpu, x_vk = randn_pair(1, 3, 8, 8)
        check(m_cpu(x_cpu), m_vk(x_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_conv2d_no_bias(self):
        torch.manual_seed(42)
        m_cpu = nn.Conv2d(4, 8, 3, bias=False, padding=1)
        m_vk = nn.Conv2d(4, 8, 3, bias=False, padding=1)
        m_vk.load_state_dict(m_cpu.state_dict())
        m_vk = m_vk.to("vulkan")
        x_cpu, x_vk = randn_pair(2, 4, 8, 8)
        check(m_cpu(x_cpu), m_vk(x_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_conv2d_stride(self):
        torch.manual_seed(42)
        m_cpu = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        m_vk = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        m_vk.load_state_dict(m_cpu.state_dict())
        m_vk = m_vk.to("vulkan")
        x_cpu, x_vk = randn_pair(1, 3, 16, 16)
        check(m_cpu(x_cpu), m_vk(x_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# Attention
# ═══════════════════════════════════════════════════════════════════

class TestAttentionCorrectness:
    """Verify SDPA matches CPU."""

    def test_sdpa_basic(self):
        torch.manual_seed(42)
        B, H, N, D = 2, 4, 8, 16
        q_cpu, q_vk = randn_pair(B, H, N, D)
        k_cpu, k_vk = randn_pair(B, H, N, D)
        v_cpu, v_vk = randn_pair(B, H, N, D)
        out_cpu = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
        out_vk = F.scaled_dot_product_attention(q_vk, k_vk, v_vk)
        check(out_cpu, out_vk, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_sdpa_different_shapes(self):
        torch.manual_seed(42)
        B, H, N, D = 1, 2, 16, 32
        q_cpu, q_vk = randn_pair(B, H, N, D)
        k_cpu, k_vk = randn_pair(B, H, N, D)
        v_cpu, v_vk = randn_pair(B, H, N, D)
        out_cpu = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
        out_vk = F.scaled_dot_product_attention(q_vk, k_vk, v_vk)
        check(out_cpu, out_vk, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════

class TestLossCorrectness:
    """Verify loss functions match CPU."""

    def test_mse_loss(self):
        a_cpu, a_vk = randn_pair(4, 8)
        b_cpu, b_vk = randn_pair(4, 8)
        check(F.mse_loss(a_cpu, b_cpu), F.mse_loss(a_vk, b_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_cross_entropy(self):
        logits_cpu, logits_vk = randn_pair(4, 10)
        target = torch.randint(0, 10, (4,))
        target_vk = target.to("vulkan")
        check(F.cross_entropy(logits_cpu, target),
              F.cross_entropy(logits_vk, target_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_nll_loss(self):
        log_probs_cpu = F.log_softmax(torch.randn(4, 10), dim=-1)
        log_probs_vk = log_probs_cpu.to("vulkan")
        target = torch.randint(0, 10, (4,))
        target_vk = target.to("vulkan")
        check(F.nll_loss(log_probs_cpu, target),
              F.nll_loss(log_probs_vk, target_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# Factories
# ═══════════════════════════════════════════════════════════════════

class TestFactoryCorrectness:
    """Verify factory ops produce correct values."""

    def test_arange(self):
        check(torch.arange(10, dtype=torch.float32),
              torch.arange(10, dtype=torch.float32, device="vulkan"))

    def test_arange_start_step(self):
        check(torch.arange(2, 10, 0.5, dtype=torch.float32),
              torch.arange(2, 10, 0.5, dtype=torch.float32, device="vulkan"))

    def test_linspace(self):
        check(torch.linspace(0, 1, 50),
              torch.linspace(0, 1, 50, device="vulkan"), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_eye(self):
        check(torch.eye(4), torch.eye(4, device="vulkan"))

    def test_eye_rect(self):
        check(torch.eye(3, 5), torch.eye(3, 5, device="vulkan"))

    def test_full(self):
        check(torch.full((4, 8), 3.14), torch.full((4, 8), 3.14, device="vulkan"))

    def test_zeros_ones(self):
        check(torch.zeros(4, 8), torch.zeros(4, 8, device="vulkan"))
        check(torch.ones(4, 8), torch.ones(4, 8, device="vulkan"))


# ═══════════════════════════════════════════════════════════════════
# Advanced ops
# ═══════════════════════════════════════════════════════════════════

class TestAdvancedOpsCorrectness:
    """Verify advanced ops match CPU."""

    def test_cumsum(self):
        cpu, vk = randn_pair(4, 8)
        check(torch.cumsum(cpu, dim=0), torch.cumsum(vk, dim=0), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(torch.cumsum(cpu, dim=1), torch.cumsum(vk, dim=1), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_sort(self):
        cpu, vk = randn_pair(4, 8)
        cpu_vals, cpu_idx = torch.sort(cpu, dim=1)
        vk_vals, vk_idx = torch.sort(vk, dim=1)
        check(cpu_vals, vk_vals)
        check(cpu_idx, vk_idx)

    def test_topk(self):
        cpu, vk = randn_pair(4, 16)
        cpu_vals, cpu_idx = torch.topk(cpu, 4, dim=1)
        vk_vals, vk_idx = torch.topk(vk, 4, dim=1)
        check(cpu_vals, vk_vals, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(cpu_idx, vk_idx)

    def test_gather(self):
        cpu, vk = randn_pair(4, 8)
        idx = torch.randint(0, 8, (4, 3))
        idx_vk = idx.to("vulkan")
        check(torch.gather(cpu, 1, idx), torch.gather(vk, 1, idx_vk))

    def test_index_select(self):
        cpu, vk = randn_pair(8, 16)
        idx = torch.tensor([0, 3, 5, 7])
        idx_vk = idx.to("vulkan")
        check(torch.index_select(cpu, 0, idx), torch.index_select(vk, 0, idx_vk))

    def test_masked_fill(self):
        cpu, vk = randn_pair(4, 8)
        mask = torch.randn(4, 8) > 0
        mask_vk = mask.to("vulkan")
        check(cpu.masked_fill(mask, -1.0), vk.masked_fill(mask_vk, -1.0))

    def test_embedding(self):
        torch.manual_seed(42)
        emb_cpu = nn.Embedding(100, 32)
        emb_vk = nn.Embedding(100, 32)
        emb_vk.load_state_dict(emb_cpu.state_dict())
        emb_vk = emb_vk.to("vulkan")
        idx = torch.randint(0, 100, (4, 8))
        idx_vk = idx.to("vulkan")
        check(emb_cpu(idx), emb_vk(idx_vk))

    def test_constant_pad(self):
        cpu, vk = randn_pair(2, 4, 8, 8)
        check(F.pad(cpu, (1, 1, 2, 2)), F.pad(vk, (1, 1, 2, 2)))

    def test_index_tensor(self):
        cpu, vk = randn_pair(8, 16)
        idx = torch.tensor([0, 2, 5])
        idx_vk = idx.to("vulkan")
        check(cpu[idx], vk[idx_vk])

    def test_upsample_nearest(self):
        cpu, vk = randn_pair(1, 4, 4, 4)
        check(F.interpolate(cpu, scale_factor=2, mode="nearest"),
              F.interpolate(vk, scale_factor=2, mode="nearest"))

    def test_upsample_bilinear(self):
        cpu, vk = randn_pair(1, 4, 4, 4)
        check(F.interpolate(cpu, scale_factor=2, mode="bilinear", align_corners=False),
              F.interpolate(vk, scale_factor=2, mode="bilinear", align_corners=False),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# Autograd correctness
# ═══════════════════════════════════════════════════════════════════

class TestAutogradCorrectness:
    """Verify backward pass produces correct gradients vs CPU."""

    def test_relu_backward(self):
        cpu = torch.randn(4, 8, requires_grad=True)
        vk = cpu.detach().to("vulkan").requires_grad_(True)
        F.relu(cpu).sum().backward()
        F.relu(vk).sum().backward()
        check(cpu.grad, vk.grad)

    def test_sigmoid_backward(self):
        cpu = torch.randn(4, 8, requires_grad=True)
        vk = cpu.detach().to("vulkan").requires_grad_(True)
        torch.sigmoid(cpu).sum().backward()
        torch.sigmoid(vk).sum().backward()
        check(cpu.grad, vk.grad, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_tanh_backward(self):
        cpu = torch.randn(4, 8, requires_grad=True)
        vk = cpu.detach().to("vulkan").requires_grad_(True)
        torch.tanh(cpu).sum().backward()
        torch.tanh(vk).sum().backward()
        check(cpu.grad, vk.grad, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_linear_backward(self):
        torch.manual_seed(42)
        m_cpu = nn.Linear(16, 8)
        m_vk = nn.Linear(16, 8)
        m_vk.load_state_dict(m_cpu.state_dict())
        m_vk = m_vk.to("vulkan")

        x_cpu = torch.randn(4, 16, requires_grad=True)
        x_vk = x_cpu.detach().to("vulkan").requires_grad_(True)

        F.mse_loss(m_cpu(x_cpu), torch.zeros(4, 8)).backward()
        F.mse_loss(m_vk(x_vk), torch.zeros(4, 8, device="vulkan")).backward()

        check(x_cpu.grad, x_vk.grad, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(m_cpu.weight.grad, m_vk.weight.grad, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(m_cpu.bias.grad, m_vk.bias.grad, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_mm_backward(self):
        a_cpu = torch.randn(4, 8, requires_grad=True)
        b_cpu = torch.randn(8, 4, requires_grad=True)
        a_vk = a_cpu.detach().to("vulkan").requires_grad_(True)
        b_vk = b_cpu.detach().to("vulkan").requires_grad_(True)

        torch.mm(a_cpu, b_cpu).sum().backward()
        torch.mm(a_vk, b_vk).sum().backward()

        check(a_cpu.grad, a_vk.grad, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(b_cpu.grad, b_vk.grad, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_conv2d_backward(self):
        torch.manual_seed(42)
        m_cpu = nn.Conv2d(3, 8, 3, padding=1)
        m_vk = nn.Conv2d(3, 8, 3, padding=1)
        m_vk.load_state_dict(m_cpu.state_dict())
        m_vk = m_vk.to("vulkan")

        x_cpu = torch.randn(1, 3, 8, 8, requires_grad=True)
        x_vk = x_cpu.detach().to("vulkan").requires_grad_(True)

        F.mse_loss(m_cpu(x_cpu), torch.zeros(1, 8, 8, 8)).backward()
        F.mse_loss(m_vk(x_vk), torch.zeros(1, 8, 8, 8, device="vulkan")).backward()

        check(x_cpu.grad, x_vk.grad, rtol=1e-2, atol=1e-2)


# ═══════════════════════════════════════════════════════════════════
# Training convergence
# ═══════════════════════════════════════════════════════════════════

class TestTrainingCorrectness:
    """Verify training produces same parameter updates as CPU."""

    def test_sgd_step_matches_cpu(self):
        """Single SGD step should produce identical parameters."""
        torch.manual_seed(42)
        m_cpu = nn.Linear(8, 4)
        m_vk = nn.Linear(8, 4)
        m_vk.load_state_dict(m_cpu.state_dict())
        m_vk = m_vk.to("vulkan")

        opt_cpu = torch.optim.SGD(m_cpu.parameters(), lr=0.01)
        opt_vk = torch.optim.SGD(m_vk.parameters(), lr=0.01)

        x = torch.randn(2, 8)
        target = torch.randn(2, 4)

        # CPU step
        loss_cpu = F.mse_loss(m_cpu(x), target)
        opt_cpu.zero_grad()
        loss_cpu.backward()
        opt_cpu.step()

        # Vulkan step
        x_vk = x.to("vulkan")
        target_vk = target.to("vulkan")
        loss_vk = F.mse_loss(m_vk(x_vk), target_vk)
        opt_vk.zero_grad()
        loss_vk.backward()
        opt_vk.step()

        # Parameters should match after one step
        for (n_cpu, p_cpu), (n_vk, p_vk) in zip(
            m_cpu.named_parameters(), m_vk.named_parameters()
        ):
            check(p_cpu.data, p_vk.data, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_adam_convergence_matches_cpu(self):
        """Adam training should converge similarly on CPU and Vulkan."""
        torch.manual_seed(42)
        m_cpu = nn.Linear(4, 2)
        m_vk = nn.Linear(4, 2)
        m_vk.load_state_dict(m_cpu.state_dict())
        m_vk = m_vk.to("vulkan")

        opt_cpu = torch.optim.Adam(m_cpu.parameters(), lr=0.01)
        opt_vk = torch.optim.Adam(m_vk.parameters(), lr=0.01)

        x = torch.randn(4, 4)
        target = torch.randn(4, 2)
        x_vk = x.to("vulkan")
        target_vk = target.to("vulkan")

        for _ in range(20):
            # CPU
            loss_cpu = F.mse_loss(m_cpu(x), target)
            opt_cpu.zero_grad()
            loss_cpu.backward()
            opt_cpu.step()
            # Vulkan
            loss_vk = F.mse_loss(m_vk(x_vk), target_vk)
            opt_vk.zero_grad()
            loss_vk.backward()
            opt_vk.step()

        # Final loss should be close
        final_cpu = F.mse_loss(m_cpu(x), target).item()
        final_vk = F.mse_loss(m_vk(x_vk), target_vk).item()
        assert abs(final_cpu - final_vk) < 0.1, f"Loss diverged: CPU={final_cpu:.4f}, VK={final_vk:.4f}"

    def test_layer_norm_backward(self):
        """LayerNorm backward gradients should match CPU."""
        torch.manual_seed(42)
        ln_cpu = nn.LayerNorm(16)
        ln_vk = nn.LayerNorm(16)
        ln_vk.load_state_dict(ln_cpu.state_dict())
        ln_vk = ln_vk.to("vulkan")

        x_cpu = torch.randn(2, 4, 16, requires_grad=True)
        x_vk = x_cpu.detach().to("vulkan").requires_grad_(True)

        ln_cpu(x_cpu).sum().backward()
        ln_vk(x_vk).sum().backward()

        check(x_cpu.grad, x_vk.grad, rtol=1e-2, atol=1e-2)
        check(ln_cpu.weight.grad, ln_vk.weight.grad, rtol=1e-2, atol=1e-2)


# ═══════════════════════════════════════════════════════════════════
# Numerical edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases: zeros, large values, inf, nan handling."""

    def test_zero_tensor_ops(self):
        z_cpu = torch.zeros(4, 8)
        z_vk = z_cpu.to("vulkan")
        check(z_cpu + 1.0, z_vk + 1.0)
        check(z_cpu * 5.0, z_vk * 5.0)
        check(torch.relu(z_cpu - 1.0), torch.relu(z_vk - 1.0))

    def test_large_values(self):
        cpu = torch.tensor([1e6, -1e6, 1e-6, -1e-6])
        vk = cpu.to("vulkan")
        check(cpu + cpu, vk + vk)
        check(cpu * cpu, vk * vk, rtol=RTOL_RELAXED, atol=1.0)

    def test_single_element(self):
        cpu = torch.tensor([3.14])
        vk = cpu.to("vulkan")
        check(cpu * 2, vk * 2)
        check(cpu.sum(), vk.sum())

    def test_empty_like(self):
        """empty_like preserves shape and device."""
        vk = torch.randn(4, 8, device="vulkan")
        e = torch.empty_like(vk)
        assert e.shape == (4, 8)
        assert e.device.type in ("privateuseone", "vulkan")

    def test_clone_preserves_data(self):
        cpu, vk = randn_pair(4, 8)
        check(cpu.clone(), vk.clone())

    def test_contiguous(self):
        cpu, vk = randn_pair(4, 8)
        t_cpu = cpu.t().contiguous()
        t_vk = vk.t().contiguous()
        check(t_cpu, t_vk)

    def test_fill_inplace(self):
        cpu = torch.empty(4, 8)
        vk = torch.empty(4, 8, device="vulkan")
        cpu.fill_(2.5)
        vk.fill_(2.5)
        check(cpu, vk)


# ── GPU Shader Ops (verify new GPU implementations) ─────────────
class TestGPUShaderOps:
    """Tests for ops recently converted from CPU fallback to GPU shaders."""

    # ── gather ──────────────────────────────────────────────────
    def test_gather_dim0(self):
        cpu = torch.randn(5, 4)
        idx = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1], [0, 0, 0, 0]])
        vk = cpu.to("vulkan")
        idx_vk = idx.to("vulkan")
        check(cpu.gather(0, idx), vk.gather(0, idx_vk))

    def test_gather_dim1(self):
        cpu = torch.randn(3, 8)
        idx = torch.tensor([[0, 7, 3], [5, 2, 1], [4, 6, 0]])
        vk = cpu.to("vulkan")
        idx_vk = idx.to("vulkan")
        check(cpu.gather(1, idx), vk.gather(1, idx_vk))

    def test_gather_3d(self):
        cpu = torch.randn(2, 4, 3)
        idx = torch.randint(0, 4, (2, 2, 3))
        vk = cpu.to("vulkan")
        idx_vk = idx.to("vulkan")
        check(cpu.gather(1, idx), vk.gather(1, idx_vk))

    # ── scatter_ ────────────────────────────────────────────────
    def test_scatter_dim0(self):
        cpu = torch.zeros(5, 4)
        src = torch.randn(3, 4)
        # No duplicate destinations to avoid GPU race condition (undefined behavior for duplicates)
        idx = torch.tensor([[0, 1, 2, 3], [4, 3, 1, 0], [2, 0, 4, 1]])
        vk = cpu.clone().to("vulkan")
        cpu.scatter_(0, idx, src)
        vk.scatter_(0, idx.to("vulkan"), src.to("vulkan"))
        check(cpu, vk)

    def test_scatter_dim1(self):
        cpu = torch.zeros(3, 8)
        src = torch.randn(3, 3)
        idx = torch.tensor([[0, 3, 7], [1, 5, 6], [2, 4, 0]])
        vk = cpu.clone().to("vulkan")
        cpu.scatter_(1, idx, src)
        vk.scatter_(1, idx.to("vulkan"), src.to("vulkan"))
        check(cpu, vk)

    # ── argmax / argmin ─────────────────────────────────────────
    def test_argmax_global(self):
        cpu = torch.randn(4, 8)
        vk = cpu.to("vulkan")
        assert cpu.argmax().item() == vk.argmax().cpu().item()

    def test_argmax_dim(self):
        cpu = torch.randn(3, 5)
        vk = cpu.to("vulkan")
        cpu_res = cpu.argmax(dim=1)
        vk_res = vk.argmax(dim=1).cpu()
        assert torch.equal(cpu_res, vk_res)

    def test_argmax_dim0_keepdim(self):
        cpu = torch.randn(4, 6)
        vk = cpu.to("vulkan")
        cpu_res = cpu.argmax(dim=0, keepdim=True)
        vk_res = vk.argmax(dim=0, keepdim=True).cpu()
        assert torch.equal(cpu_res, vk_res)

    def test_argmin_global(self):
        cpu = torch.randn(4, 8)
        vk = cpu.to("vulkan")
        assert cpu.argmin().item() == vk.argmin().cpu().item()

    def test_argmin_dim(self):
        cpu = torch.randn(3, 5)
        vk = cpu.to("vulkan")
        cpu_res = cpu.argmin(dim=1)
        vk_res = vk.argmin(dim=1).cpu()
        assert torch.equal(cpu_res, vk_res)

    # ── upsample_nearest2d ──────────────────────────────────────
    def test_upsample_nearest2d_2x(self):
        cpu = torch.randn(1, 3, 4, 4)
        vk = cpu.to("vulkan")
        check(F.interpolate(cpu, scale_factor=2, mode="nearest"),
              F.interpolate(vk, scale_factor=2, mode="nearest"))

    def test_upsample_nearest2d_asymmetric(self):
        cpu = torch.randn(2, 1, 3, 5)
        vk = cpu.to("vulkan")
        check(F.interpolate(cpu, size=(6, 15), mode="nearest"),
              F.interpolate(vk, size=(6, 15), mode="nearest"))

    # ── logical_not / bitwise_not ───────────────────────────────
    def test_logical_not(self):
        cpu = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        vk = cpu.to("vulkan")
        cpu_res = cpu.logical_not()
        vk_res = vk.logical_not()
        check(cpu_res.float(), vk_res)

    def test_bitwise_not_bool(self):
        cpu = torch.tensor([True, False, True, False])
        vk = cpu.float().to("vulkan")
        # Bitwise not on bool: True->False, False->True
        cpu_res = (~cpu).float()
        vk_res = vk.bitwise_not()
        check(cpu_res, vk_res)

    # ── mse_loss ────────────────────────────────────────────────
    def test_mse_loss_mean(self):
        cpu_input = torch.randn(4, 8)
        cpu_target = torch.randn(4, 8)
        vk_input = cpu_input.to("vulkan")
        vk_target = cpu_target.to("vulkan")
        check(F.mse_loss(cpu_input, cpu_target, reduction="mean"),
              F.mse_loss(vk_input, vk_target, reduction="mean"),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_mse_loss_sum(self):
        cpu_input = torch.randn(4, 8)
        cpu_target = torch.randn(4, 8)
        vk_input = cpu_input.to("vulkan")
        vk_target = cpu_target.to("vulkan")
        check(F.mse_loss(cpu_input, cpu_target, reduction="sum"),
              F.mse_loss(vk_input, vk_target, reduction="sum"),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_mse_loss_none(self):
        cpu_input = torch.randn(4, 8)
        cpu_target = torch.randn(4, 8)
        vk_input = cpu_input.to("vulkan")
        vk_target = cpu_target.to("vulkan")
        check(F.mse_loss(cpu_input, cpu_target, reduction="none"),
              F.mse_loss(vk_input, vk_target, reduction="none"))

    def test_mse_loss_backward(self):
        cpu_input = torch.randn(4, 8, requires_grad=True)
        cpu_target = torch.randn(4, 8)
        vk_input = cpu_input.detach().to("vulkan").requires_grad_(True)
        vk_target = cpu_target.to("vulkan")

        loss_cpu = F.mse_loss(cpu_input, cpu_target)
        loss_vk = F.mse_loss(vk_input, vk_target)
        loss_cpu.backward()
        loss_vk.backward()

        check(cpu_input.grad, vk_input.grad.cpu(),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    # ── fmod / remainder (added in previous session) ────────────
    def test_fmod(self):
        cpu_a = torch.randn(4, 8) * 5
        cpu_b = torch.rand(4, 8) * 2 + 0.5
        vk_a = cpu_a.to("vulkan")
        vk_b = cpu_b.to("vulkan")
        check(torch.fmod(cpu_a, cpu_b), torch.fmod(vk_a, vk_b))

    def test_remainder(self):
        cpu_a = torch.randn(4, 8) * 5
        cpu_b = torch.rand(4, 8) * 2 + 0.5
        vk_a = cpu_a.to("vulkan")
        vk_b = cpu_b.to("vulkan")
        check(torch.remainder(cpu_a, cpu_b), torch.remainder(vk_a, vk_b))

    # ── cumprod (added in previous session) ─────────────────────
    def test_cumprod(self):
        cpu = torch.rand(3, 5) + 0.5  # positive values for stability
        vk = cpu.to("vulkan")
        check(cpu.cumprod(dim=1), vk.cumprod(dim=1),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    # ── sort ────────────────────────────────────────────────────
    def test_sort_ascending(self):
        cpu = torch.randn(4, 8)
        vk = cpu.to("vulkan")
        cpu_vals, cpu_idx = cpu.sort(dim=1)
        vk_vals, vk_idx = vk.sort(dim=1)
        check(cpu_vals, vk_vals)
        assert torch.equal(cpu_idx, vk_idx.cpu())

    def test_sort_descending(self):
        cpu = torch.randn(3, 6)
        vk = cpu.to("vulkan")
        cpu_vals, cpu_idx = cpu.sort(dim=1, descending=True)
        vk_vals, vk_idx = vk.sort(dim=1, descending=True)
        check(cpu_vals, vk_vals)
        assert torch.equal(cpu_idx, vk_idx.cpu())

    def test_sort_dim0(self):
        cpu = torch.randn(5, 3)
        vk = cpu.to("vulkan")
        cpu_vals, cpu_idx = cpu.sort(dim=0)
        vk_vals, vk_idx = vk.sort(dim=0)
        check(cpu_vals, vk_vals)
        assert torch.equal(cpu_idx, vk_idx.cpu())

    # ── topk ────────────────────────────────────────────────────
    def test_topk_largest(self):
        cpu = torch.randn(3, 10)
        vk = cpu.to("vulkan")
        cpu_vals, cpu_idx = cpu.topk(3, dim=1)
        vk_vals, vk_idx = vk.topk(3, dim=1)
        check(cpu_vals, vk_vals)
        assert torch.equal(cpu_idx, vk_idx.cpu())

    def test_topk_smallest(self):
        cpu = torch.randn(3, 10)
        vk = cpu.to("vulkan")
        cpu_vals, cpu_idx = cpu.topk(3, dim=1, largest=False)
        vk_vals, vk_idx = vk.topk(3, dim=1, largest=False)
        check(cpu_vals, vk_vals)
        assert torch.equal(cpu_idx, vk_idx.cpu())

    def test_topk_k1(self):
        cpu = torch.randn(4, 8)
        vk = cpu.to("vulkan")
        cpu_vals, cpu_idx = cpu.topk(1, dim=1)
        vk_vals, vk_idx = vk.topk(1, dim=1)
        check(cpu_vals, vk_vals)
        assert torch.equal(cpu_idx, vk_idx.cpu())

    # ── any / all ───────────────────────────────────────────────
    def test_any_true(self):
        cpu = torch.tensor([0.0, 0.0, 1.0, 0.0])
        vk = cpu.to("vulkan")
        cpu_res = cpu.any()
        vk_res = vk.any()
        assert cpu_res.item() == (vk_res.cpu().item() != 0.0)

    def test_any_false(self):
        cpu = torch.zeros(4, 8)
        vk = cpu.to("vulkan")
        assert vk.any().cpu().item() == 0.0

    def test_any_dim(self):
        cpu = torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
        vk = cpu.to("vulkan")
        cpu_res = cpu.any(dim=1).float()
        vk_res = vk.any(dim=1)
        check(cpu_res, vk_res)

    def test_all_true(self):
        cpu = torch.ones(4, 8)
        vk = cpu.to("vulkan")
        assert vk.all().cpu().item() != 0.0

    def test_all_false(self):
        cpu = torch.tensor([1.0, 1.0, 0.0, 1.0])
        vk = cpu.to("vulkan")
        assert vk.all().cpu().item() == 0.0

    def test_all_dim(self):
        cpu = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        vk = cpu.to("vulkan")
        cpu_res = cpu.all(dim=1).float()
        vk_res = vk.all(dim=1)
        check(cpu_res, vk_res)

    # ── norm ────────────────────────────────────────────────────
    def test_norm_l2(self):
        cpu = torch.randn(4, 8)
        vk = cpu.to("vulkan")
        check(torch.linalg.vector_norm(cpu, ord=2),
              torch.linalg.vector_norm(vk, ord=2),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_norm_l1(self):
        cpu = torch.randn(4, 8)
        vk = cpu.to("vulkan")
        check(torch.linalg.vector_norm(cpu, ord=1),
              torch.linalg.vector_norm(vk, ord=1),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_norm_l2_dim(self):
        cpu = torch.randn(3, 5)
        vk = cpu.to("vulkan")
        check(torch.linalg.vector_norm(cpu, ord=2, dim=1),
              torch.linalg.vector_norm(vk, ord=2, dim=1),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    # ── adaptive_avg_pool2d ─────────────────────────────────────
    def test_adaptive_avg_pool2d_divisible(self):
        cpu = torch.randn(1, 3, 8, 8)
        vk = cpu.to("vulkan")
        check(F.adaptive_avg_pool2d(cpu, (4, 4)),
              F.adaptive_avg_pool2d(vk, (4, 4)),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_adaptive_avg_pool2d_non_divisible(self):
        cpu = torch.randn(1, 3, 7, 7)
        vk = cpu.to("vulkan")
        check(F.adaptive_avg_pool2d(cpu, (3, 3)),
              F.adaptive_avg_pool2d(vk, (3, 3)),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_adaptive_avg_pool2d_1x1(self):
        cpu = torch.randn(2, 4, 5, 5)
        vk = cpu.to("vulkan")
        check(F.adaptive_avg_pool2d(cpu, (1, 1)),
              F.adaptive_avg_pool2d(vk, (1, 1)),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    # ── upsample_bilinear2d ─────────────────────────────────────
    def test_upsample_bilinear2d(self):
        cpu = torch.randn(1, 3, 4, 4)
        vk = cpu.to("vulkan")
        check(F.interpolate(cpu, scale_factor=2, mode="bilinear", align_corners=False),
              F.interpolate(vk, scale_factor=2, mode="bilinear", align_corners=False),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_upsample_bilinear2d_align_corners(self):
        cpu = torch.randn(1, 2, 3, 3)
        vk = cpu.to("vulkan")
        check(F.interpolate(cpu, size=(6, 6), mode="bilinear", align_corners=True),
              F.interpolate(vk, size=(6, 6), mode="bilinear", align_corners=True),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    # ── index_select all dims ───────────────────────────────────
    def test_index_select_dim1(self):
        cpu = torch.randn(3, 8)
        idx = torch.tensor([1, 3, 5])
        vk = cpu.to("vulkan")
        check(cpu.index_select(1, idx),
              vk.index_select(1, idx.to("vulkan")))

    def test_index_select_dim2(self):
        cpu = torch.randn(2, 3, 8)
        idx = torch.tensor([0, 2, 7])
        vk = cpu.to("vulkan")
        check(cpu.index_select(2, idx),
              vk.index_select(2, idx.to("vulkan")))

    # ── conv2d with dilation ────────────────────────────────────
    def test_conv2d_dilation(self):
        cpu_input = torch.randn(1, 3, 8, 8)
        cpu_weight = torch.randn(4, 3, 3, 3)
        vk_input = cpu_input.to("vulkan")
        vk_weight = cpu_weight.to("vulkan")
        cpu_out = F.conv2d(cpu_input, cpu_weight, dilation=2, padding=2)
        vk_out = F.conv2d(vk_input, vk_weight, dilation=2, padding=2)
        check(cpu_out, vk_out, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    # ── conv_transpose2d ────────────────────────────────────────
    def test_conv_transpose2d(self):
        # conv_transpose2d with zero bias (None bias needs monkey-patching)
        cpu_input = torch.randn(1, 4, 4, 4)
        cpu_weight = torch.randn(4, 2, 3, 3)
        cpu_bias = torch.zeros(2)
        vk_input = cpu_input.to("vulkan")
        vk_weight = cpu_weight.to("vulkan")
        vk_bias = cpu_bias.to("vulkan")
        cpu_out = F.conv_transpose2d(cpu_input, cpu_weight, bias=cpu_bias)
        vk_out = F.conv_transpose2d(vk_input, vk_weight, bias=vk_bias)
        check(cpu_out, vk_out, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_conv_transpose2d_with_bias(self):
        cpu_input = torch.randn(1, 4, 3, 3)
        cpu_weight = torch.randn(4, 2, 3, 3)
        cpu_bias = torch.randn(2)
        vk_input = cpu_input.to("vulkan")
        vk_weight = cpu_weight.to("vulkan")
        vk_bias = cpu_bias.to("vulkan")
        cpu_out = F.conv_transpose2d(cpu_input, cpu_weight, bias=cpu_bias)
        vk_out = F.conv_transpose2d(vk_input, vk_weight, bias=vk_bias)
        check(cpu_out, vk_out, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# New loss functions
# ═══════════════════════════════════════════════════════════════════

class TestNewLossCorrectness:
    """Verify BCE, BCE with logits, smooth L1, and Huber loss."""

    def test_bce_loss_mean(self):
        cpu_input = torch.sigmoid(torch.randn(4, 8))
        cpu_target = torch.rand(4, 8)
        vk_input = cpu_input.to("vulkan")
        vk_target = cpu_target.to("vulkan")
        check(F.binary_cross_entropy(cpu_input, cpu_target),
              F.binary_cross_entropy(vk_input, vk_target),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_bce_loss_sum(self):
        cpu_input = torch.sigmoid(torch.randn(4, 8))
        cpu_target = torch.rand(4, 8)
        vk_input = cpu_input.to("vulkan")
        vk_target = cpu_target.to("vulkan")
        check(F.binary_cross_entropy(cpu_input, cpu_target, reduction='sum'),
              F.binary_cross_entropy(vk_input, vk_target, reduction='sum'),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_bce_loss_none(self):
        cpu_input = torch.sigmoid(torch.randn(4, 8))
        cpu_target = torch.rand(4, 8)
        vk_input = cpu_input.to("vulkan")
        vk_target = cpu_target.to("vulkan")
        check(F.binary_cross_entropy(cpu_input, cpu_target, reduction='none'),
              F.binary_cross_entropy(vk_input, vk_target, reduction='none'),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_bce_loss_backward(self):
        cpu_input = torch.sigmoid(torch.randn(4, 8)).detach().requires_grad_(True)
        cpu_target = torch.rand(4, 8)
        loss_cpu = F.binary_cross_entropy(cpu_input, cpu_target)
        loss_cpu.backward()
        grad_cpu = cpu_input.grad.clone()

        vk_input = cpu_input.detach().clone().to("vulkan").requires_grad_(True)
        vk_target = cpu_target.to("vulkan")
        loss_vk = F.binary_cross_entropy(vk_input, vk_target)
        loss_vk.backward()
        check(grad_cpu, vk_input.grad, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_bce_with_logits_mean(self):
        cpu_input = torch.randn(4, 8)
        cpu_target = torch.rand(4, 8)
        vk_input = cpu_input.to("vulkan")
        vk_target = cpu_target.to("vulkan")
        check(F.binary_cross_entropy_with_logits(cpu_input, cpu_target),
              F.binary_cross_entropy_with_logits(vk_input, vk_target),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_bce_with_logits_sum(self):
        cpu_input = torch.randn(4, 8)
        cpu_target = torch.rand(4, 8)
        vk_input = cpu_input.to("vulkan")
        vk_target = cpu_target.to("vulkan")
        check(F.binary_cross_entropy_with_logits(cpu_input, cpu_target, reduction='sum'),
              F.binary_cross_entropy_with_logits(vk_input, vk_target, reduction='sum'),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_bce_with_logits_none(self):
        cpu_input = torch.randn(4, 8)
        cpu_target = torch.rand(4, 8)
        vk_input = cpu_input.to("vulkan")
        vk_target = cpu_target.to("vulkan")
        check(F.binary_cross_entropy_with_logits(cpu_input, cpu_target, reduction='none'),
              F.binary_cross_entropy_with_logits(vk_input, vk_target, reduction='none'),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_smooth_l1_loss_mean(self):
        a_cpu, a_vk = randn_pair(4, 8)
        b_cpu, b_vk = randn_pair(4, 8)
        check(F.smooth_l1_loss(a_cpu, b_cpu),
              F.smooth_l1_loss(a_vk, b_vk),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_smooth_l1_loss_sum(self):
        a_cpu, a_vk = randn_pair(4, 8)
        b_cpu, b_vk = randn_pair(4, 8)
        check(F.smooth_l1_loss(a_cpu, b_cpu, reduction='sum'),
              F.smooth_l1_loss(a_vk, b_vk, reduction='sum'),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_smooth_l1_loss_none(self):
        a_cpu, a_vk = randn_pair(4, 8)
        b_cpu, b_vk = randn_pair(4, 8)
        check(F.smooth_l1_loss(a_cpu, b_cpu, reduction='none'),
              F.smooth_l1_loss(a_vk, b_vk, reduction='none'),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_smooth_l1_loss_custom_beta(self):
        a_cpu, a_vk = randn_pair(4, 8)
        b_cpu, b_vk = randn_pair(4, 8)
        check(F.smooth_l1_loss(a_cpu, b_cpu, beta=0.5),
              F.smooth_l1_loss(a_vk, b_vk, beta=0.5),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_huber_loss_mean(self):
        a_cpu, a_vk = randn_pair(4, 8)
        b_cpu, b_vk = randn_pair(4, 8)
        check(F.huber_loss(a_cpu, b_cpu),
              F.huber_loss(a_vk, b_vk),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_huber_loss_sum(self):
        a_cpu, a_vk = randn_pair(4, 8)
        b_cpu, b_vk = randn_pair(4, 8)
        check(F.huber_loss(a_cpu, b_cpu, reduction='sum'),
              F.huber_loss(a_vk, b_vk, reduction='sum'),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_huber_loss_none(self):
        a_cpu, a_vk = randn_pair(4, 8)
        b_cpu, b_vk = randn_pair(4, 8)
        check(F.huber_loss(a_cpu, b_cpu, reduction='none'),
              F.huber_loss(a_vk, b_vk, reduction='none'),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# New activations
# ═══════════════════════════════════════════════════════════════════

class TestNewActivationCorrectness:
    """Verify hardtanh, hardswish, hardsigmoid, softplus."""

    def test_hardtanh(self):
        a_cpu, a_vk = randn_pair(16, 16)
        check(F.hardtanh(a_cpu), F.hardtanh(a_vk))

    def test_hardtanh_custom_bounds(self):
        a_cpu, a_vk = randn_pair(16, 16)
        check(F.hardtanh(a_cpu, min_val=-2.0, max_val=2.0),
              F.hardtanh(a_vk, min_val=-2.0, max_val=2.0))

    def test_hardtanh_inplace(self):
        cpu = torch.randn(16, 16)
        vk = cpu.clone().to("vulkan")
        F.hardtanh_(cpu)
        F.hardtanh_(vk)
        check(cpu, vk)

    def test_hardtanh_backward(self):
        cpu_x = torch.randn(8, 8, requires_grad=True)
        out_cpu = F.hardtanh(cpu_x)
        out_cpu.sum().backward()
        grad_cpu = cpu_x.grad.clone()

        vk_x = cpu_x.detach().clone().to("vulkan").requires_grad_(True)
        out_vk = F.hardtanh(vk_x)
        out_vk.sum().backward()
        check(grad_cpu, vk_x.grad)

    def test_hardswish(self):
        a_cpu, a_vk = randn_pair(16, 16)
        check(F.hardswish(a_cpu), F.hardswish(a_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_hardswish_inplace(self):
        cpu = torch.randn(16, 16)
        vk = cpu.clone().to("vulkan")
        F.hardswish(cpu, inplace=True)
        F.hardswish(vk, inplace=True)
        check(cpu, vk, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_hardswish_backward(self):
        cpu_x = torch.randn(8, 8, requires_grad=True)
        out_cpu = F.hardswish(cpu_x)
        out_cpu.sum().backward()
        grad_cpu = cpu_x.grad.clone()

        vk_x = cpu_x.detach().clone().to("vulkan").requires_grad_(True)
        out_vk = F.hardswish(vk_x)
        out_vk.sum().backward()
        check(grad_cpu, vk_x.grad, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_hardsigmoid(self):
        a_cpu, a_vk = randn_pair(16, 16)
        check(F.hardsigmoid(a_cpu), F.hardsigmoid(a_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_hardsigmoid_inplace(self):
        cpu = torch.randn(16, 16)
        vk = cpu.clone().to("vulkan")
        F.hardsigmoid(cpu, inplace=True)
        F.hardsigmoid(vk, inplace=True)
        check(cpu, vk, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_hardsigmoid_backward(self):
        cpu_x = torch.randn(8, 8, requires_grad=True)
        out_cpu = F.hardsigmoid(cpu_x)
        out_cpu.sum().backward()
        grad_cpu = cpu_x.grad.clone()

        vk_x = cpu_x.detach().clone().to("vulkan").requires_grad_(True)
        out_vk = F.hardsigmoid(vk_x)
        out_vk.sum().backward()
        check(grad_cpu, vk_x.grad, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_softplus(self):
        a_cpu, a_vk = randn_pair(16, 16)
        check(F.softplus(a_cpu), F.softplus(a_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_softplus_custom_params(self):
        a_cpu, a_vk = randn_pair(16, 16)
        check(F.softplus(a_cpu, beta=2.0, threshold=10.0),
              F.softplus(a_vk, beta=2.0, threshold=10.0),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_softplus_backward(self):
        cpu_x = torch.randn(8, 8, requires_grad=True)
        out_cpu = F.softplus(cpu_x)
        out_cpu.sum().backward()
        grad_cpu = cpu_x.grad.clone()

        vk_x = cpu_x.detach().clone().to("vulkan").requires_grad_(True)
        out_vk = F.softplus(vk_x)
        out_vk.sum().backward()
        check(grad_cpu, vk_x.grad, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_relu6(self):
        """relu6 = hardtanh(x, 0, 6)"""
        a_cpu = torch.randn(16, 16) * 4  # scale to have values outside [0, 6]
        a_vk = a_cpu.to("vulkan")
        check(F.relu6(a_cpu), F.relu6(a_vk))


# ═══════════════════════════════════════════════════════════════════
# Instance norm, F.normalize, Dropout2d, clamp_min/max, bernoulli
# ═══════════════════════════════════════════════════════════════════

class TestTrainingOpsCorrectness:
    """Verify instance_norm, F.normalize, Dropout2d, and related ops."""

    def test_instance_norm(self):
        a_cpu = torch.randn(2, 4, 8, 8)
        a_vk = a_cpu.to("vulkan")
        check(F.instance_norm(a_cpu), F.instance_norm(a_vk),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_instance_norm_affine(self):
        a_cpu = torch.randn(2, 4, 8, 8)
        a_vk = a_cpu.to("vulkan")
        weight = torch.randn(4)
        bias = torch.randn(4)
        w_vk = weight.to("vulkan")
        b_vk = bias.to("vulkan")
        check(F.instance_norm(a_cpu, weight=weight, bias=bias),
              F.instance_norm(a_vk, weight=w_vk, bias=b_vk),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_normalize_dim1(self):
        a_cpu = torch.randn(4, 8)
        a_vk = a_cpu.to("vulkan")
        check(F.normalize(a_cpu, dim=1), F.normalize(a_vk, dim=1),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_normalize_dim0(self):
        a_cpu = torch.randn(4, 8)
        a_vk = a_cpu.to("vulkan")
        check(F.normalize(a_cpu, dim=0), F.normalize(a_vk, dim=0),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_normalize_4d(self):
        a_cpu = torch.randn(2, 4, 8, 8)
        a_vk = a_cpu.to("vulkan")
        check(F.normalize(a_cpu, dim=1), F.normalize(a_vk, dim=1),
              rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_clamp_min(self):
        a_cpu, a_vk = randn_pair(16, 16)
        check(a_cpu.clamp_min(0.0), a_vk.clamp_min(0.0))

    def test_clamp_max(self):
        a_cpu, a_vk = randn_pair(16, 16)
        check(a_cpu.clamp_max(0.5), a_vk.clamp_max(0.5))

    def test_bernoulli_float(self):
        """Bernoulli_ should produce 0s and 1s with roughly correct proportions."""
        torch.manual_seed(42)
        t = torch.empty(1000, device="vulkan").fill_(0.0)
        t.bernoulli_(0.3)
        mean = t.cpu().mean().item()
        assert 0.15 < mean < 0.45, f"Bernoulli mean {mean} out of expected range"

    def test_dropout2d(self):
        """Dropout2d should zero out entire channels."""
        torch.manual_seed(42)
        x_vk = torch.randn(2, 8, 4, 4, device="vulkan")
        d = nn.Dropout2d(p=0.5)
        d.train()
        out = d(x_vk)
        out_cpu = out.cpu()
        # Check some channels are fully zeroed
        channel_sums = out_cpu.abs().sum(dim=(2, 3))  # [2, 8]
        n_zeroed = (channel_sums == 0).sum().item()
        assert n_zeroed > 0, "Dropout2d should zero out some channels"


# ═══════════════════════════════════════════════════════════════════
# New ops: math, check, loss, activation, conv1d, cosine_similarity
# ═══════════════════════════════════════════════════════════════════

class TestMathOpsCorrectness:
    """Tests for tan, atan, atan2, log2, log10, log1p."""

    def test_tan(self):
        x = torch.clamp(torch.randn(4, 8), -1.0, 1.0)
        r_v = torch.tan(x.vulkan()).cpu()
        assert torch.allclose(r_v, torch.tan(x), atol=5e-5)

    def test_atan(self):
        x = torch.randn(4, 8)
        r_v = torch.atan(x.vulkan()).cpu()
        assert torch.allclose(r_v, torch.atan(x), atol=ATOL)

    def test_atan2(self):
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        r_v = torch.atan2(x.vulkan(), y.vulkan()).cpu()
        assert torch.allclose(r_v, torch.atan2(x, y), atol=ATOL)

    def test_log2(self):
        x = torch.abs(torch.randn(4, 8)) + 0.01
        r_v = torch.log2(x.vulkan()).cpu()
        assert torch.allclose(r_v, torch.log2(x), atol=ATOL)

    def test_log10(self):
        x = torch.abs(torch.randn(4, 8)) + 0.01
        r_v = torch.log10(x.vulkan()).cpu()
        assert torch.allclose(r_v, torch.log10(x), atol=ATOL)

    def test_log1p(self):
        x = torch.abs(torch.randn(4, 8))
        r_v = torch.log1p(x.vulkan()).cpu()
        assert torch.allclose(r_v, torch.log1p(x), atol=ATOL)

    def test_log2_large(self):
        x = torch.tensor([1.0, 2.0, 4.0, 8.0, 1024.0])
        r_v = torch.log2(x.vulkan()).cpu()
        assert torch.allclose(r_v, torch.log2(x), atol=ATOL)

    def test_log10_powers(self):
        x = torch.tensor([1.0, 10.0, 100.0, 1000.0])
        r_v = torch.log10(x.vulkan()).cpu()
        expected = torch.tensor([0.0, 1.0, 2.0, 3.0])
        assert torch.allclose(r_v, expected, atol=ATOL)


class TestCheckOpsCorrectness:
    """Tests for isnan, isinf, isfinite."""

    def test_isnan(self):
        x = torch.tensor([1.0, float('nan'), 3.0, float('nan'), 0.0])
        r_v = torch.isnan(x.vulkan()).cpu()
        assert torch.equal(r_v, torch.isnan(x))

    def test_isnan_no_nans(self):
        x = torch.randn(4, 8)
        r_v = torch.isnan(x.vulkan()).cpu()
        assert torch.equal(r_v, torch.isnan(x))

    def test_isinf(self):
        x = torch.tensor([1.0, float('inf'), -float('inf'), 0.0, float('nan')])
        r_v = torch.isinf(x.vulkan()).cpu()
        assert torch.equal(r_v, torch.isinf(x))

    def test_isinf_no_infs(self):
        x = torch.randn(4, 8)
        r_v = torch.isinf(x.vulkan()).cpu()
        assert torch.equal(r_v, torch.isinf(x))

    def test_isfinite(self):
        """isfinite = not isnan and not isinf — verify components work."""
        x = torch.tensor([1.0, float('inf'), -float('inf'), 0.0, float('nan')])
        xv = x.vulkan()
        is_nan = torch.isnan(xv).cpu()
        is_inf = torch.isinf(xv).cpu()
        # Manually compute isfinite on CPU from Vulkan results
        r_v = ~is_nan & ~is_inf
        assert torch.equal(r_v, torch.isfinite(x))

    def test_isfinite_all_finite(self):
        x = torch.randn(4, 8)
        xv = x.vulkan()
        assert not torch.isnan(xv).cpu().any()
        assert not torch.isinf(xv).cpu().any()


class TestL1LossCorrectness:
    """Tests for L1 loss forward and backward."""

    def test_l1_loss_mean(self):
        pred = torch.randn(4, 8)
        target = torch.randn(4, 8)
        r_v = F.l1_loss(pred.vulkan(), target.vulkan()).cpu()
        r_c = F.l1_loss(pred, target)
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)

    def test_l1_loss_sum(self):
        pred = torch.randn(4, 8)
        target = torch.randn(4, 8)
        r_v = F.l1_loss(pred.vulkan(), target.vulkan(), reduction='sum').cpu()
        r_c = F.l1_loss(pred, target, reduction='sum')
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)

    def test_l1_loss_none(self):
        pred = torch.randn(4, 8)
        target = torch.randn(4, 8)
        r_v = F.l1_loss(pred.vulkan(), target.vulkan(), reduction='none').cpu()
        r_c = F.l1_loss(pred, target, reduction='none')
        assert torch.allclose(r_v, r_c, atol=ATOL)

    def test_l1_loss_backward(self):
        pred = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(4, 8)
        loss_cpu = F.l1_loss(pred, target)
        loss_cpu.backward()
        grad_cpu = pred.grad.clone()

        pred_v = pred.detach().clone().requires_grad_(True)
        loss_v = F.l1_loss(pred_v.vulkan(), target.vulkan())
        loss_v.backward()
        grad_v = pred_v.grad.cpu()
        assert torch.allclose(grad_v, grad_cpu, atol=ATOL_RELAXED)


class TestKLDivCorrectness:
    """Tests for KL divergence loss."""

    def test_kl_div_sum(self):
        log_p = F.log_softmax(torch.randn(4, 8), dim=-1)
        q = F.softmax(torch.randn(4, 8), dim=-1)
        r_v = F.kl_div(log_p.vulkan(), q.vulkan(), reduction='sum').cpu()
        r_c = F.kl_div(log_p, q, reduction='sum')
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)

    def test_kl_div_mean(self):
        log_p = F.log_softmax(torch.randn(4, 8), dim=-1)
        q = F.softmax(torch.randn(4, 8), dim=-1)
        r_v = F.kl_div(log_p.vulkan(), q.vulkan(), reduction='mean').cpu()
        r_c = F.kl_div(log_p, q, reduction='mean')
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)

    def test_kl_div_none(self):
        log_p = F.log_softmax(torch.randn(4, 8), dim=-1)
        q = F.softmax(torch.randn(4, 8), dim=-1)
        r_v = F.kl_div(log_p.vulkan(), q.vulkan(), reduction='none').cpu()
        r_c = F.kl_div(log_p, q, reduction='none')
        assert torch.allclose(r_v, r_c, atol=ATOL)

    def test_kl_div_log_target(self):
        log_p = F.log_softmax(torch.randn(4, 8), dim=-1)
        log_q = F.log_softmax(torch.randn(4, 8), dim=-1)
        r_v = F.kl_div(log_p.vulkan(), log_q.vulkan(), reduction='sum', log_target=True).cpu()
        r_c = F.kl_div(log_p, log_q, reduction='sum', log_target=True)
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)


class TestMishCorrectness:
    """Tests for Mish activation forward and backward."""

    def test_mish_forward(self):
        x = torch.randn(4, 8)
        r_v = F.mish(x.vulkan()).cpu()
        r_c = F.mish(x)
        assert torch.allclose(r_v, r_c, atol=ATOL)

    def test_mish_large_values(self):
        x = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0, 25.0])
        r_v = F.mish(x.vulkan()).cpu()
        r_c = F.mish(x)
        assert torch.allclose(r_v, r_c, atol=ATOL)

    def test_mish_backward(self):
        x = torch.randn(4, 8, requires_grad=True)
        loss_cpu = F.mish(x).sum()
        loss_cpu.backward()
        grad_cpu = x.grad.clone()

        x_v = x.detach().clone().requires_grad_(True)
        loss_v = F.mish(x_v.vulkan()).sum()
        loss_v.backward()
        grad_v = x_v.grad.cpu()
        assert torch.allclose(grad_v, grad_cpu, atol=ATOL_RELAXED)


class TestConv1dCorrectness:
    """Tests for Conv1d via conv2d unsqueeze/squeeze."""

    def test_conv1d_basic(self):
        inp = torch.randn(2, 3, 16)
        w = torch.randn(8, 3, 5)
        r_v = F.conv1d(inp.vulkan(), w.vulkan(), padding=2).cpu()
        r_c = F.conv1d(inp, w, padding=2)
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)

    def test_conv1d_with_bias(self):
        inp = torch.randn(2, 3, 16)
        w = torch.randn(8, 3, 5)
        b = torch.randn(8)
        r_v = F.conv1d(inp.vulkan(), w.vulkan(), b.vulkan(), padding=2).cpu()
        r_c = F.conv1d(inp, w, b, padding=2)
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)

    def test_conv1d_stride(self):
        inp = torch.randn(2, 3, 32)
        w = torch.randn(8, 3, 3)
        r_v = F.conv1d(inp.vulkan(), w.vulkan(), stride=2, padding=1).cpu()
        r_c = F.conv1d(inp, w, stride=2, padding=1)
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)

    def test_conv1d_dilation(self):
        inp = torch.randn(2, 3, 32)
        w = torch.randn(8, 3, 3)
        r_v = F.conv1d(inp.vulkan(), w.vulkan(), dilation=2, padding=2).cpu()
        r_c = F.conv1d(inp, w, dilation=2, padding=2)
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)

    def test_conv1d_groups(self):
        inp = torch.randn(2, 6, 16)
        w = torch.randn(6, 3, 3)  # groups=2: 6 in_ch, 6 out_ch, 3 ch/group
        r_v = F.conv1d(inp.vulkan(), w.vulkan(), groups=2, padding=1).cpu()
        r_c = F.conv1d(inp, w, groups=2, padding=1)
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)

    def test_conv1d_nn_module(self):
        conv = nn.Conv1d(3, 8, kernel_size=5, padding=2)
        inp = torch.randn(2, 3, 16)
        r_c = conv(inp)
        conv_v = conv.vulkan()
        r_v = conv_v(inp.vulkan()).cpu()
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)


class TestCosineSimilarityCorrectness:
    """Tests for cosine_similarity."""

    def test_cosine_similarity_dim1(self):
        a = torch.randn(4, 8)
        b = torch.randn(4, 8)
        r_v = F.cosine_similarity(a.vulkan(), b.vulkan()).cpu()
        r_c = F.cosine_similarity(a, b)
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)

    def test_cosine_similarity_dim0(self):
        a = torch.randn(4, 8)
        b = torch.randn(4, 8)
        r_v = F.cosine_similarity(a.vulkan(), b.vulkan(), dim=0).cpu()
        r_c = F.cosine_similarity(a, b, dim=0)
        assert torch.allclose(r_v, r_c, atol=ATOL_RELAXED)

    def test_cosine_similarity_identical(self):
        a = torch.randn(4, 8)
        r_v = F.cosine_similarity(a.vulkan(), a.vulkan()).cpu()
        expected = torch.ones(4)
        assert torch.allclose(r_v, expected, atol=ATOL_RELAXED)

    def test_cosine_similarity_orthogonal(self):
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        r_v = F.cosine_similarity(a.vulkan(), b.vulkan()).cpu()
        expected = torch.tensor([0.0, 0.0])
        assert torch.allclose(r_v, expected, atol=ATOL_RELAXED)


# ═══════════════════════════════════════════════════════════════════
# Qwen3 building blocks
# ═══════════════════════════════════════════════════════════════════

class TestRMSNormCorrectness:
    """Test RMSNorm forward and backward against CPU reference."""

    @staticmethod
    def rms_norm_cpu(x, weight, eps=1e-6):
        """CPU reference implementation matching HuggingFace Qwen3RMSNorm."""
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x.to(input_dtype)

    def test_rms_norm_basic(self):
        import torch_vulkan
        x = torch.randn(2, 8)
        w = torch.randn(8).abs() + 0.1
        ref = self.rms_norm_cpu(x, w)
        out = torch_vulkan.rms_norm(x.vulkan(), w.vulkan(), 1e-6).cpu()
        check(ref, out, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_rms_norm_3d(self):
        """3D input [batch, seq, hidden] — typical LLM shape."""
        import torch_vulkan
        x = torch.randn(2, 4, 16)
        w = torch.randn(16).abs() + 0.1
        ref = self.rms_norm_cpu(x, w)
        out = torch_vulkan.rms_norm(x.vulkan(), w.vulkan(), 1e-6).cpu()
        check(ref, out, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_rms_norm_qwen3_dims(self):
        """Qwen3-0.6B hidden_size=1024."""
        import torch_vulkan
        x = torch.randn(1, 8, 1024)
        w = torch.ones(1024)
        ref = self.rms_norm_cpu(x, w)
        out = torch_vulkan.rms_norm(x.vulkan(), w.vulkan(), 1e-6).cpu()
        check(ref, out, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_rms_norm_backward(self):
        """Test backward pass produces correct gradients."""
        import torch_vulkan
        x_cpu = torch.randn(2, 4, 16, requires_grad=True)
        w_cpu = torch.randn(16).abs() + 0.1
        w_cpu.requires_grad_(True)

        # CPU reference
        out_cpu = self.rms_norm_cpu(x_cpu, w_cpu)
        loss_cpu = out_cpu.sum()
        loss_cpu.backward()
        grad_x_cpu = x_cpu.grad.clone()
        grad_w_cpu = w_cpu.grad.clone()

        # Vulkan
        x_vk = x_cpu.detach().clone().vulkan().requires_grad_(True)
        w_vk = w_cpu.detach().clone().vulkan().requires_grad_(True)
        out_vk = torch_vulkan.rms_norm(x_vk, w_vk, 1e-6)
        loss_vk = out_vk.sum()
        loss_vk.backward()
        grad_x_vk = x_vk.grad.cpu()
        grad_w_vk = w_vk.grad.cpu()

        check(grad_x_cpu, grad_x_vk, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(grad_w_cpu, grad_w_vk, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestNLLLossCorrectness:
    """Test nll_loss with ignore_index and backward."""

    def test_nll_loss_ignore_index(self):
        """ignore_index=-100 should be skipped in loss computation."""
        N, C = 4, 10
        log_probs = F.log_softmax(torch.randn(N, C), dim=-1)
        target = torch.tensor([3, -100, 7, 2])  # second sample ignored

        ref = F.nll_loss(log_probs, target, ignore_index=-100)
        vk = F.nll_loss(log_probs.vulkan(), target.vulkan(), ignore_index=-100).cpu()
        check(ref, vk, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_nll_loss_all_ignored(self):
        """All targets are ignore_index — loss should be nan (0/0)."""
        N, C = 3, 5
        log_probs = F.log_softmax(torch.randn(N, C), dim=-1)
        target = torch.tensor([-100, -100, -100])

        ref = F.nll_loss(log_probs, target, ignore_index=-100)
        vk = F.nll_loss(log_probs.vulkan(), target.vulkan(), ignore_index=-100).cpu()
        # Both should be nan (0/0 division when all targets ignored)
        assert torch.isnan(ref), f"Expected CPU nan, got {ref}"
        assert torch.isnan(vk), f"Expected Vulkan nan, got {vk}"

    def test_nll_loss_backward(self):
        """nll_loss backward should produce correct gradients on GPU."""
        N, C = 8, 10
        logits = torch.randn(N, C, requires_grad=True)
        target = torch.randint(0, C, (N,))

        # CPU
        loss_cpu = F.cross_entropy(logits, target)
        loss_cpu.backward()
        grad_cpu = logits.grad.clone()

        # Vulkan
        logits_vk = logits.detach().clone().vulkan().requires_grad_(True)
        target_vk = target.vulkan()
        loss_vk = F.cross_entropy(logits_vk, target_vk)
        loss_vk.backward()
        grad_vk = logits_vk.grad.cpu()

        check(grad_cpu, grad_vk, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_cross_entropy_backward_ignore_index(self):
        """cross_entropy backward with ignore_index=-100."""
        N, C = 8, 10
        logits = torch.randn(N, C, requires_grad=True)
        target = torch.randint(0, C, (N,))
        target[2] = -100
        target[5] = -100

        # CPU
        loss_cpu = F.cross_entropy(logits, target, ignore_index=-100)
        loss_cpu.backward()
        grad_cpu = logits.grad.clone()

        # Vulkan
        logits_vk = logits.detach().clone().vulkan().requires_grad_(True)
        target_vk = target.vulkan()
        loss_vk = F.cross_entropy(logits_vk, target_vk, ignore_index=-100)
        loss_vk.backward()
        grad_vk = logits_vk.grad.cpu()

        check(grad_cpu, grad_vk, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestSwiGLUCorrectness:
    """Test SwiGLU MLP pattern: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def test_swiglu_forward(self):
        B, S, D, I = 2, 4, 32, 64
        x_cpu = torch.randn(B, S, D)
        gate_w = torch.randn(I, D)
        up_w = torch.randn(I, D)
        down_w = torch.randn(D, I)

        # CPU reference
        gate = F.silu(x_cpu @ gate_w.T)
        up = x_cpu @ up_w.T
        ref = (gate * up) @ down_w.T

        # Vulkan
        x_vk = x_cpu.vulkan()
        gate_vk = F.silu(x_vk @ gate_w.vulkan().T)
        up_vk = x_vk @ up_w.vulkan().T
        out_vk = (gate_vk * up_vk) @ down_w.vulkan().T

        check(ref, out_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_swiglu_backward(self):
        B, S, D, I = 2, 4, 16, 32
        x_cpu = torch.randn(B, S, D, requires_grad=True)
        gate_w = torch.randn(I, D, requires_grad=True)
        up_w = torch.randn(I, D, requires_grad=True)
        down_w = torch.randn(D, I, requires_grad=True)

        # CPU forward + backward
        gate = F.silu(x_cpu @ gate_w.T)
        up = x_cpu @ up_w.T
        out_cpu = (gate * up) @ down_w.T
        out_cpu.sum().backward()
        grad_x_cpu = x_cpu.grad.clone()

        # Vulkan forward + backward
        x_vk = x_cpu.detach().clone().vulkan().requires_grad_(True)
        gw_vk = gate_w.detach().clone().vulkan().requires_grad_(True)
        uw_vk = up_w.detach().clone().vulkan().requires_grad_(True)
        dw_vk = down_w.detach().clone().vulkan().requires_grad_(True)

        gate_vk = F.silu(x_vk @ gw_vk.T)
        up_vk = x_vk @ uw_vk.T
        out_vk = (gate_vk * up_vk) @ dw_vk.T
        out_vk.sum().backward()

        check(grad_x_cpu, x_vk.grad.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestFusedSwiGLUCorrectness:
    """Test fused SwiGLU shader via torch_vulkan.swiglu()."""

    def test_fused_swiglu_forward(self):
        import torch_vulkan
        B, D = 4, 64
        torch.manual_seed(42)
        gate = torch.randn(B, D)
        up = torch.randn(B, D)

        ref = F.silu(gate) * up
        out = torch_vulkan.swiglu(gate.vulkan(), up.vulkan())
        check(ref, out.cpu(), rtol=RTOL, atol=ATOL)

    def test_fused_swiglu_backward(self):
        import torch_vulkan
        B, D = 4, 64
        torch.manual_seed(42)

        # CPU reference
        gate_cpu = torch.randn(B, D, requires_grad=True)
        up_cpu = torch.randn(B, D, requires_grad=True)
        out_cpu = F.silu(gate_cpu) * up_cpu
        out_cpu.sum().backward()

        # Vulkan fused
        gate_vk = gate_cpu.detach().clone().vulkan().requires_grad_(True)
        up_vk = up_cpu.detach().clone().vulkan().requires_grad_(True)
        out_vk = torch_vulkan.swiglu(gate_vk, up_vk)
        out_vk.sum().backward()

        check(out_cpu, out_vk.cpu(), rtol=RTOL, atol=ATOL)
        check(gate_cpu.grad, gate_vk.grad.cpu(), rtol=RTOL, atol=ATOL)
        check(up_cpu.grad, up_vk.grad.cpu(), rtol=RTOL, atol=ATOL)

    def test_fused_swiglu_mlp(self):
        """Full Qwen3-style MLP using fused SwiGLU."""
        import torch_vulkan
        B, S, D, I = 2, 4, 32, 64
        torch.manual_seed(42)
        x = torch.randn(B, S, D)
        gate_w = torch.randn(I, D) * 0.02
        up_w = torch.randn(I, D) * 0.02
        down_w = torch.randn(D, I) * 0.02

        # CPU: silu(gate_proj(x)) * up_proj(x) then down_proj
        gate_cpu = x @ gate_w.T
        up_cpu = x @ up_w.T
        hidden_cpu = F.silu(gate_cpu) * up_cpu
        out_cpu = hidden_cpu @ down_w.T

        # Vulkan: same but using fused swiglu
        x_vk = x.vulkan()
        gate_vk = x_vk @ gate_w.vulkan().T
        up_vk = x_vk @ up_w.vulkan().T
        hidden_vk = torch_vulkan.swiglu(gate_vk, up_vk)
        out_vk = hidden_vk @ down_w.vulkan().T

        check(out_cpu, out_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestGQACorrectness:
    """Test Grouped Query Attention pattern (repeat_kv + matmul)."""

    @staticmethod
    def repeat_kv(hidden_states, n_rep):
        """Standard repeat_kv from HuggingFace."""
        B, H, S, D = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        return hidden_states[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)

    def test_repeat_kv_forward(self):
        B, KVH, S, D = 1, 8, 4, 16
        n_rep = 2  # 16 query heads / 8 kv heads
        kv = torch.randn(B, KVH, S, D)

        ref = self.repeat_kv(kv, n_rep)
        vk = self.repeat_kv(kv.vulkan(), n_rep).cpu()
        check(ref, vk, rtol=RTOL, atol=ATOL)

    def test_gqa_attention(self):
        """Full GQA: Q[B,16,S,D] @ K[B,8,S,D].repeat(2).T -> softmax -> @ V."""
        B, S, D = 1, 4, 16
        Q_heads, KV_heads = 4, 2

        q = torch.randn(B, Q_heads, S, D)
        k = torch.randn(B, KV_heads, S, D)
        v = torch.randn(B, KV_heads, S, D)

        # Expand KV
        k_exp = self.repeat_kv(k, Q_heads // KV_heads)
        v_exp = self.repeat_kv(v, Q_heads // KV_heads)

        # CPU reference
        scale = D ** -0.5
        attn = (q @ k_exp.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        ref = attn @ v_exp

        # Vulkan
        k_exp_vk = self.repeat_kv(k.vulkan(), Q_heads // KV_heads)
        v_exp_vk = self.repeat_kv(v.vulkan(), Q_heads // KV_heads)
        attn_vk = (q.vulkan() @ k_exp_vk.transpose(-2, -1)) * scale
        attn_vk = F.softmax(attn_vk, dim=-1)
        out_vk = attn_vk @ v_exp_vk

        check(ref, out_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestRoPEDecomposedCorrectness:
    """Test Qwen3-style RoPE via decomposed tensor ops (not custom kernel)."""

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin):
        q_embed = (q * cos) + (TestRoPEDecomposedCorrectness.rotate_half(q) * sin)
        k_embed = (k * cos) + (TestRoPEDecomposedCorrectness.rotate_half(k) * sin)
        return q_embed, k_embed

    def test_rope_decomposed_forward(self):
        B, H, S, D = 1, 4, 8, 16
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)

        # Pre-compute cos/sin (like Qwen3RotaryEmbedding)
        theta = 10000.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
        positions = torch.arange(S).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
        sin = emb.sin().unsqueeze(0).unsqueeze(0)

        # CPU reference
        q_ref, k_ref = self.apply_rotary_pos_emb(q, k, cos, sin)

        # Vulkan
        q_vk, k_vk = self.apply_rotary_pos_emb(
            q.vulkan(), k.vulkan(), cos.vulkan(), sin.vulkan())

        check(q_ref, q_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(k_ref, k_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestCausalMaskCorrectness:
    """Test causal mask creation on Vulkan."""

    def test_causal_mask(self):
        S = 8
        mask = torch.triu(torch.full((S, S), float('-inf')), diagonal=1)
        mask_vk = torch.triu(
            torch.full((S, S), float('-inf'), device="vulkan"), diagonal=1).cpu()
        check(mask, mask_vk, rtol=0, atol=0)


class TestQwen3DecoderLayerCorrectness:
    """End-to-end test of a single Qwen3-style decoder layer."""

    def test_decoder_layer_forward(self):
        """Simplified Qwen3 decoder layer: RMSNorm → Attention → Residual → RMSNorm → SwiGLU → Residual."""
        import torch_vulkan

        B, S, D = 1, 4, 64
        I = D * 3  # intermediate_size
        num_heads = 4
        head_dim = D // num_heads
        num_kv_heads = 2
        n_rep = num_heads // num_kv_heads

        torch.manual_seed(42)

        # Weights
        ln1_w = torch.ones(D)
        ln2_w = torch.ones(D)
        q_w = torch.randn(num_heads * head_dim, D) * 0.02
        k_w = torch.randn(num_kv_heads * head_dim, D) * 0.02
        v_w = torch.randn(num_kv_heads * head_dim, D) * 0.02
        o_w = torch.randn(D, num_heads * head_dim) * 0.02
        gate_w = torch.randn(I, D) * 0.02
        up_w = torch.randn(I, D) * 0.02
        down_w = torch.randn(D, I) * 0.02

        x = torch.randn(B, S, D) * 0.1

        def decoder_layer(x, device="cpu"):
            def to_dev(t):
                return t.to(device) if device != "cpu" else t

            h = to_dev(x)
            # RMSNorm 1
            if device == "cpu":
                var = h.pow(2).mean(-1, keepdim=True)
                h_normed = h * torch.rsqrt(var + 1e-6) * to_dev(ln1_w)
            else:
                h_normed = torch_vulkan.rms_norm(h, to_dev(ln1_w), 1e-6)

            # Attention
            q = h_normed @ to_dev(q_w).T  # [B, S, num_heads*head_dim]
            k = h_normed @ to_dev(k_w).T
            v = h_normed @ to_dev(v_w).T

            q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(B, S, num_kv_heads, head_dim).transpose(1, 2)

            # GQA expand
            k = k[:, :, None, :, :].expand(B, num_kv_heads, n_rep, S, head_dim).reshape(B, num_heads, S, head_dim)
            v = v[:, :, None, :, :].expand(B, num_kv_heads, n_rep, S, head_dim).reshape(B, num_heads, S, head_dim)

            # Scaled dot product attention with causal mask
            scale = head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=h.device), diagonal=1)
            attn = attn + causal_mask
            attn = F.softmax(attn, dim=-1)
            attn_out = attn @ v

            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
            attn_out = attn_out @ to_dev(o_w).T

            h = to_dev(x) + attn_out  # residual

            # RMSNorm 2
            if device == "cpu":
                var2 = h.pow(2).mean(-1, keepdim=True)
                h_normed2 = h * torch.rsqrt(var2 + 1e-6) * to_dev(ln2_w)
            else:
                h_normed2 = torch_vulkan.rms_norm(h, to_dev(ln2_w), 1e-6)

            # SwiGLU MLP
            gate = F.silu(h_normed2 @ to_dev(gate_w).T)
            up = h_normed2 @ to_dev(up_w).T
            mlp_out = (gate * up) @ to_dev(down_w).T

            return h + mlp_out  # residual

        ref = decoder_layer(x, device="cpu")
        out = decoder_layer(x, device="vulkan").cpu()
        check(ref, out, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_decoder_layer_backward(self):
        """Test backward through simplified Qwen3 decoder layer."""
        import torch_vulkan

        B, S, D = 1, 4, 32
        I = D * 3
        num_heads = 4
        head_dim = D // num_heads

        torch.manual_seed(42)

        x_cpu = torch.randn(B, S, D, requires_grad=True)
        w_cpu = torch.ones(D, requires_grad=True)
        gate_w = torch.randn(I, D, requires_grad=True) * 0.02
        up_w = torch.randn(I, D, requires_grad=True) * 0.02
        down_w = torch.randn(D, I, requires_grad=True) * 0.02

        # CPU forward+backward: RMSNorm → SwiGLU (skip attention for simplicity)
        var = x_cpu.pow(2).mean(-1, keepdim=True)
        h = x_cpu * torch.rsqrt(var + 1e-6) * w_cpu
        gate = F.silu(h @ gate_w.T)
        up = h @ up_w.T
        out_cpu = (gate * up) @ down_w.T
        out_cpu.sum().backward()
        grad_x_cpu = x_cpu.grad.clone()

        # Vulkan forward+backward
        x_vk = x_cpu.detach().clone().vulkan().requires_grad_(True)
        w_vk = w_cpu.detach().clone().vulkan().requires_grad_(True)
        gw_vk = gate_w.detach().clone().vulkan().requires_grad_(True)
        uw_vk = up_w.detach().clone().vulkan().requires_grad_(True)
        dw_vk = down_w.detach().clone().vulkan().requires_grad_(True)

        h_vk = torch_vulkan.rms_norm(x_vk, w_vk, 1e-6)
        gate_vk = F.silu(h_vk @ gw_vk.T)
        up_vk = h_vk @ uw_vk.T
        out_vk = (gate_vk * up_vk) @ dw_vk.T
        out_vk.sum().backward()

        check(grad_x_cpu, x_vk.grad.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestQwen3TrainingStepCorrectness:
    """End-to-end Qwen3 training step: forward -> cross_entropy -> backward -> optimizer.step()."""

    def test_training_step(self):
        """One training step of a minimal Qwen3-style model."""
        import torch_vulkan

        B, S = 2, 8
        D, I = 32, 96
        V = 64
        num_heads = 4
        head_dim = D // num_heads
        num_kv_heads = 2
        n_rep = num_heads // num_kv_heads

        class MiniQwen3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(V, D)
                self.ln1_w = torch.nn.Parameter(torch.ones(D))
                self.ln2_w = torch.nn.Parameter(torch.ones(D))
                self.q_proj = torch.nn.Linear(D, num_heads * head_dim, bias=False)
                self.k_proj = torch.nn.Linear(D, num_kv_heads * head_dim, bias=False)
                self.v_proj = torch.nn.Linear(D, num_kv_heads * head_dim, bias=False)
                self.o_proj = torch.nn.Linear(num_heads * head_dim, D, bias=False)
                self.gate_proj = torch.nn.Linear(D, I, bias=False)
                self.up_proj = torch.nn.Linear(D, I, bias=False)
                self.down_proj = torch.nn.Linear(I, D, bias=False)
                self.lm_head = torch.nn.Linear(D, V, bias=False)
                for p in self.parameters():
                    if p.dim() > 1:
                        torch.nn.init.normal_(p, std=0.02)

            def rms_norm(self, x, w, device):
                if device == "cpu":
                    var = x.pow(2).mean(-1, keepdim=True)
                    return x * torch.rsqrt(var + 1e-6) * w
                return torch_vulkan.rms_norm(x, w, 1e-6)

            def forward(self, input_ids, device="cpu"):
                h = self.embed(input_ids)
                h_normed = self.rms_norm(h, self.ln1_w, device)

                q = self.q_proj(h_normed).view(B, S, num_heads, head_dim).transpose(1, 2)
                k = self.k_proj(h_normed).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
                v = self.v_proj(h_normed).view(B, S, num_kv_heads, head_dim).transpose(1, 2)

                k = k[:, :, None, :, :].expand(B, num_kv_heads, n_rep, S, head_dim).reshape(B, num_heads, S, head_dim)
                v = v[:, :, None, :, :].expand(B, num_kv_heads, n_rep, S, head_dim).reshape(B, num_heads, S, head_dim)

                scale = head_dim ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                causal = torch.triu(torch.full((S, S), float('-inf'), device=h.device), diagonal=1)
                attn = F.softmax(attn + causal, dim=-1)
                attn_out = (attn @ v).transpose(1, 2).reshape(B, S, D)
                attn_out = self.o_proj(attn_out)

                h = h + attn_out
                h_normed2 = self.rms_norm(h, self.ln2_w, device)

                gate = F.silu(self.gate_proj(h_normed2))
                up = self.up_proj(h_normed2)
                h = h + self.down_proj(gate * up)

                return self.lm_head(h)

        torch.manual_seed(123)
        model_cpu = MiniQwen3()
        torch.manual_seed(123)
        model_vk = MiniQwen3().vulkan()

        torch.manual_seed(456)
        input_ids = torch.randint(0, V, (B, S))
        targets = torch.randint(0, V, (B, S))

        # CPU forward + backward
        logits_cpu = model_cpu(input_ids, device="cpu")
        loss_cpu = F.cross_entropy(logits_cpu.view(-1, V), targets.view(-1))
        loss_cpu.backward()

        # Vulkan forward + backward
        # Flatten targets on CPU before Vulkan — int64 view on Vulkan corrupts data
        logits_vk = model_vk(input_ids.vulkan(), device="vulkan")
        loss_vk = F.cross_entropy(logits_vk.view(-1, V), targets.view(-1).vulkan())
        loss_vk.backward()

        check(loss_cpu, loss_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

        for name, p_vk in model_vk.named_parameters():
            assert p_vk.grad is not None, f"Missing gradient for {name}"

        # Optimizer step + verify weights match
        opt_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
        opt_vk = torch.optim.SGD(model_vk.parameters(), lr=0.01)
        opt_cpu.step()
        opt_vk.step()

        lm_cpu = dict(model_cpu.named_parameters())["lm_head.weight"]
        lm_vk = dict(model_vk.named_parameters())["lm_head.weight"]
        check(lm_cpu, lm_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_training_step_with_ignore_index(self):
        """Training step with padding tokens using ignore_index=-100."""
        B, S, D, V = 2, 8, 32, 64

        torch.manual_seed(789)
        embed_w = torch.randn(V, D) * 0.02
        lm_w = torch.randn(V, D) * 0.02

        input_ids = torch.randint(0, V, (B, S))
        targets = torch.randint(0, V, (B, S))
        targets[0, -3:] = -100
        targets[1, -2:] = -100

        # CPU
        embed_cpu = torch.nn.Embedding(V, D)
        embed_cpu.weight.data.copy_(embed_w)
        lm_cpu = torch.nn.Linear(D, V, bias=False)
        lm_cpu.weight.data.copy_(lm_w)
        logits_cpu = lm_cpu(embed_cpu(input_ids))
        loss_cpu = F.cross_entropy(logits_cpu.view(-1, V), targets.view(-1), ignore_index=-100)
        loss_cpu.backward()
        grad_lm_cpu = lm_cpu.weight.grad.clone()

        # Vulkan
        embed_vk = torch.nn.Embedding(V, D).vulkan()
        embed_vk.weight.data.copy_(embed_w.vulkan())
        lm_vk = torch.nn.Linear(D, V, bias=False).vulkan()
        lm_vk.weight.data.copy_(lm_w.vulkan())
        logits_vk = lm_vk(embed_vk(input_ids.vulkan()))
        # Flatten targets on CPU before Vulkan — int64 view on Vulkan corrupts data
        targets_flat = targets.view(-1).vulkan()
        loss_vk = F.cross_entropy(logits_vk.view(-1, V), targets_flat, ignore_index=-100)
        loss_vk.backward()

        check(loss_cpu, loss_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(grad_lm_cpu, lm_vk.weight.grad.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestTiedEmbeddingsCorrectness:
    """Verify tied embeddings (lm_head.weight = embed_tokens.weight) work correctly."""

    def test_tied_embeddings_forward(self):
        """Forward pass with shared embedding/lm_head weights."""
        B, S, D, V = 2, 8, 32, 64
        torch.manual_seed(42)
        shared_w = torch.randn(V, D) * 0.02

        # CPU
        embed_cpu = torch.nn.Embedding(V, D)
        embed_cpu.weight.data.copy_(shared_w)
        input_ids = torch.randint(0, V, (B, S))
        h_cpu = embed_cpu(input_ids)
        # lm_head uses same weight: logits = h @ weight.T
        logits_cpu = F.linear(h_cpu, embed_cpu.weight)

        # Vulkan
        embed_vk = torch.nn.Embedding(V, D).vulkan()
        embed_vk.weight.data.copy_(shared_w.vulkan())
        h_vk = embed_vk(input_ids.vulkan())
        logits_vk = F.linear(h_vk, embed_vk.weight)

        check(logits_cpu, logits_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_tied_embeddings_backward(self):
        """Backward pass: gradient accumulates from both embedding and lm_head usage."""
        B, S, D, V = 2, 8, 32, 64
        torch.manual_seed(42)
        shared_w = torch.randn(V, D) * 0.02
        input_ids = torch.randint(0, V, (B, S))
        targets = torch.randint(0, V, (B, S))

        # CPU — tied weights
        embed_cpu = torch.nn.Embedding(V, D)
        embed_cpu.weight.data.copy_(shared_w)
        embed_cpu.weight.requires_grad_(True)
        h_cpu = embed_cpu(input_ids)
        logits_cpu = F.linear(h_cpu, embed_cpu.weight)
        loss_cpu = F.cross_entropy(logits_cpu.view(-1, V), targets.view(-1))
        loss_cpu.backward()
        grad_cpu = embed_cpu.weight.grad.clone()

        # Vulkan — tied weights
        embed_vk = torch.nn.Embedding(V, D).vulkan()
        embed_vk.weight.data.copy_(shared_w.vulkan())
        embed_vk.weight.requires_grad_(True)
        h_vk = embed_vk(input_ids.vulkan())
        logits_vk = F.linear(h_vk, embed_vk.weight)
        loss_vk = F.cross_entropy(logits_vk.view(-1, V), targets.view(-1).vulkan())
        loss_vk.backward()

        check(loss_cpu, loss_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        # Gradient accumulates from both embedding lookup AND lm_head matmul
        check(grad_cpu, embed_vk.weight.grad.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_tied_embeddings_optimizer_step(self):
        """SGD step with tied weights — single weight tensor gets one update."""
        B, S, D, V = 2, 8, 32, 64
        torch.manual_seed(42)
        shared_w = torch.randn(V, D) * 0.02
        input_ids = torch.randint(0, V, (B, S))
        targets = torch.randint(0, V, (B, S))

        # CPU
        embed_cpu = torch.nn.Embedding(V, D)
        embed_cpu.weight.data.copy_(shared_w)
        h = embed_cpu(input_ids)
        logits = F.linear(h, embed_cpu.weight)
        loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        loss.backward()
        opt_cpu = torch.optim.SGD([embed_cpu.weight], lr=0.01)
        opt_cpu.step()

        # Vulkan
        embed_vk = torch.nn.Embedding(V, D).vulkan()
        embed_vk.weight.data.copy_(shared_w.vulkan())
        h = embed_vk(input_ids.vulkan())
        logits = F.linear(h, embed_vk.weight)
        loss = F.cross_entropy(logits.view(-1, V), targets.view(-1).vulkan())
        loss.backward()
        opt_vk = torch.optim.SGD([embed_vk.weight], lr=0.01)
        opt_vk.step()

        check(embed_cpu.weight, embed_vk.weight.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_tied_embeddings_qwen3_model(self):
        """Full Qwen3-style model with lm_head.weight = embed.weight (tied).

        NOTE: .vulkan() copies all tensors independently (opaque allocator, no shared storage),
        so weight tying must be re-established AFTER moving to Vulkan. This matches the pattern
        used by HuggingFace models: tie_weights() is called after .to(device).
        """
        import torch_vulkan
        B, S, D, V = 2, 8, 32, 64

        class TiedQwen3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(V, D)
                self.ln_w = torch.nn.Parameter(torch.ones(D))
                self.proj = torch.nn.Linear(D, D, bias=False)
                self.lm_head = torch.nn.Linear(V, D, bias=False)  # separate initially
                for p in self.parameters():
                    if p.dim() > 1:
                        torch.nn.init.normal_(p, std=0.02)
                self.tie_weights()

            def tie_weights(self):
                """Tie lm_head weight to embedding weight — call after .to(device)."""
                self.lm_head.weight = self.embed.weight

            def rms_norm(self, x, w, device):
                if device == "cpu":
                    var = x.pow(2).mean(-1, keepdim=True)
                    return x * torch.rsqrt(var + 1e-6) * w
                return torch_vulkan.rms_norm(x, w, 1e-6)

            def forward(self, input_ids, device="cpu"):
                h = self.embed(input_ids)
                h = self.rms_norm(h, self.ln_w, device)
                h = self.proj(h)
                return F.linear(h, self.embed.weight)

        torch.manual_seed(99)
        model_cpu = TiedQwen3()
        torch.manual_seed(99)
        model_vk = TiedQwen3().vulkan()
        model_vk.tie_weights()  # Re-establish tie after .vulkan()

        input_ids = torch.randint(0, V, (B, S))
        targets = torch.randint(0, V, (B, S))

        # CPU
        logits_cpu = model_cpu(input_ids, device="cpu")
        loss_cpu = F.cross_entropy(logits_cpu.view(-1, V), targets.view(-1))
        loss_cpu.backward()

        # Vulkan
        logits_vk = model_vk(input_ids.vulkan(), device="vulkan")
        loss_vk = F.cross_entropy(logits_vk.view(-1, V), targets.view(-1).vulkan())
        loss_vk.backward()

        check(loss_cpu, loss_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

        # The shared embed weight should have accumulated gradients from both paths
        grad_cpu = model_cpu.embed.weight.grad
        grad_vk = model_vk.embed.weight.grad
        assert grad_cpu is not None, "CPU embed grad is None"
        assert grad_vk is not None, "Vulkan embed grad is None"
        check(grad_cpu, grad_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

        # Verify tie is preserved: lm_head.weight IS embed.weight
        assert model_vk.lm_head.weight is model_vk.embed.weight, "Tie broken after .vulkan()"

        # Optimizer step
        opt_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
        opt_vk = torch.optim.SGD(model_vk.parameters(), lr=0.01)
        opt_cpu.step()
        opt_vk.step()
        check(model_cpu.embed.weight, model_vk.embed.weight.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestLargeVocabCorrectness:
    """Verify operations work at Qwen3 vocab scale (151936 classes)."""

    def test_large_vocab_embedding(self):
        """Embedding lookup with 151936-entry vocab."""
        V, D = 151936, 64  # Full Qwen3 vocab, reduced D for speed
        B, S = 1, 4
        torch.manual_seed(42)
        w = torch.randn(V, D) * 0.02
        input_ids = torch.randint(0, V, (B, S))

        embed_cpu = torch.nn.Embedding(V, D)
        embed_cpu.weight.data.copy_(w)
        out_cpu = embed_cpu(input_ids)

        embed_vk = torch.nn.Embedding(V, D).vulkan()
        embed_vk.weight.data.copy_(w.vulkan())
        out_vk = embed_vk(input_ids.vulkan())

        check(out_cpu, out_vk.cpu(), rtol=RTOL, atol=ATOL)

    def test_large_vocab_linear(self):
        """Linear projection to 151936 output classes."""
        V, D = 151936, 64
        B, S = 1, 4
        torch.manual_seed(42)
        x = torch.randn(B * S, D) * 0.1
        w = torch.randn(V, D) * 0.02

        # CPU: [4, 64] @ [64, 151936] = [4, 151936]
        out_cpu = F.linear(x, w)

        # Vulkan
        out_vk = F.linear(x.vulkan(), w.vulkan())

        check(out_cpu, out_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_large_vocab_log_softmax(self):
        """log_softmax over 151936 classes (wide row reduction)."""
        V = 151936
        B_S = 4  # batch*seq
        torch.manual_seed(42)
        x = torch.randn(B_S, V) * 0.1

        out_cpu = F.log_softmax(x, dim=-1)
        out_vk = F.log_softmax(x.vulkan(), dim=-1)

        check(out_cpu, out_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_large_vocab_cross_entropy(self):
        """Cross-entropy loss with 151936 classes."""
        V = 151936
        B_S = 4
        torch.manual_seed(42)
        logits = torch.randn(B_S, V) * 0.1
        targets = torch.randint(0, V, (B_S,))

        loss_cpu = F.cross_entropy(logits, targets)
        loss_vk = F.cross_entropy(logits.vulkan(), targets.vulkan())

        check(loss_cpu, loss_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_large_vocab_cross_entropy_backward(self):
        """Backward through cross-entropy with 151936 classes."""
        V, D = 151936, 64
        B_S = 4
        torch.manual_seed(42)

        # CPU
        x_cpu = torch.randn(B_S, D, requires_grad=True)
        w_cpu = torch.randn(V, D) * 0.02
        w_cpu = torch.nn.Parameter(w_cpu)
        logits_cpu = F.linear(x_cpu, w_cpu)
        targets = torch.randint(0, V, (B_S,))
        loss_cpu = F.cross_entropy(logits_cpu, targets)
        loss_cpu.backward()

        # Vulkan
        x_vk = torch.randn(B_S, D, requires_grad=True)
        x_vk.data.copy_(x_cpu.data)
        w_vk_data = w_cpu.data.clone().vulkan()
        w_vk = torch.nn.Parameter(w_vk_data)
        logits_vk = F.linear(x_vk.vulkan(), w_vk)
        loss_vk = F.cross_entropy(logits_vk, targets.vulkan())
        loss_vk.backward()

        check(loss_cpu, loss_vk.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(w_cpu.grad, w_vk.grad.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_large_vocab_embedding_backward(self):
        """embedding_dense_backward with 151936-row gradient accumulation."""
        V, D = 151936, 64
        B, S = 1, 4
        torch.manual_seed(42)
        w = torch.randn(V, D) * 0.02
        input_ids = torch.randint(0, V, (B, S))

        # CPU
        embed_cpu = torch.nn.Embedding(V, D)
        embed_cpu.weight.data.copy_(w)
        out = embed_cpu(input_ids)
        loss = out.sum()
        loss.backward()
        grad_cpu = embed_cpu.weight.grad.clone()

        # Vulkan
        embed_vk = torch.nn.Embedding(V, D).vulkan()
        embed_vk.weight.data.copy_(w.vulkan())
        out = embed_vk(input_ids.vulkan())
        loss = out.sum()
        loss.backward()

        check(grad_cpu, embed_vk.weight.grad.cpu(), rtol=RTOL, atol=ATOL)
        # Verify sparse pattern: only input_ids rows should have nonzero grad
        nonzero_rows_cpu = (grad_cpu.abs().sum(dim=1) > 0).sum().item()
        nonzero_rows_vk = (embed_vk.weight.grad.cpu().abs().sum(dim=1) > 0).sum().item()
        assert nonzero_rows_cpu == nonzero_rows_vk
        assert nonzero_rows_cpu <= B * S  # at most B*S unique rows


class TestGradientCheckpointingCorrectness:
    """Verify torch.utils.checkpoint works with Vulkan tensors."""

    def test_checkpoint_basic(self):
        """Gradient checkpointing produces same gradients as without."""
        from torch.utils.checkpoint import checkpoint

        B, D = 4, 32
        torch.manual_seed(42)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(D, D, bias=False)
                self.fc2 = torch.nn.Linear(D, D, bias=False)

            def forward(self, x):
                return self.fc2(F.relu(self.fc1(x)))

        # Without checkpointing (reference)
        torch.manual_seed(42)
        block_ref = Block().vulkan()
        x_ref = torch.randn(B, D).vulkan().requires_grad_(True)
        out_ref = block_ref(x_ref)
        out_ref.sum().backward()

        # With checkpointing
        torch.manual_seed(42)
        block_cp = Block().vulkan()
        x_cp = torch.randn(B, D).vulkan().requires_grad_(True)
        out_cp = checkpoint(block_cp, x_cp, use_reentrant=False)
        out_cp.sum().backward()

        check(out_ref.cpu(), out_cp.cpu(), rtol=RTOL, atol=ATOL)
        check(block_ref.fc1.weight.grad.cpu(), block_cp.fc1.weight.grad.cpu(), rtol=RTOL, atol=ATOL)
        check(block_ref.fc2.weight.grad.cpu(), block_cp.fc2.weight.grad.cpu(), rtol=RTOL, atol=ATOL)

    def test_checkpoint_multi_layer(self):
        """Multiple checkpointed layers in sequence."""
        from torch.utils.checkpoint import checkpoint

        B, D, V = 2, 32, 16
        torch.manual_seed(42)

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(D, D, bias=False)
            def forward(self, x):
                return F.silu(self.fc(x))

        class Model(torch.nn.Module):
            def __init__(self, use_cp=False):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer() for _ in range(4)])
                self.head = torch.nn.Linear(D, V, bias=False)
                self.use_cp = use_cp
            def forward(self, x):
                for layer in self.layers:
                    if self.use_cp:
                        x = checkpoint(layer, x, use_reentrant=False)
                    else:
                        x = layer(x)
                return self.head(x)

        w = torch.randn(D, D) * 0.02
        head_w = torch.randn(V, D) * 0.02

        # Without checkpointing
        torch.manual_seed(42)
        m_ref = Model(use_cp=False).vulkan()
        # With checkpointing
        torch.manual_seed(42)
        m_cp = Model(use_cp=True).vulkan()

        x = torch.randn(B, D)
        targets = torch.randint(0, V, (B,))

        # Reference
        logits_ref = m_ref(x.vulkan())
        loss_ref = F.cross_entropy(logits_ref, targets.vulkan())
        loss_ref.backward()

        # Checkpointed
        logits_cp = m_cp(x.vulkan())
        loss_cp = F.cross_entropy(logits_cp, targets.vulkan())
        loss_cp.backward()

        check(loss_ref.cpu(), loss_cp.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_cp.named_parameters()):
            assert p1.grad is not None, f"ref grad None for {n1}"
            assert p2.grad is not None, f"cp grad None for {n2}"
            check(p1.grad.cpu(), p2.grad.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestGradientClippingCorrectness:
    """Verify gradient clipping works on Vulkan tensors."""

    def test_clip_grad_norm(self):
        """torch.nn.utils.clip_grad_norm_ works on Vulkan parameters."""
        D = 32
        torch.manual_seed(42)

        # CPU reference
        model_cpu = torch.nn.Linear(D, D, bias=False)
        x_cpu = torch.randn(4, D)
        (model_cpu(x_cpu) * 100).sum().backward()  # large grads
        total_norm_cpu = torch.nn.utils.clip_grad_norm_(model_cpu.parameters(), max_norm=1.0)

        # Vulkan
        torch.manual_seed(42)
        model_vk = torch.nn.Linear(D, D, bias=False).vulkan()
        x_vk = torch.randn(4, D).vulkan()
        (model_vk(x_vk) * 100).sum().backward()
        total_norm_vk = torch.nn.utils.clip_grad_norm_(model_vk.parameters(), max_norm=1.0)

        check(torch.tensor(total_norm_cpu), torch.tensor(total_norm_vk), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        check(model_cpu.weight.grad, model_vk.weight.grad.cpu(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_clip_grad_value(self):
        """torch.nn.utils.clip_grad_value_ works on Vulkan parameters."""
        D = 32
        torch.manual_seed(42)

        model_cpu = torch.nn.Linear(D, D, bias=False)
        x_cpu = torch.randn(4, D)
        (model_cpu(x_cpu) * 100).sum().backward()
        torch.nn.utils.clip_grad_value_(model_cpu.parameters(), clip_value=0.5)

        torch.manual_seed(42)
        model_vk = torch.nn.Linear(D, D, bias=False).vulkan()
        x_vk = torch.randn(4, D).vulkan()
        (model_vk(x_vk) * 100).sum().backward()
        torch.nn.utils.clip_grad_value_(model_vk.parameters(), clip_value=0.5)

        check(model_cpu.weight.grad, model_vk.weight.grad.cpu(), rtol=RTOL, atol=ATOL)
