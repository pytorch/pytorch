"""Tests for Stage 2 ops: comparisons, in-place, activations, reductions, shape ops."""

import torch
import pytest

RTOL = 1e-4
ATOL = 1e-5


@pytest.fixture(autouse=True)
def setup():
    try:
        import torch_vulkan
        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
    except ImportError:
        pytest.skip("torch_vulkan not installed")


def to_vulkan(t):
    return t.to("vulkan:0")


def assert_close(vulkan_result, expected, rtol=RTOL, atol=ATOL):
    actual = vulkan_result.cpu()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


# ── Comparison Ops ───────────────────────────────────────────────


class TestComparison:
    def test_eq(self):
        a = torch.tensor([1.0, 2.0, 3.0, 2.0])
        b = torch.tensor([1.0, 3.0, 3.0, 1.0])
        result = torch.eq(to_vulkan(a), to_vulkan(b))
        expected = torch.eq(a, b)
        assert result.cpu().equal(expected)

    def test_ne(self):
        a = torch.tensor([1.0, 2.0, 3.0, 2.0])
        b = torch.tensor([1.0, 3.0, 3.0, 1.0])
        result = torch.ne(to_vulkan(a), to_vulkan(b))
        expected = torch.ne(a, b)
        assert result.cpu().equal(expected)

    def test_lt(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 2.0, 1.0])
        result = torch.lt(to_vulkan(a), to_vulkan(b))
        expected = torch.lt(a, b)
        assert result.cpu().equal(expected)

    def test_gt(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 2.0, 1.0])
        result = torch.gt(to_vulkan(a), to_vulkan(b))
        expected = torch.gt(a, b)
        assert result.cpu().equal(expected)

    def test_le(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 2.0, 1.0])
        result = torch.le(to_vulkan(a), to_vulkan(b))
        expected = torch.le(a, b)
        assert result.cpu().equal(expected)

    def test_ge(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 2.0, 1.0])
        result = torch.ge(to_vulkan(a), to_vulkan(b))
        expected = torch.ge(a, b)
        assert result.cpu().equal(expected)

    def test_where(self):
        cond = torch.tensor([True, False, True, False])
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([5.0, 6.0, 7.0, 8.0])
        result = torch.where(to_vulkan(cond), to_vulkan(a), to_vulkan(b))
        expected = torch.where(cond, a, b)
        assert_close(result, expected)


# ── In-place Ops ─────────────────────────────────────────────────


class TestInplace:
    def test_add_(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        va = to_vulkan(a.clone())
        va.add_(to_vulkan(b))
        expected = a + b
        assert_close(va, expected)

    def test_sub_(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        va = to_vulkan(a.clone())
        va.sub_(to_vulkan(b))
        expected = a - b
        assert_close(va, expected)

    def test_mul_(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        va = to_vulkan(a.clone())
        va.mul_(to_vulkan(b))
        expected = a * b
        assert_close(va, expected)

    def test_div_(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4).abs() + 0.1
        va = to_vulkan(a.clone())
        va.div_(to_vulkan(b))
        expected = a / b
        assert_close(va, expected)

    def test_add_alpha_(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        va = to_vulkan(a.clone())
        va.add_(to_vulkan(b), alpha=2.0)
        expected = a + 2.0 * b
        assert_close(va, expected)


# ── Extended Activations ─────────────────────────────────────────


class TestActivations:
    def test_sigmoid(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).sigmoid()
        assert_close(result, a.sigmoid(), rtol=1e-3, atol=1e-4)

    def test_sigmoid_range(self):
        a = torch.randn(100)
        result = to_vulkan(a).sigmoid().cpu()
        assert (result >= 0).all() and (result <= 1).all()

    def test_tanh(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).tanh()
        assert_close(result, a.tanh(), rtol=1e-3, atol=1e-4)

    def test_tanh_range(self):
        a = torch.randn(100)
        result = to_vulkan(a).tanh().cpu()
        assert (result >= -1).all() and (result <= 1).all()

    def test_gelu(self):
        a = torch.randn(4, 4)
        result = torch.nn.functional.gelu(to_vulkan(a))
        expected = torch.nn.functional.gelu(a)
        assert_close(result, expected, rtol=1e-2, atol=1e-3)

    def test_silu(self):
        a = torch.randn(4, 4)
        result = torch.nn.functional.silu(to_vulkan(a))
        expected = torch.nn.functional.silu(a)
        assert_close(result, expected, rtol=1e-3, atol=1e-4)

    def test_leaky_relu(self):
        a = torch.randn(4, 4)
        result = torch.nn.functional.leaky_relu(to_vulkan(a), 0.01)
        expected = torch.nn.functional.leaky_relu(a, 0.01)
        assert_close(result, expected)

    def test_leaky_relu_custom_slope(self):
        a = torch.randn(4, 4)
        result = torch.nn.functional.leaky_relu(to_vulkan(a), 0.2)
        expected = torch.nn.functional.leaky_relu(a, 0.2)
        assert_close(result, expected)

    def test_elu(self):
        a = torch.randn(4, 4)
        result = torch.nn.functional.elu(to_vulkan(a))
        expected = torch.nn.functional.elu(a)
        assert_close(result, expected, rtol=1e-3, atol=1e-4)

    def test_clamp(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).clamp(-0.5, 0.5)
        expected = a.clamp(-0.5, 0.5)
        assert_close(result, expected)

    def test_clamp_min_only(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).clamp(min=0.0)
        expected = a.clamp(min=0.0)
        assert_close(result, expected)

    def test_clamp_max_only(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).clamp(max=0.0)
        expected = a.clamp(max=0.0)
        assert_close(result, expected)


# ── Pow ──────────────────────────────────────────────────────────


class TestPow:
    def test_tensor_tensor(self):
        a = torch.rand(4, 4) + 0.1
        b = torch.rand(4, 4) + 0.1
        result = to_vulkan(a).pow(to_vulkan(b))
        assert_close(result, a.pow(b), rtol=1e-3, atol=1e-4)

    def test_tensor_scalar(self):
        a = torch.rand(4, 4) + 0.1
        result = to_vulkan(a).pow(2.0)
        assert_close(result, a.pow(2.0), rtol=1e-3, atol=1e-4)

    def test_square(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).pow(2.0)
        expected = a * a
        assert_close(result, expected, rtol=1e-3, atol=1e-4)


# ── Reductions ───────────────────────────────────────────────────


class TestReductions:
    def test_sum_full(self):
        a = torch.randn(64)
        result = to_vulkan(a).sum()
        assert_close(result, a.sum(), rtol=1e-3, atol=1e-2)

    def test_sum_2d(self):
        a = torch.randn(8, 8)
        result = to_vulkan(a).sum()
        assert_close(result, a.sum(), rtol=1e-3, atol=1e-2)

    def test_sum_dim(self):
        a = torch.randn(4, 8)
        result = to_vulkan(a).sum(dim=1)
        assert_close(result, a.sum(dim=1), rtol=1e-3, atol=1e-2)

    def test_sum_dim_keepdim(self):
        a = torch.randn(4, 8)
        result = to_vulkan(a).sum(dim=1, keepdim=True)
        expected = a.sum(dim=1, keepdim=True)
        assert result.cpu().shape == expected.shape
        assert_close(result, expected, rtol=1e-3, atol=1e-2)

    def test_mean_full(self):
        a = torch.randn(64)
        result = to_vulkan(a).mean()
        assert_close(result, a.mean(), rtol=1e-3, atol=1e-2)

    def test_mean_dim(self):
        a = torch.randn(4, 8)
        result = to_vulkan(a).mean(dim=1)
        assert_close(result, a.mean(dim=1), rtol=1e-3, atol=1e-2)

    def test_amax(self):
        a = torch.randn(64)
        result = to_vulkan(a).amax(dim=[])
        assert_close(result, a.amax(dim=[]), rtol=1e-4, atol=1e-5)

    def test_amin(self):
        a = torch.randn(64)
        result = to_vulkan(a).amin(dim=[])
        assert_close(result, a.amin(dim=[]), rtol=1e-4, atol=1e-5)


# ── Shape Ops ────────────────────────────────────────────────────


class TestShapeOps:
    def test_view(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).view(16)
        assert_close(result, a.view(16))

    def test_view_infer(self):
        a = torch.randn(2, 3, 4)
        result = to_vulkan(a).view(2, -1)
        assert_close(result, a.view(2, -1))

    def test_reshape(self):
        a = torch.randn(6, 4)
        result = to_vulkan(a).reshape(2, 3, 4)
        assert_close(result, a.reshape(2, 3, 4))

    def test_unsqueeze(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).unsqueeze(0)
        assert result.cpu().shape == (1, 4, 4)
        assert_close(result, a.unsqueeze(0))

    def test_unsqueeze_last(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).unsqueeze(-1)
        assert result.cpu().shape == (4, 4, 1)

    def test_squeeze(self):
        a = torch.randn(1, 4, 1, 4)
        result = to_vulkan(a).squeeze()
        assert result.cpu().shape == (4, 4)
        assert_close(result, a.squeeze())

    def test_squeeze_dim(self):
        a = torch.randn(1, 4, 4)
        result = to_vulkan(a).squeeze(0)
        assert result.cpu().shape == (4, 4)

    def test_permute(self):
        a = torch.randn(2, 3, 4)
        result = to_vulkan(a).permute(2, 0, 1)
        assert_close(result, a.permute(2, 0, 1))

    def test_transpose(self):
        a = torch.randn(3, 5)
        result = to_vulkan(a).t()
        assert_close(result, a.t())

    def test_expand(self):
        a = torch.randn(1, 4)
        result = to_vulkan(a).expand(3, 4)
        assert_close(result, a.expand(3, 4))

    def test_cat(self):
        a = torch.randn(2, 4)
        b = torch.randn(3, 4)
        result = torch.cat([to_vulkan(a), to_vulkan(b)], dim=0)
        expected = torch.cat([a, b], dim=0)
        assert_close(result, expected)

    def test_cat_dim1(self):
        a = torch.randn(4, 2)
        b = torch.randn(4, 3)
        result = torch.cat([to_vulkan(a), to_vulkan(b)], dim=1)
        expected = torch.cat([a, b], dim=1)
        assert_close(result, expected)

    def test_select(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).select(0, 2)
        assert_close(result, a.select(0, 2))

    def test_slice(self):
        a = torch.randn(8, 4)
        result = to_vulkan(a)[2:6]
        assert_close(result, a[2:6])
