"""Tests for GPU-dispatched compute operators (binary, unary, activations, BLAS)."""

import torch
import pytest

# SwiftShader tolerance (CPU Vulkan is exact for simple ops, but use rtol for safety)
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


# ── Binary Ops ───────────────────────────────────────────────────


class TestAdd:
    def test_basic(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        result = to_vulkan(a) + to_vulkan(b)
        assert_close(result, a + b)

    def test_large(self):
        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        result = to_vulkan(a) + to_vulkan(b)
        assert_close(result, a + b)

    def test_1d(self):
        a = torch.randn(100)
        b = torch.randn(100)
        result = to_vulkan(a) + to_vulkan(b)
        assert_close(result, a + b)

    def test_scalar_alpha(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        result = torch.add(to_vulkan(a), to_vulkan(b), alpha=2.0)
        assert_close(result, torch.add(a, b, alpha=2.0))


class TestSub:
    def test_basic(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        result = to_vulkan(a) - to_vulkan(b)
        assert_close(result, a - b)

    def test_scalar_alpha(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        result = torch.sub(to_vulkan(a), to_vulkan(b), alpha=0.5)
        assert_close(result, torch.sub(a, b, alpha=0.5))


class TestMul:
    def test_basic(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        result = to_vulkan(a) * to_vulkan(b)
        assert_close(result, a * b)

    def test_large(self):
        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        result = to_vulkan(a) * to_vulkan(b)
        assert_close(result, a * b)


class TestDiv:
    def test_basic(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4).abs() + 0.1  # avoid division by zero
        result = to_vulkan(a) / to_vulkan(b)
        assert_close(result, a / b)


# ── Unary Ops ────────────────────────────────────────────────────


class TestUnaryOps:
    def test_neg(self):
        a = torch.randn(4, 4)
        result = -to_vulkan(a)
        assert_close(result, -a)

    def test_abs(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).abs()
        assert_close(result, a.abs())

    def test_exp(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).exp()
        assert_close(result, a.exp(), rtol=1e-3, atol=1e-4)

    def test_log(self):
        a = torch.rand(4, 4) + 0.01  # positive values
        result = to_vulkan(a).log()
        assert_close(result, a.log())

    def test_sqrt(self):
        a = torch.rand(4, 4) + 0.01  # positive values
        result = to_vulkan(a).sqrt()
        assert_close(result, a.sqrt())

    def test_rsqrt(self):
        a = torch.rand(4, 4) + 0.01
        result = to_vulkan(a).rsqrt()
        assert_close(result, a.rsqrt(), rtol=1e-3, atol=1e-4)

    def test_ceil(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).ceil()
        assert_close(result, a.ceil())

    def test_floor(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).floor()
        assert_close(result, a.floor())

    def test_round(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).round()
        assert_close(result, a.round())

    def test_sign(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).sign()
        assert_close(result, a.sign())


# ── Activations ──────────────────────────────────────────────────


class TestReLU:
    def test_basic(self):
        a = torch.randn(4, 4)
        result = to_vulkan(a).relu()
        assert_close(result, a.relu())

    def test_all_negative(self):
        a = -torch.rand(4, 4) - 0.1
        result = to_vulkan(a).relu()
        assert_close(result, torch.zeros(4, 4))

    def test_all_positive(self):
        a = torch.rand(4, 4) + 0.1
        result = to_vulkan(a).relu()
        assert_close(result, a)

    def test_large(self):
        a = torch.randn(64, 64)
        result = to_vulkan(a).relu()
        assert_close(result, a.relu())


# ── Matrix Multiplication ────────────────────────────────────────


class TestMM:
    def test_square(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        result = torch.mm(to_vulkan(a), to_vulkan(b))
        assert_close(result, torch.mm(a, b), rtol=1e-3, atol=1e-4)

    def test_non_square(self):
        a = torch.randn(3, 5)
        b = torch.randn(5, 7)
        result = torch.mm(to_vulkan(a), to_vulkan(b))
        assert_close(result, torch.mm(a, b), rtol=1e-3, atol=1e-4)

    def test_64x64(self):
        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        result = torch.mm(to_vulkan(a), to_vulkan(b))
        assert_close(result, torch.mm(a, b), rtol=1e-3, atol=1e-3)

    def test_vector_matrix(self):
        a = torch.randn(1, 8)
        b = torch.randn(8, 4)
        result = torch.mm(to_vulkan(a), to_vulkan(b))
        assert_close(result, torch.mm(a, b), rtol=1e-3, atol=1e-4)


class TestAddMM:
    def test_basic(self):
        bias = torch.randn(4)
        a = torch.randn(3, 5)
        b = torch.randn(5, 4)
        result = torch.addmm(to_vulkan(bias), to_vulkan(a), to_vulkan(b))
        expected = torch.addmm(bias, a, b)
        assert_close(result, expected, rtol=1e-3, atol=1e-4)


# ── Clone ────────────────────────────────────────────────────────


class TestClone:
    def test_clone(self):
        a = torch.randn(4, 4)
        v = to_vulkan(a)
        c = v.clone()
        assert_close(c, a)

    def test_clone_independence(self):
        a = torch.randn(4, 4)
        v = to_vulkan(a)
        c = v.clone()
        # Modifying clone shouldn't affect original
        c.fill_(0.0)
        assert_close(v, a)


# ── Fill with GPU shader ────────────────────────────────────────


class TestFillGPU:
    def test_fill_scalar(self):
        t = torch.empty(4, 4, device="vulkan:0")
        t.fill_(42.0)
        assert_close(t, torch.full((4, 4), 42.0))

    def test_fill_zero(self):
        t = torch.empty(8, 8, device="vulkan:0")
        t.zero_()
        assert_close(t, torch.zeros(8, 8))

    def test_fill_negative(self):
        t = torch.empty(4, 4, device="vulkan:0")
        t.fill_(-1.5)
        assert_close(t, torch.full((4, 4), -1.5))


# ── Combined operations (smoke tests) ───────────────────────────


class TestCombinedOps:
    def test_add_mul_chain(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        va, vb, vc = to_vulkan(a), to_vulkan(b), to_vulkan(c)
        result = (va + vb) * vc
        assert_close(result, (a + b) * c)

    def test_neg_relu(self):
        a = torch.randn(4, 4)
        result = (-to_vulkan(a)).relu()
        assert_close(result, (-a).relu())

    def test_mm_add(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        result = torch.mm(to_vulkan(a), to_vulkan(b)) + to_vulkan(c)
        expected = torch.mm(a, b) + c
        assert_close(result, expected, rtol=1e-3, atol=1e-3)
