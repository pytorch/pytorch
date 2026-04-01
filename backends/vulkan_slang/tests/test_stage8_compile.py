"""Tests for Stage 8: torch.compile / AOT Autograd compatibility."""

import torch
import torch.nn.functional as F
import pytest


RTOL = 1e-3
ATOL = 1e-3


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


# ── Meta Kernel Shape Tests ──────────────────────────────────────
# These verify that FakeTensor tracing produces correct output shapes.


class TestMetaKernels:
    """Test that meta (FakeTensor) kernels produce correct shapes."""

    def test_meta_unary(self):
        x = torch.randn(4, 8, device="meta")
        result = torch.neg(x)
        assert result.shape == (4, 8)
        assert result.device == torch.device("meta")

    def test_meta_binary(self):
        a = torch.randn(4, 8, device="meta")
        b = torch.randn(4, 8, device="meta")
        result = a + b
        assert result.shape == (4, 8)

    def test_meta_mm(self):
        a = torch.randn(4, 8, device="meta")
        b = torch.randn(8, 16, device="meta")
        result = torch.mm(a, b)
        assert result.shape == (4, 16)

    def test_meta_conv2d(self):
        x = torch.randn(1, 3, 32, 32, device="meta")
        w = torch.randn(16, 3, 3, 3, device="meta")
        result = F.conv2d(x, w, padding=1)
        assert result.shape == (1, 16, 32, 32)

    def test_meta_softmax(self):
        x = torch.randn(4, 10, device="meta")
        result = F.softmax(x, dim=-1)
        assert result.shape == (4, 10)

    def test_meta_reshape(self):
        x = torch.randn(2, 3, 4, device="meta")
        result = x.reshape(6, 4)
        assert result.shape == (6, 4)

    def test_meta_cat(self):
        a = torch.randn(2, 3, device="meta")
        b = torch.randn(4, 3, device="meta")
        result = torch.cat([a, b], dim=0)
        assert result.shape == (6, 3)

    def test_meta_sum(self):
        x = torch.randn(4, 8, device="meta")
        result = x.sum(dim=1)
        assert result.shape == (4,)

    def test_meta_topk(self):
        x = torch.randn(4, 32, device="meta")
        values, indices = torch.topk(x, k=5, dim=-1)
        assert values.shape == (4, 5)
        assert indices.shape == (4, 5)


# ── torch.compile Tests ──────────────────────────────────────────


class TestTorchCompile:
    """Test that torch.compile works with Vulkan backend.

    These tests verify that compiled functions produce correct results
    when running on the Vulkan device. torch.compile uses meta/FakeTensor
    kernels for tracing and AOT Autograd for generating backward graphs.
    """

    def test_compile_simple_add(self):
        """Basic compile test: element-wise add."""
        @torch.compile(backend="eager")
        def f(a, b):
            return a + b

        a = torch.randn(8)
        b = torch.randn(8)
        expected = a + b

        va = to_vulkan(a)
        vb = to_vulkan(b)
        result = f(va, vb)
        assert_close(result, expected)

    def test_compile_chain(self):
        """Compile with chained ops including scalar promotion."""
        @torch.compile(backend="eager")
        def f(x):
            return torch.relu(x * 2 + 1)

        x = torch.randn(16)
        expected = torch.relu(x * 2 + 1)
        result = f(to_vulkan(x))
        assert_close(result, expected)

    def test_compile_linear(self):
        """Compile with linear layer."""
        @torch.compile(backend="eager")
        def f(x, w, b):
            return F.linear(x, w, b)

        x = torch.randn(4, 8)
        w = torch.randn(16, 8)
        b = torch.randn(16)
        expected = F.linear(x, w, b)

        result = f(to_vulkan(x), to_vulkan(w), to_vulkan(b))
        assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_compile_reduction(self):
        """Compile with reduction."""
        @torch.compile(backend="eager")
        def f(x):
            return x.sum()

        x = torch.randn(32)
        expected = x.sum()
        result = f(to_vulkan(x))
        assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_compile_sigmoid(self):
        """Compile with sigmoid activation."""
        @torch.compile(backend="eager")
        def f(x):
            return torch.sigmoid(x)

        x = torch.randn(32)
        expected = torch.sigmoid(x)
        result = f(to_vulkan(x))
        assert_close(result, expected)

    def test_compile_tanh(self):
        """Compile with tanh activation."""
        @torch.compile(backend="eager")
        def f(x):
            return torch.tanh(x * 0.5)

        x = torch.randn(32)
        expected = torch.tanh(x * 0.5)
        result = f(to_vulkan(x))
        assert_close(result, expected)

    def test_compile_matmul(self):
        """Compile with matrix multiply."""
        @torch.compile(backend="eager")
        def f(a, b):
            return torch.mm(a, b)

        a = torch.randn(8, 16)
        b = torch.randn(16, 4)
        expected = torch.mm(a, b)
        result = f(to_vulkan(a), to_vulkan(b))
        assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_compile_softmax(self):
        """Compile with softmax."""
        @torch.compile(backend="eager")
        def f(x):
            return F.softmax(x, dim=-1)

        x = torch.randn(4, 10)
        expected = F.softmax(x, dim=-1)
        result = f(to_vulkan(x))
        assert_close(result, expected)

    def test_compile_multi_op(self):
        """Compile with multiple chained operations."""
        @torch.compile(backend="eager")
        def f(x, w):
            h = torch.mm(x, w)
            return torch.sigmoid(h).sum()

        x = torch.randn(4, 8)
        w = torch.randn(8, 16)
        expected = torch.sigmoid(torch.mm(x, w)).sum()
        result = f(to_vulkan(x), to_vulkan(w))
        assert_close(result, expected, rtol=1e-2, atol=1e-2)
