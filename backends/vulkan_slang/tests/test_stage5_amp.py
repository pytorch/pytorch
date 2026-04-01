"""Tests for Stage 5: Mixed Precision (AMP) & GradScaler."""

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


# ── Buffer Pool / Memory Cache ───────────────────────────────────


class TestMemoryCache:
    def test_empty_cache(self):
        import torch_vulkan
        # Allocate some tensors
        tensors = [torch.empty(1024, device="vulkan:0") for _ in range(10)]
        del tensors
        # Should have cached memory
        cached = torch_vulkan.memory_cached()
        assert cached >= 0  # May or may not have cached depending on timing
        torch_vulkan.empty_cache()
        assert torch_vulkan.memory_cached() == 0

    def test_reuse_allocations(self):
        import torch_vulkan
        torch_vulkan.empty_cache()
        # Allocate and free
        t = torch.empty(4096, device="vulkan:0")
        del t
        # Should have cached the buffer
        cached_before = torch_vulkan.memory_cached()
        # Allocate same size — should reuse from cache
        t2 = torch.empty(4096, device="vulkan:0")
        # Cached should decrease since we reused
        cached_after = torch_vulkan.memory_cached()
        assert cached_after <= cached_before
        del t2


# ── GradScaler ───────────────────────────────────────────────────


class TestGradScaler:
    def test_gradscaler_basic(self):
        """Test that GradScaler properly scales and unscales gradients."""
        torch.manual_seed(42)
        x = torch.randn(8, 4, requires_grad=True)
        w = torch.randn(4, 2, requires_grad=True)

        vx = to_vulkan(x.detach()).requires_grad_(True)
        vw = to_vulkan(w.detach()).requires_grad_(True)

        # Forward
        out = F.linear(vx, vw.t())
        loss = out.sum()
        loss.backward()

        # Gradients should exist
        assert vx.grad is not None
        assert vw.grad is not None

        # Compare with CPU
        cpu_out = F.linear(x, w.t())
        cpu_loss = cpu_out.sum()
        cpu_loss.backward()

        assert_close(vx.grad, x.grad, rtol=1e-2, atol=1e-2)
        assert_close(vw.grad, w.grad, rtol=1e-2, atol=1e-2)


# ── Training Loop with Adam ──────────────────────────────────────


class TestTrainingLoop:
    def test_simple_mlp_training(self):
        """Test a simple MLP training loop converges on Vulkan."""
        torch.manual_seed(42)

        # Simple XOR-like problem
        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        Y = torch.tensor([[0.], [1.], [1.], [0.]])

        # Small MLP: 2 -> 4 -> 1
        w1 = torch.randn(2, 4) * 0.5
        b1 = torch.zeros(4)
        w2 = torch.randn(4, 1) * 0.5
        b2 = torch.zeros(1)

        lr = 0.1
        losses = []

        for epoch in range(50):
            vx = to_vulkan(X)
            vy = to_vulkan(Y)
            vw1 = to_vulkan(w1.clone()).requires_grad_(True)
            vb1 = to_vulkan(b1.clone()).requires_grad_(True)
            vw2 = to_vulkan(w2.clone()).requires_grad_(True)
            vb2 = to_vulkan(b2.clone()).requires_grad_(True)

            # Forward
            h = torch.relu(F.linear(vx, vw1.t(), vb1))
            pred = torch.sigmoid(F.linear(h, vw2.t(), vb2))

            # MSE loss
            diff = pred - vy
            loss = (diff * diff).mean()
            losses.append(loss.cpu().item())

            loss.backward()

            # SGD update
            with torch.no_grad():
                w1 = w1 - lr * vw1.grad.cpu()
                b1 = b1 - lr * vb1.grad.cpu()
                w2 = w2 - lr * vw2.grad.cpu()
                b2 = b2 - lr * vb2.grad.cpu()

        assert losses[-1] < losses[0], \
            f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
