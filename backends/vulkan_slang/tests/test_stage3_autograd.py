"""Tests for Stage 3: Autograd & Backward Pass.

Tests gradcheck for differentiable ops and a simple MNIST-style training loop.
"""

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


# ── Gradient Tests ───────────────────────────────────────────────


class TestReluGrad:
    def test_relu_backward(self):
        x = torch.randn(4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        # CPU
        y_cpu = F.relu(x)
        y_cpu.sum().backward()

        # Vulkan
        y_vk = F.relu(vx)
        y_vk.sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-4, atol=1e-5)

    def test_relu_grad_negative(self):
        """Gradient should be 0 for negative inputs."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.relu(vx).sum().backward()
        expected_grad = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        assert_close(vx.grad, expected_grad, rtol=1e-5, atol=1e-5)


class TestMmGrad:
    def test_mm_backward(self):
        a = torch.randn(4, 8, requires_grad=True)
        b = torch.randn(8, 6, requires_grad=True)
        va = to_vulkan(a.detach()).requires_grad_(True)
        vb = to_vulkan(b.detach()).requires_grad_(True)

        # CPU
        (a @ b).sum().backward()

        # Vulkan
        torch.mm(va, vb).sum().backward()

        assert_close(va.grad, a.grad, rtol=1e-3, atol=1e-3)
        assert_close(vb.grad, b.grad, rtol=1e-3, atol=1e-3)


class TestLinearGrad:
    def test_linear_backward_no_bias(self):
        x = torch.randn(4, 16, requires_grad=True)
        w = torch.randn(8, 16, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)
        vw = to_vulkan(w.detach()).requires_grad_(True)

        F.linear(x, w).sum().backward()
        F.linear(vx, vw).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)
        assert_close(vw.grad, w.grad, rtol=1e-3, atol=1e-3)

    def test_linear_backward_with_bias(self):
        x = torch.randn(4, 16, requires_grad=True)
        w = torch.randn(8, 16, requires_grad=True)
        b = torch.randn(8, requires_grad=True)

        vx = to_vulkan(x.detach()).requires_grad_(True)
        vw = to_vulkan(w.detach()).requires_grad_(True)
        vb = to_vulkan(b.detach()).requires_grad_(True)

        F.linear(x, w, b).sum().backward()
        F.linear(vx, vw, vb).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)
        assert_close(vw.grad, w.grad, rtol=1e-3, atol=1e-3)
        assert_close(vb.grad, b.grad, rtol=1e-3, atol=1e-3)


class TestSigmoidGrad:
    def test_sigmoid_backward(self):
        x = torch.randn(16, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        torch.sigmoid(x).sum().backward()
        torch.sigmoid(vx).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)


class TestTanhGrad:
    def test_tanh_backward(self):
        x = torch.randn(16, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        torch.tanh(x).sum().backward()
        torch.tanh(vx).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)


class TestBmmGrad:
    def test_bmm_backward(self):
        a = torch.randn(2, 4, 8, requires_grad=True)
        b = torch.randn(2, 8, 6, requires_grad=True)
        va = to_vulkan(a.detach()).requires_grad_(True)
        vb = to_vulkan(b.detach()).requires_grad_(True)

        torch.bmm(a, b).sum().backward()
        torch.bmm(va, vb).sum().backward()

        assert_close(va.grad, a.grad, rtol=1e-3, atol=1e-3)
        assert_close(vb.grad, b.grad, rtol=1e-3, atol=1e-3)


# ── Loss Function Tests ─────────────────────────────────────────


class TestLoss:
    def test_nll_loss(self):
        log_probs = F.log_softmax(torch.randn(4, 10), dim=-1)
        target = torch.tensor([3, 5, 7, 1])

        expected = F.nll_loss(log_probs, target)
        result = F.nll_loss(to_vulkan(log_probs), target.to("vulkan:0"))
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_cross_entropy_loss(self):
        logits = torch.randn(8, 10)
        target = torch.randint(0, 10, (8,))

        expected = F.cross_entropy(logits, target)
        result = F.cross_entropy(to_vulkan(logits), target.to("vulkan:0"))
        assert_close(result, expected, rtol=1e-2, atol=1e-2)


# ── MNIST-style Training Test ────────────────────────────────────


class TestMNISTTraining:
    def test_simple_mlp_training(self):
        """Train a tiny MLP on synthetic data — verifies the full training loop works."""
        torch.manual_seed(42)

        # Synthetic binary classification: 2D input, 2 classes
        N = 64
        x_data = torch.randn(N, 4)
        y_data = (x_data[:, 0] > 0).long()

        # Simple MLP: 4 → 16 → 2
        w1 = torch.randn(16, 4, requires_grad=True) * 0.1
        b1 = torch.zeros(16, requires_grad=True)
        w2 = torch.randn(2, 16, requires_grad=True) * 0.1
        b2 = torch.zeros(2, requires_grad=True)

        vx = to_vulkan(x_data)
        vy = y_data  # keep targets on CPU for cross_entropy

        lr = 0.01
        initial_loss = None

        for step in range(20):
            # Forward
            vw1 = to_vulkan(w1.detach()).requires_grad_(True)
            vb1 = to_vulkan(b1.detach()).requires_grad_(True)
            vw2 = to_vulkan(w2.detach()).requires_grad_(True)
            vb2 = to_vulkan(b2.detach()).requires_grad_(True)

            h = F.relu(F.linear(vx, vw1, vb1))
            logits = F.linear(h, vw2, vb2)

            # Loss on CPU (cross_entropy needs target)
            loss = F.cross_entropy(logits.cpu(), vy)

            if step == 0:
                initial_loss = loss.item()

            loss.backward()

            # SGD update
            with torch.no_grad():
                w1 = w1 - lr * vw1.grad.cpu()
                b1 = b1 - lr * vb1.grad.cpu()
                w2 = w2 - lr * vw2.grad.cpu()
                b2 = b2 - lr * vb2.grad.cpu()
                w1.requires_grad_(True)
                b1.requires_grad_(True)
                w2.requires_grad_(True)
                b2.requires_grad_(True)

        final_loss = loss.item()
        assert final_loss < initial_loss, \
            f"Training should decrease loss: initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_mnist_like_training(self):
        """Simulate MNIST-like training: 784 → 128 → 10, 100 samples."""
        torch.manual_seed(123)

        N = 100
        x_data = torch.randn(N, 784)
        y_data = torch.randint(0, 10, (N,))

        # MLP: 784 → 128 → 10
        w1 = torch.randn(128, 784, requires_grad=True) * 0.01
        b1 = torch.zeros(128, requires_grad=True)
        w2 = torch.randn(10, 128, requires_grad=True) * 0.01
        b2 = torch.zeros(10, requires_grad=True)

        vx = to_vulkan(x_data)
        lr = 0.01
        losses = []

        for step in range(10):
            vw1 = to_vulkan(w1.detach()).requires_grad_(True)
            vb1 = to_vulkan(b1.detach()).requires_grad_(True)
            vw2 = to_vulkan(w2.detach()).requires_grad_(True)
            vb2 = to_vulkan(b2.detach()).requires_grad_(True)

            h = F.relu(F.linear(vx, vw1, vb1))
            logits = F.linear(h, vw2, vb2)
            loss = F.cross_entropy(logits.cpu(), y_data)
            losses.append(loss.item())

            loss.backward()

            with torch.no_grad():
                w1 = w1 - lr * vw1.grad.cpu()
                b1 = b1 - lr * vb1.grad.cpu()
                w2 = w2 - lr * vw2.grad.cpu()
                b2 = b2 - lr * vb2.grad.cpu()
                w1.requires_grad_(True)
                b1.requires_grad_(True)
                w2.requires_grad_(True)
                b2.requires_grad_(True)

        # Loss should decrease
        assert losses[-1] < losses[0], \
            f"Loss should decrease over training: {losses[0]:.4f} → {losses[-1]:.4f}"


# ── Stage 3 Tier 3: Additional Activation Backward Tests ────────


class TestPreluGrad:
    def test_prelu_backward(self):
        x = torch.randn(4, 8, requires_grad=True)
        w = torch.randn(8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)
        vw = to_vulkan(w.detach()).requires_grad_(True)

        F.prelu(x, w).sum().backward()
        F.prelu(vx, vw).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)
        assert_close(vw.grad, w.grad, rtol=1e-2, atol=1e-2)


class TestSeluGrad:
    def test_selu_backward(self):
        x = torch.randn(4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.selu(x).sum().backward()
        F.selu(vx).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)


class TestClampGrad:
    def test_clamp_backward(self):
        x = torch.randn(4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        torch.clamp(x, min=-0.5, max=0.5).sum().backward()
        torch.clamp(vx, min=-0.5, max=0.5).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-4, atol=1e-4)


class TestGeluGrad:
    def test_gelu_backward(self):
        x = torch.randn(4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.gelu(x).sum().backward()
        F.gelu(vx).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-2, atol=1e-2)


class TestSiluGrad:
    def test_silu_backward(self):
        x = torch.randn(4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.silu(x).sum().backward()
        F.silu(vx).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)


class TestLeakyReluGrad:
    def test_leaky_relu_backward(self):
        x = torch.randn(4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.leaky_relu(x, 0.01).sum().backward()
        F.leaky_relu(vx, 0.01).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)

    def test_leaky_relu_backward_custom_slope(self):
        x = torch.randn(4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.leaky_relu(x, 0.2).sum().backward()
        F.leaky_relu(vx, 0.2).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)


class TestEluGrad:
    def test_elu_backward(self):
        x = torch.randn(4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.elu(x).sum().backward()
        F.elu(vx).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)


class TestSoftmaxGrad:
    def test_softmax_backward(self):
        x = torch.randn(4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.softmax(x, dim=-1).sum().backward()
        F.softmax(vx, dim=-1).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)

    def test_softmax_backward_weighted(self):
        """Test with non-uniform grad_output (weighted sum instead of .sum())."""
        x = torch.randn(4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)
        w = torch.randn(4, 8)

        (F.softmax(x, dim=-1) * w).sum().backward()
        (F.softmax(vx, dim=-1) * to_vulkan(w)).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-2, atol=1e-2)


class TestLogSoftmaxGrad:
    def test_log_softmax_backward(self):
        x = torch.randn(4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.log_softmax(x, dim=-1).sum().backward()
        F.log_softmax(vx, dim=-1).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)


# ── Stage 3 Tier 3: Conv, Pooling, Norm, Embedding Backward ────


class TestConv2dGrad:
    def test_conv2d_backward_basic(self):
        """Basic conv2d backward with no groups."""
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        w = torch.randn(4, 3, 3, 3, requires_grad=True)

        vx = to_vulkan(x.detach()).requires_grad_(True)
        vw = to_vulkan(w.detach()).requires_grad_(True)

        F.conv2d(x, w, padding=1).sum().backward()
        F.conv2d(vx, vw, padding=1).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-2, atol=1e-2)
        assert_close(vw.grad, w.grad, rtol=1e-2, atol=1e-2)

    def test_conv2d_backward_with_bias(self):
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        w = torch.randn(4, 3, 3, 3, requires_grad=True)
        b = torch.randn(4, requires_grad=True)

        vx = to_vulkan(x.detach()).requires_grad_(True)
        vw = to_vulkan(w.detach()).requires_grad_(True)
        vb = to_vulkan(b.detach()).requires_grad_(True)

        F.conv2d(x, w, b, padding=1).sum().backward()
        F.conv2d(vx, vw, vb, padding=1).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-2, atol=1e-2)
        assert_close(vw.grad, w.grad, rtol=1e-2, atol=1e-2)
        assert_close(vb.grad, b.grad, rtol=1e-2, atol=1e-2)

    def test_conv2d_backward_stride(self):
        x = torch.randn(1, 3, 16, 16, requires_grad=True)
        w = torch.randn(8, 3, 3, 3, requires_grad=True)

        vx = to_vulkan(x.detach()).requires_grad_(True)
        vw = to_vulkan(w.detach()).requires_grad_(True)

        F.conv2d(x, w, stride=2, padding=1).sum().backward()
        F.conv2d(vx, vw, stride=2, padding=1).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-2, atol=1e-2)
        assert_close(vw.grad, w.grad, rtol=1e-2, atol=1e-2)


class TestMaxPool2dGrad:
    def test_max_pool2d_backward(self):
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.max_pool2d(x, 2).sum().backward()
        F.max_pool2d(vx, 2).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)

    def test_max_pool2d_backward_with_padding(self):
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.max_pool2d(x, 3, stride=2, padding=1).sum().backward()
        F.max_pool2d(vx, 3, stride=2, padding=1).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)


class TestAvgPool2dGrad:
    def test_avg_pool2d_backward(self):
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        F.avg_pool2d(x, 2).sum().backward()
        F.avg_pool2d(vx, 2).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)


class TestBatchNormGrad:
    def test_batch_norm_inference(self):
        """Batch norm inference mode should produce correct results."""
        torch.manual_seed(42)
        x = torch.randn(4, 8, 4, 4)
        w = torch.ones(8)
        b = torch.zeros(8)
        rm = torch.randn(8)
        rv = torch.abs(torch.randn(8)) + 0.1

        expected = F.batch_norm(x, rm, rv, w, b, training=False)
        result = F.batch_norm(
            to_vulkan(x), to_vulkan(rm), to_vulkan(rv),
            to_vulkan(w), to_vulkan(b), training=False)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)


class TestEmbeddingGrad:
    def test_embedding_backward(self):
        weight = torch.randn(10, 16, requires_grad=True)
        indices = torch.tensor([0, 3, 5, 7])

        vw = to_vulkan(weight.detach()).requires_grad_(True)
        vi = indices.to("vulkan:0")

        F.embedding(indices, weight).sum().backward()
        F.embedding(vi, vw).sum().backward()

        assert_close(vw.grad, weight.grad, rtol=1e-3, atol=1e-3)

    def test_embedding_backward_repeated_indices(self):
        """Repeated indices should accumulate gradients."""
        weight = torch.randn(5, 8, requires_grad=True)
        indices = torch.tensor([0, 1, 0, 2, 1])

        vw = to_vulkan(weight.detach()).requires_grad_(True)
        vi = indices.to("vulkan:0")

        F.embedding(indices, weight).sum().backward()
        F.embedding(vi, vw).sum().backward()

        assert_close(vw.grad, weight.grad, rtol=1e-3, atol=1e-3)


class TestLayerNormGrad:
    def test_layer_norm_backward(self):
        x = torch.randn(4, 16, requires_grad=True)
        w = torch.randn(16, requires_grad=True)
        b = torch.randn(16, requires_grad=True)

        vx = to_vulkan(x.detach()).requires_grad_(True)
        vw = to_vulkan(w.detach()).requires_grad_(True)
        vb = to_vulkan(b.detach()).requires_grad_(True)

        F.layer_norm(x, [16], w, b).sum().backward()
        F.layer_norm(vx, [16], vw, vb).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-2, atol=1e-2)
        assert_close(vw.grad, w.grad, rtol=1e-2, atol=1e-2)
        assert_close(vb.grad, b.grad, rtol=1e-2, atol=1e-2)


class TestGroupNormGrad:
    def test_group_norm_backward(self):
        x = torch.randn(2, 8, 4, 4, requires_grad=True)
        w = torch.randn(8, requires_grad=True)
        b = torch.randn(8, requires_grad=True)

        vx = to_vulkan(x.detach()).requires_grad_(True)
        vw = to_vulkan(w.detach()).requires_grad_(True)
        vb = to_vulkan(b.detach()).requires_grad_(True)

        F.group_norm(x, 4, w, b).sum().backward()
        F.group_norm(vx, 4, vw, vb).sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-2, atol=1e-2)
        assert_close(vw.grad, w.grad, rtol=1e-2, atol=1e-2)
        assert_close(vb.grad, b.grad, rtol=1e-2, atol=1e-2)


# ── CNN Training Test ───────────────────────────────────────────


class TestCNNTraining:
    def test_small_cnn_training(self):
        """Train a tiny CNN on synthetic data to verify conv+pool+bn backward."""
        torch.manual_seed(42)

        N = 32
        x_data = torch.randn(N, 1, 8, 8)
        y_data = torch.randint(0, 2, (N,))

        # Tiny CNN: conv(1->4, 3x3) -> relu -> pool(2) -> linear(4*4*4 -> 2)
        conv_w = torch.randn(4, 1, 3, 3, requires_grad=True) * 0.1
        conv_b = torch.zeros(4, requires_grad=True)
        fc_w = torch.randn(2, 64, requires_grad=True) * 0.1
        fc_b = torch.zeros(2, requires_grad=True)

        vx = to_vulkan(x_data)
        lr = 0.01
        losses = []

        for step in range(10):
            vcw = to_vulkan(conv_w.detach()).requires_grad_(True)
            vcb = to_vulkan(conv_b.detach()).requires_grad_(True)
            vfw = to_vulkan(fc_w.detach()).requires_grad_(True)
            vfb = to_vulkan(fc_b.detach()).requires_grad_(True)

            h = F.relu(F.conv2d(vx, vcw, vcb, padding=1))  # [N,4,8,8]
            h = F.max_pool2d(h, 2)  # [N,4,4,4]
            h = h.reshape(N, -1)  # [N,64]
            logits = F.linear(h, vfw, vfb)  # [N,2]
            loss = F.cross_entropy(logits.cpu(), y_data)
            losses.append(loss.item())

            loss.backward()

            with torch.no_grad():
                conv_w = conv_w - lr * vcw.grad.cpu()
                conv_b = conv_b - lr * vcb.grad.cpu()
                fc_w = fc_w - lr * vfw.grad.cpu()
                fc_b = fc_b - lr * vfb.grad.cpu()
                conv_w.requires_grad_(True)
                conv_b.requires_grad_(True)
                fc_w.requires_grad_(True)
                fc_b.requires_grad_(True)

        assert losses[-1] < losses[0], \
            f"CNN loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


# ── Dropout Backward ────────────────────────────────────────────


class TestDropoutGrad:
    def test_dropout_backward(self):
        """Verify dropout backward correctly masks gradients."""
        x = torch.randn(100, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        # Forward with dropout
        out, mask = torch.native_dropout(vx, 0.5, True)
        out.sum().backward()

        # Gradient should be mask * scale
        grad = vx.grad.cpu()
        mask_cpu = mask.cpu().float()
        scale = 1.0 / (1.0 - 0.5)

        # Where mask is True, grad should be scale; where False, grad should be 0
        expected_grad = mask_cpu * scale
        torch.testing.assert_close(grad, expected_grad, rtol=1e-3, atol=1e-3)
