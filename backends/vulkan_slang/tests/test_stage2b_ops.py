"""Tests for Stage 2b ops: softmax, normalization, pooling, indexing, bmm, linear, conv2d."""

import torch
import torch.nn.functional as F
import pytest

RTOL = 1e-3
ATOL = 1e-4


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


# ── Softmax ──────────────────────────────────────────────────────


class TestSoftmax:
    def test_softmax_1d(self):
        x = torch.randn(64)
        result = F.softmax(to_vulkan(x), dim=0)
        expected = F.softmax(x, dim=0)
        assert_close(result, expected)

    def test_softmax_2d_last_dim(self):
        x = torch.randn(8, 32)
        result = F.softmax(to_vulkan(x), dim=-1)
        expected = F.softmax(x, dim=-1)
        assert_close(result, expected)

    def test_softmax_2d_first_dim(self):
        x = torch.randn(32, 16)
        result = F.softmax(to_vulkan(x), dim=0)
        expected = F.softmax(x, dim=0)
        assert_close(result, expected)

    def test_softmax_sums_to_one(self):
        x = torch.randn(4, 128)
        result = F.softmax(to_vulkan(x), dim=-1).cpu()
        sums = result.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4), rtol=1e-3, atol=1e-3)

    def test_softmax_3d(self):
        x = torch.randn(2, 4, 64)
        result = F.softmax(to_vulkan(x), dim=-1)
        expected = F.softmax(x, dim=-1)
        assert_close(result, expected)


class TestLogSoftmax:
    def test_log_softmax_1d(self):
        x = torch.randn(64)
        result = F.log_softmax(to_vulkan(x), dim=0)
        expected = F.log_softmax(x, dim=0)
        assert_close(result, expected)

    def test_log_softmax_2d(self):
        x = torch.randn(8, 128)
        result = F.log_softmax(to_vulkan(x), dim=-1)
        expected = F.log_softmax(x, dim=-1)
        assert_close(result, expected)

    def test_log_softmax_consistency(self):
        """log_softmax should equal log(softmax)."""
        x = torch.randn(4, 64)
        vx = to_vulkan(x)
        log_sm = F.log_softmax(vx, dim=-1).cpu()
        sm = F.softmax(vx, dim=-1).cpu()
        torch.testing.assert_close(log_sm, torch.log(sm), rtol=1e-3, atol=1e-3)


# ── Layer Norm ───────────────────────────────────────────────────


class TestLayerNorm:
    def test_layer_norm_basic(self):
        x = torch.randn(4, 64)
        ln = torch.nn.LayerNorm(64)
        weight = ln.weight.data
        bias = ln.bias.data

        result = F.layer_norm(to_vulkan(x), [64],
                              weight=to_vulkan(weight),
                              bias=to_vulkan(bias))
        expected = F.layer_norm(x, [64], weight=weight, bias=bias)
        assert_close(result, expected, rtol=1e-2, atol=1e-3)

    def test_layer_norm_no_affine(self):
        x = torch.randn(4, 128)
        result = F.layer_norm(to_vulkan(x), [128])
        expected = F.layer_norm(x, [128])
        assert_close(result, expected, rtol=1e-2, atol=1e-3)

    def test_layer_norm_normalized_stats(self):
        """Output should have mean ~0, std ~1 along normalized dim."""
        x = torch.randn(8, 64)
        result = F.layer_norm(to_vulkan(x), [64]).cpu()
        assert result.mean(dim=-1).abs().max() < 0.1
        assert (result.std(dim=-1) - 1.0).abs().max() < 0.2


# ── Batch Norm ───────────────────────────────────────────────────


class TestBatchNorm:
    def test_batch_norm_eval(self):
        x = torch.randn(2, 3, 4, 4)
        bn = torch.nn.BatchNorm2d(3)
        bn.eval()

        result = F.batch_norm(to_vulkan(x),
                              to_vulkan(bn.running_mean),
                              to_vulkan(bn.running_var),
                              to_vulkan(bn.weight),
                              to_vulkan(bn.bias),
                              training=False, momentum=0.1, eps=1e-5)
        expected = F.batch_norm(x, bn.running_mean, bn.running_var,
                                bn.weight, bn.bias,
                                training=False, momentum=0.1, eps=1e-5)
        assert_close(result, expected)

    def test_batch_norm_no_affine(self):
        x = torch.randn(2, 4, 8, 8)
        running_mean = torch.zeros(4)
        running_var = torch.ones(4)

        result = F.batch_norm(to_vulkan(x),
                              to_vulkan(running_mean),
                              to_vulkan(running_var),
                              training=False)
        expected = F.batch_norm(x, running_mean, running_var, training=False)
        assert_close(result, expected)


# ── Pooling ──────────────────────────────────────────────────────


class TestMaxPool2d:
    def test_max_pool2d_basic(self):
        x = torch.randn(1, 1, 8, 8)
        result = F.max_pool2d(to_vulkan(x), kernel_size=2)
        expected = F.max_pool2d(x, kernel_size=2)
        assert_close(result, expected)
        assert result.shape == expected.shape

    def test_max_pool2d_stride_padding(self):
        x = torch.randn(2, 3, 16, 16)
        result = F.max_pool2d(to_vulkan(x), kernel_size=3, stride=2, padding=1)
        expected = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        assert_close(result, expected)

    def test_max_pool2d_non_square(self):
        x = torch.randn(1, 2, 12, 8)
        result = F.max_pool2d(to_vulkan(x), kernel_size=(3, 2), stride=(2, 1))
        expected = F.max_pool2d(x, kernel_size=(3, 2), stride=(2, 1))
        assert_close(result, expected)


class TestAvgPool2d:
    def test_avg_pool2d_basic(self):
        x = torch.randn(1, 1, 8, 8)
        result = F.avg_pool2d(to_vulkan(x), kernel_size=2)
        expected = F.avg_pool2d(x, kernel_size=2)
        assert_close(result, expected)

    def test_avg_pool2d_stride_padding(self):
        x = torch.randn(2, 3, 16, 16)
        result = F.avg_pool2d(to_vulkan(x), kernel_size=3, stride=2, padding=1)
        expected = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        assert_close(result, expected)


class TestAdaptiveAvgPool2d:
    def test_adaptive_avg_pool2d_divisible(self):
        x = torch.randn(1, 3, 8, 8)
        result = F.adaptive_avg_pool2d(to_vulkan(x), (4, 4))
        expected = F.adaptive_avg_pool2d(x, (4, 4))
        assert_close(result, expected)

    def test_adaptive_avg_pool2d_to_1x1(self):
        x = torch.randn(2, 16, 8, 8)
        result = F.adaptive_avg_pool2d(to_vulkan(x), (1, 1))
        expected = F.adaptive_avg_pool2d(x, (1, 1))
        assert_close(result, expected)


# ── Embedding ────────────────────────────────────────────────────


class TestEmbedding:
    def test_embedding_basic(self):
        weight = torch.randn(10, 32)
        indices = torch.tensor([0, 3, 5, 7])
        result = F.embedding(indices, to_vulkan(weight))
        expected = F.embedding(indices, weight)
        assert_close(result, expected)

    def test_embedding_2d_indices(self):
        weight = torch.randn(20, 16)
        indices = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = F.embedding(indices, to_vulkan(weight))
        expected = F.embedding(indices, weight)
        assert_close(result, expected)

    def test_embedding_large(self):
        weight = torch.randn(1000, 64)
        indices = torch.randint(0, 1000, (8, 32))
        result = F.embedding(indices, to_vulkan(weight))
        expected = F.embedding(indices, weight)
        assert_close(result, expected)


# ── Index Select ─────────────────────────────────────────────────


class TestIndexSelect:
    def test_index_select_dim0(self):
        x = torch.randn(10, 8)
        idx = torch.tensor([0, 3, 7])
        result = torch.index_select(to_vulkan(x), 0, idx.to("vulkan:0"))
        expected = torch.index_select(x, 0, idx)
        assert_close(result, expected)

    def test_index_select_dim0_3d(self):
        x = torch.randn(8, 4, 16)
        idx = torch.tensor([1, 5, 3])
        result = torch.index_select(to_vulkan(x), 0, idx.to("vulkan:0"))
        expected = torch.index_select(x, 0, idx)
        assert_close(result, expected)


# ── Masked Fill ──────────────────────────────────────────────────


class TestMaskedFill:
    def test_masked_fill_basic(self):
        x = torch.randn(4, 4)
        mask = torch.tensor([[True, False, False, True],
                              [False, True, False, False],
                              [True, True, False, False],
                              [False, False, True, True]])
        vx = to_vulkan(x.clone())
        vx.masked_fill_(mask.to("vulkan:0"), -999.0)
        expected = x.clone()
        expected.masked_fill_(mask, -999.0)
        assert_close(vx, expected)

    def test_masked_fill_all_true(self):
        x = torch.randn(8)
        mask = torch.ones(8, dtype=torch.bool)
        vx = to_vulkan(x.clone())
        vx.masked_fill_(mask.to("vulkan:0"), 42.0)
        expected = torch.full((8,), 42.0)
        assert_close(vx, expected)


# ── BMM ──────────────────────────────────────────────────────────


class TestBMM:
    def test_bmm_basic(self):
        a = torch.randn(4, 8, 16)
        b = torch.randn(4, 16, 12)
        result = torch.bmm(to_vulkan(a), to_vulkan(b))
        expected = torch.bmm(a, b)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_bmm_square(self):
        a = torch.randn(2, 32, 32)
        b = torch.randn(2, 32, 32)
        result = torch.bmm(to_vulkan(a), to_vulkan(b))
        expected = torch.bmm(a, b)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_bmm_single_batch(self):
        a = torch.randn(1, 16, 8)
        b = torch.randn(1, 8, 4)
        result = torch.bmm(to_vulkan(a), to_vulkan(b))
        expected = torch.bmm(a, b)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)


# ── Linear ───────────────────────────────────────────────────────


class TestLinear:
    def test_linear_no_bias(self):
        x = torch.randn(4, 16)
        w = torch.randn(8, 16)
        result = F.linear(to_vulkan(x), to_vulkan(w))
        expected = F.linear(x, w)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_linear_with_bias(self):
        x = torch.randn(4, 16)
        w = torch.randn(8, 16)
        b = torch.randn(8)
        result = F.linear(to_vulkan(x), to_vulkan(w), to_vulkan(b))
        expected = F.linear(x, w, b)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_linear_batched(self):
        x = torch.randn(2, 4, 16)
        w = torch.randn(8, 16)
        b = torch.randn(8)
        result = F.linear(to_vulkan(x), to_vulkan(w), to_vulkan(b))
        expected = F.linear(x, w, b)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_nn_linear(self):
        """Test via nn.Linear module."""
        layer = torch.nn.Linear(32, 16)
        x = torch.randn(4, 32)
        expected = layer(x)
        # Move weights to vulkan
        result = F.linear(to_vulkan(x), to_vulkan(layer.weight),
                          to_vulkan(layer.bias))
        assert_close(result, expected, rtol=1e-3, atol=1e-3)


# ── Conv2D ───────────────────────────────────────────────────────


class TestConv2d:
    def test_conv2d_basic(self):
        x = torch.randn(1, 1, 8, 8)
        w = torch.randn(1, 1, 3, 3)
        result = F.conv2d(to_vulkan(x), to_vulkan(w))
        expected = F.conv2d(x, w)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)
        assert result.shape == expected.shape

    def test_conv2d_with_bias(self):
        x = torch.randn(1, 3, 16, 16)
        w = torch.randn(8, 3, 3, 3)
        b = torch.randn(8)
        result = F.conv2d(to_vulkan(x), to_vulkan(w), to_vulkan(b))
        expected = F.conv2d(x, w, b)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_conv2d_stride(self):
        x = torch.randn(1, 3, 16, 16)
        w = torch.randn(4, 3, 3, 3)
        result = F.conv2d(to_vulkan(x), to_vulkan(w), stride=2)
        expected = F.conv2d(x, w, stride=2)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_conv2d_padding(self):
        x = torch.randn(2, 3, 8, 8)
        w = torch.randn(4, 3, 3, 3)
        result = F.conv2d(to_vulkan(x), to_vulkan(w), padding=1)
        expected = F.conv2d(x, w, padding=1)
        assert result.shape == expected.shape
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_conv2d_groups(self):
        x = torch.randn(1, 4, 8, 8)
        w = torch.randn(4, 2, 3, 3)  # groups=2: 4 out, 4/2=2 in per group
        result = F.conv2d(to_vulkan(x), to_vulkan(w), groups=2)
        expected = F.conv2d(x, w, groups=2)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_conv2d_depthwise(self):
        """Depthwise conv: groups == C_in."""
        x = torch.randn(1, 8, 16, 16)
        w = torch.randn(8, 1, 3, 3)  # groups=8
        result = F.conv2d(to_vulkan(x), to_vulkan(w), groups=8)
        expected = F.conv2d(x, w, groups=8)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_conv2d_1x1(self):
        x = torch.randn(2, 16, 8, 8)
        w = torch.randn(32, 16, 1, 1)
        result = F.conv2d(to_vulkan(x), to_vulkan(w))
        expected = F.conv2d(x, w)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_nn_conv2d(self):
        """Test via nn.Conv2d module."""
        conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        x = torch.randn(1, 3, 8, 8)
        expected = conv(x)
        result = F.conv2d(to_vulkan(x), to_vulkan(conv.weight),
                          to_vulkan(conv.bias), padding=1)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)


# ── Integration: Mini Models ─────────────────────────────────────


class TestMiniModels:
    def test_conv_relu_pool(self):
        """Conv2d → ReLU → MaxPool2d pipeline."""
        x = torch.randn(1, 3, 16, 16)
        w = torch.randn(8, 3, 3, 3)
        b = torch.randn(8)

        # CPU
        expected = F.max_pool2d(F.relu(F.conv2d(x, w, b, padding=1)), 2)

        # Vulkan
        vx = to_vulkan(x)
        vw = to_vulkan(w)
        vb = to_vulkan(b)
        result = F.max_pool2d(F.relu(F.conv2d(vx, vw, vb, padding=1)), 2)

        assert result.shape == expected.shape
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_linear_relu_linear(self):
        """Linear → ReLU → Linear pipeline."""
        x = torch.randn(4, 32)
        w1 = torch.randn(16, 32)
        b1 = torch.randn(16)
        w2 = torch.randn(8, 16)
        b2 = torch.randn(8)

        expected = F.linear(F.relu(F.linear(x, w1, b1)), w2, b2)

        vx = to_vulkan(x)
        result = F.linear(F.relu(F.linear(vx, to_vulkan(w1), to_vulkan(b1))),
                          to_vulkan(w2), to_vulkan(b2))
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_embedding_layernorm(self):
        """Embedding → LayerNorm pipeline."""
        weight = torch.randn(100, 64)
        indices = torch.tensor([5, 10, 20, 50])

        x = F.embedding(indices, weight)
        expected = F.layer_norm(x, [64])

        vx = F.embedding(indices, to_vulkan(weight))
        result = F.layer_norm(vx, [64])
        assert_close(result, expected, rtol=1e-2, atol=1e-3)


# ── SELU ─────────────────────────────────────────────────────────


class TestSELU:
    def test_selu_basic(self):
        x = torch.randn(64)
        result = F.selu(to_vulkan(x))
        expected = F.selu(x)
        assert_close(result, expected)

    def test_selu_negative(self):
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = F.selu(to_vulkan(x))
        expected = F.selu(x)
        assert_close(result, expected)


# ── PReLU ────────────────────────────────────────────────────────


class TestPReLU:
    def test_prelu_scalar_weight(self):
        x = torch.randn(32)
        w = torch.tensor([0.25])
        result = F.prelu(to_vulkan(x), to_vulkan(w))
        expected = F.prelu(x, w)
        assert_close(result, expected)

    def test_prelu_per_channel(self):
        x = torch.randn(1, 4, 8, 8)
        w = torch.randn(4)
        result = F.prelu(to_vulkan(x), to_vulkan(w))
        expected = F.prelu(x, w)
        assert_close(result, expected)


# ── Group Norm ───────────────────────────────────────────────────


class TestGroupNorm:
    def test_group_norm_basic(self):
        """Small group_size that fits within 256 threads."""
        x = torch.randn(2, 4, 2, 2)  # group_size = (4/2) * 2 * 2 = 8
        gn = torch.nn.GroupNorm(2, 4)
        expected = gn(x)
        result = F.group_norm(to_vulkan(x), 2,
                              to_vulkan(gn.weight), to_vulkan(gn.bias))
        assert_close(result, expected, rtol=1e-2, atol=1e-3)


# ── Prod ─────────────────────────────────────────────────────────


class TestProd:
    def test_prod_dim(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.prod(to_vulkan(x), dim=1)
        expected = torch.prod(x, dim=1)
        assert_close(result, expected)

    def test_prod_dim_keepdim(self):
        x = torch.randn(4, 8).abs() + 0.1  # positive to avoid huge products
        result = torch.prod(to_vulkan(x), dim=0, keepdim=True)
        expected = torch.prod(x, dim=0, keepdim=True)
        assert_close(result, expected, rtol=1e-2, atol=1e-2)


# ── Argmax / Argmin ──────────────────────────────────────────────


class TestArgmaxArgmin:
    def test_argmax_global(self):
        x = torch.randn(32)
        result = torch.argmax(to_vulkan(x))
        expected = torch.argmax(x)
        assert result.cpu().item() == expected.item()

    def test_argmax_dim(self):
        x = torch.randn(4, 8)
        result = torch.argmax(to_vulkan(x), dim=1)
        expected = torch.argmax(x, dim=1)
        assert result.cpu().equal(expected)

    def test_argmin_global(self):
        x = torch.randn(32)
        result = torch.argmin(to_vulkan(x))
        expected = torch.argmin(x)
        assert result.cpu().item() == expected.item()

    def test_argmin_dim(self):
        x = torch.randn(4, 8)
        result = torch.argmin(to_vulkan(x), dim=1)
        expected = torch.argmin(x, dim=1)
        assert result.cpu().equal(expected)
