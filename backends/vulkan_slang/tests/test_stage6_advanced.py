"""Tests for Stage 6: Advanced Operators & Model Coverage."""

import torch
import torch.nn.functional as F
import pytest
import math

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


# ── Scaled Dot-Product Attention ─────────────────────────────────


class TestSDPA:
    def test_sdpa_basic(self):
        B, H, N, D = 1, 1, 8, 16
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)

        expected = F.scaled_dot_product_attention(q, k, v)
        result = F.scaled_dot_product_attention(
            to_vulkan(q), to_vulkan(k), to_vulkan(v))
        assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_sdpa_multi_head(self):
        B, H, N, D = 2, 4, 16, 32
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)

        expected = F.scaled_dot_product_attention(q, k, v)
        result = F.scaled_dot_product_attention(
            to_vulkan(q), to_vulkan(k), to_vulkan(v))
        assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_sdpa_causal(self):
        """Causal attention should match CPU reference."""
        B, H, N, D = 1, 2, 8, 16
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)

        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        result = F.scaled_dot_product_attention(
            to_vulkan(q), to_vulkan(k), to_vulkan(v), is_causal=True)
        assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_sdpa_custom_scale(self):
        B, H, N, D = 1, 2, 8, 16
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)

        scale = 0.1
        expected = F.scaled_dot_product_attention(q, k, v, scale=scale)
        result = F.scaled_dot_product_attention(
            to_vulkan(q), to_vulkan(k), to_vulkan(v), scale=scale)
        assert_close(result, expected, rtol=1e-2, atol=1e-2)


# ── Cumulative Sum ───────────────────────────────────────────────


class TestCumsum:
    def test_cumsum_1d(self):
        x = torch.randn(64)
        result = torch.cumsum(to_vulkan(x), dim=0)
        expected = torch.cumsum(x, dim=0)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_cumsum_2d_last_dim(self):
        x = torch.randn(4, 32)
        result = torch.cumsum(to_vulkan(x), dim=-1)
        expected = torch.cumsum(x, dim=-1)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_cumsum_2d_first_dim(self):
        x = torch.randn(16, 8)
        result = torch.cumsum(to_vulkan(x), dim=0)
        expected = torch.cumsum(x, dim=0)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)


# ── Sort / TopK ──────────────────────────────────────────────────


class TestSortTopK:
    def test_sort(self):
        x = torch.randn(4, 8)
        values, indices = torch.sort(to_vulkan(x), dim=-1)
        expected_values, expected_indices = torch.sort(x, dim=-1)
        assert_close(values, expected_values)
        assert indices.cpu().equal(expected_indices)

    def test_topk(self):
        x = torch.randn(4, 32)
        values, indices = torch.topk(to_vulkan(x), k=5, dim=-1)
        expected_values, expected_indices = torch.topk(x, k=5, dim=-1)
        assert_close(values, expected_values)
        assert indices.cpu().equal(expected_indices)


# ── Gather / Scatter ─────────────────────────────────────────────


class TestGatherScatter:
    def test_gather(self):
        x = torch.randn(4, 8)
        idx = torch.tensor([[0, 2, 4], [1, 3, 5], [0, 1, 2], [6, 7, 0]])
        result = torch.gather(to_vulkan(x), 1, idx.to("vulkan:0"))
        expected = torch.gather(x, 1, idx)
        assert_close(result, expected)

    def test_scatter(self):
        x = torch.zeros(4, 8)
        idx = torch.tensor([[0], [2], [1], [3]])
        src = torch.ones(4, 1)

        vx = to_vulkan(x.clone())
        vx.scatter_(1, idx.to("vulkan:0"), src.to("vulkan:0"))
        expected = x.clone().scatter_(1, idx, src)
        assert_close(vx, expected)


# ── Integration: Transformer Block ──────────────────────────────


class TestTransformerBlock:
    def test_attention_block(self):
        """Q/K/V projection → SDPA → output projection."""
        torch.manual_seed(42)
        B, N, D = 1, 8, 32
        num_heads = 4
        head_dim = D // num_heads

        x = torch.randn(B, N, D)
        wq = torch.randn(D, D) * 0.1
        wk = torch.randn(D, D) * 0.1
        wv = torch.randn(D, D) * 0.1
        wo = torch.randn(D, D) * 0.1

        def run(x, wq, wk, wv, wo):
            q = F.linear(x, wq).reshape(B, N, num_heads, head_dim).transpose(1, 2)
            k = F.linear(x, wk).reshape(B, N, num_heads, head_dim).transpose(1, 2)
            v = F.linear(x, wv).reshape(B, N, num_heads, head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v)
            out = attn.transpose(1, 2).reshape(B, N, D)
            return F.linear(out, wo)

        expected = run(x, wq, wk, wv, wo)
        result = run(to_vulkan(x), to_vulkan(wq), to_vulkan(wk),
                     to_vulkan(wv), to_vulkan(wo))
        assert_close(result, expected, rtol=1e-2, atol=1e-2)


# ── Interpolate ─────────────────────────────────────────────────


class TestInterpolate:
    def test_nearest_upsample(self):
        x = torch.randn(1, 3, 4, 4)
        expected = F.interpolate(x, scale_factor=2, mode='nearest')
        result = F.interpolate(to_vulkan(x), scale_factor=2, mode='nearest')
        assert_close(result, expected)

    def test_bilinear_upsample(self):
        x = torch.randn(1, 3, 4, 4)
        expected = F.interpolate(x, size=(8, 8), mode='bilinear', align_corners=False)
        result = F.interpolate(to_vulkan(x), size=(8, 8), mode='bilinear', align_corners=False)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)


# ── Grid Sample ─────────────────────────────────────────────────


class TestGridSample:
    def test_grid_sample_bilinear(self):
        x = torch.randn(1, 1, 4, 4)
        grid = torch.randn(1, 2, 2, 2)  # sample 2x2 output
        grid = grid.clamp(-1, 1)  # valid grid coordinates

        expected = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        result = F.grid_sample(to_vulkan(x), to_vulkan(grid), mode='bilinear',
                               padding_mode='zeros', align_corners=True)
        assert_close(result, expected, rtol=1e-3, atol=1e-3)


# ── Index Put ───────────────────────────────────────────────────


class TestIndexPut:
    def test_index_put_1d(self):
        x = torch.zeros(10)
        indices = [torch.tensor([1, 3, 5])]
        values = torch.tensor([1.0, 2.0, 3.0])

        expected = x.clone().index_put_(indices, values)
        vx = to_vulkan(x.clone())
        vi = [idx.to("vulkan:0") for idx in indices]
        vx.index_put_(vi, to_vulkan(values))
        assert_close(vx, expected)


# ── Any / All ───────────────────────────────────────────────────


class TestAnyAll:
    def test_any(self):
        x = torch.tensor([0.0, 0.0, 1.0, 0.0])
        expected = x.any()
        result = to_vulkan(x).any()
        assert result.cpu().item() == expected.item()

    def test_all_true(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        expected = x.all()
        result = to_vulkan(x).all()
        assert result.cpu().item() == expected.item()

    def test_all_false(self):
        x = torch.tensor([1.0, 0.0, 3.0])
        expected = x.all()
        result = to_vulkan(x).all()
        assert result.cpu().item() == expected.item()


# ── Split ───────────────────────────────────────────────────────


class TestSplit:
    def test_split(self):
        x = torch.randn(10, 4)
        expected = x.split(3, dim=0)
        result = to_vulkan(x).split(3, dim=0)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_close(r, e)


# ── SDPA Backward ───────────────────────────────────────────────


class TestSDPAGrad:
    def test_sdpa_backward(self):
        B, H, N, D = 1, 2, 8, 16
        q = torch.randn(B, H, N, D, requires_grad=True)
        k = torch.randn(B, H, N, D, requires_grad=True)
        v = torch.randn(B, H, N, D, requires_grad=True)

        vq = to_vulkan(q.detach()).requires_grad_(True)
        vk = to_vulkan(k.detach()).requires_grad_(True)
        vv = to_vulkan(v.detach()).requires_grad_(True)

        F.scaled_dot_product_attention(q, k, v).sum().backward()
        F.scaled_dot_product_attention(vq, vk, vv).sum().backward()

        assert_close(vq.grad, q.grad, rtol=1e-2, atol=1e-2)
        assert_close(vk.grad, k.grad, rtol=1e-2, atol=1e-2)
        assert_close(vv.grad, v.grad, rtol=1e-2, atol=1e-2)


# ── RoPE ───────────────────────────────────────────────────────


class TestRoPE:
    def _cpu_rope(self, x, theta=10000.0):
        """Reference CPU implementation of RoPE."""
        B, H, N, D = x.shape
        result = torch.empty_like(x)
        for b in range(B):
            for h in range(H):
                for n in range(N):
                    for d in range(D // 2):
                        freq = n / (theta ** (2.0 * d / D))
                        cos_val = math.cos(freq)
                        sin_val = math.sin(freq)
                        result[b, h, n, 2*d] = x[b, h, n, 2*d] * cos_val - x[b, h, n, 2*d+1] * sin_val
                        result[b, h, n, 2*d+1] = x[b, h, n, 2*d] * sin_val + x[b, h, n, 2*d+1] * cos_val
        return result

    def test_rope_forward(self):
        import torch_vulkan
        x = torch.randn(1, 2, 8, 16)
        expected = self._cpu_rope(x)
        result = torch_vulkan.rope(to_vulkan(x))
        assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_rope_backward(self):
        import torch_vulkan
        x = torch.randn(1, 2, 4, 8, requires_grad=True)
        vx = to_vulkan(x.detach()).requires_grad_(True)

        # Forward + backward on CPU reference
        cpu_out = self._cpu_rope(x)
        cpu_out.sum().backward()

        # Forward + backward on Vulkan
        vk_out = torch_vulkan.rope(vx)
        vk_out.sum().backward()

        assert_close(vx.grad, x.grad, rtol=1e-3, atol=1e-3)
