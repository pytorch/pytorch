"""Tests for Phase 3: Model coverage ops (triu/tril, pad, index, repeat, etc.)."""

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


# ── triu / tril ─────────────────────────────────────────────────

class TestTriuTril:
    def test_triu_basic(self):
        x = torch.randn(4, 4)
        expected = x.triu()
        assert_close(to_vulkan(x).triu(), expected)

    def test_triu_diagonal(self):
        x = torch.randn(4, 4)
        expected = x.triu(1)
        assert_close(to_vulkan(x).triu(1), expected)

    def test_triu_negative_diagonal(self):
        x = torch.randn(4, 4)
        expected = x.triu(-1)
        assert_close(to_vulkan(x).triu(-1), expected)

    def test_tril_basic(self):
        x = torch.randn(4, 4)
        expected = x.tril()
        assert_close(to_vulkan(x).tril(), expected)

    def test_tril_diagonal(self):
        x = torch.randn(4, 4)
        expected = x.tril(1)
        assert_close(to_vulkan(x).tril(1), expected)

    def test_triu_3d(self):
        """Batch triu for causal masks."""
        x = torch.ones(2, 4, 4)
        expected = x.triu()
        assert_close(to_vulkan(x).triu(), expected)


# ── F.pad / constant_pad_nd ────────────────────────────────────

class TestPad:
    def test_pad_1d(self):
        x = torch.randn(1, 3, 8)
        expected = F.pad(x, (1, 2))
        assert_close(F.pad(to_vulkan(x), (1, 2)), expected)

    def test_pad_2d(self):
        x = torch.randn(1, 3, 8, 8)
        expected = F.pad(x, (1, 1, 1, 1))
        assert_close(F.pad(to_vulkan(x), (1, 1, 1, 1)), expected)

    def test_pad_value(self):
        x = torch.randn(1, 3, 4)
        expected = F.pad(x, (2, 0), value=-1.0)
        assert_close(F.pad(to_vulkan(x), (2, 0), value=-1.0), expected)


# ── index.Tensor ────────────────────────────────────────────────

class TestIndexTensor:
    def test_index_1d(self):
        x = torch.randn(10)
        idx = torch.tensor([0, 3, 7])
        expected = x[idx]
        assert_close(to_vulkan(x)[to_vulkan(idx)], expected)

    def test_index_2d_rows(self):
        x = torch.randn(5, 3)
        idx = torch.tensor([0, 2, 4])
        expected = x[idx]
        assert_close(to_vulkan(x)[to_vulkan(idx)], expected)

    def test_index_advanced(self):
        x = torch.randn(4, 8)
        row_idx = torch.tensor([0, 1, 2])
        col_idx = torch.tensor([3, 5, 7])
        expected = x[row_idx, col_idx]
        assert_close(to_vulkan(x)[to_vulkan(row_idx), to_vulkan(col_idx)], expected)


# ── repeat / repeat_interleave ──────────────────────────────────

class TestRepeat:
    def test_repeat_1d(self):
        x = torch.randn(4)
        expected = x.repeat(3)
        assert_close(to_vulkan(x).repeat(3), expected)

    def test_repeat_2d(self):
        x = torch.randn(2, 3)
        expected = x.repeat(2, 3)
        assert_close(to_vulkan(x).repeat(2, 3), expected)

    def test_repeat_expand_dims(self):
        x = torch.randn(3)
        expected = x.repeat(2, 1)
        assert_close(to_vulkan(x).repeat(2, 1), expected)

    def test_repeat_interleave(self):
        x = torch.randn(2, 3)
        expected = x.repeat_interleave(2, dim=0)
        assert_close(to_vulkan(x).repeat_interleave(2, dim=0), expected)


# ── stack ───────────────────────────────────────────────────────

class TestStack:
    def test_stack_basic(self):
        tensors = [torch.randn(3) for _ in range(4)]
        expected = torch.stack(tensors)
        vulkan_tensors = [to_vulkan(t) for t in tensors]
        assert_close(torch.stack(vulkan_tensors), expected)

    def test_stack_dim1(self):
        tensors = [torch.randn(2, 3) for _ in range(4)]
        expected = torch.stack(tensors, dim=1)
        vulkan_tensors = [to_vulkan(t) for t in tensors]
        assert_close(torch.stack(vulkan_tensors, dim=1), expected)


# ── erf ─────────────────────────────────────────────────────────

class TestErf:
    def test_erf_basic(self):
        x = torch.randn(16)
        expected = x.erf()
        assert_close(to_vulkan(x).erf(), expected)

    def test_erf_inplace(self):
        x = torch.randn(16)
        expected = x.erf()
        vx = to_vulkan(x.clone())
        vx.erf_()
        assert_close(vx, expected)


# ── flip / roll ─────────────────────────────────────────────────

class TestFlipRoll:
    def test_flip(self):
        x = torch.randn(3, 4)
        expected = x.flip(0)
        assert_close(to_vulkan(x).flip(0), expected)

    def test_flip_multi_dims(self):
        x = torch.randn(2, 3, 4)
        expected = x.flip(0, 2)
        assert_close(to_vulkan(x).flip(0, 2), expected)

    def test_roll(self):
        x = torch.randn(8)
        expected = x.roll(2)
        assert_close(to_vulkan(x).roll(2), expected)

    def test_roll_dim(self):
        x = torch.randn(3, 4)
        expected = x.roll(1, 0)
        assert_close(to_vulkan(x).roll(1, 0), expected)


# ── _to_copy / as_strided / resize_ ────────────────────────────

class TestInfraOps:
    def test_to_copy_dtype(self):
        x = torch.randn(4, device="vulkan")
        # float32 -> float16 -> float32 round-trip
        half = x.to(torch.float16)
        assert half.dtype == torch.float16
        back = half.to(torch.float32)
        assert back.dtype == torch.float32

    def test_as_strided(self):
        x = torch.arange(12, dtype=torch.float32)
        expected = x.as_strided([3, 4], [4, 1])
        vx = to_vulkan(x)
        result = torch.as_strided(vx, [3, 4], [4, 1])
        assert_close(result, expected)


# ── Generator / manual_seed ─────────────────────────────────────

class TestGenerator:
    def test_generator_create(self):
        gen = torch.Generator(device="vulkan")
        assert gen.device.type == "vulkan"

    def test_manual_seed_reproducible(self):
        torch.manual_seed(42)
        a = torch.randn(8, device="vulkan")
        torch.manual_seed(42)
        b = torch.randn(8, device="vulkan")
        assert_close(b, a.cpu())

    def test_manual_seed_no_warning(self):
        """torch.manual_seed should not produce warnings."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.manual_seed(99)
            seed_warns = [x for x in w if "manual_seed_all" in str(x.message)]
            assert len(seed_warns) == 0

    def test_manual_seed_different(self):
        torch.manual_seed(42)
        a = torch.randn(8, device="vulkan")
        torch.manual_seed(123)
        b = torch.randn(8, device="vulkan")
        assert not torch.allclose(a.cpu(), b.cpu())


# ── Autocast ────────────────────────────────────────────────────

class TestAutocast:
    def test_autocast_context(self):
        """torch.autocast('vulkan') should work without errors."""
        with torch.autocast("vulkan", dtype=torch.float16):
            x = torch.randn(4, 8, device="vulkan")
            y = torch.randn(8, 4, device="vulkan")
            z = torch.mm(x, y)
            # Autocast casts mm inputs to f16 (lower_precision_fp policy)
            # Output is f16 (widen-compute-narrow: f16->f32->compute->f32->f16)
            assert z.dtype == torch.float16
            assert z.shape == (4, 4)

    def test_autocast_training(self):
        """Autocast + backward should work."""
        model = torch.nn.Linear(8, 4).vulkan()
        x = torch.randn(2, 8, device="vulkan")
        with torch.autocast("vulkan", dtype=torch.float16):
            y = model(x)
            loss = y.sum()
        loss.backward()
        assert model.weight.grad is not None

    def test_gradscaler(self):
        """GradScaler should work with vulkan device."""
        scaler = torch.amp.GradScaler("vulkan")
        model = torch.nn.Linear(4, 2).vulkan()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(2, 4, device="vulkan")

        with torch.autocast("vulkan", dtype=torch.float16):
            y = model(x)
            loss = y.sum()

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        assert model.weight.grad is not None


# ── masked_scatter ───────────────────────────────────────────────

class TestMaskedScatter:
    def test_basic(self):
        x = torch.randn(3, 4)
        mask = torch.tensor([[True, False, True, False],
                             [False, True, False, True],
                             [True, True, False, False]])
        source = torch.randn(6)
        expected = x.clone().masked_scatter_(mask, source)
        result = to_vulkan(x).masked_scatter_(to_vulkan(mask), to_vulkan(source))
        assert_close(result, expected)

    def test_1d(self):
        x = torch.randn(8)
        mask = torch.tensor([True, False, True, True, False, False, True, False])
        source = torch.randn(4)
        expected = x.clone().masked_scatter_(mask, source)
        result = to_vulkan(x).masked_scatter_(to_vulkan(mask), to_vulkan(source))
        assert_close(result, expected)

    def test_all_true(self):
        x = torch.randn(2, 3)
        mask = torch.ones(2, 3, dtype=torch.bool)
        source = torch.randn(6)
        expected = x.clone().masked_scatter_(mask, source)
        result = to_vulkan(x).masked_scatter_(to_vulkan(mask), to_vulkan(source))
        assert_close(result, expected)

    def test_all_false(self):
        x = torch.randn(2, 3)
        mask = torch.zeros(2, 3, dtype=torch.bool)
        source = torch.randn(6)
        expected = x.clone().masked_scatter_(mask, source)
        result = to_vulkan(x).masked_scatter_(to_vulkan(mask), to_vulkan(source))
        assert_close(result, expected)

    def test_vl_pattern(self):
        """Simulate Qwen2-VL vision embedding injection."""
        seq_len = 32
        hidden = 16
        num_vision = 8

        # Text embeddings
        text_embeds = torch.randn(1, seq_len, hidden)
        # Vision embeddings
        vision_embeds = torch.randn(num_vision, hidden)
        # Mask: True at vision token positions
        mask = torch.zeros(1, seq_len, dtype=torch.bool)
        mask[0, 4:4+num_vision] = True  # Vision tokens at positions 4-11

        # Expand mask to match hidden dim
        mask_3d = mask.unsqueeze(-1).expand_as(text_embeds)

        expected = text_embeds.clone().masked_scatter_(mask_3d, vision_embeds.reshape(-1))
        result = to_vulkan(text_embeds).masked_scatter_(
            to_vulkan(mask_3d), to_vulkan(vision_embeds.reshape(-1)))
        assert_close(result, expected)


# ── Conv3d ───────────────────────────────────────────────────────

class TestConv3d:
    def test_basic(self):
        """Basic Conv3d with stride==kernel (Qwen2-VL patch embedding)."""
        x = torch.randn(1, 3, 2, 14, 14)
        w = torch.randn(16, 3, 2, 14, 14)
        b = torch.randn(16)
        expected = F.conv3d(x, w, b, stride=(2, 14, 14))
        result = F.conv3d(to_vulkan(x), to_vulkan(w), to_vulkan(b), stride=(2, 14, 14))
        assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_no_bias(self):
        x = torch.randn(1, 2, 4, 6, 6)
        w = torch.randn(3, 2, 2, 3, 3)
        expected = F.conv3d(x, w, stride=(2, 3, 3))
        result = F.conv3d(to_vulkan(x), to_vulkan(w), stride=(2, 3, 3))
        assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_batch(self):
        x = torch.randn(4, 2, 4, 6, 6)
        w = torch.randn(3, 2, 2, 3, 3)
        b = torch.randn(3)
        expected = F.conv3d(x, w, b, stride=(2, 3, 3))
        result = F.conv3d(to_vulkan(x), to_vulkan(w), to_vulkan(b), stride=(2, 3, 3))
        assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_padding(self):
        x = torch.randn(1, 2, 4, 8, 8)
        w = torch.randn(3, 2, 3, 3, 3)
        expected = F.conv3d(x, w, stride=1, padding=1)
        result = F.conv3d(to_vulkan(x), to_vulkan(w), stride=1, padding=1)
        assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_nn_conv3d(self):
        """Test nn.Conv3d module."""
        conv = torch.nn.Conv3d(3, 8, kernel_size=(2, 7, 7), stride=(2, 7, 7), bias=True)
        x = torch.randn(1, 3, 2, 14, 14)
        expected = conv(x)
        result = conv.vulkan()(to_vulkan(x))
        assert_close(result, expected, atol=1e-3, rtol=1e-3)


# ── unbind / outer (decomposition tests) ─────────────────────────

class TestDecomposedOps:
    def test_unbind(self):
        x = torch.randn(3, 4, 5)
        expected = torch.unbind(x, 0)
        result = torch.unbind(to_vulkan(x), 0)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_close(r, e)

    def test_unbind_dim1(self):
        x = torch.randn(3, 4, 5)
        expected = torch.unbind(x, 1)
        result = torch.unbind(to_vulkan(x), 1)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_close(r, e)

    def test_outer(self):
        a = torch.randn(5)
        b = torch.randn(7)
        expected = torch.outer(a, b)
        result = torch.outer(to_vulkan(a), to_vulkan(b))
        assert_close(result, expected)

    def test_outer_rope_pattern(self):
        """Vision RoPE: torch.outer(seq_positions, inv_freq)."""
        seq = torch.arange(16, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
        expected = torch.outer(seq, inv_freq)
        result = torch.outer(to_vulkan(seq), to_vulkan(inv_freq))
        assert_close(result, expected)
