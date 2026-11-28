# Owner(s): ["module: nn"]

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


# ---------------------------------------------------------------------------
# Numpy reference implementations
# ---------------------------------------------------------------------------


def _ref_rope_non_interleaved(x, cos, sin):
    """Non-interleaved (half-half) RoPE in numpy."""
    D = x.shape[-1]
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def _ref_rope_interleaved(x, cos, sin):
    """Interleaved (adjacent-pair) RoPE in numpy."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    real = x1 * cos - x2 * sin
    imag = x1 * sin + x2 * cos
    out = np.stack([real, imag], axis=-1)
    return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cos_sin(S, half_D, dtype=torch.float32, device="cpu"):
    """Return (cos, sin) caches of shape (S, half_D)."""
    t = torch.arange(S, device=device, dtype=torch.float32)
    theta = 1.0 / (
        10000.0
        ** (
            torch.arange(0, half_D * 2, 2, device=device, dtype=torch.float32)
            / (half_D * 2)
        )
    )
    freqs = torch.outer(t, theta)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


# ---------------------------------------------------------------------------
# Functional tests (device-generic)
# ---------------------------------------------------------------------------


class TestRotaryEmbeddingFunctional(TestCase):
    # -- 5a: correctness vs. numpy reference ----------------------------------

    @dtypes(torch.float32)
    def test_correctness_non_interleaved(self, device, dtype):
        B, H, S, D = 2, 4, 8, 16
        x = torch.randn(B, H, S, D, device=device, dtype=dtype)
        cos, sin = _make_cos_sin(S, D // 2, dtype=dtype, device=device)

        out = F.rotary_embedding(x, cos, sin)

        x_np = x.cpu().float().numpy()
        cos_np = cos.cpu().float().numpy()
        sin_np = sin.cpu().float().numpy()
        expected = _ref_rope_non_interleaved(x_np, cos_np, sin_np)
        self.assertEqual(out, torch.from_numpy(expected).to(device=device, dtype=dtype))

    @dtypes(torch.float32)
    def test_correctness_interleaved(self, device, dtype):
        B, H, S, D = 2, 4, 8, 16
        x = torch.randn(B, H, S, D, device=device, dtype=dtype)
        cos, sin = _make_cos_sin(S, D // 2, dtype=dtype, device=device)

        out = F.rotary_embedding(x, cos, sin, interleaved=True)

        x_np = x.cpu().float().numpy()
        cos_np = cos.cpu().float().numpy()
        sin_np = sin.cpu().float().numpy()
        expected = _ref_rope_interleaved(x_np, cos_np, sin_np)
        self.assertEqual(out, torch.from_numpy(expected).to(device=device, dtype=dtype))

    @dtypes(torch.float32)
    def test_correctness_with_position_ids(self, device, dtype):
        B, H, S, D = 2, 4, 8, 16
        cache_len = 32
        x = torch.randn(B, H, S, D, device=device, dtype=dtype)
        cos_cache, sin_cache = _make_cos_sin(
            cache_len, D // 2, dtype=dtype, device=device
        )
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        out = F.rotary_embedding(x, cos_cache, sin_cache, position_ids)

        # Reference: index into cache then apply non-interleaved rotation
        cos_indexed = cos_cache[position_ids]  # (B, S, D//2)
        sin_indexed = sin_cache[position_ids]
        x_np = x.cpu().float().numpy()
        cos_np = cos_indexed.cpu().float().numpy()[:, np.newaxis, :, :]  # add H dim
        sin_np = sin_indexed.cpu().float().numpy()[:, np.newaxis, :, :]
        expected = _ref_rope_non_interleaved(x_np, cos_np, sin_np)
        self.assertEqual(out, torch.from_numpy(expected).to(device=device, dtype=dtype))

    # -- new: mathematical equivalence of the two rotation modes ---------------

    @dtypes(torch.float32)
    def test_interleaved_noninterleaved_equivalence(self, device, dtype):
        # Both modes apply identical rotations; they differ only in which
        # element within each head is paired with which rotation angle.
        # If we rearrange x so that adjacent-pair mode and half-half mode see
        # the same pairs, their outputs must be identical up to the inverse
        # rearrangement.
        #
        # For D=8, half-half pairs: (x0,x4),(x1,x5),(x2,x6),(x3,x7)
        # Adjacent-pair mode pairs:  (x0,x1),(x2,x3),(x4,x5),(x6,x7)
        #
        # To make interleaved mode see the same pairs as half-half mode we
        # interleave the two halves:
        #   x_il[..., 0::2] = x_ni[..., :D//2]   (first of each half-half pair)
        #   x_il[..., 1::2] = x_ni[..., D//2:]   (second of each half-half pair)
        # Then the interleaved output, de-interleaved back to half-half order,
        # must equal the non-interleaved output.
        B, H, S, D = 2, 4, 8, 16
        x_ni = torch.randn(B, H, S, D, device=device, dtype=dtype)
        cos, sin = _make_cos_sin(S, D // 2, dtype=dtype, device=device)

        # Rearrange x_ni into adjacent-pair layout so interleaved mode pairs
        # the same elements as non-interleaved mode does on x_ni.
        x_il = torch.stack([x_ni[..., : D // 2], x_ni[..., D // 2 :]], dim=-1).flatten(
            -2
        )

        out_ni = F.rotary_embedding(x_ni, cos, sin, interleaved=False)
        out_il = F.rotary_embedding(x_il, cos, sin, interleaved=True)

        # De-interleave the interleaved output back to half-half order.
        out_il_as_ni = torch.cat([out_il[..., 0::2], out_il[..., 1::2]], dim=-1)

        self.assertEqual(out_ni, out_il_as_ni)

    # -- new: position_ids with arbitrary non-sequential positions --------------

    @dtypes(torch.float32)
    def test_position_ids_arbitrary_positions(self, device, dtype):
        # position_ids need not be 0..S-1; verify that arbitrary indices
        # select the correct rows of the cos/sin cache.
        B, H, S, D = 2, 4, 6, 16
        cache_len = 64
        x = torch.randn(B, H, S, D, device=device, dtype=dtype)
        cos_cache, sin_cache = _make_cos_sin(
            cache_len, D // 2, dtype=dtype, device=device
        )

        # Non-monotonic indices with different values per batch element.
        position_ids = torch.tensor(
            [[5, 0, 31, 7, 2, 15], [10, 3, 63, 1, 20, 9]], device=device
        )  # (B, S)

        out = F.rotary_embedding(x, cos_cache, sin_cache, position_ids)

        # Reference: index manually then apply non-interleaved rotation.
        cos_idx = cos_cache[position_ids].unsqueeze(1)  # (B, 1, S, D//2)
        sin_idx = sin_cache[position_ids].unsqueeze(1)
        x1, x2 = x.chunk(2, dim=-1)
        expected = torch.cat(
            [x1 * cos_idx - x2 * sin_idx, x1 * sin_idx + x2 * cos_idx], dim=-1
        )
        self.assertEqual(out, expected)

    # -- new: sequential mode == position_ids with arange(S) -------------------

    @dtypes(torch.float32)
    def test_sequential_mode_equals_position_ids_arange(self, device, dtype):
        # Passing cos_cache[:S] / sin_cache[:S] directly (sequential mode)
        # must give bit-identical results to passing position_ids=arange(S).
        B, H, S, D = 2, 4, 8, 32
        cache_len = 64
        x = torch.randn(B, H, S, D, device=device, dtype=dtype)
        cos_cache, sin_cache = _make_cos_sin(
            cache_len, D // 2, dtype=dtype, device=device
        )

        out_sequential = F.rotary_embedding(x, cos_cache[:S], sin_cache[:S])

        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        out_indexed = F.rotary_embedding(x, cos_cache, sin_cache, position_ids)

        self.assertEqual(out_sequential, out_indexed)

    # -- new: float16/bfloat16 numerical accuracy vs float32 reference ---------

    @dtypes(torch.float16, torch.bfloat16)
    def test_dtype_numerical_accuracy(self, device, dtype):
        # Reduced-precision outputs must be close to the float32 reference.
        B, H, S, D = 2, 4, 8, 32
        x_fp32 = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
        cos_fp32, sin_fp32 = _make_cos_sin(
            S, D // 2, dtype=torch.float32, device=device
        )
        ref = F.rotary_embedding(x_fp32, cos_fp32, sin_fp32)

        x_low = x_fp32.to(dtype)
        cos_low, sin_low = cos_fp32.to(dtype), sin_fp32.to(dtype)
        out_low = F.rotary_embedding(x_low, cos_low, sin_low)

        # float16 is exact to ~3 decimal places; bfloat16 to ~2.
        atol = 1e-2 if dtype == torch.float16 else 2e-2
        rtol = 1e-2 if dtype == torch.float16 else 2e-2
        self.assertEqual(out_low.float(), ref, atol=atol, rtol=rtol)

    # -- 5b: shape preservation -----------------------------------------------

    @parametrize("shape", [(1, 1, 8, 64), (2, 8, 16, 64), (4, 32, 128, 128)])
    @dtypes(torch.float32)
    def test_shape_preservation(self, device, dtype, shape):
        B, H, S, D = shape
        x = torch.randn(*shape, device=device, dtype=dtype)
        cos, sin = _make_cos_sin(S, D // 2, dtype=dtype, device=device)
        out = F.rotary_embedding(x, cos, sin)
        self.assertEqual(out.shape, x.shape)

    # -- 5c: dtype coverage ---------------------------------------------------

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    def test_dtype_coverage(self, device, dtype):
        B, H, S, D = 2, 4, 8, 32
        x = torch.randn(B, H, S, D, device=device, dtype=dtype)
        cos, sin = _make_cos_sin(S, D // 2, dtype=dtype, device=device)
        out = F.rotary_embedding(x, cos, sin)
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(out.shape, x.shape)

    # -- 5d: gradient check ---------------------------------------------------

    @dtypes(torch.float64)
    def test_gradcheck_non_interleaved(self, device, dtype):
        B, H, S, D = 1, 2, 4, 8
        x = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        cos, sin = _make_cos_sin(S, D // 2, dtype=dtype, device=device)
        cos = cos.requires_grad_(False)
        sin = sin.requires_grad_(False)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_: F.rotary_embedding(x_, cos, sin),
                (x,),
            )
        )

    @dtypes(torch.float64)
    def test_gradcheck_interleaved(self, device, dtype):
        B, H, S, D = 1, 2, 4, 8
        x = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        cos, sin = _make_cos_sin(S, D // 2, dtype=dtype, device=device)
        cos = cos.requires_grad_(False)
        sin = sin.requires_grad_(False)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_: F.rotary_embedding(x_, cos, sin, interleaved=True),
                (x,),
            )
        )

    # -- 5e: torch.compile - no graph breaks ----------------------------------

    def test_compile_no_graph_break_non_interleaved(self, device):
        B, H, S, D = 2, 4, 8, 32
        x = torch.randn(B, H, S, D, device=device)
        cos, sin = _make_cos_sin(S, D // 2, device=device)

        def f(x, cos, sin):
            return F.rotary_embedding(x, cos, sin, interleaved=False)

        explanation = torch._dynamo.explain(f)(x, cos, sin)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            msg=f"Expected 0 graph breaks, got {explanation.graph_break_count}",
        )

    def test_compile_no_graph_break_interleaved(self, device):
        B, H, S, D = 2, 4, 8, 32
        x = torch.randn(B, H, S, D, device=device)
        cos, sin = _make_cos_sin(S, D // 2, device=device)

        def f(x, cos, sin):
            return F.rotary_embedding(x, cos, sin, interleaved=True)

        explanation = torch._dynamo.explain(f)(x, cos, sin)
        self.assertEqual(
            explanation.graph_break_count,
            0,
            msg=f"Expected 0 graph breaks, got {explanation.graph_break_count}",
        )

    def test_compile_correctness(self, device):
        B, H, S, D = 2, 8, 16, 64
        x = torch.randn(B, H, S, D, device=device)
        cos, sin = _make_cos_sin(S, D // 2, device=device)

        eager = F.rotary_embedding(x, cos, sin)
        compiled = torch.compile(F.rotary_embedding)(x, cos, sin)
        self.assertEqual(eager, compiled)

    # -- 5f: torch.export - sequential mode (no position_ids) ----------------

    def test_export_sequential_mode(self, device):
        if device != "cpu":
            self.skipTest("export test runs on CPU only")

        B, H, S, D = 2, 4, 8, 32

        class Model(nn.Module):
            def forward(self, x, cos, sin):
                return F.rotary_embedding(x, cos, sin)

        model = Model()
        x = torch.randn(B, H, S, D)
        cos, sin = _make_cos_sin(S, D // 2)

        B_dim = torch.export.Dim("B")
        S_dim = torch.export.Dim("S", min=1, max=512)
        ep = torch.export.export(
            model,
            (x, cos, sin),
            dynamic_shapes={
                "x": {0: B_dim, 2: S_dim},
                "cos": {0: S_dim},
                "sin": {0: S_dim},
            },
        )

        # Verify at a different seq_len
        S2 = 16
        x2 = torch.randn(B, H, S2, D)
        cos2, sin2 = _make_cos_sin(S2, D // 2)
        out = ep.module()(x2, cos2, sin2)
        self.assertEqual(out.shape, (B, H, S2, D))

    # -- 5g: torch.export - position_ids mode ---------------------------------

    def test_export_position_ids_mode(self, device):
        if device != "cpu":
            self.skipTest("export test runs on CPU only")

        B, H, S, D = 2, 4, 8, 32
        cache_len = 64

        class Model(nn.Module):
            def forward(self, x, cos, sin, position_ids):
                return F.rotary_embedding(x, cos, sin, position_ids)

        model = Model()
        x = torch.randn(B, H, S, D)
        cos_cache, sin_cache = _make_cos_sin(cache_len, D // 2)
        position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

        B_dim = torch.export.Dim("B")
        S_dim = torch.export.Dim("S", min=1, max=cache_len)
        ep = torch.export.export(
            model,
            (x, cos_cache, sin_cache, position_ids),
            dynamic_shapes={
                "x": {0: B_dim, 2: S_dim},
                "cos": None,
                "sin": None,
                "position_ids": {0: B_dim, 1: S_dim},
            },
        )

        # Verify at a different seq_len
        S2 = 16
        x2 = torch.randn(B, H, S2, D)
        pos2 = torch.arange(S2).unsqueeze(0).expand(B, -1)
        out = ep.module()(x2, cos_cache, sin_cache, pos2)
        self.assertEqual(out.shape, (B, H, S2, D))


# ---------------------------------------------------------------------------
# Module tests (device-generic)
# ---------------------------------------------------------------------------


class TestRotaryEmbeddingModule(TestCase):
    # -- 5h: basic module correctness and device transfer ---------------------

    @dtypes(torch.float32)
    def test_module_matches_functional(self, device, dtype):
        B, H, S, D = 2, 4, 16, 64
        x = torch.randn(B, H, S, D, device=device, dtype=dtype)
        rope = nn.RotaryEmbedding(dim=D, max_seq_len=S).to(device=device, dtype=dtype)

        # Sequential mode (no position_ids)
        cos = rope.cos_cache[:S]
        sin = rope.sin_cache[:S]
        expected = F.rotary_embedding(x, cos, sin)
        self.assertEqual(rope(x), expected)

    @dtypes(torch.float32)
    def test_module_position_ids_matches_functional(self, device, dtype):
        B, H, S, D = 2, 4, 8, 32
        cache_len = 64
        x = torch.randn(B, H, S, D, device=device, dtype=dtype)
        rope = nn.RotaryEmbedding(dim=D, max_seq_len=cache_len).to(
            device=device, dtype=dtype
        )
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        # Module passes the full cache + position_ids to the functional
        out = rope(x, position_ids=position_ids)
        expected = F.rotary_embedding(x, rope.cos_cache, rope.sin_cache, position_ids)
        self.assertEqual(out, expected)

    def test_module_buffers_move_to_device(self, device):
        rope = nn.RotaryEmbedding(dim=64, max_seq_len=32).to(device)
        self.assertEqual(rope.cos_cache.device.type, torch.device(device).type)
        self.assertEqual(rope.sin_cache.device.type, torch.device(device).type)

    def test_module_buffers_not_in_state_dict(self, device):
        rope = nn.RotaryEmbedding(dim=64, max_seq_len=32)
        self.assertNotIn("cos_cache", rope.state_dict())
        self.assertNotIn("sin_cache", rope.state_dict())

    # -- 5i: max_seq_len overflow and rebuild ---------------------------------

    def test_overflow_raises(self, device):
        rope = nn.RotaryEmbedding(dim=64, max_seq_len=128).to(device)
        x_too_long = torch.randn(1, 8, 256, 64, device=device)
        with self.assertRaisesRegex(RuntimeError, "exceeds max_seq_len"):
            rope(x_too_long)

    def test_rebuild_cache_then_forward_succeeds(self, device):
        rope = nn.RotaryEmbedding(dim=64, max_seq_len=128).to(device)
        x_long = torch.randn(1, 8, 256, 64, device=device)

        rope.build_rope_cache(256)
        out = rope(x_long)
        self.assertEqual(out.shape, x_long.shape)
        self.assertEqual(rope.max_seq_len, 256)
        self.assertEqual(rope.cos_cache.shape[0], 256)

    def test_rebuild_cache_preserves_device(self, device):
        # build_rope_cache called after .to(device) must land the new buffers
        # on the same device as the module, not silently fall back to CPU.
        rope = nn.RotaryEmbedding(dim=64, max_seq_len=64).to(device)
        self.assertEqual(rope.cos_cache.device.type, torch.device(device).type)

        rope.build_rope_cache(128)

        self.assertEqual(rope.cos_cache.device.type, torch.device(device).type)
        self.assertEqual(rope.sin_cache.device.type, torch.device(device).type)
        self.assertEqual(rope.cos_cache.shape[0], 128)
        # Module must still produce output on the correct device.
        x = torch.randn(1, 4, 128, 64, device=device)
        self.assertEqual(rope(x).device.type, torch.device(device).type)

    def test_odd_dim_raises(self, device):
        with self.assertRaisesRegex(ValueError, "dim must be even"):
            nn.RotaryEmbedding(dim=63)

    def test_output_shape(self, device):
        rope = nn.RotaryEmbedding(dim=64, max_seq_len=128).to(device)
        for B, H, S in [(1, 1, 8), (2, 8, 32), (4, 16, 128)]:
            x = torch.randn(B, H, S, 64, device=device)
            self.assertEqual(rope(x).shape, (B, H, S, 64))


# ---------------------------------------------------------------------------
# Register device-type tests
# ---------------------------------------------------------------------------

instantiate_device_type_tests(
    TestRotaryEmbeddingFunctional, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(
    TestRotaryEmbeddingModule, globals(), only_for=("cpu", "cuda")
)


if __name__ == "__main__":
    run_tests()
