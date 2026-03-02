# Owner(s): ["module: sdpa"]

import importlib
import unittest

from _fa_test_common import FlashAttentionTestMixin, SdpaShape

import torch
import torch.nn.functional as F
from torch.backends.cuda import SDPBackend
from torch.nn.attention import activate_flash_attention_impl, sdpa_kernel
from torch.nn.attention.experimental._scaled_dot_product_attention_quantized import (
    _scaled_dot_product_attention_quantized,
    DescaleType,
)
from torch.nn.attention.varlen import varlen_attn
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def _fa3_dependencies_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major != 9:  # FA3 requires Hopper (SM90)
        return False
    try:
        importlib.import_module("flash_attn_interface")
    except ModuleNotFoundError:
        return False
    return True


class TestFlashAttentionFA3(FlashAttentionTestMixin, TestCase):
    # Mixin configuration
    impl_name = "FA3"
    fwd_kernel_patterns = ["flash_attn_3::fwd"]
    bwd_kernel_patterns = ["flash_attn_3::bwd"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _fa3_dependencies_available():
            return
        activate_flash_attention_impl("FA3")

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("batch", [1, 2])
    @parametrize(
        "seq_len",
        [
            512,
            1024,
        ],
    )
    @parametrize("heads", [4, 8])
    @parametrize("head_dim", [64, 128])
    @parametrize(
        "is_causal",
        [False, True],
    )
    def test_flash_attention_matches_math(
        self, device, dtype, batch, seq_len, heads, head_dim, is_causal
    ):
        shape = SdpaShape(batch, heads, seq_len, head_dim)
        self._assert_flash_matches_math(
            device,
            shape=shape,
            dtype=dtype,
            is_causal=is_causal,
            test_backward=True,
        )

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fa3_kernel_called(self, device, dtype):
        self._test_kernel_called(device, dtype)

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    def test_multiple_activate(self):
        self._test_multiple_activate_impl()

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa3_metadata(self, device, dtype, is_causal):
        """Test that torch.compile preserves tensor metadata (shape, stride, dtype)."""
        self._test_compiled_sdpa_metadata(device, dtype, is_causal)

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa3_matches_math(self, device, dtype, is_causal):
        """Test compiled FA3 numerical correctness against math backend."""
        self._test_compiled_sdpa_matches_math(device, dtype, is_causal)

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa3_backward_matches_math(self, device, dtype, is_causal):
        """Test compiled FA3 backward numerical correctness against math backend."""
        min_atol = 0.05 if dtype == torch.bfloat16 else 0.01
        self._test_compiled_sdpa_backward_matches_math(
            device, dtype, is_causal, head_dim=128, min_atol=min_atol
        )

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    def test_attention_preserves_query_layout(self, device):
        """Test that FA3 output has the same layout as the query tensor."""
        self._test_attention_preserves_query_layout(device)

    # ==================== FP8 TESTS ====================

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("batch", [1, 2])
    @parametrize("seq_len", [512, 1024])
    @parametrize("heads", [4, 8])
    @parametrize("head_dim", [64, 128])
    @parametrize("is_causal", [False, True])
    def test_fp8_forward_runs(self, device, batch, seq_len, heads, head_dim, is_causal):
        """Test that FP8 forward pass runs without errors."""
        shape = SdpaShape(batch, heads, seq_len, head_dim)

        # Create FP8 inputs
        q = torch.randn(shape, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )
        k = torch.randn(shape, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )
        v = torch.randn(shape, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )

        # Create descale tensors (per-head scaling)
        descale_q = torch.ones(batch, heads, dtype=torch.float32, device=device)
        descale_k = torch.ones(batch, heads, dtype=torch.float32, device=device)
        descale_v = torch.ones(batch, heads, dtype=torch.float32, device=device)

        with torch.no_grad():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = _scaled_dot_product_attention_quantized(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    q_descale=descale_q,
                    k_descale=descale_k,
                    v_descale=descale_v,
                    q_descale_type=DescaleType.PER_HEAD,
                    k_descale_type=DescaleType.PER_HEAD,
                    v_descale_type=DescaleType.PER_HEAD,
                )

        # Check output properties
        self.assertEqual(out.shape, shape)
        self.assertEqual(out.dtype, torch.bfloat16)  # FP8 outputs bfloat16
        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("is_causal", [False, True])
    def test_fp8_forward_correctness(self, device, is_causal):
        """Test FP8 forward numerical correctness against bf16 reference."""
        shape = SdpaShape(2, 8, 512, 64)

        # Create bf16 reference inputs
        q_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)
        k_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)
        v_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)

        # Convert to FP8
        q_fp8 = q_bf16.to(torch.float8_e4m3fn)
        k_fp8 = k_bf16.to(torch.float8_e4m3fn)
        v_fp8 = v_bf16.to(torch.float8_e4m3fn)

        # Identity descales (scale factor = 1.0)
        batch, heads = shape.batch, shape.num_heads
        descale_q = torch.ones(batch, heads, dtype=torch.float32, device=device)
        descale_k = torch.ones(batch, heads, dtype=torch.float32, device=device)
        descale_v = torch.ones(batch, heads, dtype=torch.float32, device=device)

        with torch.no_grad():
            # FP8 forward
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out_fp8 = _scaled_dot_product_attention_quantized(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    is_causal=is_causal,
                    q_descale=descale_q,
                    k_descale=descale_k,
                    v_descale=descale_v,
                    q_descale_type=DescaleType.PER_HEAD,
                    k_descale_type=DescaleType.PER_HEAD,
                    v_descale_type=DescaleType.PER_HEAD,
                )

            # bf16 reference via FA3
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                # Use the dequantized fp8 values for reference
                q_deq = q_fp8.to(torch.bfloat16)
                k_deq = k_fp8.to(torch.bfloat16)
                v_deq = v_fp8.to(torch.bfloat16)
                out_bf16 = F.scaled_dot_product_attention(
                    q_deq, k_deq, v_deq, is_causal=is_causal
                )

        # FP8 has lower precision, so use relaxed tolerance
        # The error should be within a reasonable range compared to bf16
        error = (out_fp8 - out_bf16).abs().max().item()
        # FP8 e4m3 has ~3 bits of mantissa, so expect ~1/8 relative error
        self.assertLessEqual(
            error,
            0.25,  # Relaxed tolerance for FP8
            f"FP8 error {error:.4f} exceeds tolerance",
        )

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    def test_fp8_forward_with_descale(self, device):
        """Test FP8 forward with non-trivial descale values."""
        shape = SdpaShape(2, 4, 256, 64)
        batch, heads = shape.batch, shape.num_heads

        # Create inputs and scale them
        scale_factor = 2.0
        q_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device) * scale_factor
        k_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device) * scale_factor
        v_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device) * scale_factor

        # Convert to FP8 (values are scaled)
        q_fp8 = q_bf16.to(torch.float8_e4m3fn)
        k_fp8 = k_bf16.to(torch.float8_e4m3fn)
        v_fp8 = v_bf16.to(torch.float8_e4m3fn)

        # Descale tensors to undo the scaling
        descale_q = torch.full(
            (batch, heads), scale_factor, dtype=torch.float32, device=device
        )
        descale_k = torch.full(
            (batch, heads), scale_factor, dtype=torch.float32, device=device
        )
        descale_v = torch.full(
            (batch, heads), scale_factor, dtype=torch.float32, device=device
        )

        with torch.no_grad():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = _scaled_dot_product_attention_quantized(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    is_causal=False,
                    q_descale=descale_q,
                    k_descale=descale_k,
                    v_descale=descale_v,
                    q_descale_type=DescaleType.PER_HEAD,
                    k_descale_type=DescaleType.PER_HEAD,
                    v_descale_type=DescaleType.PER_HEAD,
                )

        # Output should be valid
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    def test_fp8_backward_not_supported(self, device):
        """Test that FP8 backward raises appropriate warning."""
        shape = SdpaShape(2, 4, 256, 64)
        batch, heads = shape.batch, shape.num_heads

        q = torch.randn(shape, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )
        k = torch.randn(shape, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )
        v = torch.randn(shape, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )

        # Enable gradients to trigger the warning path
        q.requires_grad_(True)

        descale_q = torch.ones(batch, heads, dtype=torch.float32, device=device)
        descale_k = torch.ones(batch, heads, dtype=torch.float32, device=device)
        descale_v = torch.ones(batch, heads, dtype=torch.float32, device=device)

        # Should warn about backward not being supported
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                _ = _scaled_dot_product_attention_quantized(
                    q,
                    k,
                    v,
                    is_causal=False,
                    q_descale=descale_q,
                    k_descale=descale_k,
                    v_descale=descale_v,
                    q_descale_type=DescaleType.PER_HEAD,
                    k_descale_type=DescaleType.PER_HEAD,
                    v_descale_type=DescaleType.PER_HEAD,
                )

            # Check that the backward warning was issued
            backward_warnings = [x for x in w if "backward" in str(x.message).lower()]
            self.assertTrue(
                len(backward_warnings) > 0,
                "Expected warning about FP8 backward not being supported",
            )

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("is_causal", [False, True])
    def test_compiled_fp8_forward(self, device, is_causal):
        """Test that FP8 forward works with torch.compile."""
        shape = SdpaShape(2, 8, 512, 64)
        batch, heads = shape.batch, shape.num_heads

        q = torch.randn(shape, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )
        k = torch.randn(shape, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )
        v = torch.randn(shape, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )

        descale_q = torch.ones(batch, heads, dtype=torch.float32, device=device)
        descale_k = torch.ones(batch, heads, dtype=torch.float32, device=device)
        descale_v = torch.ones(batch, heads, dtype=torch.float32, device=device)

        def fp8_sdpa(q, k, v, dq, dk, dv):
            return _scaled_dot_product_attention_quantized(
                q,
                k,
                v,
                is_causal=is_causal,
                q_descale=dq,
                k_descale=dk,
                v_descale=dv,
                q_descale_type=DescaleType.PER_HEAD,
                k_descale_type=DescaleType.PER_HEAD,
                v_descale_type=DescaleType.PER_HEAD,
            )

        with torch.no_grad():
            # Eager execution
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out_eager = fp8_sdpa(q, k, v, descale_q, descale_k, descale_v)

            # Compiled execution
            compiled_fn = torch.compile(fp8_sdpa, fullgraph=True)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out_compiled = compiled_fn(q, k, v, descale_q, descale_k, descale_v)

        # Metadata must match
        self.assertEqual(out_eager.shape, out_compiled.shape)
        self.assertEqual(out_eager.stride(), out_compiled.stride())
        self.assertEqual(out_eager.dtype, out_compiled.dtype)

        # Values should be identical (deterministic)
        self.assertTrue(
            torch.allclose(out_eager, out_compiled),
            f"Compiled output differs from eager. Max diff: {(out_eager - out_compiled).abs().max().item()}",
        )

    # ==================== VARLEN TESTS ====================

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_varlen_basic_functionality(self, device, dtype):
        torch.manual_seed(42)

        num_seqs = 2
        seq_len = 512
        heads = 16
        head_dim = 64

        total_tokens = num_seqs * seq_len
        q = torch.randn(
            total_tokens,
            heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.randn(
            total_tokens,
            heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        v = torch.randn(
            total_tokens,
            heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        cu_seq = torch.tensor(
            [0, seq_len, total_tokens], device=device, dtype=torch.int32
        )

        output = varlen_attn(q, k, v, cu_seq, cu_seq, seq_len, seq_len)

        self.assertEqual(output.shape, (total_tokens, heads, head_dim))
        self.assertEqual(output.device, torch.device(device))
        self.assertEqual(output.dtype, dtype)

        grad_out = torch.ones_like(output)
        dq, dk, dv = torch.autograd.grad(output, (q, k, v), grad_out)

        self.assertEqual(dq.shape, q.shape)
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)


instantiate_device_type_tests(TestFlashAttentionFA3, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
