# Owner(s): ["module: sdpa"]

import importlib
import unittest
from unittest.mock import patch

from _fa_test_common import flash_vs_math, FlashAttentionTestMixin, SdpaShape

import torch
import torch.nn.functional as F
from torch.backends.cuda import can_use_flash_attention, SDPAParams, SDPBackend
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
    sdpa_kernel,
)
from torch.nn.attention.experimental._scaled_dot_product_attention_quantized import (
    _scaled_dot_product_attention_quantized,
)
from torch.testing._internal.common_cuda import IS_SM10X
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


def _fa4_dependencies_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major not in (9, 10):
        return False
    try:
        importlib.import_module("flash_attn.cute.interface")
    except ModuleNotFoundError:
        return False
    return True


class TestFlashAttentionFA4(FlashAttentionTestMixin, TestCase):
    hw_classification = HardwareClassification.CUDA

    # Mixin configuration
    impl_name = "FA4"
    fwd_kernel_patterns = ["flash_attncute", "flash_fwd"]
    bwd_kernel_patterns = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _fa4_dependencies_available():
            return
        # This might pollute tests.. TODO
        activate_flash_attention_impl("FA4")

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
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
        # TODO: Getting bad TMA setup on dO w/ headdim = 64, will take a look
        test_backward = head_dim == 128 and dtype == torch.float16
        shape = SdpaShape(batch, heads, seq_len, head_dim)
        self._assert_flash_matches_math(
            device,
            shape=shape,
            dtype=dtype,
            is_causal=is_causal,
            # Bwd is consistently erroring
            test_backward=test_backward,
        )

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fa4_kernel_called(self, device, dtype):
        self._test_kernel_called(device, dtype)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    def test_mixed_head_dims(self, device):
        q = (
            torch.randn(1, 128, 2, 192, dtype=torch.float16, device=device)
            .transpose(1, 2)
            .requires_grad_()
        )
        k = torch.randn_like(q, memory_format=torch.preserve_format).requires_grad_()
        v = (
            torch.randn(1, 128, 2, 128, dtype=torch.float16, device=device)
            .transpose(1, 2)
            .requires_grad_()
        )

        out, _, _ = flash_vs_math(self, q, k, v, is_causal=True)
        self.assertEqual(out.shape, v.shape)
        self.assertTrue(out.transpose(1, 2).is_contiguous())

        grads = torch.autograd.grad(out, (q, k, v), torch.randn_like(out))
        for grad, expected in zip(grads, (q, k, v)):
            self.assertEqual(grad.shape, expected.shape)
            self.assertTrue(torch.isfinite(grad).all())

        def sdpa(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            compiled_out = torch.compile(sdpa, fullgraph=True)(q, k, v)
        self.assertEqual(compiled_out.shape, out.shape)
        self.assertEqual(compiled_out.stride(), out.stride())
        self.assertEqual(compiled_out, out)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    def test_mixed_head_dims_with_expanded_query(self, device):
        q = torch.randn(1, 1, 128, 192, dtype=torch.float16, device=device).expand(
            2, 4, -1, -1
        )
        k = torch.randn(1, 1, 128, 192, dtype=torch.float16, device=device).expand(
            2, 4, -1, -1
        )
        v = torch.randn(1, 1, 128, 128, dtype=torch.float16, device=device).expand(
            2, 4, -1, -1
        )

        out, _, _ = flash_vs_math(self, q, k, v, is_causal=True)
        self.assertEqual(out.shape, v.shape)
        self.assertEqual(out.stride(-1), 1)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    def test_head_dim_selection_uses_fa4_contract(self, device):
        q = torch.empty(1, 1, 128, 192, dtype=torch.float16, device=device)
        k = torch.empty_like(q)
        v = torch.empty(1, 1, 128, 128, dtype=torch.float16, device=device)
        params = SDPAParams(q, k, v, None, 0.0, True, False)
        self.assertTrue(can_use_flash_attention(params))
        dropout_params = SDPAParams(q, k, v, None, 0.1, True, False)
        self.assertFalse(can_use_flash_attention(dropout_params))

        if torch.cuda.get_device_capability(device)[0] == 10:
            unsupported_params = SDPAParams(q, k, q, None, 0.0, True, False)
            self.assertFalse(can_use_flash_attention(unsupported_params))

        restore_flash_attention_impl()
        try:
            self.assertFalse(can_use_flash_attention(params))
        finally:
            activate_flash_attention_impl("FA4")

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    def test_multiple_activate(self):
        self._test_multiple_activate_impl()

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_metadata(self, device, dtype, is_causal):
        """Test that torch.compile preserves tensor metadata (shape, stride, dtype)."""
        self._test_compiled_sdpa_metadata(device, dtype, is_causal)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_matches_math(self, device, dtype, is_causal):
        """Test compiled FA4 numerical correctness against math backend."""
        self._test_compiled_sdpa_matches_math(device, dtype, is_causal)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_backward_matches_math(self, device, dtype, is_causal):
        """Test compiled FA4 backward numerical correctness against math backend."""
        # FA4 uses head_dim=128, no min_atol override
        self._test_compiled_sdpa_backward_matches_math(
            device, dtype, is_causal, head_dim=128
        )

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    def test_attention_preserves_query_layout(self, device):
        """Test that FA4 output has the same layout as the query tensor."""
        self._test_attention_preserves_query_layout(device)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("deterministic", [False, True])
    def test_deterministic_flag_passed_to_backward(self, device, deterministic):
        """Test that deterministic flag is correctly passed through to FA4 backward kernel."""
        from torch.nn.attention import _fa4

        shape = SdpaShape(2, 4, 512, 128)
        q = torch.randn(shape, dtype=torch.float16, device=device, requires_grad=True)
        k = torch.randn(shape, dtype=torch.float16, device=device, requires_grad=True)
        v = torch.randn(shape, dtype=torch.float16, device=device, requires_grad=True)

        torch.use_deterministic_algorithms(deterministic)

        try:
            _fa4._fa4_import_module.cache_clear()

            with patch("torch.nn.attention._fa4._fa4_import_module") as mock_import:
                mock_module = mock_import.return_value

                # FA4 uses BSHD layout internally, so mock returns BSHD
                q_transposed = q.transpose(1, 2)
                mock_module._flash_attn_fwd.return_value = (
                    torch.randn_like(q_transposed),
                    torch.randn(
                        q.size(0),
                        q.size(2),
                        q.size(1),
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                mock_module._flash_attn_bwd.return_value = (
                    torch.randn_like(q_transposed),
                    torch.randn_like(q_transposed),
                    torch.randn_like(q_transposed),
                )

                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
                    )
                    grad_out = torch.randn_like(out)
                    out.backward(grad_out)

                self.assertTrue(mock_module._flash_attn_bwd.called)
                call_kwargs = mock_module._flash_attn_bwd.call_args.kwargs
                self.assertIn("deterministic", call_kwargs)
                self.assertEqual(
                    call_kwargs["deterministic"],
                    deterministic,
                    lambda msg: f"{msg}\nExpected deterministic={deterministic} but got {call_kwargs['deterministic']}",
                )
        finally:
            torch.use_deterministic_algorithms(False)
            _fa4._fa4_import_module.cache_clear()

    @unittest.skipUnless(
        _fa4_dependencies_available() and IS_SM10X,
        "FA4's fp8 forward needs FA4 and an SM100-family GPU",
    )
    @parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @parametrize("is_causal", [False, True])
    def test_fp8_sdpa_matches_dequantized_reference(self, device, dtype, is_causal):
        torch.manual_seed(0)
        batch, num_heads, num_kv_heads, seq_len, head_dim = 2, 8, 2, 256, 128
        heads_per_kv_head = num_heads // num_kv_heads
        # Powers of two, so the dequantized reference inputs are exact.
        descales = {
            "q_descale": torch.tensor([[0.5, 2.0], [1.0, 4.0]], device=device),
            "k_descale": torch.tensor([[2.0, 1.0], [0.5, 4.0]], device=device),
            "v_descale": torch.tensor([[1.0, 0.5], [2.0, 1.0]], device=device),
        }
        # Max error per unit of v_descale: FA4 rounds P to the fp8 dtype for P @ V.
        # e5m2 has one mantissa bit less than e4m3, so its rounding step and bound are twice e4m3's.
        bound = {torch.float8_e4m3fn: 0.25, torch.float8_e5m2: 0.5}[dtype]

        def fp8_randn(heads):
            shape = (batch, heads, seq_len, head_dim)
            return torch.randn(shape, device=device, dtype=torch.bfloat16).to(dtype)

        def dequantize(x, descale):
            per_head = descale.repeat_interleave(x.size(1) // descale.size(1), dim=1)
            return x.float() * per_head[:, :, None, None]

        q, k, v = fp8_randn(num_heads), fp8_randn(num_kv_heads), fp8_randn(num_kv_heads)
        with torch.no_grad():
            out = _scaled_dot_product_attention_quantized(
                q, k, v, is_causal=is_causal, **descales
            )
            with sdpa_kernel(SDPBackend.MATH):
                expected = F.scaled_dot_product_attention(
                    dequantize(q, descales["q_descale"]),
                    dequantize(k, descales["k_descale"]),
                    dequantize(v, descales["v_descale"]),
                    is_causal=is_causal,
                    enable_gqa=True,
                )

        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, q.shape)
        # Max error per (batch, KV head) block.
        error = (
            (out.float() - expected)
            .abs()
            .unflatten(1, (num_kv_heads, heads_per_kv_head))
            .amax(dim=(2, 3, 4))
        )
        self.assertTrue(
            (error <= bound * descales["v_descale"]).all(),
            f"max error per (batch, KV head): {error}",
        )


instantiate_device_type_tests(TestFlashAttentionFA4, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
