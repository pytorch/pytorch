# Owner(s): ["module: sdpa"]

import importlib
import itertools
import unittest
from collections import namedtuple
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.backends.cuda import SDPBackend
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
    register_flash_attention_impl,
    restore_flash_attention_impl,
    sdpa_kernel,
)
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


SdpaShape = namedtuple("Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"])


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


@contextmanager
def cuda_kernel_profiler(kernel_pattern="flash_attn_3"):
    result = {"found": False, "kernel_names": []}

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        yield result

    # Get all event names (both CPU and CUDA activities)
    kernel_names = [evt.name for evt in prof.events() if evt.name]
    result["kernel_names"] = kernel_names
    result["found"] = any(kernel_pattern in name for name in kernel_names)


def flash_vs_math(test_case, q, k, v, is_causal=False, rtol=2):
    """
    Compare flash-attention backend against math backend in fp32 and low precision.

    Similar to flash_vs_triton from test_flex_flash.py but for SDPA backends.
    Compares:
    - Flash backend in low precision (fp16/bf16)
    - Math backend in fp32 (reference)
    - Math backend in low precision (fp16/bf16)

    Args:
        test_case: TestCase instance for assertions
        q, k, v: Input tensors in low precision (fp16/bf16) with requires_grad=True
        is_causal: Whether to use causal masking
        rtol: Relative tolerance multiplier for error comparison

    Returns:
        out_flash, out_math_low, out_math_fp32
    """
    # Flash attention in low precision
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out_flash = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
        )

    # Math backend in fp32 (reference)
    with sdpa_kernel(SDPBackend.MATH):
        out_math_fp32 = F.scaled_dot_product_attention(
            q.to(torch.float32),
            k.to(torch.float32),
            v.to(torch.float32),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        ).to(q.dtype)

    # Math backend in low precision
    with sdpa_kernel(SDPBackend.MATH):
        out_math_low = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
        )

    # sanity checks
    test_case.assertEqual(out_flash.shape, out_math_fp32.shape)
    test_case.assertEqual(out_flash.shape, out_math_low.shape)
    test_case.assertFalse(torch.isnan(out_flash).any())
    test_case.assertFalse(torch.isnan(out_math_low).any())
    test_case.assertFalse(torch.isnan(out_math_fp32).any())
    test_case.assertTrue(torch.isfinite(out_flash).all())
    test_case.assertTrue(torch.isfinite(out_math_low).all())
    test_case.assertTrue(torch.isfinite(out_math_fp32).all())

    # Calculate forward tolerance based on fp32 reference
    fwd_atol = 2 * (out_math_fp32 + 0.3 - 0.3 - out_math_fp32).abs().max().item()

    # Calculate errors
    math_low_error = (out_math_low - out_math_fp32).abs().max().item()
    flash_error = (out_flash - out_math_fp32).abs().max().item()

    # Assert flash error is within tolerance of math low precision error
    test_case.assertLessEqual(
        flash_error,
        rtol * math_low_error + fwd_atol,
        f"Flash error {flash_error:.2e} exceeds {rtol}x Math-low error {math_low_error:.2e} + {fwd_atol:.2e}",
    )

    return out_flash, out_math_low, out_math_fp32


class DummyHandle:
    def __init__(self, name):
        self.name = name
        self.removed = False

    def remove(self):
        self.removed = True


class TestFlashAttentionFA3(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _fa3_dependencies_available():
            return
        activate_flash_attention_impl("FA3")

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    def _assert_flash_matches_math(
        self,
        device,
        shape: SdpaShape,
        dtype: torch.dtype,
        is_causal: bool,
        rtol: int = 2,
        test_backward: bool = True,
    ) -> None:
        q = torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)

        # Forward pass comparison
        out_flash, out_math_low, out_math_fp32 = flash_vs_math(
            self, q, k, v, is_causal=is_causal, rtol=rtol
        )

        if test_backward:
            # Backward pass comparison
            g = torch.randn_like(out_flash)

            # Flash gradients
            dq_flash, dk_flash, dv_flash = torch.autograd.grad(
                out_flash, (q, k, v), g, retain_graph=True
            )

            # Math fp32 gradients (reference)
            dq_math_fp32, dk_math_fp32, dv_math_fp32 = torch.autograd.grad(
                out_math_fp32, (q, k, v), g, retain_graph=True
            )

            # Math low precision gradients
            dq_math_low, dk_math_low, dv_math_low = torch.autograd.grad(
                out_math_low, (q, k, v), g
            )

            # Calculate gradient tolerances (similar to flash-attention tests)
            dq_atol = 2 * (dq_math_fp32 + 0.3 - 0.3 - dq_math_fp32).abs().max().item()
            dk_atol = 2 * (dk_math_fp32 + 0.3 - 0.3 - dk_math_fp32).abs().max().item()
            dv_atol = 2 * (dv_math_fp32 + 0.3 - 0.3 - dv_math_fp32).abs().max().item()

            # Check flash gradients are within tolerance of math low precision
            dq_math_low_error = (dq_math_low - dq_math_fp32).abs().max().item()
            dq_flash_error = (dq_flash - dq_math_fp32).abs().max().item()
            self.assertLessEqual(
                dq_flash_error,
                rtol * dq_math_low_error + dq_atol,
                f"dQ: Flash error {dq_flash_error:.2e} exceeds {rtol}x Math-low error {dq_math_low_error:.2e} + {dq_atol:.2e}",
            )

            dk_math_low_error = (dk_math_low - dk_math_fp32).abs().max().item()
            dk_flash_error = (dk_flash - dk_math_fp32).abs().max().item()
            self.assertLessEqual(
                dk_flash_error,
                rtol * dk_math_low_error + dk_atol,
                f"dK: Flash error {dk_flash_error:.2e} exceeds {rtol}x Math-low error {dk_math_low_error:.2e} + {dk_atol:.2e}",
            )

            dv_math_low_error = (dv_math_low - dv_math_fp32).abs().max().item()
            dv_flash_error = (dv_flash - dv_math_fp32).abs().max().item()
            self.assertLessEqual(
                dv_flash_error,
                rtol * (dv_math_low_error + dv_atol),
                f"dV: Flash error {dv_flash_error:.2e} exceeds {rtol}x (Math-low error {dv_math_low_error:.2e} + {dv_atol:.2e})",
            )

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
        try:
            shape = SdpaShape(2, 4, 512, 128)
            q = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
            k = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
            v = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)

            activate_flash_attention_impl("FA3")

            # FA3 forward shows as flash_attn_3::fwd in profiler
            for kernel_pattern in ["flash_attn_3::fwd", "flash_attn_3::bwd"]:
                with cuda_kernel_profiler(kernel_pattern) as prof_result:
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        out = F.scaled_dot_product_attention(
                            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
                        )
                        out.sum().backward()

                self.assertTrue(
                    prof_result["found"],
                    f"FA3 kernel not found in forward/backward. Available kernels: {prof_result['kernel_names']}",
                )

                q.grad = None
                k.grad = None
                v.grad = None
            restore_flash_attention_impl()

            with cuda_kernel_profiler("flash_attn_3") as prof_result:
                with sdpa_kernel(SDPBackend.MATH):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
                    )
                    out.sum().backward()

            self.assertFalse(
                prof_result["found"],
                f"FA3 kernel unexpectedly found with MATH backend. Kernels: {prof_result['kernel_names']}",
            )
        finally:
            activate_flash_attention_impl("FA3")  # reset for next test

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    def test_multiple_activate(self):
        try:
            handles = {}

            def make_dummy_impl(name):
                def impl():
                    handle = DummyHandle(name)
                    handles[name] = handle
                    return handle

                return impl

            restore_flash_attention_impl()  # back to default

            register_flash_attention_impl(
                "dummy_impl_1", register_fn=make_dummy_impl("dummy_impl_1")
            )
            register_flash_attention_impl(
                "dummy_impl_2", register_fn=make_dummy_impl("dummy_impl_2")
            )

            self.assertIsNone(current_flash_attention_impl())

            activate_flash_attention_impl("dummy_impl_1")
            self.assertEqual(current_flash_attention_impl(), "dummy_impl_1")
            self.assertIn("dummy_impl_1", handles)

            activate_flash_attention_impl("dummy_impl_2")
            self.assertEqual(current_flash_attention_impl(), "dummy_impl_2")
            self.assertIn("dummy_impl_2", handles)

            # with every subsequent activate() call, the previously registered custom impl should be removed
            self.assertTrue(
                handles["dummy_impl_1"].removed, "dummy_impl_1 should be removed"
            )
        finally:
            activate_flash_attention_impl("FA3")  # reset for next test

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa3_metadata(self, device, dtype, is_causal):
        """Test that torch.compile preserves tensor metadata (shape, stride, dtype)."""
        shape = SdpaShape(2, 8, 512, 64)
        q = torch.randn(shape, dtype=dtype, device=device)
        k = torch.randn(shape, dtype=dtype, device=device)
        v = torch.randn(shape, dtype=dtype, device=device)

        def sdpa_fn(q, k, v):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
            )

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out_eager = sdpa_fn(q, k, v)

        compiled_sdpa = torch.compile(sdpa_fn, fullgraph=True)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out_compiled = compiled_sdpa(q, k, v)

        # Metadata must match between eager and compiled
        self.assertEqual(out_eager.shape, out_compiled.shape)
        self.assertEqual(out_eager.stride(), out_compiled.stride())
        self.assertEqual(out_eager.dtype, out_compiled.dtype)
        self.assertEqual(out_eager.device, out_compiled.device)

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa3_matches_math(self, device, dtype, is_causal):
        """Test compiled FA3 numerical correctness against math backend."""
        shape = SdpaShape(2, 8, 512, 64)
        q = torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)

        def sdpa_fn(q, k, v):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
            )

        # Compiled FA3
        compiled_sdpa = torch.compile(sdpa_fn, fullgraph=True)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out_compiled = compiled_sdpa(q, k, v)

        # Math backend in fp32 (reference)
        with sdpa_kernel(SDPBackend.MATH):
            out_math_fp32 = F.scaled_dot_product_attention(
                q.to(torch.float32),
                k.to(torch.float32),
                v.to(torch.float32),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
            ).to(dtype)

        # Math backend in low precision
        with sdpa_kernel(SDPBackend.MATH):
            out_math_low = sdpa_fn(q, k, v)

        # Sanity checks
        self.assertFalse(torch.isnan(out_compiled).any())
        self.assertTrue(torch.isfinite(out_compiled).all())

        # Calculate tolerance based on fp32 reference
        fwd_atol = 2 * (out_math_fp32 + 0.3 - 0.3 - out_math_fp32).abs().max().item()
        math_low_error = (out_math_low - out_math_fp32).abs().max().item()
        compiled_error = (out_compiled - out_math_fp32).abs().max().item()

        rtol = 2
        self.assertLessEqual(
            compiled_error,
            rtol * math_low_error + fwd_atol,
            f"Compiled error {compiled_error:.2e} exceeds "
            f"{rtol}x Math-low error {math_low_error:.2e} + {fwd_atol:.2e}",
        )

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa3_backward_matches_math(self, device, dtype, is_causal):
        """Test compiled FA3 backward numerical correctness against math backend."""
        shape = SdpaShape(2, 8, 512, 128)

        def make_tensors():
            return (
                torch.randn(shape, dtype=dtype, device=device, requires_grad=True),
                torch.randn(shape, dtype=dtype, device=device, requires_grad=True),
                torch.randn(shape, dtype=dtype, device=device, requires_grad=True),
            )

        def sdpa_fn(q, k, v):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
            )

        # Compiled FA3 forward + backward
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        q, k, v = make_tensors()
        compiled_sdpa = torch.compile(sdpa_fn, fullgraph=True)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out_compiled = compiled_sdpa(q, k, v)
        grad_out = torch.randn_like(out_compiled)
        dq_compiled, dk_compiled, dv_compiled = torch.autograd.grad(
            out_compiled, (q, k, v), grad_out
        )

        # Math fp32 reference
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        q_ref, k_ref, v_ref = make_tensors()
        with sdpa_kernel(SDPBackend.MATH):
            out_math_fp32 = F.scaled_dot_product_attention(
                q_ref.to(torch.float32),
                k_ref.to(torch.float32),
                v_ref.to(torch.float32),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        dq_math_fp32, dk_math_fp32, dv_math_fp32 = torch.autograd.grad(
            out_math_fp32, (q_ref, k_ref, v_ref), grad_out.to(torch.float32)
        )

        # Math low precision
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        q_low, k_low, v_low = make_tensors()
        with sdpa_kernel(SDPBackend.MATH):
            out_math_low = sdpa_fn(q_low, k_low, v_low)
        dq_math_low, dk_math_low, dv_math_low = torch.autograd.grad(
            out_math_low, (q_low, k_low, v_low), grad_out
        )

        # Sanity checks
        for t in [dq_compiled, dk_compiled, dv_compiled]:
            self.assertFalse(torch.isnan(t).any())
            self.assertTrue(torch.isfinite(t).all())

        rtol = 2
        # Minimum tolerance to handle edge cases where math_low_error is 0
        min_atol = 0.05 if dtype == torch.bfloat16 else 0.01
        for name, dcompiled, dmath_fp32, dmath_low in [
            ("dQ", dq_compiled, dq_math_fp32.to(dtype), dq_math_low),
            ("dK", dk_compiled, dk_math_fp32.to(dtype), dk_math_low),
            ("dV", dv_compiled, dv_math_fp32.to(dtype), dv_math_low),
        ]:
            atol = max(
                2 * (dmath_fp32 + 0.3 - 0.3 - dmath_fp32).abs().max().item(),
                min_atol,
            )
            math_low_error = (dmath_low - dmath_fp32).abs().max().item()
            compiled_error = (dcompiled - dmath_fp32).abs().max().item()
            self.assertLessEqual(
                compiled_error,
                rtol * math_low_error + atol,
                f"{name}: Compiled error {compiled_error:.2e} exceeds "
                f"{rtol}x Math-low error {math_low_error:.2e} + {atol:.2e}",
            )

    @unittest.skipUnless(_fa3_dependencies_available(), "FA3 backend unavailable")
    def test_attention_preserves_query_layout(self, device):
        """Test that FA3 output has the same layout as the query tensor."""

        def test_attention(permute_order: list[int]):
            BHSqD = [4, 16, 256, 64]
            BHSkvD = [4, 16, 512, 64]

            shape_q = [BHSqD[idx] for idx in permute_order]
            shape_kv = [BHSkvD[idx] for idx in permute_order]
            reverse = [permute_order.index(idx) for idx in range(4)]
            q = torch.randn(
                *shape_q, dtype=torch.bfloat16, device=device, requires_grad=False
            ).permute(reverse)
            k = torch.randn(
                *shape_kv, dtype=torch.bfloat16, device=device, requires_grad=False
            ).permute(reverse)
            v = torch.randn(
                *shape_kv, dtype=torch.bfloat16, device=device, requires_grad=False
            ).permute(reverse)
            self.assertEqual(q.shape, BHSqD)
            self.assertEqual(k.shape, BHSkvD)
            self.assertEqual(v.shape, BHSkvD)

            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(q, k, v)
            self.assertTrue(
                out.permute(permute_order).is_contiguous(),
                f"Output layout mismatch for permute_order={permute_order}, "
                f"q.stride()={q.stride()}, out.stride()={out.stride()}",
            )

        permutable = [0, 1, 2]
        permute_orders = itertools.permutations(permutable)

        for permute_order in permute_orders:
            test_attention(list(permute_order) + [3])

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
                out = F._scaled_dot_product_attention_fp8(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    q_descale=descale_q,
                    k_descale=descale_k,
                    v_descale=descale_v,
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
                out_fp8 = F._scaled_dot_product_attention_fp8(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    is_causal=is_causal,
                    q_descale=descale_q,
                    k_descale=descale_k,
                    v_descale=descale_v,
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
                out = F._scaled_dot_product_attention_fp8(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    is_causal=False,
                    q_descale=descale_q,
                    k_descale=descale_k,
                    v_descale=descale_v,
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
                _ = F._scaled_dot_product_attention_fp8(
                    q,
                    k,
                    v,
                    is_causal=False,
                    q_descale=descale_q,
                    k_descale=descale_k,
                    v_descale=descale_v,
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
            return F._scaled_dot_product_attention_fp8(
                q, k, v, is_causal=is_causal, q_descale=dq, k_descale=dk, v_descale=dv
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


instantiate_device_type_tests(TestFlashAttentionFA3, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
