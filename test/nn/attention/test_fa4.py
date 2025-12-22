# Owner(s): ["module: sdpa"]

import importlib
import itertools
import unittest
from collections import namedtuple
from contextlib import contextmanager
from unittest.mock import patch

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


@contextmanager
def cuda_kernel_profiler(kernel_pattern="flash_attncute"):
    result = {"found": False, "kernel_names": []}

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        yield result

    kernel_names = [
        evt.name
        for evt in prof.events()
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.name
    ]
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


class TestFlashAttentionFA4(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _fa4_dependencies_available():
            return
        # This might pollute tests.. TODO
        activate_flash_attention_impl("FA4")

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
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
        try:
            shape = SdpaShape(2, 4, 512, 128)
            q = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
            k = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
            v = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)

            for backend in ["flash_attncute", "flash_fwd"]:
                with cuda_kernel_profiler(backend) as prof_result:
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        out = F.scaled_dot_product_attention(
                            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
                        )
                        out.sum().backward()

                self.assertTrue(
                    prof_result["found"],
                    f"FA4 CUTE kernel not found in forward/backward. Available kernels: {prof_result['kernel_names']}",
                )

                q.grad = None
                k.grad = None
                v.grad = None
                restore_flash_attention_impl()

            with cuda_kernel_profiler(backend) as prof_result:
                with sdpa_kernel(SDPBackend.MATH):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
                    )
                    out.sum().backward()

            self.assertFalse(
                prof_result["found"],
                f"FA4 CUTE kernel unexpectedly found with MATH backend. Kernels: {prof_result['kernel_names']}",
            )
        finally:
            activate_flash_attention_impl("FA4")  # reset for next test

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
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
            activate_flash_attention_impl("FA4")  # reset for next test

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_metadata(self, device, dtype, is_causal):
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

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_matches_math(self, device, dtype, is_causal):
        """Test compiled FA4 numerical correctness against math backend."""
        shape = SdpaShape(2, 8, 512, 64)
        q = torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)

        def sdpa_fn(q, k, v):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
            )

        # Compiled FA4
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

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_backward_matches_math(self, device, dtype, is_causal):
        """Test compiled FA4 backward numerical correctness against math backend."""
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

        # Compiled FA4 forward + backward
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
        for name, dcompiled, dmath_fp32, dmath_low in [
            ("dQ", dq_compiled, dq_math_fp32.to(dtype), dq_math_low),
            ("dK", dk_compiled, dk_math_fp32.to(dtype), dk_math_low),
            ("dV", dv_compiled, dv_math_fp32.to(dtype), dv_math_low),
        ]:
            atol = 2 * (dmath_fp32 + 0.3 - 0.3 - dmath_fp32).abs().max().item()
            math_low_error = (dmath_low - dmath_fp32).abs().max().item()
            compiled_error = (dcompiled - dmath_fp32).abs().max().item()
            self.assertLessEqual(
                compiled_error,
                rtol * math_low_error + atol,
                f"{name}: Compiled error {compiled_error:.2e} exceeds "
                f"{rtol}x Math-low error {math_low_error:.2e} + {atol:.2e}",
            )

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    def test_attention_preserves_query_layout(self, device):
        """Test that FA4 output has the same layout as the query tensor."""

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
                    f"Expected deterministic={deterministic} but got {call_kwargs['deterministic']}",
                )
        finally:
            torch.use_deterministic_algorithms(False)
            _fa4._fa4_import_module.cache_clear()


instantiate_device_type_tests(TestFlashAttentionFA4, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
