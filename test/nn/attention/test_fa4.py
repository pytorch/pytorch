# Owner(s): ["module: sdpa"]

import importlib
import unittest
from collections import namedtuple
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.backends.cuda import SDPBackend
from torch.nn.attention import activate_flash_attention_impl, sdpa_kernel
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
        shape = SdpaShape(2, 4, 512, 128)
        q = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
        v = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)

        with cuda_kernel_profiler("flash_attncute") as prof_result:
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

        with cuda_kernel_profiler("flash_attncute") as prof_result:
            with sdpa_kernel(SDPBackend.MATH):
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                out.sum().backward()

        self.assertFalse(
            prof_result["found"],
            f"FA4 CUTE kernel unexpectedly found with MATH backend. Kernels: {prof_result['kernel_names']}",
        )


instantiate_device_type_tests(TestFlashAttentionFA4, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
