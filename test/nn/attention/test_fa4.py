# Owner(s): ["module: sdpa"]

import importlib
import unittest
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.backends.cuda import SDPBackend
from torch.nn.attention import install_flash_attention_impl, sdpa_kernel
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


SdpaShape = namedtuple("Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"])


def _shape_to_fa(shape: SdpaShape) -> tuple[int, int, int, int]:
    return shape.batch, shape.seq_len, shape.num_heads, shape.head_dim


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


def flash_vs_math(q, k, v, is_causal=False, rtol=2):
    """
    Compare flash-attention backend against math backend in fp32 and low precision.

    Similar to flash_vs_triton from test_flex_flash.py but for SDPA backends.
    Compares:
    - Flash backend in low precision (fp16/bf16)
    - Math backend in fp32 (reference)
    - Math backend in low precision (fp16/bf16)

    Args:
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
    assert out_flash.shape == out_math_fp32.shape == out_math_low.shape
    assert not torch.isnan(out_flash).any()
    assert not torch.isnan(out_math_low).any()
    assert not torch.isnan(out_math_fp32).any()
    assert torch.isfinite(out_flash).all()
    assert torch.isfinite(out_math_low).all()
    assert torch.isfinite(out_math_fp32).all()

    # Calculate forward tolerance based on fp32 reference
    fwd_atol = 2 * (out_math_fp32 + 0.3 - 0.3 - out_math_fp32).abs().max().item()

    # Calculate errors
    math_low_error = (out_math_low - out_math_fp32).abs().max().item()
    flash_error = (out_flash - out_math_fp32).abs().max().item()

    # Assert flash error is within tolerance of math low precision error
    assert flash_error <= rtol * math_low_error + fwd_atol, (
        f"Flash error {flash_error:.2e} exceeds {rtol}x Math-low error {math_low_error:.2e} + {fwd_atol:.2e}"
    )

    return out_flash, out_math_low, out_math_fp32


class TestFlashAttentionFA4(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _fa4_dependencies_available():
            return
        # This might pollute tests.. TODO
        install_flash_attention_impl("FA4")

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    def _assert_flash_matches_math(
        self,
        device,
        shape: SdpaShape,
        dtype: torch.dtype,
        is_causal: bool,
        rtol: int = 2,
    ) -> None:
        fa_shape = _shape_to_fa(shape)
        q = torch.randn(
            fa_shape, dtype=dtype, device=device, requires_grad=True
        ).transpose(1, 2)
        k = torch.randn(
            fa_shape, dtype=dtype, device=device, requires_grad=True
        ).transpose(1, 2)
        v = torch.randn(
            fa_shape, dtype=dtype, device=device, requires_grad=True
        ).transpose(1, 2)

        # Forward pass comparison
        out_flash, out_math_low, out_math_fp32 = flash_vs_math(
            q, k, v, is_causal=is_causal, rtol=rtol
        )

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
        assert dq_flash_error <= rtol * dq_math_low_error + dq_atol, (
            f"dQ: Flash error {dq_flash_error:.2e} exceeds {rtol}x Math-low error {dq_math_low_error:.2e} + {dq_atol:.2e}"
        )

        dk_math_low_error = (dk_math_low - dk_math_fp32).abs().max().item()
        dk_flash_error = (dk_flash - dk_math_fp32).abs().max().item()
        assert dk_flash_error <= rtol * dk_math_low_error + dk_atol, (
            f"dK: Flash error {dk_flash_error:.2e} exceeds {rtol}x Math-low error {dk_math_low_error:.2e} + {dk_atol:.2e}"
        )

        dv_math_low_error = (dv_math_low - dv_math_fp32).abs().max().item()
        dv_flash_error = (dv_flash - dv_math_fp32).abs().max().item()
        assert dv_flash_error <= rtol * dv_math_low_error + dv_atol, (
            f"dV: Flash error {dv_flash_error:.2e} exceeds {rtol}x Math-low error {dv_math_low_error:.2e} + {dv_atol:.2e}"
        )

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("batch", [1, 2])
    @parametrize(
        "seq_len",
        [
            256,
        ],
    )
    @parametrize("heads", [4, 8])
    @parametrize("head_dim", [64, 128])
    @parametrize(
        "is_causal",
        [
            False,
        ],
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
        )


instantiate_device_type_tests(TestFlashAttentionFA4, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
