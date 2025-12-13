# Owner(s): ["module: inductor"]
"""
Test SDPA autotuning with custom op infrastructure.

This test demonstrates autotuning between:
1. Triton SDPA kernel (from sdpa.py)
2. PyTorch's torch.nn.functional.scaled_dot_product_attention

Tests that the register_custom_op_autotuning API correctly handles
non-tensor kwargs like is_causal by preserving parameter names from op schema.
"""

import unittest
from typing import Optional

import torch
import torch.nn.functional as F
from torch._inductor.kernel.custom_op import (
    CustomOpConfig,
    register_custom_op_autotuning,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import skipIfRocm
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


# Skip if Triton is not available
try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# Import the Triton SDPA kernel from the same directory
# Need to add the test directory to path for the import
if HAS_TRITON and HAS_GPU:
    import sys
    from pathlib import Path

    test_dir = Path(__file__).parent
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))

    from test_sdpa_triton_kernel import sdpa as triton_sdpa


def decomp_triton_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    """Decomposition using Triton SDPA kernel."""
    triton_scale = 0.0 if scale is None else scale
    return triton_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=triton_scale,
        enable_gqa=False,
    )


@torch.library.custom_op("sdpa_autotune_test::pytorch_sdpa", mutates_args=())
def pytorch_sdpa_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """PyTorch's standard SDPA wrapped as a custom op."""
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@pytorch_sdpa_op.register_fake
def pytorch_sdpa_op_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return torch.empty_like(query)


def decomp_pytorch_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    """Decomposition using PyTorch's standard SDPA."""
    return pytorch_sdpa_op(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@torch.library.custom_op("sdpa_autotune_test::sdpa", mutates_args=())
def autotuned_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Autotuned Scaled Dot-Product Attention.

    Benchmarks Triton SDPA vs PyTorch SDPA and uses the fastest.
    Default (eager) uses PyTorch SDPA.
    """
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@autotuned_sdpa.register_fake
def autotuned_sdpa_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return torch.empty_like(query)


# Register for autotuning
if HAS_TRITON and HAS_GPU:
    register_custom_op_autotuning(
        custom_op=autotuned_sdpa,
        configs=[
            CustomOpConfig(decomp_triton_sdpa),
            CustomOpConfig(decomp_pytorch_sdpa),
        ],
        name="sdpa_autotuned_test",
        input_gen_fns={
            "query": lambda fake: torch.randn(
                fake.shape, device=GPU_TYPE, dtype=torch.bfloat16
            ),
            "key": lambda fake: torch.randn(
                fake.shape, device=GPU_TYPE, dtype=torch.bfloat16
            ),
            "value": lambda fake: torch.randn(
                fake.shape, device=GPU_TYPE, dtype=torch.bfloat16
            ),
        },
    )


@unittest.skipIf(not HAS_GPU, "GPU required")
@unittest.skipIf(not HAS_TRITON, "Triton required")
class TestSDPAAutotuning(TestCase):
    """Test SDPA autotuning with custom op infrastructure."""

    @skipIfRocm
    def test_sdpa_autotuning_basic(self):
        """Test basic SDPA autotuning between Triton and PyTorch."""
        B, H, L_q, L_kv, D = 2, 8, 512, 512, 64

        query = torch.randn(B, H, L_q, D, device=GPU_TYPE, dtype=torch.bfloat16)
        key = torch.randn(B, H, L_kv, D, device=GPU_TYPE, dtype=torch.bfloat16)
        value = torch.randn(B, H, L_kv, D, device=GPU_TYPE, dtype=torch.bfloat16)

        with torch.no_grad():
            expected = F.scaled_dot_product_attention(query, key, value)

        # Test eager execution
        with torch.no_grad():
            result_eager = autotuned_sdpa(query, key, value)
        torch.testing.assert_close(result_eager, expected, atol=1e-2, rtol=1e-2)

        # Test compiled execution (triggers autotuning)
        @torch.compile
        def compiled_sdpa(q, k, v):
            return autotuned_sdpa(q, k, v)

        with torch.no_grad():
            result_compiled = compiled_sdpa(query, key, value)
        torch.testing.assert_close(result_compiled, expected, atol=1e-2, rtol=1e-2)

    @skipIfRocm
    def test_sdpa_causal_attention(self):
        """Test SDPA with causal masking (tests non-tensor kwarg handling)."""
        B, H, L, D = 2, 8, 256, 64

        query = torch.randn(B, H, L, D, device=GPU_TYPE, dtype=torch.bfloat16)
        key = torch.randn(B, H, L, D, device=GPU_TYPE, dtype=torch.bfloat16)
        value = torch.randn(B, H, L, D, device=GPU_TYPE, dtype=torch.bfloat16)

        with torch.no_grad():
            expected = F.scaled_dot_product_attention(query, key, value, is_causal=True)

        @torch.compile
        def compiled_causal_sdpa(q, k, v):
            return autotuned_sdpa(q, k, v, is_causal=True)

        with torch.no_grad():
            result = compiled_causal_sdpa(query, key, value)

        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
