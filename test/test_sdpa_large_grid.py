# Owner(s): ["module: sdpa"]

"""
Tests for SDPA mem-efficient attention with large batch*num_heads that
previously exceeded CUDA grid dimension limits (issue #142228).

The fix combines num_batches and num_heads into a single grid.y dimension
(grid.y = num_heads * num_batches) instead of mapping them to separate
grid.y and grid.z axes, each of which is limited to 65535.
"""

import torch
import unittest
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(
    not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    "Does not support mem-efficient attention",
)
class TestSDPALargeGrid(TestCase):
    """Verify that SDPA mem-efficient attention works when batch*num_heads > 65535."""

    def _run_sdpa(self, batch, num_heads, seq_len, head_dim, dtype=torch.float16):
        """Run forward pass through mem-efficient attention and return output."""
        q = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)

        out, _, _, _ = torch._scaled_dot_product_efficient_attention(
            q, k, v, attn_bias=None, compute_log_sumexp=False, dropout_p=0.0,
        )
        return out

    def _run_sdpa_and_verify(self, batch, num_heads, seq_len=2, head_dim=8):
        """Run SDPA and verify output against CPU reference."""
        dtype = torch.float32
        q = torch.randn(batch, seq_len, num_heads, head_dim, device="cpu", dtype=dtype)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device="cpu", dtype=dtype)
        v = torch.randn(batch, seq_len, num_heads, head_dim, device="cpu", dtype=dtype)

        q_cuda = q.cuda()
        k_cuda = k.cuda()
        v_cuda = v.cuda()

        out_cuda, _, _, _ = torch._scaled_dot_product_efficient_attention(
            q_cuda, k_cuda, v_cuda,
            attn_bias=None, compute_log_sumexp=False, dropout_p=0.0,
        )

        # CPU reference via math attention
        q_ref = q.transpose(1, 2)  # [B, H, S, D]
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        scale = head_dim ** -0.5
        attn = torch.matmul(q_ref * scale, k_ref.transpose(-2, -1))
        attn = torch.softmax(attn, dim=-1)
        out_ref = torch.matmul(attn, v_ref).transpose(1, 2)  # [B, S, H, D]

        self.assertEqual(
            out_cuda.cpu(), out_ref, atol=1e-2, rtol=1e-2,
            msg=f"Mismatch for batch={batch}, num_heads={num_heads}",
        )

    def test_large_batch_overflow_z(self):
        """batch=65536 exceeds old grid.z limit of 65535."""
        self._run_sdpa(batch=65536, num_heads=1, seq_len=2, head_dim=8)

    def test_large_heads_overflow_y(self):
        """num_heads=65536 exceeds old grid.y limit of 65535."""
        self._run_sdpa(batch=1, num_heads=65536, seq_len=2, head_dim=8)

    def test_combined_overflow(self):
        """batch*heads=65536, each individually within limit but combined exceeds 65535."""
        self._run_sdpa(batch=256, num_heads=256, seq_len=2, head_dim=8)

    def test_large_batch_correctness(self):
        """Verify numerical correctness with large batch (subset check)."""
        self._run_sdpa_and_verify(batch=256, num_heads=256)

    def test_large_heads_correctness(self):
        """Verify numerical correctness with large num_heads."""
        self._run_sdpa_and_verify(batch=1, num_heads=512)


if __name__ == "__main__":
    run_tests()
