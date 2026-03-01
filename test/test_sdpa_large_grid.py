"""
Test for SDPA memory-efficient attention with large batch*num_heads.

When num_batches or num_heads exceeds 65535, the CUDA grid dimensions
overflow because getBlocksGrid() maps num_heads and num_batches to
grid.y and grid.z axes, each of which is limited to 65535.
"""

import unittest

import torch
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestSDPALargeGrid(TestCase):

    def _run_sdpa_mem_eff(
        self, batch, num_heads, seq_len=1, head_dim=1, dtype=torch.float16
    ):
        """Run SDPA memory-efficient attention and return output."""
        q = torch.randn(
            batch, seq_len, num_heads, head_dim, device="cuda", dtype=dtype
        )
        k = torch.randn(
            batch, seq_len, num_heads, head_dim, device="cuda", dtype=dtype
        )
        v = torch.randn(
            batch, seq_len, num_heads, head_dim, device="cuda", dtype=dtype
        )

        out, _, _, _ = torch._scaled_dot_product_efficient_attention(
            q,
            k,
            v,
            attn_bias=None,
            compute_log_sumexp=False,
            dropout_p=0.0,
        )
        return out

    def _run_correctness_check(
        self, batch, num_heads, seq_len=1, head_dim=16, dtype=torch.float32
    ):
        """Compare CUDA mem-eff attention result against CPU reference."""
        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype)
        k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype)
        v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype)

        q_cuda = q.cuda()
        k_cuda = k.cuda()
        v_cuda = v.cuda()

        out_cuda, _, _, _ = torch._scaled_dot_product_efficient_attention(
            q_cuda,
            k_cuda,
            v_cuda,
            attn_bias=None,
            compute_log_sumexp=False,
            dropout_p=0.0,
        )

        # CPU reference via math attention
        q_ref = q.transpose(1, 2)  # [B, H, S, D]
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        scale = head_dim**-0.5
        attn = torch.matmul(q_ref * scale, k_ref.transpose(-2, -1))
        attn = torch.softmax(attn, dim=-1)
        out_ref = torch.matmul(attn, v_ref).transpose(1, 2)  # [B, S, H, D]

        self.assertEqual(
            out_cuda.cpu(),
            out_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Mismatch for batch={batch}, num_heads={num_heads}",
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Memory efficient attention not supported",
    )
    def test_large_batch_overflow_z(self):
        """Test batch=65536 which would overflow grid.z (was limited to 65535)."""
        self._run_sdpa_mem_eff(batch=65536, num_heads=1)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Memory efficient attention not supported",
    )
    def test_large_heads_overflow_y(self):
        """Test num_heads=65536 which would overflow grid.y (was limited to 65535)."""
        self._run_sdpa_mem_eff(batch=1, num_heads=65536)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Memory efficient attention not supported",
    )
    def test_combined_overflow(self):
        """Test batch=256, heads=256 (combined=65536) exceeding old grid limits."""
        self._run_sdpa_mem_eff(batch=256, num_heads=256)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Memory efficient attention not supported",
    )
    def test_large_batch_correctness(self):
        """Test correctness with large batch by comparing to CPU reference."""
        self._run_correctness_check(batch=65536, num_heads=1)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Memory efficient attention not supported",
    )
    def test_large_heads_correctness(self):
        """Test correctness with large num_heads by comparing to CPU reference."""
        self._run_correctness_check(batch=1, num_heads=65536)


if __name__ == "__main__":
    run_tests()
