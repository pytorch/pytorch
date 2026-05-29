# Owner(s): ["module: mta"]
"""Tests for aten::_foreach_mm and the nvmath cublasLt override."""

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestForeachMM(TestCase):
    def _ref(self, A, B):
        return [torch.mm(a, b) for a, b in zip(A, B)]

    def _make(self, shapes, dtype=torch.bfloat16, device="cpu"):
        A = [torch.randn(M, K, dtype=dtype, device=device) for M, _, K in shapes]
        B = [torch.randn(K, N, dtype=dtype, device=device) for _, N, K in shapes]
        return A, B

    def _check(self, A, B, atol=0.5, rtol=0.1):
        out = torch._foreach_mm(A, B)
        ref = self._ref(A, B)
        self.assertEqual(len(out), len(ref))
        for i, (r, o) in enumerate(zip(ref, out)):
            self.assertEqual(r.shape, o.shape, msg=f"shape mismatch at {i}")
            self.assertTrue(
                torch.allclose(r, o, atol=atol, rtol=rtol),
                msg=f"mismatch at {i}: max_diff={((r - o).abs().max().item())}",
            )

    # --- CPU (CompositeImplicitAutograd fallback) ---

    def test_cpu_uniform(self):
        shapes = [(256, 256, 256)] * 4
        A, B = self._make(shapes, dtype=torch.float32, device="cpu")
        self._check(A, B, atol=1e-4, rtol=1e-4)

    def test_cpu_mixed(self):
        shapes = [(128, 64, 256), (256, 128, 512), (64, 32, 128)]
        A, B = self._make(shapes, dtype=torch.float32, device="cpu")
        self._check(A, B, atol=1e-4, rtol=1e-4)

    # --- CUDA ---

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_bf16_1024(self):
        shapes = [(1024, 1024, 1024)] * 16
        A, B = self._make(shapes, device="cuda")
        self._check(A, B)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_bf16_2048(self):
        shapes = [(2048, 2048, 2048)] * 8
        A, B = self._make(shapes, device="cuda")
        self._check(A, B)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_bf16_4096(self):
        shapes = [(4096, 4096, 4096)] * 4
        A, B = self._make(shapes, device="cuda")
        self._check(A, B, atol=1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_bf16_mixed_shapes(self):
        shapes = [
            (1024, 1024, 1024),
            (1024, 1024, 2752),
            (1024, 2752, 1024),
            (512, 512, 512),
        ]
        A, B = self._make(shapes, device="cuda")
        self._check(A, B)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_bf16_muon_4096_attn(self):
        shapes = [(4096, 4096, 4096)] * 16
        A, B = self._make(shapes, device="cuda")
        self._check(A, B, atol=1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_bf16_muon_4096_mlp(self):
        shapes = [(4096, 4096, 11008)] * 8
        A, B = self._make(shapes, device="cuda")
        self._check(A, B, atol=1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_fp32_fallback(self):
        shapes = [(256, 256, 256)] * 4
        A, B = self._make(shapes, dtype=torch.float32, device="cuda")
        self._check(A, B, atol=1e-4, rtol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_single_pair(self):
        shapes = [(512, 512, 512)]
        A, B = self._make(shapes, device="cuda")
        self._check(A, B)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_col_major(self):
        A = [
            torch.randn(256, 128, device="cuda", dtype=torch.bfloat16).T.contiguous().T
            for _ in range(4)
        ]
        B = [
            torch.randn(128, 64, device="cuda", dtype=torch.bfloat16) for _ in range(4)
        ]
        self._check(A, B)


if __name__ == "__main__":
    run_tests()
