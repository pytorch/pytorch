# Owner(s): ["module: linalg"]

"""
Tests for asymmetric broadcasting matmul optimization.

This tests the fix for https://github.com/pytorch/pytorch/issues/110858
where broadcasting matmul was much slower than equivalent einsum for
asymmetric batch dimensions.

The fix detects cases where one operand has significantly more batch
dimensions than the other and uses einsum internally for better performance.
"""

import unittest
import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
)
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    slowTest,
)


class TestMatmulBroadcastPerf(TestCase):
    """Tests for matmul asymmetric broadcasting optimization."""

    def test_original_issue_correctness(self, device):
        """Test correctness for the original issue case."""
        # Original issue: A(3, 64, 64) @ B(4096, 3, 64, 1)
        A = torch.randn(3, 64, 64, device=device)
        B = torch.randn(4096, 3, 64, 1, device=device)

        result_matmul = A @ B
        result_einsum = torch.einsum('...ij,...jk->...ik', A, B)

        self.assertEqual(result_matmul.shape, (4096, 3, 64, 1))
        self.assertTrue(torch.allclose(result_matmul, result_einsum, rtol=1e-4, atol=1e-5))

    def test_2d_3d_broadcast_correctness(self, device):
        """Test 2D @ 3D broadcasting correctness."""
        A = torch.randn(64, 64, device=device)
        B = torch.randn(4096, 64, 2, device=device)

        result_matmul = A @ B
        result_einsum = torch.einsum('...ij,...jk->...ik', A, B)

        self.assertEqual(result_matmul.shape, (4096, 64, 2))
        self.assertTrue(torch.allclose(result_matmul, result_einsum, rtol=1e-4, atol=1e-5))

    def test_3d_4d_broadcast_correctness(self, device):
        """Test 3D @ 4D broadcasting correctness."""
        A = torch.randn(4, 64, 64, device=device)
        B = torch.randn(1024, 4, 64, 1, device=device)

        result_matmul = A @ B
        result_einsum = torch.einsum('...ij,...jk->...ik', A, B)

        self.assertEqual(result_matmul.shape, (1024, 4, 64, 1))
        self.assertTrue(torch.allclose(result_matmul, result_einsum, rtol=1e-4, atol=1e-5))

    def test_no_broadcast_unchanged(self, device):
        """Test that non-broadcast cases are unchanged."""
        A = torch.randn(32, 64, 64, device=device)
        B = torch.randn(32, 64, 64, device=device)

        result_matmul = A @ B
        result_einsum = torch.einsum('...ij,...jk->...ik', A, B)

        self.assertEqual(result_matmul.shape, (32, 64, 64))
        self.assertTrue(torch.allclose(result_matmul, result_einsum, rtol=1e-4, atol=1e-5))

    def test_simple_2d_unchanged(self, device):
        """Test that simple 2D @ 2D is unchanged."""
        A = torch.randn(64, 64, device=device)
        B = torch.randn(64, 64, device=device)

        result_matmul = A @ B
        result_mm = torch.mm(A, B)

        self.assertEqual(result_matmul.shape, (64, 64))
        self.assertTrue(torch.allclose(result_matmul, result_mm, rtol=1e-4, atol=1e-4))

    def test_large_output_columns_unchanged(self, device):
        """Test that large output column cases use standard matmul."""
        # With p=64, matmul should still use the standard bmm path
        A = torch.randn(64, 64, device=device)
        B = torch.randn(4096, 64, 64, device=device)

        result_matmul = A @ B
        result_einsum = torch.einsum('...ij,...jk->...ik', A, B)

        self.assertEqual(result_matmul.shape, (4096, 64, 64))
        self.assertTrue(torch.allclose(result_matmul, result_einsum, rtol=1e-4, atol=1e-5))

    def test_vector_rhs_correctness(self, device):
        """Test correctness for vector right-hand side."""
        A = torch.randn(4, 64, 64, device=device)
        B = torch.randn(1024, 4, 64, device=device)

        result_matmul = A @ B
        result_einsum = torch.einsum('...ij,...j->...i', A, B)

        self.assertEqual(result_matmul.shape, (1024, 4, 64))
        self.assertTrue(torch.allclose(result_matmul, result_einsum, rtol=1e-4, atol=1e-5))

    def test_small_asymmetry_unchanged(self, device):
        """Test that small asymmetry doesn't trigger einsum path."""
        A = torch.randn(1, 64, 64, device=device)
        B = torch.randn(8, 64, 64, device=device)

        result_matmul = A @ B
        result_einsum = torch.einsum('...ij,...jk->...ik', A, B)

        self.assertEqual(result_matmul.shape, (8, 64, 64))
        self.assertTrue(torch.allclose(result_matmul, result_einsum, rtol=1e-4, atol=1e-5))

    @dtypes(torch.float32, torch.float64)
    def test_dtypes_correctness(self, device, dtype):
        """Test correctness across different dtypes."""
        A = torch.randn(3, 64, 64, device=device, dtype=dtype)
        B = torch.randn(512, 3, 64, 1, device=device, dtype=dtype)

        result_matmul = A @ B
        result_einsum = torch.einsum('...ij,...jk->...ik', A, B)

        self.assertEqual(result_matmul.shape, (512, 3, 64, 1))
        self.assertEqual(result_matmul.dtype, dtype)
        self.assertTrue(torch.allclose(result_matmul, result_einsum, rtol=1e-4, atol=1e-5))

    def test_requires_grad_correctness(self, device):
        """Test correctness with gradients."""
        A = torch.randn(3, 64, 64, device=device, requires_grad=True)
        B = torch.randn(512, 3, 64, 1, device=device, requires_grad=True)

        result = A @ B
        loss = result.sum()
        loss.backward()

        # Verify gradients exist and have correct shapes
        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(B.grad)
        self.assertEqual(A.grad.shape, A.shape)
        self.assertEqual(B.grad.shape, B.shape)

    def test_edge_cases(self, device):
        """Test edge cases."""
        # Empty tensor
        A = torch.randn(3, 0, 64, device=device)
        B = torch.randn(4096, 3, 64, 1, device=device)
        result = A @ B
        self.assertEqual(result.shape, (4096, 3, 0, 1))

        # Single element
        A = torch.randn(1, 1, 1, device=device)
        B = torch.randn(100, 1, 1, 1, device=device)
        result = A @ B
        self.assertEqual(result.shape, (100, 1, 1, 1))


instantiate_device_type_tests(TestMatmulBroadcastPerf, globals())

if __name__ == '__main__':
    run_tests()

