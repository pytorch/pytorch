# Owner(s): ["module: inductor"]
"""Tests for aten._foreach_mm kernel and inductor foreach_mm_pass."""

import torch
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestForeachMM(TestCase):
    """Numerical correctness tests for aten._foreach_mm."""

    def _reference(self, a_list, b_list):
        return [torch.mm(a, b) for a, b in zip(a_list, b_list)]

    @onlyCUDA
    @dtypes(torch.bfloat16, torch.float16, torch.float32)
    @parametrize("batch_size", [1, 3, 8, 16])
    @parametrize("m,k,n", [(64, 64, 64), (128, 256, 128), (2048, 2048, 2048)])
    def test_correctness(self, device, dtype, batch_size, m, k, n):
        a_list = [
            torch.randn(m, k, device=device, dtype=dtype) for _ in range(batch_size)
        ]
        b_list = [
            torch.randn(k, n, device=device, dtype=dtype) for _ in range(batch_size)
        ]

        result = torch.ops.aten._foreach_mm(a_list, b_list)
        expected = self._reference(a_list, b_list)

        self.assertEqual(len(result), batch_size)
        for i, (res, exp) in enumerate(zip(result, expected)):
            self.assertEqual(
                res, exp, atol=1e-2, rtol=1e-2, msg=f"mismatch at index {i}"
            )

    @onlyCUDA
    def test_bf16_cutlass_path(self, device):
        """Verify bf16 uses CUTLASS grouped GEMM on SM90+."""
        if not SM90OrLater:
            self.skipTest("CUTLASS grouped GEMM requires SM90+")
        a = [
            torch.randn(256, 256, device=device, dtype=torch.bfloat16) for _ in range(8)
        ]
        b = [
            torch.randn(256, 256, device=device, dtype=torch.bfloat16) for _ in range(8)
        ]
        result = torch.ops.aten._foreach_mm(a, b)
        expected = self._reference(a, b)
        for r, e in zip(result, expected):
            self.assertEqual(r, e, atol=1e-2, rtol=1e-2)

    @onlyCUDA
    def test_transposed_inputs(self, device):
        """Verify correctness with transposed (column-major) inputs."""
        if not SM90OrLater:
            self.skipTest("CUTLASS transpose dispatch requires SM90+")
        m, k, n, batch = 128, 64, 128, 4
        a_list = [
            torch.randn(k, m, device=device, dtype=torch.bfloat16).T
            for _ in range(batch)
        ]
        b_list = [
            torch.randn(n, k, device=device, dtype=torch.bfloat16).T
            for _ in range(batch)
        ]
        result = torch.ops.aten._foreach_mm(a_list, b_list)
        expected = self._reference(a_list, b_list)
        for r, e in zip(result, expected):
            self.assertEqual(r, e, atol=1e-2, rtol=1e-2)

    @onlyCUDA
    def test_single_element(self, device):
        """Batch of 1 should match regular mm exactly."""
        a = [torch.randn(128, 64, device=device, dtype=torch.float32)]
        b = [torch.randn(64, 256, device=device, dtype=torch.float32)]
        result = torch.ops.aten._foreach_mm(a, b)
        expected = [torch.mm(a[0], b[0])]
        self.assertEqual(result[0], expected[0])

    @onlyCUDA
    def test_error_mismatched_list_sizes(self, device):
        a = [torch.randn(4, 4, device=device) for _ in range(3)]
        b = [torch.randn(4, 4, device=device) for _ in range(2)]
        with self.assertRaisesRegex(RuntimeError, "same number of tensors"):
            torch.ops.aten._foreach_mm(a, b)

    @onlyCUDA
    def test_error_empty_lists(self, device):
        with self.assertRaisesRegex(RuntimeError, "non-empty"):
            torch.ops.aten._foreach_mm([], [])  # type: ignore[arg-type]

    @onlyCUDA
    def test_error_shape_mismatch(self, device):
        a = [torch.randn(4, 4, device=device), torch.randn(8, 8, device=device)]
        b = [torch.randn(4, 4, device=device), torch.randn(8, 8, device=device)]
        with self.assertRaisesRegex(RuntimeError, "must have shape"):
            torch.ops.aten._foreach_mm(a, b)

    @onlyCUDA
    def test_error_incompatible_dims(self, device):
        a = [torch.randn(4, 8, device=device)]
        b = [torch.randn(4, 4, device=device)]  # K mismatch: 8 vs 4
        with self.assertRaisesRegex(RuntimeError, "contraction dimension"):
            torch.ops.aten._foreach_mm(a, b)

    @onlyCUDA
    def test_error_not_2d(self, device):
        a = [torch.randn(2, 3, 4, device=device)]
        b = [torch.randn(2, 4, 5, device=device)]
        with self.assertRaisesRegex(RuntimeError, "2D"):
            torch.ops.aten._foreach_mm(a, b)

    @onlyCUDA
    def test_error_mixed_dtypes(self, device):
        a = [
            torch.randn(4, 4, device=device, dtype=torch.float32),
            torch.randn(4, 4, device=device, dtype=torch.bfloat16),
        ]
        b = [
            torch.randn(4, 4, device=device, dtype=torch.float32),
            torch.randn(4, 4, device=device, dtype=torch.bfloat16),
        ]
        with self.assertRaisesRegex(RuntimeError, "same dtype"):
            torch.ops.aten._foreach_mm(a, b)

    @onlyCUDA
    def test_error_stride_mismatch(self, device):
        a = [
            torch.randn(4, 4, device=device),
            torch.randn(4, 4, device=device).T.contiguous().T,
        ]
        b = [torch.randn(4, 4, device=device) for _ in range(2)]
        with self.assertRaisesRegex(RuntimeError, "same strides"):
            torch.ops.aten._foreach_mm(a, b)

    @onlyCUDA
    def test_error_batch_size_limit(self, device):
        a = [
            torch.randn(4, 4, device=device, dtype=torch.bfloat16) for _ in range(1024)
        ]
        b = [
            torch.randn(4, 4, device=device, dtype=torch.bfloat16) for _ in range(1024)
        ]
        if SM90OrLater:
            with self.assertRaisesRegex(RuntimeError, "1024"):
                torch.ops.aten._foreach_mm(a, b)
        else:
            # cuBLAS fallback has no batch limit
            result = torch.ops.aten._foreach_mm(a, b)
            self.assertEqual(len(result), 1024)


instantiate_device_type_tests(TestForeachMM, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
