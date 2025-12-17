# Owner(s): ["module: inductor"]


import unittest

import torch
from torch._inductor import config
from torch._inductor.codegen.cuda.cuda_env import is_datacenter_blackwell_arch
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import ensure_cutlass_api_available
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@unittest.skipIf(
    not (ensure_cutlass_api_available() and is_datacenter_blackwell_arch()),
    "cutlass_api library or Blackwell device not available",
)
@instantiate_parametrized_tests
class TestCutlassAPIGemm(TestCase):
    """Test cases for cutlass_api GEMM functionality."""

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("m,n,k", ((512, 512, 512), (1024, 1024, 1024), (2048, 2048, 512)))
    def test_basic_matmul(self, dtype, m, n, k):
        """Test basic matmul with cutlass_api backend."""

        def matmul(a, b):
            return a @ b

        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTEDSL",
                "cuda.cutlass_api_max_profiling_configs": 3,
            }
        ):
            compiled_fn = torch.compile(matmul)
            result = compiled_fn(a, b)
            expected = matmul(a, b)

            torch.testing.assert_close(result, expected)

    @parametrize("layout_a", ("contiguous", "aligned_offset", "view"))
    @parametrize("layout_b", ("contiguous", "aligned_offset", "view"))
    def test_assorted_layouts(self, layout_a, layout_b):
        """Test matmul with various tensor layouts (offset, view).

        These layouts all have 16-byte aligned strides and should work with cutlass_api.
        """
        m, n, k = 512, 512, 512
        dtype = torch.bfloat16
        device = "cuda"

        def matmul(a, b):
            return a @ b

        def create_tensor_with_layout(layout, rows, cols):
            """Create a tensor with the specified layout."""
            if layout == "contiguous":
                return torch.randn(rows, cols, device=device, dtype=dtype)
            elif layout == "aligned_offset":
                # Allocate bigger buffer than needed, use nonzero storage offset
                storage = torch.randn(rows * cols + 512, device=device, dtype=dtype)
                offset = 128
                return torch.as_strided(storage[offset:], (rows, cols), (cols, 1))
            elif layout == "view":
                storage = torch.randn(rows * cols, device=device, dtype=dtype)
                return storage.view(rows, cols)

        a = create_tensor_with_layout(layout_a, m, k)
        b = create_tensor_with_layout(layout_b, k, n)

        expected = matmul(a, b)

        torch._dynamo.reset()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTEDSL",
                "cuda.cutlass_api_max_profiling_configs": 3,
            }
        ):
            compiled_fn = torch.compile(matmul)
            result = compiled_fn(a, b)

        torch.testing.assert_close(result, expected)

    @parametrize(
        "layout_a,layout_b",
        (
            ("padded", "contiguous"),
            ("contiguous", "padded"),
            ("offset", "contiguous"),
            ("contiguous", "offset"),
        ),
    )
    def test_unaligned_layouts_rejected(self, layout_a, layout_b):
        """Test that matmul with unaligned layouts is rejected by cutlass_api.

        Padded layouts have strides that aren't 16-byte aligned,
        the guard function should reject cutlass_api and we should get no choices
        when CUTEDSL is the only backend.
        """
        m, n, k = 512, 512, 512
        dtype = torch.bfloat16
        device = "cuda"

        def matmul(a, b):
            return a @ b

        def create_tensor_with_layout(layout, rows, cols):
            """Create a tensor with the specified layout."""
            if layout == "contiguous":
                return torch.randn(rows, cols, device=device, dtype=dtype)
            elif layout == "offset":
                # Allocate bigger buffer than needed, use nonzero storage offset
                storage = torch.randn(rows * cols + 512, device=device, dtype=dtype)
                offset = 117
                return torch.as_strided(storage[offset:], (rows, cols), (cols, 1))
            elif layout == "padded":
                # Simulate row pitch > cols with non-16-byte-aligned stride
                # row_stride = cols + 8 = 520, 520 * 2 bytes = 1040 bytes (not 16-byte aligned)
                row_pitch = cols + 8
                storage = torch.randn(rows * row_pitch, device=device, dtype=dtype)
                return torch.as_strided(storage, (rows, cols), (row_pitch, 1))

        a = create_tensor_with_layout(layout_a, m, k)
        b = create_tensor_with_layout(layout_b, k, n)

        torch._dynamo.reset()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTEDSL",
                "cuda.cutlass_api_max_profiling_configs": 3,
            }
        ):
            compiled_fn = torch.compile(matmul)
            with self.assertRaisesRegex(
                Exception, "NoValidChoicesError|no valid choice"
            ):
                compiled_fn(a, b)

    def test_non_divisible_m_works(self):
        """Test that matmul with m not divisible by 16 still works.

        cutlass_api only requires n and k to be divisible by 16, not n.
        """
        m, n, k = 513, 512, 512
        dtype = torch.bfloat16
        device = "cuda"

        def matmul(a, b):
            return a @ b

        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)

        torch._dynamo.reset()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTEDSL",
                "cuda.cutlass_api_max_profiling_configs": 3,
            }
        ):
            compiled_fn = torch.compile(matmul)
            result = compiled_fn(a, b)
            expected = matmul(a, b)

        torch.testing.assert_close(result, expected)

    @parametrize("m,n,k", ((512, 513, 512), (512, 512, 513)))
    def test_non_divisible_m_or_k_rejected(self, m, n, k):
        """Test that matmul with n or k not divisible by 16 is rejected.

        cutlass_api requires n and k to be divisible by 16. When they're not,
        the guard function should reject cutlass_api and we should get no choices
        when CUTEDSL is the only backend.
        """
        dtype = torch.bfloat16
        device = "cuda"

        def matmul(a, b):
            return a @ b

        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)

        torch._dynamo.reset()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTEDSL",
                "cuda.cutlass_api_max_profiling_configs": 3,
            }
        ):
            compiled_fn = torch.compile(matmul)
            with self.assertRaisesRegex(
                Exception, "NoValidChoicesError|no valid choice"
            ):
                compiled_fn(a, b)


@unittest.skipIf(
    not (ensure_cutlass_api_available() and is_datacenter_blackwell_arch()),
    "cutlass_api library or Blackwell device not available",
)
class TestCutlassAPIMetadataFiltering(TestCase):
    """Test cases for cutlass_api metadata filtering logic."""

    def test_dtype_conversion(self):
        """Test torch dtype to cutlass dtype conversion."""
        import cutlass

        from torch._inductor.codegen.cuda.cutlass_api_gemm import (
            _torch_dtype_to_cutlass,
        )

        self.assertEqual(_torch_dtype_to_cutlass(torch.float32), cutlass.Float32)
        self.assertEqual(_torch_dtype_to_cutlass(torch.float16), cutlass.Float16)
        self.assertEqual(_torch_dtype_to_cutlass(torch.bfloat16), cutlass.BFloat16)
        self.assertEqual(_torch_dtype_to_cutlass(torch.int8), cutlass.Int8)
        self.assertEqual(_torch_dtype_to_cutlass(torch.int32), cutlass.Int32)

    def test_stride_compatible_same_rank(self):
        """Test stride compatibility check with same rank tensors."""
        from torch._inductor.codegen.cuda.cutlass_api_gemm import _stride_compatible

        self.assertTrue(_stride_compatible((0, 1), (512, 1)))
        self.assertTrue(_stride_compatible((0, 1), (1024, 1)))

        self.assertTrue(_stride_compatible((1, 0), (1, 512)))

        self.assertFalse(_stride_compatible((0, 1), (1, 512)))

    def test_stride_compatible_different_rank(self):
        """Test stride compatibility with batched vs unbatched."""
        from torch._inductor.codegen.cuda.cutlass_api_gemm import _stride_compatible

        self.assertTrue(_stride_compatible((0, 0, 1), (512, 1)))
        self.assertTrue(_stride_compatible((0, 1, 0), (1, 512)))

    def test_stride_compatible_all_zeros(self):
        """Test stride compatibility with broadcast dimensions."""
        from torch._inductor.codegen.cuda.cutlass_api_gemm import _stride_compatible

        self.assertTrue(_stride_compatible((0, 0), (0, 0)))

    def test_stride_compatible_no_constraint(self):
        """Test when kernel has no stride constraint."""
        from torch._inductor.codegen.cuda.cutlass_api_gemm import _stride_compatible

        self.assertTrue(_stride_compatible((0, 512), (256, 128)))


if __name__ == "__main__":
    run_tests()
