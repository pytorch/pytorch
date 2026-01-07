# Owner(s): ["module: inductor"]


import unittest

import torch
from torch._inductor import config
from torch._inductor.codegen.cuda.cuda_env import is_datacenter_blackwell_arch
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import ensure_nv_universal_gemm_available
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


# TODO(nikhilap): Remove Blackwell restriction once cutlass_api includes H100 kernels
@unittest.skipIf(
    not (ensure_nv_universal_gemm_available() and is_datacenter_blackwell_arch()),
    "NVIDIA Universal GEMM (cutlass_api) library not available or not on Blackwell",
)
@instantiate_parametrized_tests
class TestNVUniversalGemm(TestCase):
    """Test cases for NVIDIA Universal GEMM functionality."""

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize(
        "layout_a,layout_b",
        (
            ("contiguous", "contiguous"),
            ("aligned_offset", "contiguous"),
            ("contiguous", "view"),
            ("aligned_offset", "view"),
        ),
    )
    def test_matmul(self, dtype, layout_a, layout_b):
        """Test matmul with various dtypes and tensor layouts.

        These layouts all have 16-byte aligned strides and should work with NVIDIA Universal GEMM.
        M=513 tests that non-divisible M dimension works (only N and K must be divisible by 16).
        """
        m, n, k = 513, 512, 512
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
                "max_autotune_gemm_backends": "NVGEMM",
                "cuda.nvgemm_max_profiling_configs": 3,
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
        """Test that matmul with unaligned layouts is rejected by NVIDIA Universal GEMM.

        Padded layouts have strides that aren't 16-byte aligned,
        the guard function should reject NVIDIA Universal GEMM and we should get no choices
        when NVGEMM is the only backend.
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
                "max_autotune_gemm_backends": "NVGEMM",
                "cuda.nvgemm_max_profiling_configs": 3,
            }
        ):
            compiled_fn = torch.compile(matmul)
            with self.assertRaisesRegex(
                Exception, "NoValidChoicesError|no valid choice"
            ):
                compiled_fn(a, b)

    @parametrize("m,n,k", ((512, 513, 512), (512, 512, 513)))
    def test_non_divisible_n_or_k_rejected(self, m, n, k):
        """Test that matmul with n or k not divisible by 16 is rejected.

        NVIDIA Universal GEMM requires n and k to be divisible by 16. When they're not,
        the guard function should reject it and we should get no choices
        when NVGEMM is the only backend.
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
                "max_autotune_gemm_backends": "NVGEMM",
                "cuda.nvgemm_max_profiling_configs": 3,
            }
        ):
            compiled_fn = torch.compile(matmul)
            with self.assertRaisesRegex(
                Exception, "NoValidChoicesError|no valid choice"
            ):
                compiled_fn(a, b)

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_reinterpret_view_from_slice(self, dtype):
        """Test that sliced tensors (creating ReinterpretViews) work correctly.

        When tensors are slices of a shared buffer (e.g., from a fused projection),
        they become ReinterpretViews with non-contiguous strides. NVIDIA Universal GEMM must
        handle these correctly.
        """
        m, n, k = 512, 512, 512
        device = "cuda"

        def fn(x, weight):
            # Fused projection creates a single large output
            projected = x @ weight  # (m, 2*n)
            # Slicing creates ReinterpretViews
            a, b = projected.split(n, dim=1)  # Each is (m, n)
            return a @ b.t()  # (m, m)

        x = torch.randn(m, k, device=device, dtype=dtype)
        # Weight projects to 2*n so we can split
        weight = torch.randn(k, 2 * n, device=device, dtype=dtype)

        expected = fn(x, weight)

        torch._dynamo.reset()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "NVGEMM",
                "cuda.nvgemm_max_profiling_configs": 3,
            }
        ):
            compiled_fn = torch.compile(fn)
            result = compiled_fn(x, weight)

        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    run_tests()
