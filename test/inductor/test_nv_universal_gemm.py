# Owner(s): ["module: inductor"]


import unittest
from unittest.mock import MagicMock, patch

import torch
from torch._inductor import config
from torch._inductor.codegen.cuda.cuda_env import is_datacenter_blackwell_arch
from torch._inductor.template_heuristics.nv_universal_gemm import (
    HeuristicConfig,
    NVUniversalGemmHeuristics,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import ensure_nv_universal_gemm_available, run_and_get_code
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

    def test_workspace_allocation(self):
        """Test that workspace allocation works correctly.

        Since no current CUTLASS kernels require a workspace, we mock the
        kernel.get_workspace_size method to return a non-zero value. This
        exercises the workspace allocation/deallocation code paths.
        """
        m, n, k = 512, 512, 512
        dtype = torch.bfloat16
        device = "cuda"

        def matmul(a, b):
            return a @ b

        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)

        expected = matmul(a, b)

        torch._dynamo.reset()

        # Patch cutlass_api.Kernel.get_workspace_size to return non-zero
        import cutlass_api

        def patched_get_workspace_size(self, args):
            return 1024

        with patch.object(
            cutlass_api.Kernel,
            "get_workspace_size",
            patched_get_workspace_size,
        ):
            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "NVGEMM",
                    "cuda.nvgemm_max_profiling_configs": 3,
                }
            ):
                result, (code,) = run_and_get_code(
                    torch.compile(matmul),
                    a,
                    b,
                )

        self.assertIn("workspace=workspace", code)

        torch.testing.assert_close(result, expected)


class TestNVUniversalGemmHeuristics(TestCase):
    """Unit tests for NVUniversalGemmHeuristics without requiring actual libraries."""

    def _create_mock_kernel(self, tile_m, tile_n, tile_k, cluster_m, cluster_n):
        """Create a mock kernel with the given tile/cluster configuration."""
        kernel = MagicMock()
        kernel.metadata.design.tile_shape = (tile_m, tile_n, tile_k)
        kernel.metadata.design.cluster_shape = (cluster_m, cluster_n)
        return kernel

    def _create_mock_inputs(self, m=512, n=512, k=512, dtype=torch.float16):
        """Create a mock MMKernelInputs."""
        inputs = MagicMock()
        inputs.mnk_hinted.return_value = (m, n, k)
        inputs.dtype.return_value = dtype
        inputs._mat1_idx = 0
        inputs._mat2_idx = 1
        inputs.strides_hinted.return_value = ((k, 1), (n, 1))
        return inputs

    def test_fallback_when_heuristics_unavailable(self):
        """Test that filter_kernels returns first N kernels when heuristics unavailable."""
        heuristics = NVUniversalGemmHeuristics()

        kernels = [self._create_mock_kernel(128, 128, 64, 1, 1) for _ in range(10)]
        inputs = self._create_mock_inputs()

        with patch.object(heuristics, "should_run", return_value=False):
            result = heuristics.filter_kernels(kernels, inputs, count=3)

        self.assertEqual(len(result), 3)
        self.assertEqual(result, kernels[:3])

    def test_fallback_when_no_configs_extracted(self):
        """Test fallback when kernel configs cannot be extracted."""
        heuristics = NVUniversalGemmHeuristics()

        kernels = []
        for _ in range(5):
            kernel = MagicMock()
            kernel.metadata.design = MagicMock(spec=[])  # No tile_shape attr
            kernels.append(kernel)

        inputs = self._create_mock_inputs()

        with patch.object(heuristics, "should_run", return_value=True):
            result = heuristics.filter_kernels(kernels, inputs, count=2)

        self.assertEqual(len(result), 2)
        self.assertEqual(result, kernels[:2])

    def test_filter_kernels_sorts_by_runtime(self):
        """Test that filter_kernels returns kernels sorted by estimated runtime and respects count."""
        heuristics = NVUniversalGemmHeuristics()

        kernel_a = self._create_mock_kernel(128, 128, 64, 1, 1)
        kernel_b = self._create_mock_kernel(256, 128, 64, 2, 1)
        kernel_c = self._create_mock_kernel(128, 256, 32, 1, 2)
        kernels = [kernel_a, kernel_b, kernel_c]

        inputs = self._create_mock_inputs()

        heuristic_configs = [
            HeuristicConfig(128, 128, 64, 1, 1, 4, 1, 64, 64, 32, 0.003),
            HeuristicConfig(256, 128, 64, 2, 1, 4, 1, 64, 64, 32, 0.001),
            HeuristicConfig(128, 256, 32, 1, 2, 4, 1, 64, 64, 32, 0.002),
        ]

        with patch.object(heuristics, "should_run", return_value=True):
            with patch.object(
                heuristics, "_get_heuristic_configs", return_value=heuristic_configs
            ):
                # Test sorting with count=3 (all kernels)
                result = heuristics.filter_kernels(kernels, inputs, count=3)
                self.assertEqual(len(result), 3)
                self.assertIs(result[0], kernel_b)
                self.assertIs(result[1], kernel_c)
                self.assertIs(result[2], kernel_a)

                # Test count limit with count=2 (should drop slowest)
                result = heuristics.filter_kernels(kernels, inputs, count=2)
                self.assertEqual(len(result), 2)
                self.assertIs(result[0], kernel_b)
                self.assertIs(result[1], kernel_c)

                # Test count > available kernels (should return all 3)
                result = heuristics.filter_kernels(kernels, inputs, count=10)
                self.assertEqual(len(result), 3)


if __name__ == "__main__":
    run_tests()
