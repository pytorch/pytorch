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
            ("padded", "contiguous"),
        ),
    )
    def test_matmul(self, dtype, layout_a, layout_b):
        """Test matmul with various dtypes and tensor layouts.

        These layouts test various alignment scenarios:
        - contiguous/view/aligned_offset: Standard aligned layouts
        - padded: Non-16-byte-aligned stride, Inductor pads to aligned size
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
                # Allocate bigger buffer than needed, use 16-byte aligned offset
                # offset=128 elements * 2 bytes = 256 bytes (16-byte aligned)
                storage = torch.randn(rows * cols + 512, device=device, dtype=dtype)
                offset = 128
                return torch.as_strided(storage[offset:], (rows, cols), (cols, 1))
            elif layout == "view":
                storage = torch.randn(rows * cols, device=device, dtype=dtype)
                return storage.view(rows, cols)
            elif layout == "padded":
                # Simulate row pitch > cols with non-16-byte-aligned stride
                # row_stride = cols + 8 = 520, 520 * 2 bytes = 1040 bytes (not 16-byte aligned)
                row_pitch = cols + 8
                storage = torch.randn(rows * row_pitch, device=device, dtype=dtype)
                return torch.as_strided(storage, (rows, cols), (row_pitch, 1))

        a = create_tensor_with_layout(layout_a, m, k)
        b = create_tensor_with_layout(layout_b, k, n)

        expected = matmul(a, b)

        torch._dynamo.reset()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "NVGEMM",
                "nvgemm_max_profiling_configs": 3,
            }
        ):
            compiled_fn = torch.compile(matmul)
            result = compiled_fn(a, b)

        torch.testing.assert_close(result, expected)

    def test_unaligned_base_pointer_rejected(self):
        """Test that matmul with unaligned base pointer is rejected.

        cutlass_api requires 16-byte aligned base pointers. Since alignment
        can't be checked at compile time (FakeTensors don't have real pointers),
        Inductor must guard against unaligned buffers.
        """
        m, n, k = 512, 512, 512
        dtype = torch.bfloat16
        device = "cuda"

        def matmul(a, b):
            return a @ b

        # Create tensor with unaligned base pointer
        # offset=117 elements * 2 bytes = 234 bytes (NOT 16-byte aligned)
        storage = torch.randn(m * k + 512, device=device, dtype=dtype)
        a = torch.as_strided(storage[117:], (m, k), (k, 1))
        b = torch.randn(k, n, device=device, dtype=dtype)

        torch._dynamo.reset()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "NVGEMM",
                "nvgemm_max_profiling_configs": 3,
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
                "nvgemm_max_profiling_configs": 3,
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
                    "nvgemm_max_profiling_configs": 3,
                }
            ):
                result, (code,) = run_and_get_code(
                    torch.compile(matmul),
                    a,
                    b,
                )

        self.assertIn("workspace=workspace", code)

        torch.testing.assert_close(result, expected)

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_bmm_non_standard_batch_stride(self, dtype):
        """Test BMM path with non-standard batch strides."""
        batch, m, n, k = 8, 64, 256, 128
        device = "cuda"

        def bmm(a, b):
            return torch.bmm(a, b)

        # Create tensors with non-largest batch stride by transposing
        # a_base shape: (m, batch, k), stride: (batch*k, k, 1)
        # After transpose: shape (batch, m, k), stride: (k, batch*k, 1)
        # batch_stride = k = 128, but m*k = 64*128 = 8192, so batch_stride < m*k
        a_base = torch.randn(m, batch, k, device=device, dtype=dtype)
        a = a_base.transpose(0, 1)  # (batch, m, k) with stride (k, batch*k, 1)

        b_base = torch.randn(k, batch, n, device=device, dtype=dtype)
        b = b_base.transpose(0, 1)  # (batch, k, n) with stride (n, batch*n, 1)

        # Verify batch stride is not largest (i.e., batch_stride_largest_or_zero would be False)
        if a.stride()[0] == a.shape[1] * a.shape[2]:
            raise AssertionError(
                "Test setup error: a should have non-standard batch stride"
            )
        if b.stride()[0] == b.shape[1] * b.shape[2]:
            raise AssertionError(
                "Test setup error: b should have non-standard batch stride"
            )

        expected = bmm(a, b)

        torch._dynamo.reset()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "NVGEMM",
                "nvgemm_max_profiling_configs": 3,
            }
        ):
            compiled_fn = torch.compile(bmm)
            result = compiled_fn(a, b)

        torch.testing.assert_close(result, expected)

    def test_scaled_gemm_mxfp8(self):
        """Test MXFP8 scaled GEMM with NVGEMM backend.

        Note: Invalid inputs (wrong shapes, dtypes, K not divisible by 16, etc.)
        are caught early by Dynamo's _check_scaled_mm_sizes in torch/_meta_registrations.py.
        NVGEMM can assume inputs are valid by the time they reach kernel selection.
        """
        from torch._inductor.utils import ceildiv

        m, n, k = 256, 512, 1024
        block_size = 32
        device = "cuda"

        def _round_up(x, multiple):
            return ((x + multiple - 1) // multiple) * multiple

        def _prep_k(K, scale_size):
            """Prepare K dimension for 32-4-4 swizzle requirements."""
            return _round_up(ceildiv(K, scale_size), 4)

        def scaled_mm(a, b, scale_a, scale_b):
            return torch._scaled_mm(
                a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32
            )

        # Create FP8 tensors
        a_fp8 = torch.randint(-1, 2, (m, k), device=device).to(torch.float8_e4m3fn)
        # B is N x K, then transposed to K x N for scaled_mm
        b_fp8 = torch.randint(-1, 2, (n, k), device=device).to(torch.float8_e4m3fn).T

        # Scale factors in float8_e8m0fnu (MXFP8 format)
        # Shape: (M, prep_k(K, 32)) for A, (prep_k(K, 32), N) for B
        scale_a = torch.rand(m, _prep_k(k, block_size), device=device).to(
            torch.float8_e8m0fnu
        )
        scale_b = torch.rand(_prep_k(k, block_size), n, device=device).to(
            torch.float8_e8m0fnu
        )

        # Get reference result from eager mode (ATen)
        expected = scaled_mm(a_fp8, b_fp8, scale_a, scale_b)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "NVGEMM",
                "nvgemm_max_profiling_configs": 3,
                "autotune_fallback_to_aten": False,
            }
        ):
            compiled_fn = torch.compile(scaled_mm)
            result = compiled_fn(a_fp8, b_fp8, scale_a, scale_b)

        torch.testing.assert_close(result, expected)

    def test_grouped_gemm(self):
        """Test grouped GEMM with NVGEMM backend.

        This test runs the same shape twice with different offsets to verify that
        different offset distributions produce correct results.

        Note: GroupedGemm currently only supports TN layout (column-major B).
        B is created with shape (g, k, n) but column-major inner layout via permute.
        """
        g, k, n = 4, 256, 256
        dtype = torch.bfloat16
        device = "cuda"

        def grouped_mm(a, b, offsets):
            return torch._grouped_mm(a, b, offs=offsets)

        b = torch.randn(g, n, k, device=device, dtype=dtype).permute(0, 2, 1)

        m_per_group_1 = [64, 64, 64, 64]
        total_m_1 = sum(m_per_group_1)
        offsets_1 = torch.tensor(
            [sum(m_per_group_1[: i + 1]) for i in range(g)],
            device=device,
            dtype=torch.int32,
        )
        a_1 = torch.randn(total_m_1, k, device=device, dtype=dtype)

        m_per_group_2 = [32, 96, 48, 80]
        total_m_2 = sum(m_per_group_2)
        if total_m_1 != total_m_2:
            raise AssertionError("Total M must match for cache key test")
        offsets_2 = torch.tensor(
            [sum(m_per_group_2[: i + 1]) for i in range(g)],
            device=device,
            dtype=torch.int32,
        )
        a_2 = torch.randn(total_m_2, k, device=device, dtype=dtype)

        expected_1 = grouped_mm(a_1, b, offsets_1)
        expected_2 = grouped_mm(a_2, b, offsets_2)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "NVGEMM",
                "autotune_fallback_to_aten": False,
            }
        ):
            compiled_fn = torch.compile(grouped_mm)

            result_1 = compiled_fn(a_1, b, offsets_1)
            torch.testing.assert_close(result_1, expected_1)

            result_2 = compiled_fn(a_2, b, offsets_2)
            torch.testing.assert_close(result_2, expected_2)

            self.assertFalse(torch.allclose(result_1, result_2))


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


@unittest.skipIf(
    not (ensure_nv_universal_gemm_available() and is_datacenter_blackwell_arch()),
    "NVIDIA Universal GEMM (cutlass_api) library not available or not on Blackwell",
)
class TestNVUniversalGemmDynamicShapes(TestCase):
    """Test cases for NVIDIA Universal GEMM with dynamic shapes."""

    @torch._dynamo.config.patch({"capture_dynamic_output_shape_ops": True})
    def test_unbacked_symint_rejected(self):
        """Test that NVGEMM rejects unbacked symbolic integers."""

        def fn(x, w):
            nz = torch.nonzero(x)  # Creates unbacked symint for nz.size(0)
            # Use unbacked symint as M dimension in matmul
            a = torch.ones(nz.size(0), w.size(0), dtype=w.dtype, device=w.device)
            return a @ w

        x = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], device="cuda")
        w = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")

        torch._dynamo.reset()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "NVGEMM",
                "nvgemm_max_profiling_configs": 2,
            }
        ):
            compiled_fn = torch.compile(fn, dynamic=True)
            with self.assertRaisesRegex(
                Exception, "NoValidChoicesError|no valid choice"
            ):
                compiled_fn(x, w)

    def test_dynamic_shapes(self):
        """Stress test dynamic shapes with extreme variations."""

        def matmul(a, b):
            return a @ b

        torch._dynamo.reset()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "NVGEMM",
                "nvgemm_max_profiling_configs": 2,
            }
        ):
            compiled_fn = torch.compile(matmul, dynamic=True)

            shapes = [
                (4, 4, 4, False),
                (16, 16, 16, True),
                (2048, 64, 128, True),
                (4, 4, 4, False),  # Unsupported again
                (64, 2048, 128, True),
                (128, 128, 2048, True),
                (2048, 2048, 512, True),
                (16, 16, 16, True),
            ]

            for m, n, k, supported in shapes:
                a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
                if not supported:
                    with self.assertRaisesRegex(
                        Exception, "NoValidChoicesError|no valid choice"
                    ):
                        compiled_fn(a, b)
                else:
                    result = compiled_fn(a, b)
                    torch.testing.assert_close(result, a @ b)


if __name__ == "__main__":
    run_tests()
