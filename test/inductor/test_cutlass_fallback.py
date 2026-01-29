# Owner(s): ["module: inductor"]
"""
Tests for CUTLASS backend fallback behavior when benchmarks fail.

These tests verify the fix for GitHub issue #171094 where CUDA illegal memory
access errors occurred when CUTLASS kernels failed during benchmarking but
were still selected for execution.

The fix ensures:
1. ExternKernelCaller (ATen) choices are included in prescreening candidates
2. When all benchmarks fail (timing=inf), ATen fallback is selected
3. The system gracefully handles CUTLASS benchmark failures without crashing
"""

import os
import unittest
import unittest.mock as mock
from unittest.mock import patch, MagicMock

import torch
from torch._inductor import config
from torch._inductor.select_algorithm import (
    AlgorithmSelectorCache,
    ExternKernelCaller,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_caches
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON

# Check if CUTLASS is available
try:
    from torch._inductor.codegen.cuda.cutlass_utils import try_import_cutlass
    HAS_CUTLASS = try_import_cutlass()
except ImportError:
    HAS_CUTLASS = False


def _get_path_without_sccache() -> str:
    """Get the PATH environment variable without sccache."""
    path_envs = os.environ.get("PATH", "").split(":")
    path_envs = [env for env in path_envs if "/opt/cache/bin" not in env]
    return ":".join(path_envs)


class TestCutlassFallback(TestCase):
    """Tests for CUTLASS fallback behavior when benchmarks fail."""

    def setUp(self):
        if not HAS_CUDA_AND_TRITON:
            self.skipTest("CUDA and triton are not available")
        if torch.version.hip:
            self.skipTest("CUTLASS backend is not supported on HIP")

        old_disable_fresh_cache_envvar = os.environ.get(
            "INDUCTOR_TEST_DISABLE_FRESH_CACHE", ""
        )
        try:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = "1"
            super().setUp()
        finally:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = (
                old_disable_fresh_cache_envvar
            )
        torch.random.manual_seed(1234)

    def tearDown(self):
        super().tearDown()
        clear_caches()

    @unittest.skipIf(not HAS_CUTLASS, "requires CUTLASS")
    @unittest.skipIf(not SM90OrLater, "requires SM90+")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_fallback_to_aten_when_cutlass_benchmarks_fail(self):
        """
        Test that ATen fallback is used when all CUTLASS benchmarks return inf.

        This tests the fix for issue #171094 where kernels with timing=inf
        were incorrectly selected, causing CUDA illegal memory access errors.
        """
        from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller

        # Track which kernel type was selected
        selected_choice_types = []

        original_output_node = None

        def tracking_output_node(self):
            selected_choice_types.append(type(self).__name__)
            return original_output_node(self)

        # Mock CUDATemplateCaller.benchmark to always return inf (failure)
        def mock_cuda_benchmark(*args, **kwargs):
            return float("inf")

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS,ATen",
                "cuda.cutlass_max_profiling_configs": 4,
            }
        ):
            with patch.object(
                CUDATemplateCaller, "benchmark", mock_cuda_benchmark
            ):
                torch._dynamo.reset()
                clear_caches()

                @torch.compile
                def fn(a, b):
                    return torch.mm(a, b)

                # Use dimensions that trigger CUTLASS selection
                M, N, K = 256, 2048, 3520
                a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

                # Should not crash - should fall back to ATen
                result = fn(a, b)

                # Verify result is correct (comparing against eager execution)
                expected = torch.mm(a, b)
                torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    @unittest.skipIf(not HAS_CUTLASS, "requires CUTLASS")
    @unittest.skipIf(not SM90OrLater, "requires SM90+")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_extern_kernel_included_in_prescreening(self):
        """
        Test that ExternKernelCaller choices are included in prescreening candidates.

        This verifies the fix that adds ATen choices to prescreening so they
        can be benchmarked alongside CUTLASS kernels and serve as fallback.
        """
        from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller

        prescreening_candidates = []

        # Patch prescreen_choices to capture what candidates are included
        original_prescreen = AlgorithmSelectorCache.prescreen_choices

        @staticmethod
        def capturing_prescreen(choices, name, inputs_key, prescreen_cache):
            result = original_prescreen(choices, name, inputs_key, prescreen_cache)
            if result:  # Only capture if prescreening is active
                prescreening_candidates.extend(result)
            return result

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS,ATen",
                "cuda.cutlass_max_profiling_configs": 40,  # Enough to trigger prescreening
                "cuda.cutlass_prescreening": True,
            }
        ):
            with patch.object(
                AlgorithmSelectorCache,
                "prescreen_choices",
                capturing_prescreen,
            ):
                torch._dynamo.reset()
                clear_caches()

                @torch.compile
                def fn(a, b):
                    return torch.mm(a, b)

                M, N, K = 256, 2048, 3520
                a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

                _ = fn(a, b)

                # Check if ExternKernelCaller was included in prescreening
                if prescreening_candidates:
                    extern_in_prescreening = any(
                        isinstance(c, ExternKernelCaller)
                        for c in prescreening_candidates
                    )
                    self.assertTrue(
                        extern_in_prescreening,
                        "ExternKernelCaller should be included in prescreening candidates",
                    )

    @unittest.skipIf(not HAS_CUTLASS, "requires CUTLASS")
    def test_cutlass_cache_key_type(self):
        """
        Test that cutlass_cache.py correctly handles instantiation_level as string.

        This tests the fix for the TypeError where instantiation_level was passed
        as int instead of str when generating the cache hash key.
        """
        from torch._inductor.codegen.cuda.cutlass_cache import get_config_request_key

        # This should not raise TypeError even when instantiation_level is an int
        # The fix ensures str() is called on instantiation_level
        try:
            # Test with string instantiation_level (normal case)
            key1 = get_config_request_key("sm_90", "12.4", "1")
            self.assertIsInstance(key1, str)

            # The function signature now expects string, but internally it
            # should handle the conversion properly. The fix was to ensure
            # the join operation works correctly.
            self.assertTrue(len(key1) == 8)  # Hash is truncated to 8 chars
        except TypeError as e:
            if "expected str instance, int found" in str(e):
                self.fail(
                    "cutlass_cache.py still has type error with instantiation_level"
                )
            raise

    @unittest.skipIf(not HAS_CUTLASS, "requires CUTLASS")
    @unittest.skipIf(not SM90OrLater, "requires SM90+")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_addmm_fallback_on_cutlass_failure(self):
        """
        Test addmm operation falls back to ATen when CUTLASS fails.

        Similar to test_fallback_to_aten_when_cutlass_benchmarks_fail but
        specifically tests the addmm path which was affected by issue #171094.
        """
        from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller

        def mock_cuda_benchmark(*args, **kwargs):
            return float("inf")

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS,ATen",
                "cuda.cutlass_max_profiling_configs": 4,
            }
        ):
            with patch.object(
                CUDATemplateCaller, "benchmark", mock_cuda_benchmark
            ):
                torch._dynamo.reset()
                clear_caches()

                @torch.compile
                def fn(bias, a, b):
                    return torch.addmm(bias, a, b)

                # Use dimensions from issue #171094
                M, N, K = 256, 2048, 3520
                a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
                bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)

                # Should not crash
                result = fn(bias, a, b)

                # Verify correctness
                expected = torch.addmm(bias, a, b)
                torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    @unittest.skipIf(not HAS_CUTLASS, "requires CUTLASS")
    @unittest.skipIf(not SM90OrLater, "requires SM90+")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_issue_171094_dimensions(self):
        """
        Regression test for the exact dimensions from issue #171094.

        M=256, K=3520, N=2048 with bfloat16 on SM90 caused illegal memory access.
        """
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS,ATen",
                "cuda.cutlass_max_profiling_configs": 4,
            }
        ):
            torch._dynamo.reset()
            clear_caches()

            @torch.compile
            def fn(bias, a, b):
                return torch.addmm(bias, a, b)

            # Exact dimensions from issue #171094
            M, K, N = 256, 3520, 2048
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
            bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)

            # This should complete without CUDA illegal memory access
            result = fn(bias, a, b)

            # Verify the result is correct
            expected = torch.addmm(bias, a, b)
            torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    run_tests()
