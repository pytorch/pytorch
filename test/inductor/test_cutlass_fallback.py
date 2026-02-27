# Owner(s): ["module: inductor"]
"""Tests for CUTLASS fallback and subprocess isolation (#171094)."""

import os
import unittest
import unittest.mock as mock
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.select_algorithm import AlgorithmSelectorCache, ExternKernelCaller
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_caches
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


# Check if CUTLASS is available
try:
    from torch._inductor.codegen.cutlass.utils import try_import_cutlass

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
        """ATen fallback is used when all CUTLASS benchmarks return inf."""
        from torch._inductor.codegen.cutlass.kernel import CUTLASSTemplateCaller

        def mock_cuda_benchmark(*args, **kwargs):
            return float("inf")

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS,ATen",
                "cuda.cutlass_max_profiling_configs": 4,
            }
        ):
            with patch.object(CUTLASSTemplateCaller, "benchmark", mock_cuda_benchmark):
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
        """Test that ExternKernelCaller choices are included in prescreening candidates."""
        prescreening_candidates = []

        original_prescreen = AlgorithmSelectorCache.prescreen_choices

        @staticmethod
        def capturing_prescreen(choices, name, inputs_key, prescreen_cache):
            result = original_prescreen(choices, name, inputs_key, prescreen_cache)
            if result:
                prescreening_candidates.extend(result)
            return result

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS,ATen",
                "cuda.cutlass_max_profiling_configs": 40,
                "cuda.cutlass_prescreening": True,
            }
        ):
            with patch.object(
                AlgorithmSelectorCache, "prescreen_choices", capturing_prescreen
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

                # Prescreening requires >= 10 CUTLASS candidates; if triggered,
                # verify ATen was included
                if prescreening_candidates:
                    extern_in_prescreening = any(
                        isinstance(c, ExternKernelCaller)
                        for c in prescreening_candidates
                    )
                    self.assertTrue(
                        extern_in_prescreening,
                        "ExternKernelCaller should be included in prescreening candidates",
                    )
                else:
                    # If prescreening wasn't triggered (< 10 CUTLASS candidates),
                    # that's OK -- ATen fallback is in the main benchmark path
                    pass

    @unittest.skipIf(not HAS_CUTLASS, "requires CUTLASS")
    @unittest.skipIf(not SM90OrLater, "requires SM90+")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_addmm_fallback_on_cutlass_failure(self):
        """addmm falls back to ATen when CUTLASS benchmarks fail."""
        from torch._inductor.codegen.cutlass.kernel import CUTLASSTemplateCaller

        def mock_cuda_benchmark(*args, **kwargs):
            return float("inf")

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS,ATen",
                "cuda.cutlass_max_profiling_configs": 4,
            }
        ):
            with patch.object(CUTLASSTemplateCaller, "benchmark", mock_cuda_benchmark):
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
        """Regression test with exact dimensions from #171094."""
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


class TestCutlassSubprocessRouting(TestCase):
    """Tests for CUTLASS subprocess benchmarking routing and error recovery."""

    def test_cutlass_forces_subprocess_benchmarking(self):
        """make_benchmark_fn routes to subprocess when CUTLASS choices are present."""
        from torch._inductor.codegen.cutlass.kernel import CUTLASSTemplateCaller

        mock_cutlass = mock.create_autospec(CUTLASSTemplateCaller, instance=True)
        mock_extern = mock.create_autospec(ExternKernelCaller, instance=True)

        with config.patch({"autotune_in_subproc": False}):
            fn = AlgorithmSelectorCache.make_benchmark_fn(
                choices=[mock_extern, mock_cutlass],
                input_nodes=[],
                layout=mock.MagicMock(),
                input_gen_fns=None,
            )
            self.assertEqual(fn.func, AlgorithmSelectorCache.benchmark_in_sub_process)

    def test_no_subprocess_without_cutlass(self):
        """make_benchmark_fn uses current process when no CUTLASS and autotune_in_subproc=False."""
        mock_extern = mock.create_autospec(ExternKernelCaller, instance=True)

        with config.patch({"autotune_in_subproc": False}):
            fn = AlgorithmSelectorCache.make_benchmark_fn(
                choices=[mock_extern],
                input_nodes=[],
                layout=mock.MagicMock(),
                input_gen_fns=None,
            )
            self.assertEqual(
                fn.func, AlgorithmSelectorCache.benchmark_in_current_process
            )

    def test_subprocess_restart_on_illegal_address(self):
        """TuningProcessPool restarts subprocess on cudaErrorIllegalAddress."""
        from torch._inductor.autotune_process import TuningProcessPool

        mock_process = mock.MagicMock()
        mock_process.get.side_effect = RuntimeError(
            "cudaErrorIllegalAddress: an illegal memory access was encountered"
        )

        mock_queue = mock.MagicMock()
        mock_queue.get.return_value = mock_process

        pool = TuningProcessPool.__new__(TuningProcessPool)
        pool.process_queue = mock_queue

        mock_choice = mock.MagicMock()
        mock_choice.bmreq = mock.MagicMock()

        result = pool.target(mock_choice)
        self.assertEqual(result, float("inf"))
        mock_process.restart.assert_called_once()

    def test_subprocess_restart_on_launch_failure(self):
        """TuningProcessPool restarts subprocess on cudaErrorLaunchFailure."""
        from torch._inductor.autotune_process import TuningProcessPool

        mock_process = mock.MagicMock()
        mock_process.get.side_effect = RuntimeError("cudaErrorLaunchFailure")

        mock_queue = mock.MagicMock()
        mock_queue.get.return_value = mock_process

        pool = TuningProcessPool.__new__(TuningProcessPool)
        pool.process_queue = mock_queue

        mock_choice = mock.MagicMock()
        mock_choice.bmreq = mock.MagicMock()

        result = pool.target(mock_choice)
        self.assertEqual(result, float("inf"))
        mock_process.restart.assert_called_once()

    def test_subprocess_no_restart_on_other_errors(self):
        """TuningProcessPool does not restart on non-sticky CUDA errors."""
        from torch._inductor.autotune_process import TuningProcessPool

        mock_process = mock.MagicMock()
        mock_process.get.side_effect = RuntimeError("some other error")

        mock_queue = mock.MagicMock()
        mock_queue.get.return_value = mock_process

        pool = TuningProcessPool.__new__(TuningProcessPool)
        pool.process_queue = mock_queue

        mock_choice = mock.MagicMock()
        mock_choice.bmreq = mock.MagicMock()

        result = pool.target(mock_choice)
        self.assertEqual(result, float("inf"))
        mock_process.restart.assert_not_called()

    def test_cutlass_ordered_after_triton_in_subprocess(self):
        """benchmark_in_sub_process orders Triton before CUTLASS."""
        from torch._inductor.codegen.cutlass.kernel import CUTLASSTemplateCaller

        mock_cutlass = mock.create_autospec(CUTLASSTemplateCaller, instance=True)
        mock_triton = mock.MagicMock()  # Not extern, not CUTLASS

        captured_choices = []

        def capture_benchmark(choices):
            captured_choices.extend(choices)
            return dict.fromkeys(choices, 1.0)

        with (
            mock.patch(
                "torch._inductor.autotune_process.benchmark_in_sub_process",
                side_effect=capture_benchmark,
            ),
            mock.patch.object(
                AlgorithmSelectorCache,
                "benchmark_in_current_process",
                return_value={},
            ),
            mock.patch.object(
                AlgorithmSelectorCache,
                "_is_extern",
                side_effect=lambda c: False,
            ),
        ):
            AlgorithmSelectorCache.benchmark_in_sub_process(
                choices=[mock_cutlass, mock_triton],
                input_nodes=[],
                layout=mock.MagicMock(),
                input_gen_fns=None,
            )

            self.assertEqual(len(captured_choices), 2)
            # Triton (non-CUTLASS) should come before CUTLASS
            self.assertNotIsInstance(captured_choices[0], CUTLASSTemplateCaller)
            self.assertIsInstance(captured_choices[1], CUTLASSTemplateCaller)

    def test_prescreen_includes_extern_with_enough_cutlass(self):
        """prescreen_choices includes ExternKernelCaller when >= 10 CUTLASS candidates."""
        from torch._inductor.codegen.cutlass.kernel import CUTLASSTemplateCaller

        def make_cutlass_mock(swizzle="2"):
            m = mock.create_autospec(CUTLASSTemplateCaller, instance=True)
            m.info_dict.return_value = {"swizzle": swizzle}
            return m

        cutlass_choices = [make_cutlass_mock() for _ in range(12)]
        extern_choice = mock.create_autospec(ExternKernelCaller, instance=True)
        choices = [extern_choice] + cutlass_choices

        with config.patch(
            {
                "cuda.cutlass_prescreening": True,
                "cuda.cutlass_max_profiling_swizzle_options": [1, 2],
            }
        ):
            result = AlgorithmSelectorCache.prescreen_choices(
                choices, "test_op", "test_key", {}
            )

        self.assertGreater(len(result), 0)
        self.assertTrue(
            any(isinstance(c, ExternKernelCaller) for c in result),
            "ExternKernelCaller should be in prescreening candidates",
        )
        # Extern should be first in the list
        self.assertIsInstance(result[0], ExternKernelCaller)

    def test_prescreen_skipped_with_few_cutlass(self):
        """prescreen_choices returns [] when < 10 CUTLASS candidates."""
        from torch._inductor.codegen.cutlass.kernel import CUTLASSTemplateCaller

        def make_cutlass_mock(swizzle="2"):
            m = mock.create_autospec(CUTLASSTemplateCaller, instance=True)
            m.info_dict.return_value = {"swizzle": swizzle}
            return m

        cutlass_choices = [make_cutlass_mock() for _ in range(5)]
        extern_choice = mock.create_autospec(ExternKernelCaller, instance=True)
        choices = [extern_choice] + cutlass_choices

        with config.patch(
            {
                "cuda.cutlass_prescreening": True,
                "cuda.cutlass_max_profiling_swizzle_options": [1, 2],
            }
        ):
            result = AlgorithmSelectorCache.prescreen_choices(
                choices, "test_op", "test_key", {}
            )

        self.assertEqual(result, [])


if __name__ == "__main__":
    run_tests()
