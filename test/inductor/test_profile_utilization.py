"""
Tests for profiler utilization annotations.

Tests:
1. Unit tests for annotation functions (no CUDA)
2. Integration tests for profiler with CUDA kernels
"""

import json
import os
import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def requires_cuda(test_func):
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")(test_func)


class TestUtilizationAnnotations(TestCase):
    """Unit tests for utilization annotation functions (no CUDA needed)."""

    def test_basic_annotation(self):
        """Test utilization annotations are added correctly."""
        from torch._inductor.analysis.profile_analysis import add_utilization_annotations

        trace_data = {
            "traceEvents": [
                {"name": "kernel", "cat": "kernel", "dur": 100,
                 "args": {"kernel_flop": 2e9, "kernel_num_gb": 0.1}},
            ],
            "deviceProperties": [{"id": 0, "name": "NVIDIA H100"}],
        }

        result = add_utilization_annotations(trace_data, device_name="NVIDIA H100", dtype=torch.float32)
        args = result["traceEvents"][0]["args"]

        self.assertIn("achieved_flops_percent", args)
        self.assertIn("achieved_bandwidth_percent", args)
        self.assertGreater(args["achieved_flops_percent"], 0)
        self.assertLess(args["achieved_flops_percent"], 100)

    def test_skips_non_kernel_events(self):
        """Test that cpu_op events are not annotated."""
        from torch._inductor.analysis.profile_analysis import add_utilization_annotations

        trace_data = {
            "traceEvents": [
                {"name": "cpu_op", "cat": "cpu_op", "dur": 100,
                 "args": {"kernel_flop": 1e9, "kernel_num_gb": 0.1}},
            ],
            "deviceProperties": [{"id": 0, "name": "NVIDIA H100"}],
        }

        result = add_utilization_annotations(trace_data, device_name="NVIDIA H100")
        self.assertNotIn("achieved_flops_percent", result["traceEvents"][0]["args"])

    def test_handles_missing_metrics(self):
        """Test that events without flop/bandwidth info are handled gracefully."""
        from torch._inductor.analysis.profile_analysis import add_utilization_annotations

        trace_data = {
            "traceEvents": [{"name": "kernel", "cat": "kernel", "dur": 100, "args": {}}],
            "deviceProperties": [{"id": 0, "name": "NVIDIA H100"}],
        }

        result = add_utilization_annotations(trace_data, device_name="NVIDIA H100")
        self.assertNotIn("achieved_flops_percent", result["traceEvents"][0]["args"])


class TestProfilerCallbacks(TestCase):
    """Test the profiler callback mechanism."""

    def setUp(self):
        from torch._inductor.analysis.profile_analysis import clear_profiler_export_callbacks
        clear_profiler_export_callbacks()

    def tearDown(self):
        from torch._inductor.analysis.profile_analysis import clear_profiler_export_callbacks
        clear_profiler_export_callbacks()

    def test_callback_registration_and_execution(self):
        """Test callbacks are registered and run in order."""
        from torch._inductor.analysis.profile_analysis import (
            register_profiler_export_callback,
            run_profiler_export_callbacks,
            _profiler_export_callbacks,
        )

        results = []
        register_profiler_export_callback(lambda d: (results.append(1), d)[1])
        register_profiler_export_callback(lambda d: (results.append(2), d)[1])

        self.assertEqual(len(_profiler_export_callbacks), 2)

        run_profiler_export_callbacks({"traceEvents": []})
        self.assertEqual(results, [1, 2])


class ProfilerIntegrationBase(TestCase):
    """Base class for profiler integration tests with common setup/teardown."""

    def setUp(self):
        self.trace_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self.trace_path = self.trace_file.name
        self.trace_file.close()

    def tearDown(self):
        if os.path.exists(self.trace_path):
            os.unlink(self.trace_path)

    def profile_and_get_kernels(self, fn, iterations=3):
        """Profile a function and return kernel events with utilization."""
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for _ in range(iterations):
                fn()
            torch.cuda.synchronize()

        prof.export_chrome_trace(self.trace_path)

        with open(self.trace_path) as f:
            data = json.load(f)

        return [
            e for e in data["traceEvents"]
            if e.get("cat") == "kernel"
            and ("achieved_flops_percent" in e.get("args", {})
                 or "achieved_bandwidth_percent" in e.get("args", {}))
        ]

    def assert_reasonable_utilization(self, kernel_events, min_util=10):
        """Assert at least one kernel achieves reasonable utilization."""
        self.assertGreater(len(kernel_events), 0, "No kernel events with utilization found")

        max_flop = max(e["args"].get("achieved_flops_percent", 0) for e in kernel_events)
        max_bw = max(e["args"].get("achieved_bandwidth_percent", 0) for e in kernel_events)

        self.assertTrue(
            max_flop >= min_util or max_bw >= min_util,
            f"Utilization too low: FLOPS={max_flop:.1f}%, BW={max_bw:.1f}%"
        )
        self.assertLessEqual(max(max_flop, max_bw), 100, "Utilization exceeds 100%")


@requires_cuda
class TestCublasUtilization(ProfilerIntegrationBase):
    """Test cuBLAS GEMM achieves reasonable utilization."""

    def test_gemm_utilization(self):
        """Test GEMM achieves reasonable FLOPS or bandwidth utilization."""
        size = 2048
        a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, size, device="cuda", dtype=torch.float32)

        # Warmup
        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()

        kernels = self.profile_and_get_kernels(lambda: torch.mm(a, b))
        self.assert_reasonable_utilization(kernels, min_util=20)


@requires_cuda
class TestInductorUtilization(ProfilerIntegrationBase):
    """Test Inductor-generated kernels achieve reasonable utilization."""

    def test_pointwise_bandwidth(self):
        """Test pointwise ops achieve reasonable bandwidth utilization."""
        @torch.compile
        def pointwise(a, b):
            return a + b * 2.0

        size = 10_000_000
        a = torch.randn(size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, device="cuda", dtype=torch.float32)

        # Warmup
        for _ in range(3):
            pointwise(a, b)
        torch.cuda.synchronize()

        kernels = self.profile_and_get_kernels(lambda: pointwise(a, b))
        self.assert_reasonable_utilization(kernels, min_util=10)

    def test_fused_kernel_bandwidth_not_inflated(self):
        """Test fused kernels don't overestimate bandwidth."""
        @torch.compile
        def fused(x):
            return torch.relu(torch.sigmoid(x)) + 1.0

        size = 10_000_000  # 40MB at fp32
        x = torch.randn(size, device="cuda", dtype=torch.float32)

        for _ in range(3):
            fused(x)
        torch.cuda.synchronize()

        kernels = self.profile_and_get_kernels(lambda: fused(x))
        self.assertGreater(len(kernels), 0)

        # Fused kernel: read 40MB + write 40MB = 80MB = 0.08GB
        # Should not be inflated by intermediate results
        for e in kernels:
            kernel_num_gb = e["args"].get("kernel_num_gb", 0)
            if kernel_num_gb > 0:
                self.assertLess(kernel_num_gb, 0.2,
                    f"Bandwidth inflated: {kernel_num_gb:.3f} GB (expected ~0.08 GB)")


@requires_cuda
class TestTritonGemmUtilization(ProfilerIntegrationBase):
    """Test Triton GEMM with max-autotune achieves reasonable utilization."""

    def test_triton_gemm_utilization(self):
        """Test max-autotune Triton GEMM achieves reasonable utilization."""
        torch._inductor.config.force_disable_caches = True
        old_backend = getattr(torch._inductor.config, "max_autotune_gemm_backends", None)

        try:
            torch._inductor.config.max_autotune_gemm_backends = "TRITON"

            @torch.compile(mode="max-autotune-no-cudagraphs")
            def triton_mm(a, b):
                return torch.mm(a, b)

            size = 2048
            a = torch.randn(size, size, device="cuda", dtype=torch.float32)
            b = torch.randn(size, size, device="cuda", dtype=torch.float32)

            # Warmup (includes autotuning)
            for _ in range(3):
                triton_mm(a, b)
            torch.cuda.synchronize()

            kernels = self.profile_and_get_kernels(lambda: triton_mm(a, b))
            self.assert_reasonable_utilization(kernels, min_util=10)

        finally:
            torch._inductor.config.force_disable_caches = False
            if old_backend is not None:
                torch._inductor.config.max_autotune_gemm_backends = old_backend


@requires_cuda
class TestRooflineMetrics(ProfilerIntegrationBase):
    """Test roofline classification (compute-bound vs memory-bound)."""

    def test_large_gemm_is_compute_bound(self):
        """Test large GEMM has high arithmetic intensity."""
        size = 4096
        a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, size, device="cuda", dtype=torch.float32)

        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()

        kernels = self.profile_and_get_kernels(lambda: torch.mm(a, b))

        # Find GEMM kernels (high FLOP count)
        gemm_kernels = [e for e in kernels if e["args"].get("kernel_flop", 0) > 1e10]
        self.assertGreater(len(gemm_kernels), 0, "No GEMM kernels found")

        for e in gemm_kernels:
            arith_intensity = e["args"].get("arithmetic_intensity", 0)
            self.assertGreater(arith_intensity, 50,
                f"Expected high arithmetic intensity, got {arith_intensity:.1f}")

    def test_pointwise_is_memory_bound(self):
        """Test pointwise ops have low arithmetic intensity."""
        @torch.compile
        def add_op(a, b):
            return a + b

        size = 10_000_000
        a = torch.randn(size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, device="cuda", dtype=torch.float32)

        for _ in range(3):
            add_op(a, b)
        torch.cuda.synchronize()

        kernels = self.profile_and_get_kernels(lambda: add_op(a, b))

        memory_bound = [e for e in kernels if e["args"].get("roofline_bound") == "memory"]
        self.assertGreater(len(memory_bound), 0, "No memory-bound kernels found")

        for e in memory_bound:
            arith_intensity = e["args"].get("arithmetic_intensity", 0)
            self.assertLess(arith_intensity, 10,
                f"Expected low arithmetic intensity, got {arith_intensity:.1f}")


if __name__ == "__main__":
    run_tests()
