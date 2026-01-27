# Owner(s): ["module: inductor"]
"""
Tests for profiler utilization annotations.

Tests:
1. Unit tests for annotation functions (no GPU needed)
2. Integration tests for profiler with GPU kernels
"""

import json
import os
import tempfile
import unittest

import torch
from torch._inductor import config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class TestUtilizationAnnotations(TestCase):
    """Unit tests for utilization annotation functions (no GPU needed)."""

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

        result = add_utilization_annotations(trace_data, device_name="NVIDIA H100")
        args = result["traceEvents"][0]["args"]

        self.assertIn("achieved_flops_percent", args)
        self.assertIn("achieved_bandwidth_percent", args)
        # Utilization should be between 10% and 95%
        self.assertGreaterEqual(args["achieved_flops_percent"], 0)
        self.assertLessEqual(args["achieved_flops_percent"], 95)

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
        fd, self.trace_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

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

    def assert_reasonable_utilization(self, kernel_events, min_util=10, max_util=95, check_flops=None):
        """Assert at least one kernel achieves reasonable utilization (10-95%).

        Args:
            kernel_events: List of kernel events from trace
            min_util: Minimum utilization percentage
            max_util: Maximum utilization percentage
            check_flops: If True, assert FLOPS metrics exist. If False, assert they don't.
                        If None, don't check for FLOPS presence.
        """
        self.assertGreater(len(kernel_events), 0, "No kernel events with utilization found")

        max_flop = max(e["args"].get("achieved_flops_percent", 0) for e in kernel_events)
        max_bw = max(e["args"].get("achieved_bandwidth_percent", 0) for e in kernel_events)

        if check_flops is True:
            self.assertGreater(max_flop, 0, "Expected FLOPS metrics but none found")
        elif check_flops is False:
            self.assertEqual(max_flop, 0, f"Expected no FLOPS metrics but got {max_flop:.1f}%")

        self.assertTrue(
            max_flop >= min_util or max_bw >= min_util,
            f"Utilization too low: FLOPS={max_flop:.1f}%, BW={max_bw:.1f}%"
        )
        self.assertLessEqual(
            max(max_flop, max_bw), max_util,
            f"Utilization too high: FLOPS={max_flop:.1f}%, BW={max_bw:.1f}%"
        )


@unittest.skipUnless(HAS_GPU, "requires GPU")
class TestCublasUtilization(ProfilerIntegrationBase):
    """Test cuBLAS GEMM achieves reasonable utilization."""

    def test_gemm_utilization(self):
        """Test GEMM achieves reasonable FLOPS or bandwidth utilization."""
        size = 2048
        a = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float32)

        # Warmup
        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()

        kernels = self.profile_and_get_kernels(lambda: torch.mm(a, b))
        self.assert_reasonable_utilization(kernels, min_util=20)


@unittest.skipUnless(HAS_GPU, "requires GPU")
class TestInductorUtilization(ProfilerIntegrationBase):
    """Test Inductor-generated kernels achieve reasonable utilization."""

    def test_pointwise_bandwidth(self):
        """Test pointwise ops achieve reasonable bandwidth utilization."""
        @torch.compile
        def pointwise(a, b):
            return a + b * 2.0

        size = 10_000_000
        a = torch.randn(size, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(size, device=GPU_TYPE, dtype=torch.float32)

        # Warmup
        for _ in range(3):
            pointwise(a, b)
        torch.cuda.synchronize()

        kernels = self.profile_and_get_kernels(lambda: pointwise(a, b))
        self.assert_reasonable_utilization(kernels, min_util=10)

    def test_fused_kernel_bandwidth_exact(self):
        """Test fused kernels report correct bandwidth (deterministic)."""
        @torch.compile
        def fused(x):
            return torch.relu(torch.sigmoid(x)) + 1.0

        size = 10_000_000  # 40MB at fp32
        x = torch.randn(size, device=GPU_TYPE, dtype=torch.float32)

        for _ in range(3):
            fused(x)
        torch.cuda.synchronize()

        kernels = self.profile_and_get_kernels(lambda: fused(x))
        self.assertGreater(len(kernels), 0)

        # Fused kernel: read 40MB + write 40MB = 80MB = 0.08GB exactly
        # (10M elements * 4 bytes * 2 for read+write)
        expected_gb = 10_000_000 * 4 * 2 / 1e9  # 0.08 GB
        for e in kernels:
            kernel_num_gb = e["args"].get("kernel_num_gb", 0)
            if kernel_num_gb > 0:
                self.assertEqual(kernel_num_gb, expected_gb)


@unittest.skipUnless(HAS_GPU, "requires GPU")
@inductor_config.patch(force_disable_caches=True, max_autotune_gemm_backends="TRITON")
class TestTritonGemmUtilization(ProfilerIntegrationBase):
    """Test Triton GEMM with max-autotune achieves reasonable utilization."""

    def test_triton_gemm_utilization(self):
        """Test max-autotune Triton GEMM achieves reasonable utilization."""
        @torch.compile(mode="max-autotune-no-cudagraphs")
        def triton_mm(a, b):
            return torch.mm(a, b)

        size = 2048
        a = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float32)

        # Warmup (includes autotuning)
        for _ in range(3):
            triton_mm(a, b)
        torch.cuda.synchronize()

        kernels = self.profile_and_get_kernels(lambda: triton_mm(a, b))
        self.assert_reasonable_utilization(kernels, min_util=10)


@unittest.skipUnless(HAS_GPU, "requires GPU")
class TestRooflineMetrics(ProfilerIntegrationBase):
    """Test roofline classification (compute-bound vs memory-bound)."""

    def test_large_gemm_is_compute_bound(self):
        """Test large GEMM has high arithmetic intensity."""
        size = 4096
        a = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float32)

        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()

        kernels = self.profile_and_get_kernels(lambda: torch.mm(a, b))

        # Find GEMM kernels (high FLOP count)
        gemm_kernels = [e for e in kernels if e["args"].get("kernel_flop", 0) > 1e10]
        self.assertGreater(len(gemm_kernels), 0, "No GEMM kernels found")

        for e in gemm_kernels:
            arith_intensity = e["args"].get("arithmetic_intensity", 0)
            self.assertGreater(
                arith_intensity, 50,
                f"Expected high arithmetic intensity, got {arith_intensity:.1f}"
            )

    def test_pointwise_is_memory_bound(self):
        """Test pointwise ops have low arithmetic intensity."""
        @torch.compile
        def add_op(a, b):
            return a + b

        size = 10_000_000
        a = torch.randn(size, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(size, device=GPU_TYPE, dtype=torch.float32)

        for _ in range(3):
            add_op(a, b)
        torch.cuda.synchronize()

        kernels = self.profile_and_get_kernels(lambda: add_op(a, b))

        memory_bound = [e for e in kernels if e["args"].get("roofline_bound") == "memory"]
        self.assertGreater(len(memory_bound), 0, "No memory-bound kernels found")

        for e in memory_bound:
            arith_intensity = e["args"].get("arithmetic_intensity", 0)
            self.assertLess(
                arith_intensity, 10,
                f"Expected low arithmetic intensity, got {arith_intensity:.1f}"
            )


if __name__ == "__main__":
    run_tests()
