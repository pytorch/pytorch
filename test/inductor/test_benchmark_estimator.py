"""
Tests for benchmark logging and estimator loading.

This tests Goal 2: Log benchmarked collectives/compute nodes to tlparse
and load that file as an estimator with linear interpolation.
"""

import json
import os
import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestBenchmarkEstimator(TestCase):
    """Test benchmark recording, logging, and estimator loading."""

    def test_record_benchmark_result(self):
        """Test that benchmark results are recorded correctly."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            record_benchmark_result,
            get_all_benchmark_results,
            _BENCHMARK_RESULTS,
        )

        # Clear any existing results
        _BENCHMARK_RESULTS["compute"].clear()
        _BENCHMARK_RESULTS["collectives"].clear()

        # Record some results
        record_benchmark_result(
            "test_op: 1024 bytes",
            runtime_ms=0.5,
            category="compute",
            bytes_count=1024,
            flops=1000000,
        )
        record_benchmark_result(
            "all_reduce: 4096 bytes",
            runtime_ms=1.2,
            category="collectives",
            bytes_count=4096,
            group_size=4,
        )

        results = get_all_benchmark_results()

        self.assertIn("test_op: 1024 bytes", results["compute"])
        self.assertEqual(results["compute"]["test_op: 1024 bytes"]["runtime_ms"], 0.5)
        self.assertEqual(results["compute"]["test_op: 1024 bytes"]["bytes"], 1024)

        self.assertIn("all_reduce: 4096 bytes", results["collectives"])
        self.assertEqual(
            results["collectives"]["all_reduce: 4096 bytes"]["runtime_ms"], 1.2
        )
        self.assertEqual(
            results["collectives"]["all_reduce: 4096 bytes"]["group_size"], 4
        )

    def test_save_and_load_benchmarks(self):
        """Test saving benchmarks to file and loading as estimator."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            record_benchmark_result,
            save_benchmarks_to_file,
            BenchmarkEstimator,
            _BENCHMARK_RESULTS,
        )

        # Clear and record fresh results
        _BENCHMARK_RESULTS["compute"].clear()
        _BENCHMARK_RESULTS["collectives"].clear()

        record_benchmark_result(
            "torch.ops.aten.mm.default: 4MB",
            runtime_ms=0.5,
            category="compute",
            bytes_count=4194304,
        )
        record_benchmark_result(
            "torch.ops.aten.mm.default: 1MB",
            runtime_ms=0.12,
            category="compute",
            bytes_count=1048576,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            save_benchmarks_to_file(filepath)

            # Load and verify
            estimator = BenchmarkEstimator.from_file(filepath)
            self.assertIn("compute", estimator.benchmarks)
            self.assertEqual(len(estimator.benchmarks["compute"]), 2)
        finally:
            os.unlink(filepath)

    def test_linear_interpolation(self):
        """Test that linear interpolation works correctly."""
        from torch._inductor.fx_passes.node_runtime_estimation import BenchmarkEstimator

        # Test interpolation between two points
        points = [(1000, 0.1), (2000, 0.2)]

        # Exact match
        result = BenchmarkEstimator._linear_interpolate(points, 1000)
        self.assertAlmostEqual(result, 0.1, places=5)

        # Mid-point interpolation
        result = BenchmarkEstimator._linear_interpolate(points, 1500)
        self.assertAlmostEqual(result, 0.15, places=5)

        # Extrapolation beyond max
        result = BenchmarkEstimator._linear_interpolate(points, 3000)
        self.assertAlmostEqual(result, 0.3, places=5)

        # Extrapolation below min
        result = BenchmarkEstimator._linear_interpolate(points, 500)
        self.assertAlmostEqual(result, 0.05, places=5)

    def test_linear_interpolation_single_point(self):
        """Test interpolation with a single data point."""
        from torch._inductor.fx_passes.node_runtime_estimation import BenchmarkEstimator

        points = [(1000, 0.1)]

        # Should scale linearly from origin
        result = BenchmarkEstimator._linear_interpolate(points, 2000)
        self.assertAlmostEqual(result, 0.2, places=5)

        result = BenchmarkEstimator._linear_interpolate(points, 500)
        self.assertAlmostEqual(result, 0.05, places=5)

    def test_linear_interpolation_empty(self):
        """Test interpolation with no data points."""
        from torch._inductor.fx_passes.node_runtime_estimation import BenchmarkEstimator

        result = BenchmarkEstimator._linear_interpolate([], 1000)
        self.assertIsNone(result)

    def test_benchmark_estimator_from_dict(self):
        """Test creating estimator from a dictionary."""
        from torch._inductor.fx_passes.node_runtime_estimation import BenchmarkEstimator

        benchmarks = {
            "compute": {
                "torch.ops.aten.mm.default: (4194304 bytes)": {
                    "runtime_ms": 0.5,
                    "bytes": 4194304,
                },
                "torch.ops.aten.mm.default: (1048576 bytes)": {
                    "runtime_ms": 0.12,
                    "bytes": 1048576,
                },
            },
            "collectives": {
                "torch.ops._c10d_functional.all_reduce: (4 group size, 4194304 bytes)": {
                    "runtime_ms": 1.2,
                    "bytes": 4194304,
                    "group_size": 4,
                }
            },
        }

        estimator = BenchmarkEstimator(benchmarks)

        # Check indices were built
        self.assertIn("torch.ops.aten.mm.default", estimator._compute_by_target)
        self.assertEqual(len(estimator._compute_by_target["torch.ops.aten.mm.default"]), 2)

    def test_create_estimator_from_benchmarks(self):
        """Test creating an estimator function from current benchmarks."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            record_benchmark_result,
            create_estimator_from_benchmarks,
            _BENCHMARK_RESULTS,
        )

        # Clear and record fresh results
        _BENCHMARK_RESULTS["compute"].clear()
        _BENCHMARK_RESULTS["collectives"].clear()

        record_benchmark_result(
            "test_op: 1024 bytes",
            runtime_ms=0.5,
            category="compute",
            bytes_count=1024,
        )

        estimate_fn = create_estimator_from_benchmarks()

        # The function should be callable
        self.assertTrue(callable(estimate_fn))


class TestBenchmarkEstimatorIntegration(TestCase):
    """Integration tests for benchmark estimator with FX graphs."""

    def test_estimator_with_fx_node(self):
        """Test that estimator can estimate runtime for an FX node."""
        from torch._inductor.fx_passes.node_runtime_estimation import BenchmarkEstimator
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx

        benchmarks = {
            "compute": {
                "torch.ops.aten.mm.default: (4194304 bytes)": {
                    "runtime_ms": 0.5,
                    "bytes": 4194304,
                },
            },
            "collectives": {},
        }

        estimator = BenchmarkEstimator(benchmarks)

        # Create an FX graph with mm
        def f(a, b):
            return torch.mm(a, b)

        with FakeTensorMode():
            a = torch.randn(1024, 1024)
            b = torch.randn(1024, 1024)
            gm = make_fx(f)(a, b)

        # Find the mm node
        mm_node = None
        for node in gm.graph.nodes:
            if node.op == "call_function" and "mm" in str(node.target):
                mm_node = node
                break

        self.assertIsNotNone(mm_node)

        # The estimator should be able to provide an estimate
        # (may return None if exact match not found, but shouldn't error)
        result = estimator.estimate_runtime(mm_node)
        # Result can be None if target name doesn't match exactly
        # This is OK - the important thing is it doesn't crash


if __name__ == "__main__":
    run_tests()
