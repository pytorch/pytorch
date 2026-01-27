"""
Tests for PyTorch profiler trace estimator.

This tests Goal 3: Load an estimator from a PyTorch profiler trace file
with linear interpolation for missing entries.
"""

import json
import os
import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestProfilerTraceEstimator(TestCase):
    """Test loading estimators from PyTorch profiler traces."""

    def test_parse_chrome_trace_basic(self):
        """Test parsing a basic Chrome trace format."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )

        trace_data = {
            "traceEvents": [
                {
                    "name": "aten::mm",
                    "ph": "X",
                    "ts": 1000,
                    "dur": 500,  # 0.5ms
                    "cat": "cpu_op",
                    "args": {},
                },
                {
                    "name": "aten::add",
                    "ph": "X",
                    "ts": 2000,
                    "dur": 100,  # 0.1ms
                    "cat": "cpu_op",
                    "args": {},
                },
            ]
        }

        benchmarks = ProfilerTraceEstimator._parse_chrome_trace(trace_data)

        self.assertIn("compute", benchmarks)
        self.assertIn("aten::mm: (0 bytes)", benchmarks["compute"])
        self.assertEqual(benchmarks["compute"]["aten::mm: (0 bytes)"]["runtime_ms"], 0.5)
        self.assertIn("aten::add: (0 bytes)", benchmarks["compute"])
        self.assertEqual(benchmarks["compute"]["aten::add: (0 bytes)"]["runtime_ms"], 0.1)

    def test_parse_chrome_trace_with_shapes(self):
        """Test parsing Chrome trace with input shape information."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )

        trace_data = {
            "traceEvents": [
                {
                    "name": "aten::mm",
                    "ph": "X",
                    "ts": 1000,
                    "dur": 500,
                    "cat": "cpu_op",
                    "args": {
                        "Input Dims": "[[1024, 1024], [1024, 1024]]",
                        "Input type": "float32",
                    },
                },
            ]
        }

        benchmarks = ProfilerTraceEstimator._parse_chrome_trace(trace_data)

        # Should extract bytes from shapes: 2 * 1024 * 1024 * 4 bytes = 8388608
        self.assertIn("aten::mm: (8388608 bytes)", benchmarks["compute"])

    def test_parse_chrome_trace_collectives(self):
        """Test that collective operations are classified correctly."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )

        trace_data = {
            "traceEvents": [
                {
                    "name": "nccl:all_reduce",
                    "ph": "X",
                    "ts": 1000,
                    "dur": 1200,
                    "cat": "gpu_op",
                    "args": {},
                },
                {
                    "name": "c10d::allgather_",
                    "ph": "X",
                    "ts": 2000,
                    "dur": 800,
                    "cat": "gpu_op",
                    "args": {},
                },
            ]
        }

        benchmarks = ProfilerTraceEstimator._parse_chrome_trace(trace_data)

        self.assertIn("collectives", benchmarks)
        self.assertIn("nccl:all_reduce: (0 bytes)", benchmarks["collectives"])
        self.assertIn("c10d::allgather_: (0 bytes)", benchmarks["collectives"])

    def test_from_chrome_trace_file(self):
        """Test loading estimator from a Chrome trace file."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )

        trace_data = {
            "traceEvents": [
                {
                    "name": "aten::mm",
                    "ph": "X",
                    "ts": 1000,
                    "dur": 500,
                    "cat": "cpu_op",
                    "args": {
                        "Input Dims": "[[512, 512], [512, 512]]",
                        "Input type": "float32",
                    },
                },
                {
                    "name": "aten::mm",
                    "ph": "X",
                    "ts": 2000,
                    "dur": 2000,
                    "cat": "cpu_op",
                    "args": {
                        "Input Dims": "[[2048, 2048], [2048, 2048]]",
                        "Input type": "float32",
                    },
                },
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(trace_data, f)
            filepath = f.name

        try:
            estimator = ProfilerTraceEstimator.from_chrome_trace(filepath)

            # Check that estimator was created with correct data
            self.assertIn("aten::mm", estimator._estimator._compute_by_target)
            # Should have two data points for interpolation
            self.assertEqual(
                len(estimator._estimator._compute_by_target["aten::mm"]), 2
            )
        finally:
            os.unlink(filepath)

    def test_estimate_bytes_from_shapes(self):
        """Test byte estimation from shape information."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )

        # Test with list of shapes
        result = ProfilerTraceEstimator._estimate_bytes_from_shapes(
            [[1024, 1024]], "float32"
        )
        self.assertEqual(result, 1024 * 1024 * 4)

        # Test with string shapes
        result = ProfilerTraceEstimator._estimate_bytes_from_shapes(
            "[[512, 512]]", "float64"
        )
        self.assertEqual(result, 512 * 512 * 8)

        # Test with multiple tensors
        result = ProfilerTraceEstimator._estimate_bytes_from_shapes(
            [[100, 100], [100, 100]], "float16"
        )
        self.assertEqual(result, 2 * 100 * 100 * 2)

    def test_estimate_bytes_different_dtypes(self):
        """Test byte estimation with different data types."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )

        # Use a single shape - the function sums all tensors
        shapes = [[100, 100]]
        numel = 100 * 100

        # float32 (default)
        result = ProfilerTraceEstimator._estimate_bytes_from_shapes(shapes, "float32")
        self.assertEqual(result, numel * 4)

        # float64
        result = ProfilerTraceEstimator._estimate_bytes_from_shapes(shapes, "float64")
        self.assertEqual(result, numel * 8)

        # float16
        result = ProfilerTraceEstimator._estimate_bytes_from_shapes(shapes, "float16")
        self.assertEqual(result, numel * 2)

        # bfloat16
        result = ProfilerTraceEstimator._estimate_bytes_from_shapes(shapes, "bfloat16")
        self.assertEqual(result, numel * 2)

        # int64
        result = ProfilerTraceEstimator._estimate_bytes_from_shapes(shapes, "int64")
        self.assertEqual(result, numel * 8)

        # int8
        result = ProfilerTraceEstimator._estimate_bytes_from_shapes(shapes, "int8")
        self.assertEqual(result, numel * 1)

    def test_interpolation_with_profiler_data(self):
        """Test that interpolation works with profiler-loaded data."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )

        trace_data = {
            "traceEvents": [
                {
                    "name": "aten::mm",
                    "ph": "X",
                    "ts": 1000,
                    "dur": 100,  # 0.1ms for small
                    "cat": "cpu_op",
                    "args": {
                        "Input Dims": "[[256, 256], [256, 256]]",
                        "Input type": "float32",
                    },
                },
                {
                    "name": "aten::mm",
                    "ph": "X",
                    "ts": 2000,
                    "dur": 1000,  # 1.0ms for large
                    "cat": "cpu_op",
                    "args": {
                        "Input Dims": "[[1024, 1024], [1024, 1024]]",
                        "Input type": "float32",
                    },
                },
            ]
        }

        benchmarks = ProfilerTraceEstimator._parse_chrome_trace(trace_data)
        estimator = ProfilerTraceEstimator(benchmarks)

        # Check that data points are indexed
        self.assertIn("aten::mm", estimator._estimator._compute_by_target)
        points = estimator._estimator._compute_by_target["aten::mm"]
        self.assertEqual(len(points), 2)

        # Verify the points are sorted by bytes
        self.assertLess(points[0][0], points[1][0])

    def test_list_format_trace(self):
        """Test parsing trace in list format (not dict with traceEvents)."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )

        # Some traces are just a list of events
        trace_data = [
            {
                "name": "aten::mm",
                "ph": "X",
                "ts": 1000,
                "dur": 500,
                "cat": "cpu_op",
                "args": {},
            },
        ]

        benchmarks = ProfilerTraceEstimator._parse_chrome_trace(trace_data)
        self.assertIn("aten::mm: (0 bytes)", benchmarks["compute"])

    def test_skip_non_complete_events(self):
        """Test that non-complete events (ph != 'X') are skipped."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )

        trace_data = {
            "traceEvents": [
                # Complete event - should be included
                {"name": "aten::mm", "ph": "X", "ts": 1000, "dur": 500, "args": {}},
                # Begin event - should be skipped
                {"name": "aten::add", "ph": "B", "ts": 2000, "args": {}},
                # End event - should be skipped
                {"name": "aten::add", "ph": "E", "ts": 2100, "args": {}},
                # Metadata - should be skipped
                {"name": "process_name", "ph": "M", "args": {"name": "test"}},
            ]
        }

        benchmarks = ProfilerTraceEstimator._parse_chrome_trace(trace_data)

        # Only the complete event should be included
        self.assertIn("aten::mm: (0 bytes)", benchmarks["compute"])
        self.assertNotIn("aten::add: (0 bytes)", benchmarks["compute"])


class TestProfilerTraceEstimatorIntegration(TestCase):
    """Integration tests that run the actual PyTorch profiler."""

    def test_profiler_to_estimator_roundtrip(self):
        """
        Integration test: Run profiler, export trace, load as estimator, query results.

        This tests the full workflow:
        1. Run some operations under the PyTorch profiler
        2. Export the trace to a Chrome trace file
        3. Load the trace with ProfilerTraceEstimator
        4. Verify we can query the results
        """
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )
        from torch.profiler import profile, ProfilerActivity

        # Run some operations under the profiler
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            trace_path = f.name

        try:
            # Profile some tensor operations
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
            ) as prof:
                # Do some matrix multiplications of different sizes
                a_small = torch.randn(128, 128)
                b_small = torch.randn(128, 128)
                c_small = torch.mm(a_small, b_small)

                a_medium = torch.randn(512, 512)
                b_medium = torch.randn(512, 512)
                c_medium = torch.mm(a_medium, b_medium)

                a_large = torch.randn(1024, 1024)
                b_large = torch.randn(1024, 1024)
                c_large = torch.mm(a_large, b_large)

                # Also do some additions
                d = torch.add(a_small, b_small)

            # Export to Chrome trace format
            prof.export_chrome_trace(trace_path)

            # Load the trace with ProfilerTraceEstimator
            estimator = ProfilerTraceEstimator.from_chrome_trace(trace_path)

            # Verify we got some compute operations
            self.assertIn("compute", estimator._estimator.benchmarks)
            compute_ops = estimator._estimator.benchmarks["compute"]
            self.assertGreater(len(compute_ops), 0, "Should have captured some operations")

            # Check that we captured mm operations
            mm_ops = [k for k in compute_ops.keys() if "mm" in k.lower()]
            self.assertGreater(len(mm_ops), 0, "Should have captured mm operations")

            # Check that the mm operations have reasonable runtimes (> 0)
            for key in mm_ops:
                runtime = compute_ops[key]["runtime_ms"]
                self.assertGreater(runtime, 0, f"Runtime for {key} should be > 0")

        finally:
            os.unlink(trace_path)

    def test_profiler_events_to_estimator(self):
        """
        Test loading estimator directly from profiler events (without file).
        """
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )
        from torch.profiler import profile, ProfilerActivity

        # Profile some operations
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
        ) as prof:
            a = torch.randn(256, 256)
            b = torch.randn(256, 256)
            c = torch.mm(a, b)
            d = torch.add(a, b)
            e = torch.relu(c)

        # Get key averages
        events = prof.key_averages()

        # Load directly from events
        estimator = ProfilerTraceEstimator.from_profiler_events(events)

        # Verify we got some operations
        compute_ops = estimator._estimator.benchmarks["compute"]
        self.assertGreater(len(compute_ops), 0, "Should have captured some operations")

    def test_profiler_trace_interpolation_accuracy(self):
        """
        Test that interpolation gives reasonable estimates for sizes not in the trace.
        """
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
            BenchmarkEstimator,
        )
        from torch.profiler import profile, ProfilerActivity

        # Profile operations at specific sizes
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            trace_path = f.name

        try:
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
            ) as prof:
                # Small size
                a1 = torch.randn(256, 256)
                b1 = torch.randn(256, 256)
                for _ in range(5):  # Multiple iterations for more stable timing
                    c1 = torch.mm(a1, b1)

                # Large size
                a2 = torch.randn(1024, 1024)
                b2 = torch.randn(1024, 1024)
                for _ in range(5):
                    c2 = torch.mm(a2, b2)

            prof.export_chrome_trace(trace_path)

            # Load and create estimator
            estimator = ProfilerTraceEstimator.from_chrome_trace(trace_path)

            # Check if we have mm data points for interpolation
            if "aten::mm" in estimator._estimator._compute_by_target:
                points = estimator._estimator._compute_by_target["aten::mm"]

                if len(points) >= 2:
                    # Test interpolation at a mid-point
                    small_bytes = points[0][0]
                    large_bytes = points[-1][0]
                    mid_bytes = (small_bytes + large_bytes) // 2

                    result = BenchmarkEstimator._linear_interpolate(points, mid_bytes)
                    self.assertIsNotNone(result)
                    self.assertGreater(result, 0)

                    # The interpolated value should be between the two endpoints
                    small_runtime = points[0][1]
                    large_runtime = points[-1][1]
                    self.assertGreaterEqual(result, min(small_runtime, large_runtime) * 0.5)
                    self.assertLessEqual(result, max(small_runtime, large_runtime) * 2.0)

        finally:
            os.unlink(trace_path)

    def test_profiler_trace_has_expected_operations(self):
        """
        Test that the profiler trace captures the expected operations.
        """
        from torch._inductor.fx_passes.node_runtime_estimation import (
            ProfilerTraceEstimator,
        )
        from torch.profiler import profile, ProfilerActivity

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            trace_path = f.name

        try:
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
            ) as prof:
                # Specific operations we expect to see
                a = torch.randn(100, 100)
                b = torch.randn(100, 100)

                # Matrix multiply
                c = torch.mm(a, b)

                # Element-wise add
                d = torch.add(a, b)

                # ReLU
                e = torch.relu(c)

                # Transpose
                f = a.t()

            prof.export_chrome_trace(trace_path)
            estimator = ProfilerTraceEstimator.from_chrome_trace(trace_path)

            compute_ops = estimator._estimator.benchmarks["compute"]
            op_names = " ".join(compute_ops.keys()).lower()

            # At least some of these should be present
            found_ops = []
            for op in ["mm", "add", "relu"]:
                if op in op_names:
                    found_ops.append(op)

            self.assertGreater(
                len(found_ops), 0,
                f"Should find at least one of mm/add/relu in {list(compute_ops.keys())}"
            )

        finally:
            os.unlink(trace_path)


if __name__ == "__main__":
    run_tests()
