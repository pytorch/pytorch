"""
Tests for torch._dynamo.profiler — per-function trace timing breakdown.

This module tests the profiler API that reports what percentage of compile
time is spent tracing each user function.
"""

import torch
import torch._dynamo
import torch._dynamo.profiler as profiler
import torch._dynamo.test_case
from torch._dynamo.testing import make_test_cls_with_patches


class ProfilerTests(torch._dynamo.test_case.TestCase):
    """Tests for the Dynamo tracing profiler."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        profiler.clear()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        profiler.clear()

    def test_basic_trace_breakdown(self):
        """Basic test: compile a simple function and get trace breakdown."""

        @torch.compile
        def simple_fn(x):
            return x.sin() + x.cos()

        x = torch.randn(10)
        simple_fn(x)

        # Get the trace breakdown
        breakdown = profiler.trace_breakdown()

        # Should have at least one entry
        self.assertGreater(len(breakdown), 0)

        # Check first entry has expected fields
        entry = breakdown[0]
        self.assertIsNotNone(entry.compile_id)
        self.assertEqual(entry.function_name, "simple_fn")
        self.assertIn("test_profiler.py", entry.filename)
        self.assertGreater(entry.tracing_time_s, 0)
        self.assertGreater(entry.total_compile_time_s, 0)
        self.assertGreaterEqual(entry.tracing_percent, 0)
        self.assertLessEqual(entry.tracing_percent, 100)

    def test_multiple_functions(self):
        """Test that multiple compiled functions each get their own entry."""

        @torch.compile
        def fn1(x):
            return x.sin()

        @torch.compile
        def fn2(x):
            return x.cos()

        x = torch.randn(10)
        fn1(x)
        fn2(x)

        breakdown = profiler.trace_breakdown()

        # Should have entries for both functions
        self.assertGreaterEqual(len(breakdown), 2)

        function_names = {e.function_name for e in breakdown}
        self.assertIn("fn1", function_names)
        self.assertIn("fn2", function_names)

    def test_recompile_tracking(self):
        """Test that recompiles are tracked with cache_size."""

        @torch.compile
        def dynamic_fn(x):
            return x.sum()

        # Trigger recompile with different shapes
        dynamic_fn(torch.randn(10))
        dynamic_fn(torch.randn(20))

        breakdown = profiler.trace_breakdown()

        # Should have multiple entries or entries with cache_size > 0
        self.assertGreater(len(breakdown), 0)

    def test_print_trace_breakdown(self):
        """Test that print_trace_breakdown runs without error."""

        @torch.compile
        def simple_fn(x):
            return x.relu()

        simple_fn(torch.randn(10))

        # Should not raise
        profiler.print_trace_breakdown()

    def test_aggregate_by_frame(self):
        """Test aggregation by frame groups compiles of same function."""

        @torch.compile
        def fn(x):
            return x + 1

        fn(torch.randn(10))
        fn(torch.randn(20))  # May trigger recompile

        # Get both views
        by_compile_id = profiler.trace_breakdown(aggregate_by="compile_id")
        by_frame = profiler.trace_breakdown(aggregate_by="frame")

        # Frame aggregation should have fewer or equal entries
        self.assertLessEqual(len(by_frame), len(by_compile_id))

    def test_clear(self):
        """Test that clear() removes all collected data."""

        @torch.compile
        def fn(x):
            return x * 2

        fn(torch.randn(10))
        self.assertGreater(len(profiler.trace_breakdown()), 0)

        profiler.clear()
        self.assertEqual(len(profiler.trace_breakdown()), 0)

    def test_tracing_percent_reasonable(self):
        """Test that tracing percentage is within reasonable bounds."""

        @torch.compile(backend="eager")
        def fn(x):
            # Simple function - tracing should be significant portion
            return x.sin().cos().tan()

        fn(torch.randn(100))

        breakdown = profiler.trace_breakdown()
        self.assertGreater(len(breakdown), 0)

        entry = breakdown[0]
        # Tracing should be at least some portion of compile time
        # With eager backend, most time should be tracing
        self.assertGreater(entry.tracing_percent, 0)

    def test_export_json(self):
        """Test JSON export functionality."""
        import json
        import tempfile
        import os

        @torch.compile
        def fn(x):
            return x + x

        fn(torch.randn(10))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            profiler.export_trace_breakdown(filepath)

            with open(filepath) as f:
                data = json.load(f)

            self.assertIn("version", data)
            self.assertIn("entries", data)
            self.assertGreater(len(data["entries"]), 0)
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
