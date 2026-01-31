# Owner(s): ["module: dynamo"]
"""
Tests for the Dynamo Profiler functionality.

These tests verify that the dynamo_profiler config flag and related profiling
infrastructure work correctly for tracking where Dynamo spends time during compilation.
"""

from pstats import SortKey

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class DynamoProfilerTests(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch(dynamo_profiler=True)
    def test_function_trace_timing(self):
        """Test that inline function timing data is captured during compilation."""
        from torch._dynamo.dynamo_profiler import FunctionTraceTiming
        from torch._guards import TracingContext

        captured_timings = []

        def helper_fn(x):
            return x * 2 + 1

        def nested_helper(x):
            return helper_fn(x) + helper_fn(x * 2)

        def main_fn(x):
            return nested_helper(x)

        def timing_capturing_backend(gm, example_inputs):
            tc = TracingContext.try_get()
            if tc and tc.profiler_state:
                timings = tc.profiler_state.get_timings()
                if timings:
                    captured_timings.extend(timings)
            return gm.forward

        torch._dynamo.reset()

        @torch.compile(backend=timing_capturing_backend)
        def test_fn(x):
            return main_fn(x)

        x = torch.randn(10)
        test_fn(x)

        # Verify timing data was captured
        self.assertGreater(len(captured_timings), 0)

        # Verify all entries are FunctionTraceTiming instances
        for t in captured_timings:
            self.assertIsInstance(t, FunctionTraceTiming)
            self.assertGreater(t.trace_time_ns, 0)
            self.assertGreater(t.bytecode_count, 0)
            self.assertGreaterEqual(t.inline_depth, 1)

        # Verify we captured the expected functions
        func_names = {t.func_name for t in captured_timings}
        self.assertIn("helper_fn", func_names)
        self.assertIn("nested_helper", func_names)
        self.assertIn("main_fn", func_names)

    @torch._dynamo.config.patch(dynamo_profiler=True)
    def test_generate_pstats_from_timings(self):
        """Test generating pstats-compatible output from trace timings."""
        import pstats
        import tempfile

        from torch._dynamo.dynamo_profiler import DynamoProfilerState
        from torch._guards import TracingContext

        trace_timings = []

        def helper_fn(x):
            return x * 2

        def main_fn(x):
            return helper_fn(x)

        def timing_backend(gm, example_inputs):
            tc = TracingContext.try_get()
            if tc and tc.profiler_state:
                timings = tc.profiler_state.get_timings()
                if timings:
                    trace_timings.extend(timings)
            return gm.forward

        torch._dynamo.reset()

        @torch.compile(backend=timing_backend)
        def compiled_fn(x):
            return main_fn(x)

        x = torch.randn(10)
        compiled_fn(x)

        # Generate pstats
        with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as f:
            profiler_state = DynamoProfilerState()
            profiler_state.timings = trace_timings
            stats = profiler_state.generate_pstats(f.name)
            print(stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())

            # Verify stats object is valid
            self.assertIsInstance(stats, pstats.Stats)
            self.assertGreater(stats.total_calls, 0)

            # Verify file can be loaded
            loaded_stats = pstats.Stats(f.name)
            self.assertEqual(loaded_stats.total_calls, stats.total_calls)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
