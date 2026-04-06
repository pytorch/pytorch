# Owner(s): ["module: dynamo"]
"""
Tests for the Dynamo Profiler functionality.

These tests verify that the dynamo_profiler config flag and related profiling
infrastructure work correctly for tracking where Dynamo spends time during compilation.
"""

import os
import pstats
import tempfile

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class DynamoProfilerTests(torch._dynamo.test_case.TestCase):
    def test_function_trace_timing(self):
        """Test that inline function timing data is captured during compilation."""

        def helper_fn(x):
            return x * 2 + 1

        def nested_helper(x):
            return helper_fn(x) + helper_fn(x * 2)

        def main_fn(x):
            return nested_helper(x)

        torch._dynamo.reset()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = os.path.join(tmpdir, "profile.prof")

            with torch._dynamo.config.patch(dynamo_profiler=profile_path):

                @torch.compile(backend="eager")
                def test_fn(x):
                    return main_fn(x)

                x = torch.randn(10)
                test_fn(x)

            # Load and verify the profile
            stats = pstats.Stats(profile_path)

            # Verify stats object is valid
            self.assertGreater(stats.total_calls, 0)

            # Verify we captured the expected functions
            func_names = {key[2] for key in stats.stats}
            self.assertIn("helper_fn", func_names)
            self.assertIn("nested_helper", func_names)
            self.assertIn("main_fn", func_names)
            self.assertIn("test_fn", func_names)  # Root function

    def test_pstats_file_loadable(self):
        """Test that the generated pstats file can be loaded and analyzed."""

        def helper_fn(x):
            return x * 2

        def main_fn(x):
            return helper_fn(x)

        torch._dynamo.reset()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = os.path.join(tmpdir, "profile.prof")

            with torch._dynamo.config.patch(dynamo_profiler=profile_path):

                @torch.compile(backend="eager")
                def compiled_fn(x):
                    return main_fn(x)

                x = torch.randn(10)
                compiled_fn(x)

            # Verify file can be loaded and analyzed
            stats = pstats.Stats(profile_path)
            self.assertGreater(stats.total_calls, 0)

            # Verify we can sort and print stats (basic pstats operations)
            stats.sort_stats("cumulative")
            # This would raise if the stats format is invalid
            stats.print_stats(5)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
