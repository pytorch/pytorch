# Owner(s): ["module: dynamo"]
"""
Tests for the Dynamo Profiler functionality.

These tests verify that the dynamo_profiler config flag and related profiling
infrastructure work correctly for tracking where Dynamo spends time during compilation.
"""

import gc
import os
import pstats
import tempfile
import weakref

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class DynamoProfilerTests(torch._dynamo.test_case.TestCase):
    def test_graph_compiler_runtime_context_survives_output_graph_teardown(self):
        from torch._dynamo.output_graph import GraphCompiler
        from torch._dynamo.source import LocalSource
        from torch.fx import Graph, GraphModule

        class FakeRootTx:
            f_code = DynamoProfilerTests.test_function_trace_timing.__code__

            def format_frame_summary(self) -> str:
                return "fake frame summary"

        class FakeOutputGraph:
            def __init__(self) -> None:
                source = LocalSource("x")
                self.param_name_to_source = {"x": source}
                self.source_to_user_stacks = {source: []}
                self.dynamo_compile_id = None
                self.has_user_defined_allowed_in_graph = False
                self.co_fields = {"co_name": "fake", "co_filename": __file__}
                self.root_tx = FakeRootTx()

        output_graph = FakeOutputGraph()
        compiler = GraphCompiler(output_graph, lambda gm, _: gm.forward)
        runtime_context = compiler.runtime_context()

        graph = Graph()
        x = graph.placeholder("x")
        x._dynamo_source = LocalSource("x")
        graph.output(x)
        gm = GraphModule(torch.nn.Module(), graph)

        output_graph_ref = weakref.ref(output_graph)
        del output_graph
        gc.collect()
        self.assertIsNone(output_graph_ref())

        compiled = compiler.call_user_compiler(
            gm, [torch.randn(2)], runtime_context=runtime_context
        )
        result = compiled(torch.ones(2))

        self.assertIs(gm._param_name_to_source, runtime_context.param_name_to_source)
        self.assertIs(gm._source_to_user_stacks, runtime_context.source_to_user_stacks)
        self.assertTrue(torch.equal(result, torch.ones(2)))

    def test_bytecode_tracing_stop_captures_root_tx_before_close(self):
        from torch._dynamo.output_graph import BytecodeEmitter

        class FakeRootTx:
            def __init__(self) -> None:
                self.f_code = DynamoProfilerTests.test_function_trace_timing.__code__

        class FakeProfilerState:
            def __init__(self) -> None:
                self.recorded_timings = []
                self.dumped_output = None

            def pop(self):
                return type(
                    "StackEntry",
                    (),
                    {
                        "func_name": "fn",
                        "filename": __file__,
                        "firstlineno": 1,
                        "start_time_ns": 10,
                        "child_time_ns": 0,
                        "is_primitive_call": False,
                    },
                )()

            def record_timing(self, timing) -> None:
                self.recorded_timings.append(timing)

            def dump_stats(self, output_file) -> None:
                self.dumped_output = output_file

        class FakeOutputGraph:
            def __init__(self) -> None:
                self.root_tx = FakeRootTx()
                self.profiler_state = FakeProfilerState()

        output_graph = FakeOutputGraph()
        emitter = BytecodeEmitter(output_graph)
        emitter.compiler_trace_stack.callback(
            lambda: setattr(output_graph, "root_tx", None)
        )

        with torch._dynamo.config.patch(dynamo_profiler=True):
            emitter.mark_bytecode_tracing_stop()

        self.assertEqual(len(output_graph.profiler_state.recorded_timings), 1)
        self.assertEqual(
            output_graph.profiler_state.recorded_timings[0].bytecode_count,
            len(FakeRootTx().f_code.co_code),
        )

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
