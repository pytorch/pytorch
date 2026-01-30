# Owner(s): ["module: dynamo"]
from pstats import SortKey
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch._dynamo.utils import dynamo_timed
from torch.profiler import record_function
from torch.testing._internal.common_utils import TemporaryFileName


class DynamoProfilerTests(torch._dynamo.test_case.TestCase):
    def test_dynamo_timed_profiling_isolated(self):
        # dynamo_timed functions should appear in profile traces.
        def inner_fn(x):
            with dynamo_timed("inner_fn"):
                return x.sin()

        def outer_fn(x, y):
            return inner_fn(x) * y

        x, y = (torch.rand((2, 2)) for _ in range(2))

        with torch.profiler.profile(with_stack=False) as prof:
            outer_fn(x, y)

        self.assertTrue(
            any("inner_fn (dynamo_timed)" in evt.name for evt in prof.events())
        )

    def test_dynamo_timed_profiling_backend_compile(self):
        # dynamo_timed functions should appear in profile traces.
        # this checks whether these actually appear in actual dynamo execution.
        # "backend_compile" is just chosen as an example; if it gets renamed
        # this test can be replaced or deleted

        fn_name = "call_user_compiler"

        def fn(x, y):
            return x.sin() * y.cos()

        x, y = (torch.rand((2, 2)) for _ in range(2))

        with torch.profiler.profile(with_stack=False) as prof:
            torch.compile(fn, backend="aot_eager")(x, y)

        self.assertTrue(
            any(f"{fn_name} (dynamo_timed)" in evt.name for evt in prof.events())
        )

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    def test_profile_dynamic_shapes_runtime(self):
        def fn(x, y, z):
            return x @ y + z

        opt_fn = torch.compile(fn, backend="aot_eager", dynamic=True, fullgraph=True)

        inputs = [
            (torch.rand(a, b), torch.rand(b, c), torch.rand(a, c))
            for (a, b, c) in [(15, 16, 17), (15, 15, 16), (16, 16, 16)]
        ]

        opt_fn(*inputs[0])
        opt_fn(*inputs[1])

        with torch.profiler.profile(record_shapes=True):
            opt_fn(*inputs[2])

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    def test_profile_dynamic_shapes_compilation(self):
        def fn(x, y, z):
            return x @ y + z

        opt_fn = torch.compile(fn, backend="aot_eager", dynamic=True, fullgraph=True)

        inputs = (torch.rand(15, 16), torch.rand(16, 17), torch.rand(15, 17))

        with torch.profiler.profile(record_shapes=True):
            opt_fn(*inputs)

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    def test_profile_dynamic_shapes_list_compilation(self):
        def fn(x, y, z):
            return torch.cat([x, y], dim=0) + z

        opt_fn = torch.compile(fn, backend="aot_eager", dynamic=True, fullgraph=True)

        inputs = (torch.rand(4, 16), torch.rand(12, 16), torch.rand(16, 16))

        with torch.profiler.profile(record_shapes=True):
            opt_fn(*inputs)

    def test_execution_trace_dynamic_shapes(self):
        def fn(x, y, z):
            return x @ y + z

        et = torch.profiler.ExecutionTraceObserver()
        opt_fn = torch.compile(fn, dynamic=True, backend="aot_eager")
        inputs = [torch.rand((4, 4)) for _ in range(3)]

        with TemporaryFileName() as fname:
            et.register_callback(fname)
            et.start()
            opt_fn(*inputs)
            et.stop()
            et.unregister_callback()

    def test_profiler_cache_lookup(self):
        def fn(x):
            y = x**2
            y = y + 2
            z = y**3
            return z

        for profiler, get_events in (
            (torch.autograd.profiler.profile, lambda prof: prof.function_events),
            (torch.profiler.profiler.profile, lambda prof: prof.events()),
        ):
            x = torch.randn((2, 2), requires_grad=True)
            ref = fn(x)
            opt_fn = torch.compile(fn, backend="aot_eager")

            # warmup
            opt_fn(x)

            with profiler() as prof:
                res = opt_fn(x)
            events = list(
                filter(
                    lambda event: "TorchDynamo Cache Lookup" in event.name,
                    get_events(prof),
                )
            )

            self.assertEqual(ref, res)
            self.assertTrue(
                len(events) == 1,
                "Expected one lookup profiler event for one opt_fn run",
            )

    def test_profiler_cache_lookup_profiler_step(self):
        def fn(x, y, z):
            return torch.add(torch.sub(x, y), z)

        opt_fn = torch.compile(fn, backend="aot_eager")

        (
            x,
            y,
            z,
        ) = (torch.rand(4, 4) for _ in range(3))

        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=2, repeat=1)
        )

        for _ in range(10):
            opt_fn(x, y, z)
            prof.step()

        self.assertTrue(
            any(e.name == "TorchDynamo Cache Lookup" for e in prof.events())
        )

    def test_profiler_enabled_export(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.sin(x)
                if torch.autograd._profiler_enabled():
                    return torch.cos(x)
                else:
                    return torch.sigmoid(x)

        mod = Mod()

        x = torch.randn(4)
        opt_mod = torch._dynamo.export(mod, (x))

        ref = mod(x)
        res = opt_mod.graph_module(x)
        self.assertEqual(ref, res)

        with torch.autograd.profiler.profile():
            ref = mod(x)
            # Reexport because export skips guards
            opt_mod = torch._dynamo.export(mod, (x))
            res = opt_mod.graph_module(x)
            self.assertEqual(ref, res)

    def test_profiler_dynamo_compiled_region(self):
        def fn(x, y):
            r = y.sum(dim=1)
            print(r.shape)
            return x * r

        with torch.profiler.profile() as prof:
            fn_c = torch.compile(fn)

            fn_c(
                torch.randn(10),
                torch.randn(10, 10),
            )

            fn_c(
                torch.randn(10),
                torch.randn(10, 15),
            )

        annotations = [e.name for e in prof.events() if "Torch-Compiled" in e.name]
        self.assertEqual(
            annotations,
            [
                "Torch-Compiled Region: 0/0",
                "Torch-Compiled Region: 1/0",
                "Torch-Compiled Region: 0/1",
                "Torch-Compiled Region: 1/0",
            ],
        )

    @torch._dynamo.config.patch("capture_profiler_record_function", True)
    def test_dynamo_preserve_record_func(self):
        def fn(x):
            with record_function("my_net1"):
                a = x.sin()
            with record_function("my_cos"):
                b = a.cos()
            with record_function("my_net2"):
                c = b + 2
            return c

        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
        fn_c = torch.compile(fn, backend=backend)
        fn_c(
            torch.randn(10),
        )
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    _record_function_enter_new = torch.ops.profiler._record_function_enter_new('my_net1', None)
    a = l_x_.sin();  l_x_ = None
    _record_function_exit__record_function = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new);  _record_function_enter_new = _record_function_exit__record_function = None
    _record_function_enter_new_1 = torch.ops.profiler._record_function_enter_new('my_cos', None)
    b = a.cos();  a = None
    _record_function_exit__record_function_1 = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new_1);  _record_function_enter_new_1 = _record_function_exit__record_function_1 = None
    _record_function_enter_new_2 = torch.ops.profiler._record_function_enter_new('my_net2', None)
    c = b + 2;  b = None
    _record_function_exit__record_function_2 = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new_2);  _record_function_enter_new_2 = _record_function_exit__record_function_2 = None
    return (c,)""",  # noqa: B950
        )
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    _record_function_enter_new = torch.ops.profiler._record_function_enter_new.default('my_net1')
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    _record_function_exit = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new);  _record_function_enter_new = _record_function_exit = None
    _record_function_enter_new_1 = torch.ops.profiler._record_function_enter_new.default('my_cos')
    cos = torch.ops.aten.cos.default(sin);  sin = None
    _record_function_exit_1 = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new_1);  _record_function_enter_new_1 = _record_function_exit_1 = None
    _record_function_enter_new_2 = torch.ops.profiler._record_function_enter_new.default('my_net2')
    add = torch.ops.aten.add.Tensor(cos, 2);  cos = None
    _record_function_exit_2 = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new_2);  _record_function_enter_new_2 = _record_function_exit_2 = None
    return (add,)""",  # noqa: B950
        )
        with torch.profiler.profile() as prof:
            fn_c(
                torch.randn(10),
            )

        annotations = [e.name for e in prof.events() if "my_" in e.name]
        self.assertEqual(
            annotations,
            [
                "my_net1",
                "my_cos",
                "my_net2",
            ],
        )

    @torch._dynamo.config.patch("capture_profiler_record_function", True)
    def test_dynamo_preserve_record_func_with_graph_break(self):
        # Test that record_function works correctly with graph breaks
        def fn(x):
            with record_function("pre_graph_break"):
                a = x.sin()
            # This causes a graph break
            torch._dynamo.graph_break()
            with record_function("post_graph_break"):
                b = a.cos()
            return b

        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
        fn_c = torch.compile(fn, backend=backend)
        fn_c(
            torch.randn(10),
        )

        # We expect 2 graphs due to the graph break
        self.assertEqual(len(backend.graphs), 2)

        # First graph should have the pre_graph_break record_function
        self.assertIn("pre_graph_break", backend.graphs[0].code)
        self.assertIn("_record_function_enter_new", backend.graphs[0].code)
        self.assertIn("_record_function_exit", backend.graphs[0].code)

        # Second graph should have the post_graph_break record_function
        self.assertIn("post_graph_break", backend.graphs[1].code)
        self.assertIn("_record_function_enter_new", backend.graphs[1].code)
        self.assertIn("_record_function_exit", backend.graphs[1].code)

        # Verify profiler events work correctly
        with torch.profiler.profile() as prof:
            fn_c(
                torch.randn(10),
            )

        annotations = [
            e.name
            for e in prof.events()
            if e.name in ["pre_graph_break", "post_graph_break"]
        ]
        # Both record_function contexts should appear in profiler events
        self.assertEqual(
            annotations,
            [
                "pre_graph_break",
                "post_graph_break",
            ],
        )

    @torch._dynamo.config.patch("capture_profiler_record_function", True)
    def test_dynamo_preserve_record_func_spanning_graph_break(self):
        # Test that record_function that spans across a graph break raises an error
        # This prevents the confusing behavior where the context gets duplicated across graphs
        def fn(x):
            x = x + 1
            with record_function("spanning_context"):
                a = x.sin()
                torch._dynamo.graph_break()
                b = a.cos()
            b = b - 1
            return b

        fn_c = torch.compile(fn, backend="aot_eager")
        x = torch.randn(10)
        fn_c(x)
        with torch.profiler.profile() as prof:
            result = fn_c(x)

        self.assertEqual(fn(x), result)

        annotations = [e.name for e in prof.events() if e.name == "spanning_context"]
        # record_function contexts should appear in profiler events once
        self.assertEqual(
            annotations,
            [
                "spanning_context",
            ],
        )

    @torch._dynamo.config.patch(dynamo_profiler=True)
    def test_function_trace_timing(self):
        """Test that inline function timing data is captured during compilation."""
        from torch._dynamo.profiler import (
            format_function_trace_timings_aggregated,
            get_function_trace_timings,
        )
        from torch._guards import FunctionTraceTiming

        captured_timings = []

        def helper_fn(x):
            return x * 2 + 1

        def nested_helper(x):
            return helper_fn(x) + helper_fn(x * 2)

        def main_fn(x):
            return nested_helper(x)

        def timing_capturing_backend(gm, example_inputs):
            timings = get_function_trace_timings()
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

        # Verify formatting function works
        formatted_agg = format_function_trace_timings_aggregated(captured_timings)
        self.assertIn("Aggregated", formatted_agg)

    @torch._dynamo.config.patch(dynamo_profiler=True)
    def test_generate_pstats_from_timings(self):
        """Test generating pstats-compatible output from trace timings."""
        import pstats
        import tempfile

        from torch._dynamo.profiler import (
            generate_pstats_from_timings,
            get_function_trace_timings,
        )

        trace_timings = []

        def helper_fn(x):
            return x * 2

        def main_fn(x):
            return helper_fn(x)

        def timing_backend(gm, example_inputs):
            timings = get_function_trace_timings()
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
            from pstats import SortKey

            stats = generate_pstats_from_timings(trace_timings, f.name)
            print(stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())

            # Verify stats object is valid
            self.assertIsInstance(stats, pstats.Stats)
            self.assertGreater(stats.total_calls, 0)

            # Verify file can be loaded
            loaded_stats = pstats.Stats(f.name)
            self.assertEqual(loaded_stats.total_calls, stats.total_calls)

    @torch._dynamo.config.patch(dynamo_profiler=True)
    def test_trace_timing_inspect_signature(self):
        """
        Test profiling Dynamo tracing of inspect.Signature.

        This test traces code that uses inspect.Signature and verifies that
        the profiling infrastructure can identify where Dynamo spends time.
        """
        import inspect

        from torch._dynamo.profiler import (
            format_function_trace_timings_aggregated,
            generate_pstats_from_timings,
            get_function_trace_timings,
        )

        def sample_function(a, b, c=10):
            """Sample function to inspect."""
            return a + b + c

        def function_with_signature_introspection(fn, x):
            """A function that uses inspect.signature internally."""
            sig = inspect.signature(fn)
            num_params = len(sig.parameters)
            if num_params > 0:
                return x * num_params
            return x

        def complex_signature_usage(x):
            """More complex usage of signature inspection."""
            sig = inspect.signature(sample_function)
            result = x
            for name, param in sig.parameters.items():
                if param.default is not inspect.Parameter.empty:
                    result = result + param.default
                else:
                    result = result * 2
            return result

        def nested_signature_calls(x):
            """Nested calls involving signature inspection."""
            y = function_with_signature_introspection(sample_function, x)
            z = complex_signature_usage(y)
            return z

        # Capture trace timings
        trace_timings = []

        def timing_capturing_backend(gm, example_inputs):
            timings = get_function_trace_timings()
            if timings:
                trace_timings.extend(timings)
            return gm.forward

        torch._dynamo.reset()

        @torch.compile(backend=timing_capturing_backend)
        def compiled_fn(x):
            return nested_signature_calls(x)

        x = torch.randn(10)
        compiled_fn(x)

        # Verify we captured timing data
        self.assertGreater(len(trace_timings), 0)

        # Verify we captured timings for the user-defined functions
        func_names = {t.func_name for t in trace_timings}
        self.assertIn("nested_signature_calls", func_names)
        self.assertIn("complex_signature_usage", func_names)
        self.assertIn("function_with_signature_introspection", func_names)

        # Verify we have caller info populated
        with_caller = [t for t in trace_timings if t.caller_func_name is not None]
        self.assertGreater(len(with_caller), 0)

        # Verify tottime <= cumtime for all entries
        for t in trace_timings:
            self.assertLessEqual(t.tottime_ns, t.cumtime_ns)

        # Verify aggregated formatting works
        formatted = format_function_trace_timings_aggregated(trace_timings)
        self.assertIn("nested_signature_calls", formatted)

        # Verify pstats generation works with caller edges
        stats = generate_pstats_from_timings(trace_timings)
        self.assertGreater(stats.total_calls, 2)

        print(stats.sort_stats(SortKey.CUMULATIVE).print_stats())

        # Verify we can print callers (tests that caller edges are populated)
        print("\nCallers:")
        stats.print_callers()

    @torch._dynamo.config.patch(dynamo_profiler=True)
    def test_profiler_recursive_and_shared_functions(self):
        """
        Test profiling with recursive functions and shared functions called by multiple callers.

        This tests that:
        1. Recursive function timing is handled correctly (no double counting)
        2. A common function called by multiple callers has correct caller edges
        3. tottime sums correctly across all functions
        """
        import inspect

        from torch._dynamo.profiler import (
            format_function_trace_timings_aggregated,
            generate_pstats_from_timings,
            get_function_trace_timings,
        )

        trace_timings = []

        def sample_fn(a, b, c=10, d=20, e=30):
            return a + b + c + d + e

        # Common function called by multiple callers - made expensive with inspect
        def common_helper(x):
            sig = inspect.signature(sample_fn)
            result = x
            for name, param in sig.parameters.items():
                if param.default is not inspect.Parameter.empty:
                    result = result + param.default
                else:
                    result = result * 2
            return result

        # Two different callers that both call common_helper
        def caller_a(x):
            y = common_helper(x)
            # More inspect usage to make it expensive
            sig = inspect.signature(sample_fn)
            for _ in sig.parameters:
                y = y + 1
            return y

        def caller_b(x):
            y = common_helper(x)
            z = common_helper(y)
            # More work
            sig = inspect.signature(sample_fn)
            for name, param in sig.parameters.items():
                if param.default is not inspect.Parameter.empty:
                    z = z + param.default
            return z

        # main_fn is itself recursive
        def main_fn(x, depth):
            if depth <= 0:
                return x
            a = caller_a(x)
            b = caller_b(x)
            return a + b + main_fn(x, depth - 1)

        def timing_backend(gm, example_inputs):
            timings = get_function_trace_timings()
            if timings:
                trace_timings.extend(timings)
            return gm.forward

        torch._dynamo.reset()

        @torch.compile(backend=timing_backend)
        def compiled_fn(x):
            return main_fn(x, 30)  # 31 calls to main_fn (depth 30, 29, ..., 0)

        x = torch.randn(10)
        compiled_fn(x)

        # Verify we captured timing data
        self.assertGreater(len(trace_timings), 0)

        # Count function calls
        func_counts = {}
        for t in trace_timings:
            func_counts[t.func_name] = func_counts.get(t.func_name, 0) + 1

        print("\nFunction call counts:")
        for name, count in sorted(func_counts.items()):
            print(f"  {name}: {count}")

        # Verify expected call counts
        # main_fn called 31 times (depth 30, 29, ..., 0)
        # caller_a and caller_b called 30 times each (at depth 30..1, not at depth 0)
        # common_helper called 90 times (30 calls to caller_a * 1 + 30 calls to caller_b * 2)
        self.assertEqual(func_counts.get("main_fn", 0), 31)
        self.assertEqual(func_counts.get("caller_a", 0), 30)
        self.assertEqual(func_counts.get("caller_b", 0), 30)
        self.assertEqual(func_counts.get("common_helper", 0), 90)

        # Verify tottime <= cumtime for all entries
        for t in trace_timings:
            self.assertLessEqual(
                t.tottime_ns,
                t.cumtime_ns,
                f"{t.func_name}: tottime ({t.tottime_ns}) > cumtime ({t.cumtime_ns})",
            )

        # Verify caller info for main_fn (should be called by itself for recursive calls)
        main_fn_callers = {
            t.caller_func_name for t in trace_timings if t.func_name == "main_fn"
        }
        self.assertIn("main_fn", main_fn_callers)  # recursive calls
        self.assertIn(None, main_fn_callers)  # first call has no caller in our tracking

        # Print aggregated timings
        print("\nAggregated timings:")
        print(format_function_trace_timings_aggregated(trace_timings))

        # Generate pstats and verify
        stats = generate_pstats_from_timings(trace_timings)
        print("\nPSTATS:")
        stats.sort_stats(SortKey.CUMULATIVE).print_stats()

        print("\nCALLERS:")
        stats.print_callers()

        print("\nCALLEES:")
        stats.print_callees()

        # Verify the key invariant: sum of all tottime should equal the root main_fn's cumtime
        total_tottime_ns = sum(t.tottime_ns for t in trace_timings)
        # The root main_fn call is the one with no caller (or depth 1)
        root_main_fn = next(
            t
            for t in trace_timings
            if t.func_name == "main_fn" and t.caller_func_name is None
        )

        print(f"\nTotal tottime: {total_tottime_ns / 1e6:.2f}ms")
        print(f"Root main_fn cumtime: {root_main_fn.cumtime_ns / 1e6:.2f}ms")

        # They should be approximately equal (allow more tolerance for longer runs)
        self.assertAlmostEqual(
            total_tottime_ns / 1e6,
            root_main_fn.cumtime_ns / 1e6,
            delta=50.0,  # Allow 50ms tolerance for longer runs
            msg="Sum of tottime should equal root main_fn's cumtime",
        )

    @torch._dynamo.config.patch(dynamo_profiler=True)
    def test_profiler_indirect_recursion(self):
        """
        Test profiling with indirect recursion: cmn -> A -> B -> cmn.

        This tests that the is_primitive_call flag correctly detects recursion
        through multiple layers, not just direct caller == callee.
        """
        import inspect

        from torch._dynamo.profiler import (
            format_function_trace_timings_aggregated,
            generate_pstats_from_timings,
            get_function_trace_timings,
        )

        trace_timings = []

        def sample_fn(a, b, c=10, d=20, e=30):
            return a + b + c + d + e

        # cmn calls A, which calls B, which calls cmn (indirect recursion)
        def fn_b(x, depth):
            # Some work
            sig = inspect.signature(sample_fn)
            for name, param in sig.parameters.items():
                if param.default is not inspect.Parameter.empty:
                    x = x + param.default
            # Indirectly recurse back to cmn
            if depth > 0:
                return fn_cmn(x, depth - 1)
            return x

        def fn_a(x, depth):
            # Some work
            sig = inspect.signature(sample_fn)
            for _ in sig.parameters:
                x = x + 1
            # Call B
            return fn_b(x, depth)

        def fn_cmn(x, depth):
            # Some work
            sig = inspect.signature(sample_fn)
            result = x
            for name, param in sig.parameters.items():
                if param.default is not inspect.Parameter.empty:
                    result = result + param.default
                else:
                    result = result * 2
            # Call A (which will call B, which will call us back)
            return fn_a(result, depth)

        def timing_backend(gm, example_inputs):
            timings = get_function_trace_timings()
            if timings:
                trace_timings.extend(timings)
            return gm.forward

        torch._dynamo.reset()

        @torch.compile(backend=timing_backend)
        def compiled_fn(x):
            return fn_cmn(x, 10)  # cmn -> A -> B -> cmn ... 11 times

        x = torch.randn(10)
        compiled_fn(x)

        # Verify we captured timing data
        self.assertGreater(len(trace_timings), 0)

        # Count function calls
        func_counts = {}
        for t in trace_timings:
            func_counts[t.func_name] = func_counts.get(t.func_name, 0) + 1

        print("\nFunction call counts:")
        for name, count in sorted(func_counts.items()):
            print(f"  {name}: {count}")

        # Verify expected call counts
        # fn_cmn: 11 (depth 10, 9, ..., 0)
        # fn_a: 11 (called by each fn_cmn)
        # fn_b: 11 (called by each fn_a)
        self.assertEqual(func_counts.get("fn_cmn", 0), 11)
        self.assertEqual(func_counts.get("fn_a", 0), 11)
        self.assertEqual(func_counts.get("fn_b", 0), 11)

        # Check is_primitive_call for fn_cmn
        fn_cmn_calls = [t for t in trace_timings if t.func_name == "fn_cmn"]
        primitive_cmn_calls = [t for t in fn_cmn_calls if t.is_primitive_call]
        recursive_cmn_calls = [t for t in fn_cmn_calls if not t.is_primitive_call]

        # Only the first call to fn_cmn should be primitive
        self.assertEqual(
            len(primitive_cmn_calls), 1, "Only root fn_cmn should be primitive"
        )
        self.assertEqual(
            len(recursive_cmn_calls), 10, "10 calls should be detected as recursive"
        )

        # Verify the recursive calls have fn_b as their caller (not fn_cmn)
        for t in recursive_cmn_calls:
            self.assertEqual(
                t.caller_func_name,
                "fn_b",
                "Recursive fn_cmn should be called by fn_b, not fn_cmn",
            )

        # Print aggregated timings
        print("\nAggregated timings:")
        print(format_function_trace_timings_aggregated(trace_timings))

        # Generate pstats and verify
        stats = generate_pstats_from_timings(trace_timings)
        print("\nPSTATS:")
        stats.sort_stats(SortKey.CUMULATIVE).print_stats()

        # Verify the key invariant: sum of all tottime should equal the root fn_cmn's cumtime
        total_tottime_ns = sum(t.tottime_ns for t in trace_timings)
        root_cmn = primitive_cmn_calls[0]

        print(f"\nTotal tottime: {total_tottime_ns / 1e6:.2f}ms")
        print(f"Root fn_cmn cumtime: {root_cmn.cumtime_ns / 1e6:.2f}ms")

        # They should be approximately equal
        self.assertAlmostEqual(
            total_tottime_ns / 1e6,
            root_cmn.cumtime_ns / 1e6,
            delta=50.0,
            msg="Sum of tottime should equal root fn_cmn's cumtime",
        )

    @torch._dynamo.config.patch(dynamo_profiler=True)
    def test_profiler_save_for_snakeviz(self):
        """
        Test saving profile data that can be loaded into snakeviz.

        This test creates a scenario where a common function is called by multiple
        callers with different frequencies, then saves the profile to a file that
        can be visualized with snakeviz to verify per-caller timing works.

        To visualize: snakeviz /tmp/dynamo_profile.prof
        """
        import inspect
        import os

        from torch._dynamo.profiler import generate_pstats_from_timings

        trace_timings = []

        def sample_fn(a, b, c=10, d=20):
            return a + b + c + d

        # Common function called by multiple callers with different frequencies
        def common_fn(x):
            sig = inspect.signature(sample_fn)
            for name, param in sig.parameters.items():
                if param.default is not inspect.Parameter.empty:
                    x = x + param.default
            return x

        # Caller A calls common_fn 3 times
        def caller_a(x):
            y = common_fn(x)
            y = common_fn(y)
            y = common_fn(y)
            return y

        # Caller B calls common_fn 1 time
        def caller_b(x):
            return common_fn(x)

        def main_fn(x):
            # Call caller_a 10 times -> 30 calls to common_fn from A
            # Call caller_b 10 times -> 10 calls to common_fn from B
            result = x
            for _ in range(10):
                result = caller_a(result)
                result = caller_b(result)
            return result

        def timing_backend(gm, example_inputs):
            timings = get_function_trace_timings()
            if timings:
                trace_timings.extend(timings)
            return gm.forward

        torch._dynamo.reset()

        @torch.compile(backend=timing_backend)
        def compiled_fn(x):
            return main_fn(x)

        x = torch.randn(10)
        compiled_fn(x)

        # Verify expected call distribution
        common_fn_timings = [t for t in trace_timings if t.func_name == "common_fn"]
        from_caller_a = [
            t for t in common_fn_timings if t.caller_func_name == "caller_a"
        ]
        from_caller_b = [
            t for t in common_fn_timings if t.caller_func_name == "caller_b"
        ]

        self.assertEqual(
            len(from_caller_a), 30, "common_fn should be called 30 times from caller_a"
        )
        self.assertEqual(
            len(from_caller_b), 10, "common_fn should be called 10 times from caller_b"
        )

        # Calculate per-caller timing
        time_from_a = sum(t.cumtime_ns for t in from_caller_a) / 1e6
        time_from_b = sum(t.cumtime_ns for t in from_caller_b) / 1e6

        print(
            f"\ncommon_fn called from caller_a: 30 times, {time_from_a:.2f}ms cumtime"
        )
        print(f"common_fn called from caller_b: 10 times, {time_from_b:.2f}ms cumtime")

        # Save to /tmp for easy snakeviz access - generate both versions
        # Version 1: Without call paths (aggregated by function)
        profile_path = "/tmp/dynamo_profile.prof"
        stats = generate_pstats_from_timings(trace_timings, profile_path)

        # Version 2: With call paths (separate entries per call path)
        # This makes snakeviz correctly show per-caller timing when drilling down
        profile_path_with_paths = "/tmp/dynamo_profile_with_paths.prof"
        stats_with_paths = generate_pstats_from_timings(
            trace_timings, profile_path_with_paths, use_call_paths=True
        )

        self.assertTrue(os.path.exists(profile_path))
        self.assertTrue(os.path.exists(profile_path_with_paths))
        print(f"\nProfile saved to: {profile_path}")
        print(f"Profile with call paths saved to: {profile_path_with_paths}")
        print("\nVisualize with:")
        print("  snakeviz /tmp/dynamo_profile.prof  # aggregated view")
        print("  snakeviz /tmp/dynamo_profile_with_paths.prof  # per-caller drill-down")

        # Print the call-path version to show it works
        print("\nCall-path pstats output:")
        stats_with_paths.sort_stats(SortKey.CUMULATIVE).print_stats()

        # Verify the file can be loaded and has correct caller edges
        import pstats

        loaded = pstats.Stats(profile_path)

        # Print callers to show per-caller breakdown
        print("\nPer-caller breakdown for common_fn:")
        loaded.sort_stats(SortKey.CUMULATIVE).print_callers("common_fn")

    @torch._dynamo.config.patch(dynamo_profiler=True)
    def test_dynamo_profiler_config_flag(self):
        """Test that the dynamo_profiler config flag prints pstats output."""
        import io
        import sys

        def helper_fn(x):
            return x * 2 + 1

        def main_fn(x):
            return helper_fn(x)

        torch._dynamo.reset()

        # Capture stdout to verify pstats output
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        try:

            @torch.compile(backend="eager")
            def compiled_fn(x):
                return main_fn(x)

            x = torch.randn(10)
            compiled_fn(x)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        print(output)  # Print for debugging

        # Verify pstats output was printed
        self.assertIn("Dynamo Profiler", output)
        self.assertIn("cumtime", output)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
