# Owner(s): ["module: dynamo"]
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

    def test_compile_profile_timing_breakdown(self):
        """Test that TORCH_COMPILE_PROFILE=1 outputs detailed timing breakdown."""
        import io
        import re
        import sys

        torch._dynamo.reset()

        captured_stderr = io.StringIO()

        def fn(x):
            return x.sin().cos()

        with patch.object(torch._dynamo.config, "compile_profile", True):
            old_stderr = sys.stderr
            sys.stderr = captured_stderr
            try:
                opt_fn = torch.compile(fn, backend="aot_eager")
                opt_fn(torch.randn(10))
            finally:
                sys.stderr = old_stderr

        output = captured_stderr.getvalue()

        self.assertIn("TORCH_COMPILE_PROFILE", output)
        self.assertIn("Compile Time Breakdown", output)
        self.assertIn("Total compile time:", output)

        self.assertTrue(
            re.search(r"\d+\.\d{4}s", output),
            f"Expected timing data in output, got: {output[:500]}",
        )

    def test_compile_profile_has_detailed_phases(self):
        """Test that compile_times returns detailed phase information when profiling."""
        torch._dynamo.reset()

        def fn(x):
            return x.sin() + x.cos()

        with patch.object(torch._dynamo.config, "compile_profile", True):
            opt_fn = torch.compile(fn, backend="aot_eager")
            opt_fn(torch.randn(10))

        self.assertIn("TorchDynamo compilation metrics", torch._dynamo.utils.compile_times(repr="str"))

        self.assertGreater(
            len(torch._dynamo.utils.compilation_time_metrics), 5,
            f"Should have detailed timing breakdown, got: {list(torch._dynamo.utils.compilation_time_metrics.keys())}"
        )

    def test_get_cache_stats(self):
        """Test that get_cache_stats returns correct cache statistics from counters."""
        from torch._dynamo.utils import counters, get_cache_stats

        counters.clear()

        stats = get_cache_stats()
        self.assertEqual(stats, {})

        counters["inductor"]["fxgraph_cache_hit"] = 3
        counters["inductor"]["fxgraph_cache_miss"] = 2

        stats = get_cache_stats()
        self.assertIn("fx_graph_cache", stats)
        self.assertEqual(stats["fx_graph_cache"]["hits"], 3)
        self.assertEqual(stats["fx_graph_cache"]["misses"], 2)

        counters["inductor"]["async_compile_cache_hit"] = 5
        counters["inductor"]["async_compile_cache_miss"] = 1

        stats = get_cache_stats()
        self.assertIn("async_compile_cache", stats)
        self.assertEqual(stats["async_compile_cache"]["hits"], 5)
        self.assertEqual(stats["async_compile_cache"]["misses"], 1)

        counters["inductor"]["generated_module_cache_hit"] = 10
        counters["inductor"]["generated_module_cache_miss"] = 0

        stats = get_cache_stats()
        self.assertIn("generated_module_cache", stats)
        self.assertEqual(stats["generated_module_cache"]["hits"], 10)
        self.assertEqual(stats["generated_module_cache"]["misses"], 0)

        counters.clear()

    def test_log_cache_stats(self):
        """Test that log_cache_stats logs cache hit/miss rates."""
        import logging

        from torch._dynamo.utils import counters, log_cache_stats

        counters.clear()

        counters["inductor"]["fxgraph_cache_hit"] = 8
        counters["inductor"]["fxgraph_cache_miss"] = 2

        log_records = []
        handler = logging.Handler()
        handler.emit = lambda record: log_records.append(record)

        logger = logging.getLogger("torch._dynamo")
        original_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        try:
            log_cache_stats()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)
            counters.clear()

        self.assertTrue(len(log_records) > 0, "Expected log output from log_cache_stats")
        log_message = log_records[0].getMessage()
        self.assertIn("Cache hit/miss statistics", log_message)
        self.assertIn("fx_graph_cache", log_message)
        self.assertIn("80.0% hit rate", log_message)

    def test_compile_profile_includes_cache_stats(self):
        """Test that print_compile_profile includes cache statistics when available."""
        import io
        import sys

        from torch._dynamo.utils import counters

        torch._dynamo.reset()
        counters.clear()

        captured_stderr = io.StringIO()

        def fn(x):
            return x.sin().cos()

        with patch.object(torch._dynamo.config, "compile_profile", True):
            old_stderr = sys.stderr
            sys.stderr = captured_stderr
            try:
                opt_fn = torch.compile(fn, backend="inductor")
                opt_fn(torch.randn(10))
            finally:
                sys.stderr = old_stderr

        output = captured_stderr.getvalue()
        self.assertIn("TORCH_COMPILE_PROFILE", output)

        if counters["inductor"]["fxgraph_cache_miss"] > 0 or counters["inductor"]["fxgraph_cache_hit"] > 0:
            self.assertIn("Cache Statistics", output)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
