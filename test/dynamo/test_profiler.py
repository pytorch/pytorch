# Owner(s): ["module: dynamo"]
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch._dynamo.utils import dynamo_timed
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

        annotations = [e.name for e in prof.events() if "Compiled" in e.name]
        self.assertEqual(
            annotations,
            [
                "Torch-Compiled Region: 0/0",
                "Torch-Compiled Region: 1/0",
                "Torch-Compiled Region: 0/1",
                "Torch-Compiled Region: 1/0",
            ],
        )

    def test_profiler_enabled(self):
        def fn(x):
            x = torch.sin(x)
            if torch.autograd._profiler_enabled():
                return torch.cos(x)
            else:
                return torch.sigmoid(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        with torch.autograd.profiler.profile():
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_profiler_record_function_ignore(self):
        def fn(x):
            x = torch.sin(x)
            if torch.autograd._profiler_enabled():
                with torch.autograd.profiler.record_function("dummy"):
                    return torch.cos(x)
            else:
                return torch.sigmoid(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        with torch.autograd.profiler.profile():
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
