# Owner(s): ["module: dynamo"]
import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils

from torch._dynamo.utils import dynamo_timed


class DynamoProfilerTests(torch._dynamo.test_case.TestCase):
    def test_dynamo_timed_profiling_isolated(self):
        # @dynamo_timed functions should appear in profile traces.
        @dynamo_timed
        def inner_fn(x):
            return x.sin()

        def outer_fn(x, y):
            return inner_fn(x) * y

        x, y = [torch.rand((2, 2)) for _ in range(2)]

        with torch.profiler.profile(with_stack=False) as prof:
            outer_fn(x, y)

        self.assertTrue(
            any("inner_fn (dynamo_timed)" in evt.name for evt in prof.events())
        )

    def test_dynamo_timed_profiling_backend_compile(self):
        # @dynamo_timed functions should appear in profile traces.
        # this checks whether these actually appear in actual dynamo execution.
        # "backend_compile" is just chosen as an example; if it gets renamed
        # this test can be replaced or deleted

        fn_name = "call_user_compiler"

        def fn(x, y):
            return x.sin() * y.cos()

        x, y = [torch.rand((2, 2)) for _ in range(2)]

        with torch.profiler.profile(with_stack=False) as prof:
            torch._dynamo.optimize("aot_eager")(fn)(x, y)

        self.assertTrue(
            any(f"{fn_name} (dynamo_timed)" in evt.name for evt in prof.events())
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
