# Owner(s): ["module: inductor"]
import json
import unittest

import torch
import torch._dynamo.test_case
import torch._inductor.utils

from torch.profiler import ProfilerActivity

from torch.testing._internal.common_utils import TemporaryFileName, TEST_WITH_ROCM

HAS_TRITON = torch._inductor.utils.has_triton()


class DynamoProfilerTests(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_triton_launch(self):
        # Verify that we get some sort of CPU-side indication of triton kernel launches
        # in the profile traces. Currently, those appear as `cuLaunchKernel`. If this
        # detail changes, the test can be updated or removed.
        @torch.compile
        def fn(x, y):
            return (x + y).sin().cos()

        x, y = (torch.rand((4, 4), device="cuda") for _ in range(2))

        with torch.profiler.profile() as prof:
            fn(x, y)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace_json = json.load(f)

        self.assertTrue("traceEvents" in trace_json)
        events = trace_json["traceEvents"]

        def nameMatchesLaunchKernel(event_name):
            return "cuLaunchKernel" in event_name

        self.assertTrue(
            any(
                ("name" in event and "cuLaunchKernel" == event["name"])
                for event in events
            )
        )

    @unittest.skipIf(not HAS_TRITON, "requires cuda & triton")
    def test_inductor_profiling_kernel_names(self):
        """
        We expect a record_function event to be added on the CPU side, surrounding
        the launch of each triton kernel.
        """

        def fn(x, y):
            return (x + y).sin().cos()

        fn_opt = torch.compile(fn)

        x, y = (torch.rand((4, 4), device="cuda") for _ in range(2))

        for _ in range(2):
            fn_opt(x, y)

        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            fn_opt(x, y)

        # The name of the kernel is expected to match the name of the kernel in debug
        # files etc. The name could change in the future, but it seems reasonable that
        # the name should always contain "triton" and "sin" - "sin" because this
        # kernel contains a sin op. If this changes in the future, feel free to change
        # the assertion here.
        # As of time of writing, the kernel name was "triton_poi_fused_add_cos_sin_0"
        # Debugging tips: you can add prof.export_chrome_trace("test.json") inline in
        # this test, and then view test.json in chrome://tracing to see the trace.
        self.assertTrue(
            any(
                (
                    hasattr(event, "name")
                    and "sin" in event.name
                    and "triton" in event.name
                )
                for event in prof.events()
            )
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not TEST_WITH_ROCM:
        run_tests()
