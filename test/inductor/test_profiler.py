# Owner(s): ["module: inductor"]
import json
import unittest

import torch
import torch._dynamo.test_case
import torch._inductor.utils

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not TEST_WITH_ROCM:
        run_tests()
