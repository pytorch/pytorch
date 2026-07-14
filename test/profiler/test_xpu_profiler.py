# Owner(s): ["oncall: profiler"]


import unittest
from collections import defaultdict

import torch
from torch.profiler import DeviceType
from torch.testing._internal.common_utils import run_tests, TEST_XPU, TestCase


Verbose = False


class XpuProfilerTest(TestCase):
    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_profiler(self):
        t = torch.empty(1000, dtype=torch.int, device="xpu")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.XPU,
            ]
        ) as p:
            for _ in range(10):
                t.zero_()

        events = defaultdict(int)
        for event in p.events():
            events[event.device_type] += 1

        if Verbose:
            print(f"{events = }")

        self.assertEqual(len(events), 2)
        self.assertTrue(DeviceType.CPU in events)
        self.assertTrue(DeviceType.XPU in events)


if __name__ == "__main__":
    run_tests()
