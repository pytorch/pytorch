# Owner(s): ["oncall: profiler"]


import json
import os
import tempfile
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

    def gen_and_check_json(self, p, json_file):
        p.export_chrome_trace(json_file)

        with open(json_file) as f:
            data = json.load(f)
            self.assertTrue("traceEvents" in data)

            trace_events = data["traceEvents"]
            self.assertTrue(isinstance(trace_events, list))
            self.assertTrue(len(trace_events) > 0)

            count_names = defaultdict(int)
            count_cats = defaultdict(int)
            for event in trace_events:
                self.assertTrue("ph" in event)
                self.assertTrue("name" in event)

                if event["ph"] == "X":
                    self.assertTrue("cat" in event)
                    count_names[event["name"]] += 1
                    count_cats[event["cat"]] += 1

            if Verbose:
                print(f"{count_names = }")
                print(f"{count_cats = }")

            self.assertTrue("xpu_runtime" in count_cats)
            self.assertTrue("xpu_driver" in count_cats)
            self.assertTrue("kernel" in count_cats)

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_profiler_xpu_driver(self):
        a = torch.rand([100, 200]).to("xpu")
        b = torch.rand([200, 300]).to("xpu")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ],
        ) as p:
            r1 = torch.matmul(a, b)
            r2 = torch.add(r1, 1.0)
            result = torch.abs(r2)

        self.assertTrue(result.numel() > 0)

        json_file = os.environ.get("JSON_FILE")

        if json_file:
            self.gen_and_check_json(p, json_file)
        else:
            with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
                self.gen_and_check_json(p, tmp.name)

        if Verbose:
            print(p.key_averages().table())


if __name__ == "__main__":
    run_tests()
