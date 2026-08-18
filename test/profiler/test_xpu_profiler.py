# Owner(s): ["oncall: profiler"]


import json
import os
import subprocess
import sys
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
            print(f"{events=}")

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
                print(f"{count_names=}")
                print(f"{count_cats=}")

            self.assertTrue("xpu_runtime" in count_cats)
            self.assertTrue("xpu_driver" in count_cats)
            self.assertTrue("kernel" in count_cats)

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_profiler_overhead(self):
        # The OVERHEAD activity type surfaces the profiler's own collection cost
        # on a dedicated track, matching CUDA behaviour. It is enabled together
        # with ProfilerActivity.XPU via kXpuTypes. PTI instruments the device
        # operations it traces, so a workload that issues device ops is expected
        # to produce overhead records.
        a = torch.rand([100, 200]).to("xpu")
        b = torch.rand([200, 300]).to("xpu")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ],
        ) as p:
            for _ in range(10):
                r = torch.matmul(a, b)

        self.assertTrue(r.numel() > 0)

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
            p.export_chrome_trace(tmp.name)
            with open(tmp.name) as f:
                data = json.load(f)

            overhead_events = [
                e
                for e in data["traceEvents"]
                if e.get("ph") == "X" and e.get("cat") == "overhead"
            ]

            if Verbose:
                print(f"{len(overhead_events) = }")

            self.assertGreater(len(overhead_events), 0)
            for e in overhead_events:
                args = e.get("args", {})
                self.assertIn("overhead cost", args)
                self.assertIn("overhead count", args)

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

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_profiler_with_xpu_graph(self):
        # Subprocess: Kineto registers its activity set once per process, and
        # the other tests in this file already profile.
        script = """
import torch
from torch.profiler import ProfilerActivity, profile

def add_one(in_: torch.Tensor):
    return in_ + 1

sample_arg = torch.zeros(10, device="xpu").requires_grad_(True)

add_one_graphed = torch.xpu.graphs.make_graphed_callables(add_one, sample_args=(sample_arg,))
zeros = torch.zeros(10, device="xpu")
out = add_one_graphed(zeros)
if out[0] != 1:
    raise AssertionError(f"Expected out[0] == 1, got {out[0]}")

with profile(activities=[ProfilerActivity.CPU]):
    add_one_graphed(zeros)

# Graph replay surfaces no device-side events, so there is nothing to assert.
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]):
    add_one_graphed(zeros)

out = add_one_graphed(zeros)
torch.xpu.synchronize()
if out[0] != 1:
    raise AssertionError(f"Expected out[0] == 1, got {out[0]}")
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            timeout=120,
            capture_output=True,
        )
        if proc.returncode != 0:
            # Native graph recording needs SYCL compiler >= 2026.1.0 on
            # non-PVC architectures (aten/src/ATen/xpu/XPUGraph.cpp);
            # older builds can fail to finalize the graph on the driver side.
            if "Failed to instantiate native UR executable graph" in proc.stderr:
                self.skipTest(
                    "torch.xpu.graphs requires oneAPI SYCL compiler >= 2026.1.0 "
                    "for native graph recording on this architecture"
                )
            self.fail(f"subprocess failed (exit {proc.returncode}):\n{proc.stderr}")


if __name__ == "__main__":
    run_tests()
