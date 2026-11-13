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


def find_ac2g_flow_finishes_off_device(events):
    """Return the ac2g flow-finish events that do not land on a GPU device track.

    ac2g means Async CPU-to-GPU: every such flow starts at a host launch and
    must end at the device op, so a flow finish (ph='f') must sit on a GPU
    device track. GPU tracks are the pids labelled "GPU ..." by the chrome-trace
    process_labels metadata; a finish on any other (host) track is malformed --
    e.g. a redundant arrow to a driver/overhead subspan, or an orphan host
    finish. Type-agnostic (does not look at record-type names).
    """
    gpu_pids = {
        e.get("pid")
        for e in events
        if e.get("ph") == "M"
        and e.get("name") == "process_labels"
        and str((e.get("args") or {}).get("labels", "")).startswith("GPU")
    }
    return [
        e
        for e in events
        if e.get("cat") == "ac2g"
        and e.get("ph") == "f"
        and e.get("pid") not in gpu_pids
    ]


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
    def test_ac2g_flows_end_on_device(self):
        # ac2g = Async CPU-to-GPU: every CPU->GPU flow arrow starts at a host
        # launch and must end at the device op, so no flow may both start and
        # end on the host (e.g. a redundant arrow from a host span to its own
        # nested host subspan, or an orphan host finish).
        a = torch.rand([10, 20], device="xpu")
        b = torch.rand([20, 30], device="xpu")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ],
        ) as p:
            for _ in range(3):
                r = torch.abs(torch.add(torch.matmul(a, b), 1.0))

        self.assertTrue(r.numel() > 0)

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
            p.export_chrome_trace(tmp.name)
            with open(tmp.name) as f:
                events = json.load(f)["traceEvents"]

        # Sanity: the workload must actually produce CPU->GPU flows, otherwise
        # the check below would pass vacuously.
        ac2g_finishes = [
            e for e in events if e.get("cat") == "ac2g" and e.get("ph") == "f"
        ]
        self.assertTrue(ac2g_finishes, "no ac2g CPU->GPU flows captured")

        off_device = find_ac2g_flow_finishes_off_device(events)
        self.assertEqual(
            len(off_device),
            0,
            "ac2g flow(s) finish off the device (should end on a GPU track): "
            f"{[(e.get('pid'), e.get('tid')) for e in off_device][:3]}",
        )


if __name__ == "__main__":
    run_tests()
