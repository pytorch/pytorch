# Owner(s): ["oncall: profiler"]


import json
import os
import tempfile
from collections import defaultdict

import torch
from torch.profiler import DeviceType, ProfilerActivity
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    ALLOW_XPU_PROFILING_TEST,
    DEVICE_LIST_SUPPORT_PROFILING_TEST,
    run_tests,
    TestCase,
)


Verbose = False


class AcceleratorProfilerTest(TestCase):
    def test_profiler(self, device):
        device_type = device.split(":")[0]
        t = torch.empty(1000, dtype=torch.int, device=device)

        # Dynamically get the ProfilerActivity for this device
        profiler_activity = getattr(ProfilerActivity, device_type.upper())

        with torch.profiler.profile(
            activities=[
                profiler_activity,
            ]
        ) as p:
            for _ in range(10):
                t.zero_()

        events = defaultdict(int)
        for event in p.events():
            events[event.device_type] += 1

        if Verbose:
            print(f"{events=}")

        # CPU profiling only shows CPU events, accelerators show CPU + device events
        if device_type == "cpu":
            self.assertEqual(len(events), 1)
            self.assertTrue(DeviceType.CPU in events)
        else:
            self.assertEqual(len(events), 2)
            self.assertTrue(DeviceType.CPU in events)
            # Dynamically check for the correct device type
            expected_device_type = getattr(DeviceType, device_type.upper())
            self.assertTrue(expected_device_type in events)

    def gen_and_check_json(self, p, json_file, device_type):
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

            # Check for device-specific categories (only for accelerators, not CPU)
            if device_type != "cpu":
                # Check that at least runtime category exists for accelerators
                self.assertTrue(f"{device_type}_runtime" in count_cats)
                # Kernel category is GPU-specific
                self.assertTrue("kernel" in count_cats)

    def test_profiler_driver(self, device):
        device_type = device.split(":")[0]
        a = torch.rand([100, 200]).to(device)
        b = torch.rand([200, 300]).to(device)

        # Dynamically get the ProfilerActivity for this device
        profiler_activity = getattr(ProfilerActivity, device_type.upper())

        # Build activities list - avoid duplicating CPU
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device_type != "cpu":
            activities.append(profiler_activity)

        with torch.profiler.profile(
            activities=activities,
        ) as p:
            r1 = torch.matmul(a, b)
            r2 = torch.add(r1, 1.0)
            result = torch.abs(r2)

        self.assertTrue(result.numel() > 0)

        json_file = os.environ.get("JSON_FILE")

        if json_file:
            self.gen_and_check_json(p, json_file, device_type)
        else:
            with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
                self.gen_and_check_json(p, tmp.name, device_type)

        if Verbose:
            print(p.key_averages().table())


instantiate_device_type_tests(
    AcceleratorProfilerTest,
    globals(),
    only_for=DEVICE_LIST_SUPPORT_PROFILING_TEST,
    allow_xpu=ALLOW_XPU_PROFILING_TEST,
)

if __name__ == "__main__":
    run_tests()
