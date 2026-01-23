# Owner(s): ["oncall: profiler"]

import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.testing._internal.common_utils import run_tests, TEST_XPU, TestCase


Verbose = False


class XpuScopeProfilerTest(TestCase):
    def __init__(self, arg):
        super().__init__(arg)

        self.metrics_names = [
            "GpuTime",
            "GpuCoreClocks",
            "AvgGpuCoreFrequencyMHz",
            "XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION",
            "XVE_ACTIVE",
            "XVE_STALL",
        ]

    def count_metrics(self, event, counter):
        self.assertTrue("args" in event)
        self.assertTrue(isinstance(event["args"], dict))

        for arg in event["args"]:
            counter[arg] += 1

    def check_metrics(self, counter):
        rev_counter = defaultdict(int)

        for metric_name_in_json in counter:
            metric_name_in_json_valid = sum(
                metric_name_in_json == metric_name
                or metric_name_in_json.startswith(metric_name + " [")
                for metric_name in self.metrics_names
            )
            self.assertTrue(metric_name_in_json_valid <= 1)
            if metric_name_in_json_valid == 1:
                rev_counter[counter[metric_name_in_json]] += 1

        if Verbose:
            print(f"{rev_counter = }")

        # keys are amounts of certain metrics
        # only one dict element means all metrics have the same amount
        self.assertEqual(len(rev_counter), 1)
        first_value = next(iter(rev_counter.values()))
        self.assertEqual(first_value, len(self.metrics_names))

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
            count_x_metrics = defaultdict(int)
            count_c_metrics = defaultdict(int)
            for event in trace_events:
                self.assertTrue("ph" in event)
                self.assertTrue("name" in event)

                if event["ph"] == "X":
                    self.assertTrue("cat" in event)
                    count_names[event["name"]] += 1
                    count_cats[event["cat"]] += 1

                    if event["name"].startswith("metrics: "):
                        self.count_metrics(event, count_x_metrics)

                elif event["ph"] == "C":
                    self.count_metrics(event, count_c_metrics)

            if Verbose:
                print(f"{count_names = }")
                print(f"{count_cats = }")
                print(f"{count_x_metrics = }")
                print(f"{count_c_metrics = }")

            metric_name_in_json_valid = sum(
                name.startswith("metrics: ") for name in count_names
            )
            self.assertTrue(metric_name_in_json_valid > 0)

            self.assertTrue("xpu_runtime" in count_cats)
            self.assertTrue("kernel" in count_cats)

            self.check_metrics(count_x_metrics)
            self.check_metrics(count_c_metrics)

    def required_env_setting(self):
        zet_enable_metrics = os.environ.get("ZET_ENABLE_METRICS", "unset")
        if Verbose:
            print(f"ZET_ENABLE_METRICS={zet_enable_metrics}")
        return zet_enable_metrics == "1"

    def required_paranoid_setting(self, paranoid_path):
        try:
            paranoid_content = Path(paranoid_path).read_text().strip()
            if Verbose:
                print(f"{paranoid_path} contains: {paranoid_content}")
            return paranoid_content == "0"

        except FileNotFoundError:
            if Verbose:
                print(f"{paranoid_path} doesn't exists")
            return False

    def required_paranoid_settings(self):
        return any(
            self.required_paranoid_setting(paranoid)
            for paranoid in [
                "/proc/sys/dev/xe/observation_paranoid",
                "/proc/sys/dev/i915/perf_stream_paranoid",
            ]
        )

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_scope_profiler(self):
        if not self.required_env_setting():
            self.skipTest("ZET_ENABLE_METRICS not set")

        if not self.required_paranoid_settings():
            self.skipTest("paranoid settings missing")

        a = torch.rand([100, 200]).to("xpu")
        b = torch.rand([200, 300]).to("xpu")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ],
            experimental_config=_ExperimentalConfig(
                profiler_metrics=self.metrics_names,
                profiler_measure_per_kernel=True,
            ),
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

        count_metrics = sum(
            event.key.startswith("metrics: ") for event in p.key_averages()
        )
        self.assertTrue(count_metrics > 0)


if __name__ == "__main__":
    run_tests()
