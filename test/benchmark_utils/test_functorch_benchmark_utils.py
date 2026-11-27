# Owner(s): ["module: unknown"]

import contextlib
import os
import tempfile
import types
from unittest.mock import patch

import torch
from torch._functorch import benchmark_utils
from torch.profiler import ProfilerActivity
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


def _fn(inp):
    return inp[0].sum()


class TestFunctorchBenchmarkUtils(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _dump(self, devices, sync_calls, path):
        def fake_sync():
            sync_calls.append(1)

        x = torch.randn(8, 8)
        with (
            patch("torch.accelerator.is_available", return_value=True),
            patch("torch.accelerator.synchronize", fake_sync),
        ):
            benchmark_utils.dump_chrome_trace(
                _fn,
                (x,),
                path,
                contextlib.nullcontext(),
                [ProfilerActivity.CPU],
                devices=devices,
            )

    def test_accelerator_sync_used_for_device_list(self):
        calls: list[int] = []
        with tempfile.TemporaryDirectory() as d:
            self._dump(["cuda"], calls, os.path.join(d, "t.json"))
        self.assertGreater(len(calls), 0)

    def test_cpu_with_index_stays_unsynced(self):
        # "cpu:0" must normalize to cpu even when an accelerator is present
        calls: list[int] = []
        with tempfile.TemporaryDirectory() as d:
            self._dump(["cpu:0"], calls, os.path.join(d, "t.json"))
        self.assertEqual(calls, [])

    def test_module_synchronize_is_not_rebound(self):
        original = benchmark_utils.synchronize
        calls: list[int] = []
        with tempfile.TemporaryDirectory() as d:
            self._dump(["cuda"], calls, os.path.join(d, "t.json"))
        self.assertIs(benchmark_utils.synchronize, original)

    def test_default_trace_settings_without_accelerator(self):
        with patch("torch.accelerator.is_available", return_value=False):
            activities, devices = benchmark_utils._default_trace_settings()
        self.assertEqual(activities, [ProfilerActivity.CUDA])
        self.assertEqual(devices, ["cuda"])

    def test_default_trace_settings_renamed_backend(self):
        with (
            patch("torch.accelerator.is_available", return_value=True),
            patch(
                "torch.accelerator.current_accelerator",
                return_value=types.SimpleNamespace(type="foo"),
            ),
        ):
            activities, devices = benchmark_utils._default_trace_settings()
        self.assertEqual(activities, [ProfilerActivity.PrivateUse1])
        self.assertEqual(devices, ["foo"])


if __name__ == "__main__":
    run_tests()
