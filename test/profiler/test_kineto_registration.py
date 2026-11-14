# Owner(s): ["oncall: profiler"]

import json
import unittest

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import (
    DeviceType,
    kineto_available,
    profile,
    ProfilerActivity,
    supported_activities,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


# NOTE: Each test needs to be run in a brand new process. The first profiling
# session in a process registers Kineto, and nothing can undo that. run_test.py
# runs this file with --subprocess.

# The device runtime and driver events that a session collecting CPU activity
# asks for. A session without a device records them only if it armed device
# tracing.
RUNTIME_ACTIVITY_TYPES = {
    "cuda_runtime",
    "cuda_driver",
    "xpu_runtime",
    "xpu_driver",
    "privateuse1_runtime",
    "privateuse1_driver",
}


@unittest.skipIf(not kineto_available(), "Kineto is required")
class TestKinetoRegistration(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_device_tracing_decided_per_session(self, device):
        # Runs several profiling sessions, some with the device and some without.
        # Sessions without the device must see no device traces, and sessions
        # with the device must see them. That a session without the device never
        # arms device tracing at all is checked by libkineto's
        # SyncActivityProfilerHandlerTest.
        device_type = torch.device(device).type
        device_activity = getattr(ProfilerActivity, device_type.upper(), None)
        if device_activity not in supported_activities():
            self.skipTest(f"The profiler cannot trace {device_type}")
        x = torch.ones(64, 64, device=device)

        def run(activities, experimental_config=None):
            with profile(
                activities=activities, experimental_config=experimental_config
            ) as prof:
                x @ x
                torch.accelerator.synchronize()
            events = prof.profiler.kineto_results.events()
            device_events = sum(e.device_type() != DeviceType.CPU for e in events)
            runtime = sum(e.activity_type() in RUNTIME_ACTIVITY_TYPES for e in events)
            return device_events, runtime

        cpu = [ProfilerActivity.CPU]
        with_device = [ProfilerActivity.CPU, device_activity]
        self.assertEqual(run(cpu), (0, 0))
        device_events, _ = run(with_device)
        self.assertGreater(device_events, 0)
        self.assertEqual(run(cpu), (0, 0))
        # Kineto stops parsing a config at its first malformed line, which a
        # multi-line custom config contains.
        multi_line = _ExperimentalConfig(
            custom_profiler_config=json.dumps({"unused": 1}, indent=2)
        )
        self.assertEqual(run(cpu, multi_line), (0, 0))


instantiate_device_type_tests(
    TestKinetoRegistration, globals(), except_for="cpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
