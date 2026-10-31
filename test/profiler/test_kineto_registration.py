# Owner(s): ["oncall: profiler"]

import unittest

import torch
from torch.profiler import DeviceType, kineto_available, profile, ProfilerActivity
from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_WINDOWS,
    run_tests,
    TEST_CUDA,
    TestCase,
)


# NOTE: Each test needs to be run in a brand new process. The first profiling
# session in a process registers Kineto, and nothing can undo that. run_test.py
# runs this file with --subprocess.


@unittest.skipIf(not TEST_CUDA, "CUDA is required")
@unittest.skipIf(not kineto_available(), "Kineto is required")
@unittest.skipIf(IS_WINDOWS, "CUDA profiler tests are skipped on Windows")
class TestKinetoRegistration(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_device_tracing_decided_per_session(self):
        # Each session decides for itself whether to trace the device, whatever
        # the session that registered Kineto asked for.
        x = torch.ones(64, 64, device="cuda")
        runtime_prefix = "cuda" if torch.version.cuda else "hip"

        def run(activities):
            with profile(activities=activities) as prof:
                x @ x
                torch.cuda.synchronize()
            events = prof.events()
            device = sum(e.device_type == DeviceType.CUDA for e in events)
            runtime = sum(e.name.startswith(runtime_prefix) for e in events)
            return device, runtime

        cpu = [ProfilerActivity.CPU]
        cuda = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        self.assertEqual(run(cpu), (0, 0))
        device, runtime = run(cuda)
        self.assertGreater(device, 0)
        self.assertGreater(runtime, 0)
        self.assertEqual(run(cpu), (0, 0))


if __name__ == "__main__":
    run_tests()
