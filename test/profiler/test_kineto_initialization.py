# Owner(s): ["oncall: profiler"]
import os
import subprocess
import sys
from unittest.mock import patch

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_profiler import skipIfProfilerUnsupported
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


class TestKinetoInitialization(TestCase):
    hw_classification = HardwareClassification.CUDA

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    def test_daemon_initialization(self, device):
        """
        This test checks whether kineto works with torch in daemon mode, please refer to issue #112389 and #131020.
        Besides that, this test will also check that kineto will not be initialized when user loads the shared library
        directly.
        """
        device_type = device.split(":")[0]
        script = f"""
import torch
if torch.{device_type}.is_available():
    device_module = torch.{device_type}
    if hasattr(device_module, 'init'):
        device_module.init()
"""
        try:
            subprocess.check_output(
                [sys.executable, "-W", "always", "-c", script],
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode != 0:
                self.assertTrue(
                    False,
                    "Kineto is not working properly with the Dynolog environment variable",
                )
        # import the shared library directly - it triggers static init but doesn't call kineto_init
        env = os.environ.copy()
        env["KINETO_USE_DAEMON"] = "1"
        if "KINETO_DAEMON_INIT_DELAY_S" in env:
            env.pop("KINETO_DAEMON_INIT_DELAY_S")
        _, stderr = TestCase.run_process_no_exception(
            f"from ctypes import CDLL; CDLL('{torch._C.__file__}')"
        )
        self.assertNotRegex(
            stderr.decode("ascii"),
            "Registering daemon config loader",
            "kineto should not be initialized when the shared library is imported directly",
        )


@skipIfProfilerUnsupported
class TestKinetoInitializationDevice(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    @patch.dict(os.environ, {"KINETO_DAEMON_INIT_DELAY_S": "1"})
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_daemon_initialization_with_profiling(self, device):
        script = (
            f"device = {device!r}\n"
            + """
import torch
import torch.nn as nn
from torch.profiler import supported_activities, profile
from torch.autograd.profiler import KinetoStepTracker

class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def payload(device="cpu"):
    x = torch.randn(10, 10).to(device)
    y = torch.randn(10, 10).to(device)
    z = torch.mm(x, y)
    z = z + y

niters = 8
net = SimpleNet()
opt = torch.optim.SGD(net.parameters(), lr=0.01)
opt.zero_grad()
inputs = torch.rand(10)

with profile(activities=supported_activities()):
    payload(device=device)

initial_step = KinetoStepTracker.current_step()

def run_batch():
    out = net(inputs)
    loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
    loss.backward()
    opt.step()

for _ in range(niters):
    run_batch()

with profile(
    activities=supported_activities(),
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
) as p:
    for _ in range(niters):
        run_batch()
        p.step()
if KinetoStepTracker.current_step() != initial_step + 2 * niters:
    raise AssertionError(f"Expected step {initial_step + 2 * niters}, got {KinetoStepTracker.current_step()}")
"""
        )
        try:
            subprocess.check_output(
                [sys.executable, "-W", "always", "-c", script],
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode != 0:
                self.assertTrue(
                    False,
                    "Kineto is not working properly with the Dynolog environment variable",
                )


instantiate_device_type_tests(TestKinetoInitializationDevice, globals(), allow_xpu=True)


instantiate_device_type_tests(
    TestKinetoInitialization,
    globals(),
    only_for=("cuda",),
)

if __name__ == "__main__":
    run_tests()
