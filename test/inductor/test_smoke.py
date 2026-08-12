# Owner(s): ["module: inductor"]
import logging

import torch
import torch._logging
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_device_type import (
    Capability,
    instantiate_device_type_tests,
    requires_capabilities,
)
from torch.testing._internal.common_utils import HardwareClassification


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(1, 6)
        self.l2 = torch.nn.Linear(6, 1)

    def forward(self, x=None):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


def _test_f(x):
    return x * x


class SmokeTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_compile_invalid_options(self):
        with self.assertRaises(RuntimeError):
            torch.compile(_test_f, mode="ha")


class SmokeTestDevice(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @requires_capabilities(Capability.lib.triton)
    def test_mlp(self, device):
        torch._logging.set_logs(
            dynamo=logging.DEBUG, inductor=logging.DEBUG, aot=logging.DEBUG
        )

        mlp = torch.compile(MLP().to(device))
        for _ in range(3):
            mlp(torch.randn(1, device=device))

        # set back to defaults
        torch._logging.set_logs()

    @requires_capabilities(Capability.lib.triton)
    def test_compile_decorator(self, device):
        @torch.compile
        def foo(x):
            return torch.sin(x) + x.min()

        @torch.compile(mode="reduce-overhead")
        def bar(x):
            return x * x

        for _ in range(3):
            foo(torch.full((3, 4), 0.7, device=device))
            bar(torch.rand((2, 2), device=device))


instantiate_device_type_tests(
    SmokeTestDevice, globals(), except_for="cpu", allow_xpu=True
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
