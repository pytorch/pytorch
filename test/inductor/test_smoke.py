# Owner(s): ["module: inductor"]
import logging

import torch
import torch._logging
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_LINUX, HardwareClassification
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON, HAS_GPU


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

    def test_mlp(self, device):
        torch._logging.set_logs(
            dynamo=logging.DEBUG, inductor=logging.DEBUG, aot=logging.DEBUG
        )

        mlp = torch.compile(MLP().to(device))
        for _ in range(3):
            mlp(torch.randn(1, device=device))

        # set back to defaults
        torch._logging.set_logs()

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


from torch.testing._internal.common_device_type import instantiate_device_type_tests

instantiate_device_type_tests(SmokeTestDevice, globals(), except_for="cpu")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if IS_LINUX and HAS_GPU:
        if (not HAS_CUDA_AND_TRITON) or torch.cuda.get_device_properties(0).major <= 5:
            run_tests()
