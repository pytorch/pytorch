# Owner(s): ["module: cuda"]

import sys
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    CudaSyncGuard,
    NoTest,
    run_tests,
    TestCase,
)


# NOTE: this needs to be run in a brand new process

if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
class Test(TestCase):
    def test_autograd_save_on_cpu_does_not_synchronize(self):
        a = torch.randn(5, requires_grad=True, device="cuda")
        b = torch.randn(5, requires_grad=True, device="cuda")
        c = torch.randn(5, requires_grad=True, device="cuda")

        def f(a, b, c):
            prod_1 = a * b
            prod_2 = prod_1 * c
            y = prod_2 * a
            return y

        with CudaSyncGuard("error"), torch.autograd.graph.save_on_cpu(pin_memory=True):
            y = f(a, b, c)
            y.sum().backward()


if __name__ == "__main__":
    run_tests()
