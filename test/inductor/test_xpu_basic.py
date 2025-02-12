# Owner(s): ["module: inductor"]
import importlib
import os
import sys
import unittest

import torch
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_xpu_basic yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

importlib.import_module("filelock")

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model_gpu,
    TestCase,
)


# TODO: Remove this file.
# This is a temporary test case to test the base functionality of first Intel GPU Inductor integration.
# We are working on reuse and pass the test cases in test/inductor/*  step by step.
# Will remove this file when pass full test in test/inductor/*.


class XpuBasicTests(TestCase):
    common = check_model_gpu
    device = "xpu"

    def test_add(self):
        def fn(a, b):
            return a + b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_sub(self):
        def fn(a, b):
            return a - b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_mul(self):
        def fn(a, b):
            return a * b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_div(self):
        def fn(a, b):
            return a / b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_XPU

    if HAS_XPU:
        run_tests(needs="filelock")
