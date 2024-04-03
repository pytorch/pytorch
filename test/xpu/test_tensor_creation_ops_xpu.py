# Owner(s): ["module: intel"]

import sys

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_XPU, TestCase

from xpu_test_utils import copy_tests, XPUPatchForImport

with XPUPatchForImport():
    from test_tensor_creation_ops import TestTensorCreation as TestTensorCreationBase

if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

TEST_MULTIXPU = torch.xpu.device_count() > 1

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTensorCreationXPU(TestCase):
    pass


copy_tests(
    TestTensorCreationXPU,
    TestTensorCreationBase,
    applicable_list=["test_empty_strided"],
)
instantiate_device_type_tests(TestTensorCreationXPU, globals(), only_for="xpu")


if __name__ == "__main__":
    run_tests()
