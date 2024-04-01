# Owner(s): ["module: intel"]

import os
import sys

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_XPU, TestCase

# Add test folder to path
current_file_path = os.path.realpath(__file__)
test_package = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(test_package)

from xpu.xpu_test_utils import copy_tests, XPUPatch

with XPUPatch():
    from test_tensor_creation_ops import Namespace

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
    Namespace.TestTensorCreationWrapper,
    applicable_list=["test_empty_strided"],
)
instantiate_device_type_tests(TestTensorCreationXPU, globals(), only_for="xpu")


if __name__ == "__main__":
    run_tests()
