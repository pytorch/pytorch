# Owner(s): ["module: intel"]

import os
import sys
import unittest

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyXPU,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import ops_and_refs
from torch.testing._internal.common_utils import (
    NoTest,
    run_tests,
    suppress_warnings,
    TEST_WITH_UBSAN,
    TEST_XPU,
    TestCase,
)

# Add test folder to path
current_file_path = os.path.realpath(__file__)
test_package = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(test_package)

from xpu.xpu_test_utils import get_wrapped_fn, XPUPatch

with XPUPatch():
    from test_ops import Namespace

if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

TEST_MULTIXPU = torch.xpu.device_count() > 1

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

any_common_cpu_xpu_one = OpDTypes.any_common_cpu_cuda_one
_xpu_computation_op_list = [
    "fill",
    "zeros",
    "zeros_like",
    "clone",
    "view_as_real",
    "view_as_complex",
    "view",
    "resize_",
    "resize_as_",
    "add",
    "sub",
    "mul",
    "div",
    "abs",
]

_xpu_computation_ops = [
    op for op in ops_and_refs if op.name in _xpu_computation_op_list
]


class TestCommonXPU(TestCase):
    @onlyXPU
    @suppress_warnings
    @ops(_xpu_computation_ops, dtypes=any_common_cpu_xpu_one)
    def test_compare_cpu(self, device, dtype, op):
        self.proxy = Namespace.TestCommonWrapper()

        test_common_test_fn = get_wrapped_fn(
            Namespace.TestCommonWrapper.test_compare_cpu
        )
        test_common_test_fn(self.proxy, device, dtype, op)

    @onlyXPU
    @ops(_xpu_computation_ops, allowed_dtypes=(torch.bool,))
    @unittest.skipIf(TEST_WITH_UBSAN, "Test uses undefined behavior")
    def test_non_standard_bool_values(self, device, dtype, op):
        self.proxy = Namespace.TestCommonWrapper()

        test_common_test_fn = get_wrapped_fn(
            Namespace.TestCommonWrapper.test_non_standard_bool_values
        )
        test_common_test_fn(self.proxy, device, dtype, op)


instantiate_device_type_tests(TestCommonXPU, globals(), only_for="xpu")


if __name__ == "__main__":
    run_tests()
