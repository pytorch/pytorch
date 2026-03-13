# Owner(s): ["module: PrivateUse1"]

import unittest

import torch
import torch.testing._internal.common_device_type as common_device_type
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    onlyOn,
    ops,
    PrivateUse1TestBase,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.opinfo.core import DecorateInfo, OpInfo


class TestBypassDeviceRestrictions(TestCase):
    """Verify that PrivateUse1 backends can run tests decorated with @onlyCUDA.

    PrivateUse1TestBase sets bypass_device_restrictions = False by default.
    Backends that are ready to run @onlyOn-gated tests must explicitly opt in
    by setting bypass_device_restrictions = True in their setUp or subclass.
    This allows out-of-tree backends to run accelerator tests that are
    currently gated behind @onlyCUDA while the long-term migration to
    device-generic tests is in progress.
    """

    executed_count = 0

    def setUp(self):
        # Explicitly opt-in: instance attribute shadows the False default on
        # PrivateUse1TestBase so @onlyOn-based decorators allow this test to run.
        self.bypass_device_restrictions = True
        super().setUp()

    @onlyCUDA
    def test_bypass_only_cuda(self, device):
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")

    @onlyOn(["cuda"])
    def test_bypass_only_on(self, device):
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")

    def test_vaildate_bypass_execution(self, device):
        # Must run last. The 'v' prefix ensures this sorts after test_bypass_* ('b') alphabetically,
        # so executed_count has been incremented by both bypass tests before we check it here.
        expected_runs = 2
        actual_runs = type(self).executed_count
        self.assertEqual(
            actual_runs,
            expected_runs,
            f"Bypass logic failed! "
            f"Expected {expected_runs} tests to run, "
            f"but only {actual_runs} executed.",
        )


dummy_op1 = OpInfo(
    "dummy_op1", op=lambda x: x, dtypes=common_device_type.get_all_dtypes()
)
dummy_op2 = OpInfo(
    "dummy_op2", op=lambda x: x, dtypes=common_device_type.get_all_dtypes()
)
dummy_op3 = OpInfo(
    "dummy_op3", op=lambda x: x, dtypes=common_device_type.get_all_dtypes()
)


class TestDeviceTypeOpenReg(TestCase):
    def test_normal(self, device):
        pass

    @ops([dummy_op1, dummy_op2, dummy_op3])
    def test_op(self, device, dtype, op):
        if op.name == "dummy_op2":
            self.fail("dummy_op2 should be skipped via op_skips")
        if op.name == "dummy_op3":
            # This should fail, but since we decorated it with expectedFailure, the test will pass!
            self.fail("dummy_op3 fails but is decorated with expectedFailure")


# Modify PrivateUse1TestBase which is automatically included for OpenReg
PrivateUse1TestBase.op_skips = {
    "dummy_op2": [DecorateInfo(unittest.skip("skip dummy_op2"))],
}
PrivateUse1TestBase.op_decorators = {
    "dummy_op3": [DecorateInfo(unittest.expectedFailure)],
}

instantiate_device_type_tests(TestDeviceTypeOpenReg, globals(), only_for=("openreg",))
instantiate_device_type_tests(
    TestBypassDeviceRestrictions, globals(), only_for="openreg"
)

if __name__ == "__main__":
    run_tests()
