# Owner(s): ["module: PrivateUse1"]

import unittest
from collections import defaultdict

import torch
import torch.testing._internal.common_device_type as common_device_type
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    onlyOn,
    ops,
    precisionOverride,
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
# dummy_op4 starts with a low precision override (1e-5) declared in the OpInfo itself.
# op_decorators will later register a higher override (1e-2) for float32 to verify
# that the op_decorators entry takes precedence (because it is appended last to
# op.decorators and therefore applied last, overwriting precision_overrides).
dummy_op4 = OpInfo(
    "dummy_op4",
    op=lambda x: x,
    dtypes=common_device_type.get_all_dtypes(),
    decorators=[DecorateInfo(precisionOverride({torch.float32: 1e-5}))],
)


class TestDeviceTypeOpenReg(TestCase):
    # Track which test variants actually executed so we can assert on them in
    # tearDownClass and catch silent regressions (e.g. accidentally skipping
    # all op variants or executing too few/many).
    _executed: dict = defaultdict(int)

    @classmethod
    def tearDownClass(cls):
        # dummy_op1: all dtypes must run (no skips/xfails).
        if cls._executed["dummy_op1"] <= 0:
            raise AssertionError(
                "dummy_op1 tests should have run, but executed count is 0"
            )
        # dummy_op2: all variants are skipped, so the test body must never run.
        if cls._executed["dummy_op2"] != 0:
            raise AssertionError(
                "dummy_op2 test body should never run because all variants are op_skipped"
            )
        # dummy_op3: all variants are expectedFailure, so the test body must always
        # fail (and therefore never reach the counter increment).
        if cls._executed["dummy_op3"] != 0:
            raise AssertionError(
                "dummy_op3 test body should always fail (expectedFailure), never increment counter"
            )
        # dummy_op4: op_decorators precisionOverride must have been applied.
        if cls._executed["dummy_op4_precision_ok"] <= 0:
            raise AssertionError(
                "dummy_op4 float32 variant must have verified op_decorators precisionOverride wins"
            )
        super().tearDownClass()

    def test_normal(self, device):
        pass

    @ops([dummy_op1, dummy_op2, dummy_op3])
    def test_op(self, device, dtype, op):
        if op.name == "dummy_op2":
            self.fail("dummy_op2 should be skipped via op_skips")
        if op.name == "dummy_op3":
            # This should fail, but since we decorated it with expectedFailure, the test will pass!
            self.fail("dummy_op3 fails but is decorated with expectedFailure")
        type(self)._executed[op.name] += 1

    @ops([dummy_op4])
    def test_op_precision_override(self, device, dtype, op):
        """Verify that a precisionOverride registered via op_decorators takes
        precedence over the one declared inside the OpInfo.

        dummy_op4's OpInfo sets precisionOverride({torch.float32: 1e-5}).
        PrivateUse1TestBase.op_decorators additionally registers
        precisionOverride({torch.float32: 1e-2}) for dummy_op4.
        Because op_decorators entries are appended *after* the OpInfo's own
        decorators, the op_decorators value is the last to be applied and
        therefore wins.
        """
        if dtype == torch.float32:
            self.assertEqual(
                self.precision,
                1e-2,
                msg=(
                    f"Expected op_decorators precisionOverride (1e-2) to win over "
                    f"OpInfo precisionOverride (1e-5), but got {self.precision}"
                ),
            )
            type(self)._executed["dummy_op4_precision_ok"] += 1

    @ops([dummy_op1])
    def test_op_narrow_ops(self, device, dtype, op):
        """Verify that having extra entries in op_skips that are NOT present in
        the @ops list does not raise a KeyError (safety check in update_op_list).

        op_skips includes dummy_op2 and PrivateUse1TestBase.op_skips also
        lists dummy_op2; here @ops only exposes dummy_op1.  The safety check
        introduced in update_op_list must silently ignore the missing keys.
        """
        # If we reach here without a KeyError the safety check worked.


# Modify PrivateUse1TestBase which is automatically included for OpenReg
PrivateUse1TestBase.op_skips = {
    "dummy_op2": [DecorateInfo(unittest.skip("skip dummy_op2"))],
}
PrivateUse1TestBase.op_decorators = {
    "dummy_op3": [DecorateInfo(unittest.expectedFailure)],
    # This overrides the 1e-5 precision already declared on dummy_op4's OpInfo.
    "dummy_op4": [DecorateInfo(precisionOverride({torch.float32: 1e-2}))],
}

instantiate_device_type_tests(TestDeviceTypeOpenReg, globals(), only_for=("openreg",))
instantiate_device_type_tests(
    TestBypassDeviceRestrictions, globals(), only_for="openreg"
)

if __name__ == "__main__":
    run_tests()
