# Owner(s): ["module: PrivateUse1"]

import unittest
from collections import defaultdict
from contextlib import contextmanager

import torch
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

    @classmethod
    def tearDownClass(cls):
        expected_runs = 2
        actual_runs = cls.executed_count
        if actual_runs != expected_runs:
            raise AssertionError(
                f"Bypass logic failed! "
                f"Expected {expected_runs} tests to run, "
                f"but only {actual_runs} executed."
            )
        super().tearDownClass()

    @onlyCUDA
    def test_bypass_only_cuda(self, device):
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")

    @onlyOn(["cuda"])
    def test_bypass_only_on(self, device):
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")


def _make_dummy_op(name, **kwargs):
    return OpInfo(
        name,
        op=lambda x: x,
        dtypes={torch.float32, torch.float64},
        sample_inputs_func=lambda op, device, dtype, requires_grad, **kw: [],
        **kwargs,
    )


op_normal = _make_dummy_op("op_normal")
op_skip = _make_dummy_op("op_skip")
op_skip_f32 = _make_dummy_op("op_skip_f32")
op_xfail = _make_dummy_op("op_xfail")
# op_precision starts with a low precision override (1e-5) declared in the OpInfo itself.
# op_overrides will later register a higher override (1e-2) for float32 to verify
# that the op_overrides entry takes precedence (because it is appended last to
# op.decorators and therefore applied last, overwriting precision_overrides).
op_precision = _make_dummy_op(
    "op_precision",
    decorators=[DecorateInfo(precisionOverride({torch.float32: 1e-5}))],
)

# Dummy ops for testing combined op_allowlist + op_overrides
op_combined_supported = _make_dummy_op("op_combined_supported")
op_combined_skip = _make_dummy_op("op_combined_skip")
op_combined_unsupported = _make_dummy_op("op_combined_unsupported")


@contextmanager
def _temp_test_configs(obj, **configs):
    backup = {k: getattr(obj, k, None) for k in configs}
    obj.set_test_configs(**configs)
    try:
        yield
    finally:
        obj.set_test_configs(**backup)


class TestDeviceTypeOpenReg(TestCase):
    # Track which test variants actually executed so we can assert on them in
    # tearDownClass and catch silent regressions (e.g. accidentally skipping
    # all op variants or executing too few/many).
    _executed: dict = defaultdict(int)

    @classmethod
    def tearDownClass(cls):
        should_run = ("op_normal", "op_precision")
        should_not_run = ("op_skip", "op_xfail")
        for op_name in should_run:
            if op_name not in cls._executed or cls._executed[op_name] == 0:
                raise AssertionError(f"{op_name} tests should have run")
        for op_name in should_not_run:
            if op_name in cls._executed and cls._executed[op_name] != 0:
                raise AssertionError(f"{op_name} test body should never run")
        if cls._executed["op_skip_f32"] != 1:
            raise AssertionError(
                "op_skip_f32 should have run exactly once,"
                f"but ran {cls._executed['op_skip_f32']} times"
            )
        super().tearDownClass()

    @ops([op_normal, op_skip, op_skip_f32, op_xfail, op_precision])
    def test_op(self, device, dtype, op):
        if op.name in ("op_skip", "op_xfail"):
            self.fail(f"{op.name} deliberately fails")
        if op.name == "op_precision" and dtype == torch.float32:
            self.assertEqual(
                self.precision,
                1e-2,
                msg=(
                    f"Expected op_overrides precisionOverride (1e-2) to win over "
                    f"OpInfo precisionOverride (1e-5), but got {self.precision}"
                ),
            )
        type(self)._executed[op.name] += 1

    @ops([op_normal])
    def test_op_narrow_ops(self, device, dtype, op):
        """Verify that having extra entries in op_overrides that are NOT present in
        the @ops list does not raise a KeyError (safety check in _apply_op_overrides).

        op_overrides includes op_skip and PrivateUse1TestBase.op_overrides also
        lists op_skip; here @ops only exposes op_normal.  The safety check
        introduced in _apply_op_overrides must silently ignore the missing keys.
        """
        # If we reach here without a KeyError the safety check worked.


class TestSkippedSpecificTestCases(TestCase):
    executed_count = 0

    @classmethod
    def tearDownClass(cls):
        expected_runs = 1
        if cls.executed_count != expected_runs:
            raise AssertionError(
                f"Skip logic failed! Expected {expected_runs} tests to run, "
                f"but {cls.executed_count} executed."
            )
        super().tearDownClass()

    def test_runs(self, device):
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")

    def test_skipped(self, device):
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")
        self.fail("This test should not be instantiated for openreg")


class TestSkippedWholeTestClass(TestCase):
    def test_skipped_class_member(self, device):
        self.fail("This class should not be instantiated for openreg")


class TestSupportedOpsWithOverrides(TestCase):
    """Verify that op_allowlist filtering works together with op_overrides.

    op_allowlist filters which ops generate variants.
    op_overrides adds decorators to those ops that pass the filter.
    """

    _executed_combined: dict = defaultdict(int)

    @classmethod
    def tearDownClass(cls):
        # op_combined_supported should run (it's in op_allowlist, no skip decorator)
        if cls._executed_combined["op_combined_supported"] == 0:
            raise AssertionError("op_combined_supported should have run")
        # op_combined_skip should NOT run (in op_allowlist but has skip decorator)
        if cls._executed_combined["op_combined_skip"] != 0:
            raise AssertionError("op_combined_skip should be skipped by op_overrides")
        # op_combined_unsupported should NOT run (not in op_allowlist)
        if cls._executed_combined["op_combined_unsupported"] != 0:
            raise AssertionError(
                "op_combined_unsupported should not run (not in op_allowlist)"
            )
        super().tearDownClass()

    @ops([op_combined_supported, op_combined_skip, op_combined_unsupported])
    def test_combined_filter(self, device, dtype, op):
        type(self)._executed_combined[op.name] += 1


OPENREG_OP_OVERRIDES = {
    "op_skip": [DecorateInfo(unittest.skip("skip op_skip"))],
    "op_skip_f32": [
        DecorateInfo(unittest.skip("skip op_skip"), dtypes=(torch.float32,))
    ],
    "op_xfail": [DecorateInfo(unittest.expectedFailure)],
    # This overrides the 1e-5 precision already declared on op_precision's OpInfo.
    "op_precision": [DecorateInfo(precisionOverride({torch.float32: 1e-2}))],
}
with _temp_test_configs(PrivateUse1TestBase, op_overrides=OPENREG_OP_OVERRIDES):
    instantiate_device_type_tests(
        TestDeviceTypeOpenReg, globals(), only_for=("openreg",)
    )

instantiate_device_type_tests(
    TestBypassDeviceRestrictions, globals(), only_for="openreg"
)

OPENREG_TEST_EXCLUSIONS = {
    "TestSkippedSpecificTestCases": ["test_skipped"],
    "TestSkippedWholeTestClass": "*",
}
with _temp_test_configs(PrivateUse1TestBase, test_exclusions=OPENREG_TEST_EXCLUSIONS):
    instantiate_device_type_tests(
        TestSkippedSpecificTestCases, globals(), only_for="openreg"
    )
    instantiate_device_type_tests(
        TestSkippedWholeTestClass, globals(), only_for="openreg"
    )

with _temp_test_configs(
    PrivateUse1TestBase,
    op_overrides={
        "op_combined_skip": [DecorateInfo(unittest.skip("skip via op_overrides"))]
    },
    op_allowlist=("op_combined_supported", "op_combined_skip"),
):
    instantiate_device_type_tests(
        TestSupportedOpsWithOverrides, globals(), only_for=("openreg",)
    )

if __name__ == "__main__":
    run_tests()
