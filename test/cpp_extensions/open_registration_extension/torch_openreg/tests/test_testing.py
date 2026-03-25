# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    onlyOn,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestBypassDeviceRestrictions(TestCase):
    """Verify that PrivateUse1 backends can run tests decorated with @onlyCUDA.

    PrivateUse1TestBase sets bypass_device_restrictions = True, which causes
    @onlyOn-based decorators to be bypassed. This allows out-of-tree backends
    to run accelerator tests that are currently gated behind @onlyCUDA while
    the long-term migration to device-generic tests is in progress.
    """

    executed_count = 0

    @onlyCUDA
    def test_bypass_only_cuda(self, device):
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")

    @onlyOn(["cuda"])
    def test_bypass_only_on(self, device):
        type(self).executed_count += 1
        self.assertEqual(torch.device(device).type, "openreg")

    def test_vaildate_bypass_execution(self, device):
        expected_runs = 2
        actual_runs = type(self).executed_count
        self.assertEqual(
            actual_runs,
            expected_runs,
            f"Bypass logic failed! "
            f"Expected {expected_runs} tests to run, but only {actual_runs} executed.",
        )


instantiate_device_type_tests(
    TestBypassDeviceRestrictions, globals(), only_for="openreg"
)


if __name__ == "__main__":
    run_tests()
