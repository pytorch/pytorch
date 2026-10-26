"""Known-failure code skips under rerun-disabled-tests, run by TestRerunCodeSkipCollection
in test/test_testing.py. Its disabled-tests JSON lists test_listed for all platforms and
test_code_skip_disabled_elsewhere for Windows only."""

import unittest

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    known_failure_skip_if,
    run_tests,
    TestCase,
)
from torch.testing._internal.rerun_code_skip import (
    mark_rerun_code_skip,
    rerun_code_skip,
)


def known_failure_on(device_type):
    return lambda fn: mark_rerun_code_skip(fn, f"{device_type} failure", device_type)


class TestCodeSkipSuite(TestCase):
    @rerun_code_skip("known failure")
    def test_code_skip_passes(self):
        pass

    @rerun_code_skip("known failure")
    def test_code_skip_fails(self):
        self.fail("still failing")

    @rerun_code_skip("known failure")
    @unittest.skip("capability skip")
    def test_code_skip_capability_skip(self):
        pass

    @rerun_code_skip("known failure")
    def test_code_skip_disabled_elsewhere(self):
        pass

    @known_failure_skip_if(lambda: True, "known failure here")
    def test_known_failure_here(self):
        pass

    @known_failure_skip_if(lambda: False, "known failure elsewhere")
    def test_known_failure_elsewhere(self):
        pass

    def test_listed(self):
        pass

    def test_not_listed(self):
        pass


class TestCodeSkipSuiteDevice(TestCase):
    @known_failure_on("cpu")
    def test_cpu_known_failure(self, device):
        pass

    @known_failure_on("cuda")
    def test_cuda_known_failure(self, device):
        pass


instantiate_device_type_tests(TestCodeSkipSuiteDevice, globals(), only_for="cpu")

if __name__ == "__main__":
    run_tests()
