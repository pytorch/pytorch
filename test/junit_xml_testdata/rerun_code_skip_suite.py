"""Known-failure code skips under rerun-disabled-tests, run by TestRerunCodeSkipCollection
in test/test_testing.py. Its disabled-tests JSON lists test_listed for all platforms and
test_code_skip_disabled_elsewhere for Windows only."""

import unittest

from torch.testing._internal.common_utils import (
    known_failure_skip_if,
    run_tests,
    TestCase,
)


class TestCodeSkipSuite(TestCase):
    @known_failure_skip_if(lambda: True, "known failure")
    def test_code_skip_passes(self):
        pass

    @known_failure_skip_if(lambda: True, "known failure")
    def test_code_skip_fails(self):
        self.fail("still failing")

    @known_failure_skip_if(lambda: True, "known failure")
    @unittest.skip("capability skip")
    def test_code_skip_capability_skip(self):
        pass

    @known_failure_skip_if(lambda: True, "known failure")
    def test_code_skip_disabled_elsewhere(self):
        pass

    @known_failure_skip_if(lambda: False, "known failure elsewhere")
    def test_known_failure_elsewhere(self):
        pass

    def test_listed(self):
        pass

    def test_not_listed(self):
        pass


if __name__ == "__main__":
    run_tests()
