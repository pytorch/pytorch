# Owner(s): ["module: ci"]
# Sanity check for CI setup in GHA.  This file is expected to fail so it can trigger reruns

import os

from torch.testing._internal.common_utils import periodic, run_tests, slowTest, TestCase


class TestCISanityCheck(TestCase):
    def test_env_vars_exist(self):
        # This check should fail and trigger reruns.  If it passes, something is wrong
        self.assertTrue(os.environ.get("CI") is None)

    @slowTest
    def test_env_vars_exist_slow(self):
        # Same as the above, but for the slow suite
        self.assertTrue(os.environ.get("CI") is None)

    @periodic
    def test_env_vars_exist_periodic(self):
        # Same as the above, but for the periodic suite (periodic-strict skips
        # every test not marked @periodic, so it needs its own failing canary)
        self.assertTrue(os.environ.get("CI") is None)


if __name__ == "__main__":
    run_tests()
