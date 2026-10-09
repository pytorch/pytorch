"""A run_tests script whose test kills its own process
(TestReportCrashes.test_subprocess_crash)."""

import os
import signal

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestCrash(TestCase):
    @parametrize("x", [1])
    def test_crash(self, x):
        os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    run_tests()
