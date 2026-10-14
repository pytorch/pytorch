# Owner(s): ["module: ci"]
"""DO NOT LAND. Crashes its test process on purpose so the torchci test run
report records crashed runs in CI (pytorch/test-infra#9103)."""

import ctypes
import os
import signal

from torch.testing._internal.common_utils import run_tests, TestCase


class TestCrashCanary(TestCase):
    # unittest orders these alphabetically: pass, SIGKILL, SIGSEGV, pass.
    def test_a_pass(self):
        pass

    def test_b_sigkill(self):
        os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))

    def test_c_sigsegv(self):
        ctypes.string_at(0)

    def test_d_pass(self):
        pass


if __name__ == "__main__":
    run_tests()
