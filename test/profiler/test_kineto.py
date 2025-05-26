# Owner(s): ["oncall: profiler"]
import os
import subprocess
import sys
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class SimpleKinetoInitializationTest(TestCase):
    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    def test_kineto_profiler_with_environment_variable(self):
        """
        This test checks whether kineto works with torch in daemon mode, please refer to issue #112389 and #131020.
        Besides that, this test will also check that kineto will not be initialized when user loads the shared library
        directly.
        """
        script = """
import torch
if torch.cuda.is_available() > 0:
    torch.cuda.init()
"""
        try:
            subprocess.check_output(
                [sys.executable, "-W", "always", "-c", script],
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode != 0:
                self.assertTrue(
                    False,
                    "Kineto is not working properly with the Dynolog environment variable",
                )
        # import the shared library directly - it triggers static init but doesn't call kineto_init
        env = os.environ.copy()
        env["KINETO_USE_DAEMON"] = "1"
        if "KINETO_DAEMON_INIT_DELAY_S" in env:
            env.pop("KINETO_DAEMON_INIT_DELAY_S")
        _, stderr = TestCase.run_process_no_exception(
            f"from ctypes import CDLL; CDLL('{torch._C.__file__}')"
        )
        self.assertNotRegex(
            stderr.decode("ascii"),
            "Registering daemon config loader",
            "kineto should not be initialized when the shared library is imported directly",
        )


if __name__ == "__main__":
    run_tests()
