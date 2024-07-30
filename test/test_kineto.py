# Owner(s): ["module: unknown"]
import os

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class LoggingTest(TestCase):
    def testKinetoUseDaemon(self):
        """
        This test verifies that libkineto_init is not triggered via static
        initialization. Since it's triggered after torch module initialized.
        """
        env = os.environ.copy()
        env["KINETO_USE_DAEMON"] = "1"
        if "KINETO_DAEMON_INIT_DELAY_S" in env:
            env.pop("KINETO_DAEMON_INIT_DELAY_S")

        script_to_run = """
import torch
if torch.cuda.is_available() > 0:
    torch.cuda.init()
"""
        _, stderr = TestCase.run_process_no_exception(script_to_run, env=env)
        self.assertRegex(stderr.decode('ascii'), f"cpuOnly =  {0 if torch.cuda.is_available() else 1}")
        self.assertRegex(stderr.decode('ascii'), "RuntimeError: Cannot initialize CUDA without ATen_cuda library")
        # import the shared library directly - it triggers static init but doesn't call kineto_init
        _, stderr = TestCase.run_process_no_exception(f"from ctypes import CDLL; CDLL('{torch._C.__file__}')", env=env)
        self.assertNotRegex(stderr.decode('ascii'), "Registering daemon config loader")


if __name__ == '__main__':
    run_tests()