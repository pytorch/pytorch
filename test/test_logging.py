import os
import sys
import subprocess
import torch
from common_utils import TestCase, run_tests


class LoggingTest(TestCase):
    @staticmethod
    def _runAndCaptureStderr(code):
        env = os.environ.copy()
        env["PYTORCH_API_USAGE_STDERR"] = "1"
        pipes = subprocess.Popen(
            [sys.executable, '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env)
        return pipes.communicate()[1].decode('ascii')

    def testApiUsage(self):
        """
        This test verifies that api usage logging is not triggered via static
        initialization. Since it's triggered at first invocation only - we just
        subprocess
        """
        s = self._runAndCaptureStderr("import torch")
        self.assertRegexpMatches(s, "PYTORCH_API_USAGE.*import")
        # import the shared library directly - it triggers static init but doesn't call anything
        s = self._runAndCaptureStderr("from ctypes import CDLL; CDLL('{}')".format(torch._C.__file__))
        self.assertNotRegexpMatches(s, "PYTORCH_API_USAGE")


if __name__ == '__main__':
    run_tests()
