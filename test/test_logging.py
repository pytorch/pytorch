# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class LoggingTest(TestCase):
    def testApiUsage(self):
        """
        This test verifies that api usage logging is not triggered via static
        initialization. Since it's triggered at first invocation only - we just
        subprocess
        """
        s = TestCase.runWithPytorchAPIUsageStderr("import torch")
        self.assertRegex(s, "PYTORCH_API_USAGE.*import")
        # import the shared library directly - it triggers static init but doesn't call anything
        s = TestCase.runWithPytorchAPIUsageStderr("from ctypes import CDLL; CDLL('{}')".format(torch._C.__file__))
        self.assertNotRegex(s, "PYTORCH_API_USAGE")


if __name__ == '__main__':
    run_tests()
