import unittest
import sys

from torch.testing._internal.common_utils import run_tests

if __name__ == '__main__':
    sys.argv.append("--jit_executor=simple")
    run_tests()
    import test_jit_py3
    suite = unittest.findTestCases(test_jit_py3)
    unittest.TextTestRunner().run(suite)
