# Owner(s): ["module: unknown"]

from torch.testing._internal.common_utils import TestCase, run_tests
import os
import subprocess
import sys

class TestMKLVerbose(TestCase):
    def test_verbose_on(self):
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkl_verbose.py --verbose-level=1', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.startswith("MKL_VERBOSE"):
                    num = num + 1
                elif line == 'Failed to set MKL into verbose mode. Please consider to disable this verbose scope.':
                    return
        self.assertTrue(num > 0, 'oneMKL verbose messages not found.')

    def test_verbose_off(self):
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkl_verbose.py --verbose-level=0', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.startswith("MKL_VERBOSE"):
                    num = num + 1
        self.assertEqual(num, 0, 'unexpected oneMKL verbose messages found.')

if __name__ == '__main__':
    run_tests()
