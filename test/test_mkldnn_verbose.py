# Owner(s): ["module: mkldnn"]

from torch.testing._internal.common_utils import TestCase, run_tests
import os
import subprocess
import sys

class TestMKLDNNVerbose(TestCase):
    def test_verbose_on(self):
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkldnn_verbose.py --verbose-level=1', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.startswith("onednn_verbose"):
                    num = num + 1
                elif line == 'Failed to set MKLDNN into verbose mode. Please consider to disable this verbose scope.':
                    return
        self.assertTrue(num > 0, 'oneDNN verbose messages not found.')

    def test_verbose_off(self):
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkldnn_verbose.py --verbose-level=0', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.startswith("onednn_verbose"):
                    num = num + 1
        self.assertEqual(num, 0, 'unexpected oneDNN verbose messages found.')

if __name__ == '__main__':
    run_tests()
