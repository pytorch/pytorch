# Owner(s): ["module: unknown"]

import os
import subprocess
import sys
import unittest

import torch
from torch.testing._internal.common_utils import TEST_ACL, TestCase, run_tests

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

    @unittest.skipIf(
        not TEST_ACL or not torch.ops.mkldnn._is_mkldnn_bf16_supported(),
        "ACL BF16 is not available",
    )
    def test_acl_bf16_linear_does_not_use_ref_inner_product(self):
        loc = os.path.dirname(os.path.abspath(__file__))
        inner_product_impls = []
        output = []
        with subprocess.Popen(
            f'{sys.executable} -u {loc}/mkldnn_verbose.py '
            '--verbose-level=1 --model=bf16-linear',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                output.append(line)
                if not line.startswith("onednn_verbose"):
                    continue
                fields = line.split(",")
                if "primitive" not in fields or "exec" not in fields:
                    continue
                try:
                    primitive_idx = fields.index("inner_product")
                    impl = fields[primitive_idx + 1]
                except (ValueError, IndexError):
                    continue
                inner_product_impls.append(impl)
            returncode = p.wait()
        self.assertEqual(returncode, 0, "\n".join(output))
        self.assertTrue(
            inner_product_impls, "BF16 inner_product verbose messages not found."
        )
        self.assertFalse(
            any(impl.startswith("ref") for impl in inner_product_impls),
            f"BF16 inner_product used oneDNN reference implementation(s): "
            f"{inner_product_impls}",
        )

if __name__ == '__main__':
    run_tests()
