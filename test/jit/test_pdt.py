import os
import sys
import torch
from torch.testing._internal.jit_utils import JitTestCase

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

def test_sum(a, b):
    return a + b

def test_sub(a, b):
    return a - b

def test_mul(a, b):
    return a * b

def test_args_complex(real, img):
    return torch.complex(real, img)

class TestPDT(JitTestCase):
    """
    A suite of tests for profile directed typing in TorchScript.
    """
    def setUp(self):
        super(TestPDT, self).setUp()

    def tearDown(self):
        super(TestPDT, self).tearDown()

    def test_pdt(self):
        scripted_fn_add = torch.jit._script_pdt(test_sum, example_inputs=[(3, 4)])
        scripted_fn_sub = torch.jit._script_pdt(test_sub, example_inputs=[(3.9, 4.10)])
        scripted_fn_mul = torch.jit._script_pdt(test_mul, example_inputs=[(-10, 9)])
        scripted_fn_complex = torch.jit._script_pdt(test_args_complex, example_inputs=[(torch.rand(3, 4), torch.rand(3, 4))])
        self.assertEqual(scripted_fn_add(10, 2), test_sum(10, 2))
        self.assertEqual(scripted_fn_sub(6.5, 2.9), test_sub(6.5, 2.9))
        self.assertEqual(scripted_fn_mul(-1, 3), test_mul(-1, 3))
        arg1, arg2 = torch.rand(3, 4), torch.rand(3, 4)
        self.assertEqual(scripted_fn_complex(arg1, arg2), test_args_complex(arg1, arg2))
