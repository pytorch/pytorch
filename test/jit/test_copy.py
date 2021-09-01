import os
import sys
import copy

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestCopy(JitTestCase):
    """
    Tests for copying behavior of TorchScript modules
    """
    def test_copy_parameter(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear_relu_stack = torch.nn.Linear(2, 4)

            def forward(self, x):
                return x

        m = torch.jit.script(Model())
        copied_m = copy.deepcopy(m)

        for (p, copied_p) in zip(m.parameters(), copied_m.parameters()):
            self.assertEqual(p.is_leaf, copied_p.is_leaf)
