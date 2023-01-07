# Owner(s): ["oncall: jit"]

import os
import sys

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# Tests for torch.jit.isinstance
class TestIValue(JitTestCase):
    def test_qscheme_ivalue(self):
        def qscheme(x: torch.Tensor):
            return x.qscheme()

        x = torch.rand(2, 2)
        self.checkScript(qscheme, (x,))
