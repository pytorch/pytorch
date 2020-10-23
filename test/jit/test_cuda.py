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


class TestCUDA(JitTestCase):
    """
    A suite of tests for the CUDA API in TorchScript.
    """
    def test_simple(self):
        @torch.jit.script
        def fn():
            s = torch.classes.cuda.Stream(0, 0)
            e = torch.classes.cuda.Event(False, False, False)
