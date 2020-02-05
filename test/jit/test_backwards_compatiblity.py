import os
import sys
from typing import List

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
# from torch.testing._internal.jit_utils import JitTestCase
import unittest

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestBackwardsCompatibility(unittest.TestCase):
    """
    Loads some old models and ensures that they can actually run
    """
    def test_bc(self):
        base_dir = os.join(os.path.dirname(os.path.realpath(__file__)), 'backwards_compatibility_models')
        os.listdir(base_dir)
