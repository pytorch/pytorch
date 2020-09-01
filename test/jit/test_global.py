import os
import sys

import torch

from . import global_test_lib

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestGlobal(JitTestCase):
    # This test must not be split into different test cases because it tests
    # handling of global variable manipulations, which isn't parallel-safe.
    def test_global(self):
        # Test case where global variable is not captured.
        global_test_lib.reset()
        global_test_lib.set_global_var("foo")

        scripted = torch.jit.script(global_test_lib.use_global_var)
        with self.assertRaisesRegexWithHighlight(RuntimeError, "not found in CapturedGlobalValuesRegistry", "global_var"):
            scripted()

        # Test case where global variable is captured
        global_test_lib.reset()
        global_test_lib.set_global_var_and_capture("foo")
        scripted = torch.jit.script(global_test_lib.use_global_var)
        self.assertEqual(scripted(), "foo")

        # Test update previously captured global variable is supported
        global_test_lib.set_global_var_and_capture("bar")
        self.assertEqual(scripted(), "bar")
