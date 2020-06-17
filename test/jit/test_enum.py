import os
import sys

import torch
from enum import Enum

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestEnum(JitTestCase):
    
    def test_enum_comp(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        def enum_comp(x: Color, y: Color) -> bool:
            return x == y

        # TODO(gmagogsfm): Re-anble hooks when serialization/deserialization
        # is supported.
        with torch.jit._disable_emit_hooks():
            scripted_enum_comp = torch.jit.script(enum_comp)

        self.assertEqual(
            scripted_enum_comp(Color.RED, Color.RED),
            enum_comp(Color.RED, Color.RED))

        self.assertEqual(
            scripted_enum_comp(Color.RED, Color.GREEN),
            enum_comp(Color.RED, Color.GREEN))