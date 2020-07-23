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
    def setUp(self):
        self.saved_enum_env_var = os.environ.get("EXPERIMENTAL_ENUM_SUPPORT", None)
        os.environ["EXPERIMENTAL_ENUM_SUPPORT"] = "1"

    def tearDown(self):
        if self.saved_enum_env_var:
            os.environ["EXPERIMENTAL_ENUM_SUPPORT"] = self.saved_enum_env_var

    def test_enum_comp(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        def enum_comp(x: Color, y: Color) -> bool:
            return x == y

        # TODO(gmagogsfm): Re-anble hooks when serialization/deserialization
        # is supported.
        with torch._jit_internal._disable_emit_hooks():
            scripted_enum_comp = torch.jit.script(enum_comp)

        self.assertEqual(
            scripted_enum_comp(Color.RED, Color.RED),
            enum_comp(Color.RED, Color.RED))

        self.assertEqual(
            scripted_enum_comp(Color.RED, Color.GREEN),
            enum_comp(Color.RED, Color.GREEN))

    def test_heterogenous_value_type_enum_error(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = "green"

        def enum_comp(x: Color, y: Color) -> bool:
            return x == y

        # TODO(gmagogsfm): Re-anble hooks when serialization/deserialization
        # is supported.
        with self.assertRaisesRegex(RuntimeError, "Could not unify type list"):
            scripted_enum_comp = torch.jit.script(enum_comp)


# Tests that Enum support features are properly guarded before they are mature.
class TestEnumFeatureGuard(JitTestCase):
    def setUp(self):
        self.saved_enum_env_var = os.environ.get("EXPERIMENTAL_ENUM_SUPPORT", None)
        if self.saved_enum_env_var:
            del os.environ["EXPERIMENTAL_ENUM_SUPPORT"]

    def tearDown(self):
        if self.saved_enum_env_var:
            os.environ["EXPERIMENTAL_ENUM_SUPPORT"] = self.saved_enum_env_var

    def test_enum_comp_disabled(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        def enum_comp(x: Color, y: Color) -> bool:
            return x == y

        with self.assertRaisesRegex(NotImplementedError,
                                    "Enum support is work in progress"):
            torch.jit.script(enum_comp)
