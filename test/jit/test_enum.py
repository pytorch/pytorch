import os
import sys

import torch
from enum import Enum
from typing import Any, List

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

    def test_enum_value_types(self):
        global IntEnum

        class IntEnum(Enum):
            FOO = 1
            BAR = 2

        global FloatEnum

        class FloatEnum(Enum):
            FOO = 1.2
            BAR = 2.3

        global StringEnum

        class StringEnum(Enum):
            FOO = "foo as in foo bar"
            BAR = "bar as in foo bar"

        def supported_enum_types(a: IntEnum, b: FloatEnum, c: StringEnum):
            return (a.name, b.name, c.name)
        # TODO(gmagogsfm): Re-enable hooks when serialization/deserialization
        # is supported.
        with torch._jit_internal._disable_emit_hooks():
            torch.jit.script(supported_enum_types)

        global TensorEnum

        class TensorEnum(Enum):
            FOO = torch.tensor(0)
            BAR = torch.tensor(1)

        def unsupported_enum_types(a: TensorEnum):
            return a.name

        with self.assertRaisesRegex(RuntimeError, "Cannot create Enum with value type 'Tensor'"):
            torch.jit.script(unsupported_enum_types)

    def test_enum_comp(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        def enum_comp(x: Color, y: Color) -> bool:
            return x == y

        # TODO(gmagogsfm): Re-enable hooks when serialization/deserialization
        # is supported.
        with torch._jit_internal._disable_emit_hooks():
            scripted_enum_comp = torch.jit.script(enum_comp)

        self.assertEqual(
            scripted_enum_comp(Color.RED, Color.RED),
            enum_comp(Color.RED, Color.RED))

        self.assertEqual(
            scripted_enum_comp(Color.RED, Color.GREEN),
            enum_comp(Color.RED, Color.GREEN))

    def test_enum_comp_diff_classes(self):
        global Foo, Bar

        class Foo(Enum):
            ITEM1 = 1
            ITEM2 = 2

        class Bar(Enum):
            ITEM1 = 1
            ITEM2 = 2

        def enum_comp(x: Foo) -> bool:
            return x == Bar.ITEM1

        # TODO(gmagogsfm): Re-enable hooks when serialization/deserialization
        # is supported.
        with torch._jit_internal._disable_emit_hooks():
            scripted_enum_comp = torch.jit.script(enum_comp)

        self.assertEqual(
            scripted_enum_comp(Foo.ITEM1),
            False)


    def test_heterogenous_value_type_enum_error(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = "green"

        def enum_comp(x: Color, y: Color) -> bool:
            return x == y

        # TODO(gmagogsfm): Re-enable hooks when serialization/deserialization
        # is supported.
        with self.assertRaisesRegex(RuntimeError, "Could not unify type list"):
            scripted_enum_comp = torch.jit.script(enum_comp)

    def test_enum_name(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        def enum_name(x: Color) -> str:
            return x.name

        # TODO(gmagogsfm): Re-enable hooks when serialization/deserialization
        # is supported.
        with torch._jit_internal._disable_emit_hooks():
            scripted_enum_name = torch.jit.script(enum_name)

        self.assertEqual(scripted_enum_name(Color.RED), Color.RED.name)
        self.assertEqual(scripted_enum_name(Color.GREEN), Color.GREEN.name)

    def test_enum_value(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        def enum_value(x: Color) -> int:
            return x.value

        # TODO(gmagogsfm): Re-enable hooks when serialization/deserialization
        # is supported.
        with torch._jit_internal._disable_emit_hooks():
            scripted_enum_value = torch.jit.script(enum_value)

        self.assertEqual(scripted_enum_value(Color.RED), Color.RED.value)
        self.assertEqual(scripted_enum_value(Color.GREEN), Color.GREEN.value)

    def test_enum_as_const(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        @torch.jit.script
        def enum_const(x: Color) -> bool:
            if x == Color.RED:
                return True
            else:
                return False

        self.assertEqual(enum_const(Color.RED), True)
        self.assertEqual(enum_const(Color.GREEN), False)

    def test_non_existent_enum_value(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        def enum_const(x: Color) -> bool:
            if x == Color.PURPLE:
                return True
            else:
                return False

        with self.assertRaisesRegexWithHighlight(RuntimeError, "has no attribute 'PURPLE'", "Color.PURPLE"):
            torch.jit.script(enum_const)

    def test_enum_ivalue_type(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        def is_color_enum(x: Any):
            if isinstance(x, Color):
                return True
            else:
                return False

        with torch._jit_internal._disable_emit_hooks():
            scripted_is_color_enum = torch.jit.script(is_color_enum)

        self.assertEqual(scripted_is_color_enum(Color.RED), True)
        self.assertEqual(scripted_is_color_enum(Color.GREEN), True)
        self.assertEqual(scripted_is_color_enum(1), False)

    def test_closed_over_enum_constant(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        a = Color

        def closed_over_aliased_type():
            return a.RED.value

        with torch._jit_internal._disable_emit_hooks():
            scripted = torch.jit.script(closed_over_aliased_type)

        self.assertEqual(scripted(), Color.RED.value)


        b = Color.RED

        def closed_over_aliased_value():
            return b.value

        with torch._jit_internal._disable_emit_hooks():
            scripted = torch.jit.script(closed_over_aliased_value)

        self.assertEqual(scripted(), Color.RED.value)

    def test_enum_as_module_attribute(self):
        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        class TestModule(torch.nn.Module):
            def __init__(self, e: Color):
                super(TestModule, self).__init__()
                self.e = e

            def forward(self):
                return self.e.value

        m = TestModule(Color.RED)
        with torch._jit_internal._disable_emit_hooks():
            scripted = torch.jit.script(m)

        self.assertEqual(scripted(), Color.RED.value)

    def test_enum_iterate(self):
        global ColorForIterate

        class ColorForIterate(Enum):
            RED = 1
            GREEN = 2
            PURPLE = 3

        def iterate_enum(x: ColorForIterate):
            res: List[int] = []
            for e in ColorForIterate:
                if e != x:
                    res.append(e.value)
            return res

        with torch._jit_internal._disable_emit_hooks():
            scripted = torch.jit.script(iterate_enum)

        # PURPLE always appear last because we follow Python's Enum definition order.
        self.assertEqual(scripted(ColorForIterate.RED), [ColorForIterate.GREEN.value, ColorForIterate.PURPLE.value])
        self.assertEqual(scripted(ColorForIterate.GREEN), [ColorForIterate.RED.value, ColorForIterate.PURPLE.value])

        # TODO(gmagogsfm): Add FileCheck test after serialization and ir representation is completed.


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

        with self.assertRaisesRegexWithHighlight(RuntimeError, "Unknown type name 'Color'", "Color"):
            torch.jit.script(enum_comp)
