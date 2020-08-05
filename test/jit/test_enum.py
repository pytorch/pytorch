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
        global ColorDiffType

        class ColorDiffType(Enum):
            RED = 1
            GREEN = "green"

        def enum_comp(x: ColorDiffType, y: ColorDiffType) -> bool:
            return x == y

        # TODO(gmagogsfm): Re-enable hooks when serialization/deserialization
        # is supported.
        with self.assertRaisesRegex(RuntimeError, "Could not unify type list"):
            scripted_enum_comp = torch.jit.script(enum_comp)

    def test_enum_name(self):
        global ColorForName

        class ColorForName(Enum):
            RED = 1
            GREEN = 2

        def enum_name(x: ColorForName) -> str:
            return x.name

        # TODO(gmagogsfm): Re-enable hooks when serialization/deserialization
        # is supported.
        with torch._jit_internal._disable_emit_hooks():
            scripted_enum_name = torch.jit.script(enum_name)

        self.assertEqual(scripted_enum_name(ColorForName.RED), ColorForName.RED.name)
        self.assertEqual(scripted_enum_name(ColorForName.GREEN), ColorForName.GREEN.name)

    def test_enum_value(self):
        global ColorForValue

        class ColorForValue(Enum):
            RED = 1
            GREEN = 2

        def enum_value(x: ColorForValue) -> int:
            return x.value

        # TODO(gmagogsfm): Re-enable hooks when serialization/deserialization
        # is supported.
        with torch._jit_internal._disable_emit_hooks():
            scripted_enum_value = torch.jit.script(enum_value)

        self.assertEqual(scripted_enum_value(ColorForValue.RED), ColorForValue.RED.value)
        self.assertEqual(scripted_enum_value(ColorForValue.GREEN), ColorForValue.GREEN.value)

    def test_enum_as_const(self):
        global ColorConst

        class ColorConst(Enum):
            RED = 1
            GREEN = 2

        @torch.jit.script
        def enum_const(x: ColorConst) -> bool:
            if x == ColorConst.RED:
                return True
            else:
                return False

        self.assertEqual(enum_const(ColorConst.RED), True)
        self.assertEqual(enum_const(ColorConst.GREEN), False)

    def test_non_existent_enum_value(self):
        global ColorNonExistentVal

        class ColorNonExistentVal(Enum):
            RED = 1
            GREEN = 2

        def enum_const(x: ColorNonExistentVal) -> bool:
            if x == ColorNonExistentVal.PURPLE:
                return True
            else:
                return False

        with self.assertRaisesRegexWithHighlight(RuntimeError, "has no attribute 'PURPLE'", "ColorNonExistentVal.PURPLE"):
            torch.jit.script(enum_const)

    def test_enum_ivalue_type(self):
        global ColorIValueType

        class ColorIValueType(Enum):
            RED = 1
            GREEN = 2

        def is_color_enum(x: Any):
            if isinstance(x, ColorIValueType):
                return True
            else:
                return False

        with torch._jit_internal._disable_emit_hooks():
            scripted_is_color_enum = torch.jit.script(is_color_enum)

        self.assertEqual(scripted_is_color_enum(ColorIValueType.RED), True)
        self.assertEqual(scripted_is_color_enum(ColorIValueType.GREEN), True)
        self.assertEqual(scripted_is_color_enum(1), False)

    def test_closed_over_enum_constant(self):
        global ColorClosedOver

        class ColorClosedOver(Enum):
            RED = 1
            GREEN = 2

        a = ColorClosedOver

        def closed_over_aliased_type():
            return a.RED.value

        with torch._jit_internal._disable_emit_hooks():
            scripted = torch.jit.script(closed_over_aliased_type)

        self.assertEqual(scripted(), ColorClosedOver.RED.value)


        b = ColorClosedOver.RED

        def closed_over_aliased_value():
            return b.value

        with torch._jit_internal._disable_emit_hooks():
            scripted = torch.jit.script(closed_over_aliased_value)

        self.assertEqual(scripted(), ColorClosedOver.RED.value)

    def test_enum_as_module_attribute(self):
        global ColorModuleAttr

        class ColorModuleAttr(Enum):
            RED = 1
            GREEN = 2

        class TestModule(torch.nn.Module):
            def __init__(self, e: ColorModuleAttr):
                super(TestModule, self).__init__()
                self.e = e

            def forward(self):
                return self.e.value

        m = TestModule(ColorModuleAttr.RED)
        with torch._jit_internal._disable_emit_hooks():
            scripted = torch.jit.script(m)

        self.assertEqual(scripted(), ColorModuleAttr.RED.value)

    def test_enum_return(self):
        global ColorForReturn

        class ColorForReturn(Enum):
            RED = 1
            GREEN = 2

        @torch.jit.script
        def return_enum(cond: bool):
            if cond:
                return ColorForReturn.RED
            else:
                return ColorForReturn.GREEN

        self.assertEqual(return_enum(True), ColorForReturn.RED)
        self.assertEqual(return_enum(False), ColorForReturn.GREEN)

    def test_enum_module_return(self):
        global ColorForModuleReturn

        class ColorForModuleReturn(Enum):
            RED = 1
            GREEN = 2

        class TestModule(torch.nn.Module):
            def __init__(self, e: ColorForModuleReturn):
                super(TestModule, self).__init__()
                self.e = e

            def forward(self):
                return self.e

        m = TestModule(ColorForModuleReturn.RED)
        with torch._jit_internal._disable_emit_hooks():
            scripted = torch.jit.script(m)

        self.assertEqual(scripted(), ColorForModuleReturn.RED)


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
        global ColorGuarded

        class ColorGuarded(Enum):
            RED = 1
            GREEN = 2

        def enum_comp(x: ColorGuarded, y: ColorGuarded) -> bool:
            return x == y

        with self.assertRaisesRegexWithHighlight(RuntimeError, "Unknown type name 'ColorGuarded'", "ColorGuarded"):
            torch.jit.script(enum_comp)
