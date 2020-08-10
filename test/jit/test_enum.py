import os
import sys

import torch
from torch.testing import FileCheck
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

        @torch.jit.script
        def supported_enum_types(a: IntEnum, b: FloatEnum, c: StringEnum):
            return (a.name, b.name, c.name)

        FileCheck() \
            .check("IntEnum") \
            .check("FloatEnum") \
            .check("StringEnum") \
            .run(str(supported_enum_types.graph))

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

        @torch.jit.script
        def enum_comp(x: Color, y: Color) -> bool:
            return x == y

        FileCheck().check("aten::eq").run(enum_comp.graph)

        self.assertEqual(
            enum_comp(Color.RED, Color.RED), True)

        self.assertEqual(
            enum_comp(Color.RED, Color.GREEN), False)

    def test_enum_comp_diff_classes(self):
        global Foo, Bar

        class Foo(Enum):
            ITEM1 = 1
            ITEM2 = 2

        class Bar(Enum):
            ITEM1 = 1
            ITEM2 = 2

        @torch.jit.script
        def enum_comp(x: Foo) -> bool:
            return x == Bar.ITEM1

        FileCheck() \
            .check("prim::Constant") \
            .check_same("Bar.ITEM1") \
            .check("aten::eq") \
            .run(enum_comp.graph)

        self.assertEqual(enum_comp(Foo.ITEM1), False)

    def test_heterogenous_value_type_enum_error(self):
        global ColorDiffType

        class ColorDiffType(Enum):
            RED = 1
            GREEN = "green"

        def enum_comp(x: ColorDiffType, y: ColorDiffType) -> bool:
            return x == y

        with self.assertRaisesRegex(RuntimeError, "Could not unify type list"):
            torch.jit.script(enum_comp)

    def test_enum_name(self):
        global ColorForName

        class ColorForName(Enum):
            RED = 1
            GREEN = 2

        @torch.jit.script
        def enum_name(x: ColorForName) -> str:
            return x.name

        FileCheck() \
            .check("ColorForName") \
            .check_next("prim::EnumName") \
            .check_next("return") \
            .run(enum_name.graph)

        self.assertEqual(enum_name(ColorForName.RED), ColorForName.RED.name)
        self.assertEqual(enum_name(ColorForName.GREEN), ColorForName.GREEN.name)

    def test_enum_value(self):
        global ColorForValue

        class ColorForValue(Enum):
            RED = 1
            GREEN = 2

        @torch.jit.script
        def enum_value(x: ColorForValue) -> int:
            return x.value

        FileCheck() \
            .check("ColorForValue") \
            .check_next("prim::EnumValue") \
            .check_next("return") \
            .run(enum_value.graph)

        self.assertEqual(enum_value(ColorForValue.RED), ColorForValue.RED.value)
        self.assertEqual(enum_value(ColorForValue.GREEN), ColorForValue.GREEN.value)

    def test_enum_as_const(self):
        global ColorConst

        class ColorConst(Enum):
            RED = 1
            GREEN = 2

        @torch.jit.script
        def enum_const(x: ColorConst) -> bool:
            return x == ColorConst.RED

        FileCheck() \
            .check("prim::Constant[value=Enum<__torch__.jit.test_enum.ColorConst.RED>]") \
            .check_next("aten::eq") \
            .check_next("return") \
            .run(enum_const.graph)

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

        @torch.jit.script
        def is_color_enum(x: Any):
            return isinstance(x, ColorIValueType)

        FileCheck() \
            .check("prim::isinstance[types=[Enum<__torch__.jit.test_enum.ColorIValueType>]]") \
            .check_next("return") \
            .run(str(is_color_enum.graph))

        self.assertEqual(is_color_enum(ColorIValueType.RED), True)
        self.assertEqual(is_color_enum(ColorIValueType.GREEN), True)
        self.assertEqual(is_color_enum(1), False)

    def test_closed_over_enum_constant(self):
        global ColorClosedOver

        class ColorClosedOver(Enum):
            RED = 1
            GREEN = 2

        a = ColorClosedOver

        @torch.jit.script
        def closed_over_aliased_type():
            return a.RED.value

        FileCheck() \
            .check("prim::Constant[value={}]".format(a.RED.value)) \
            .check_next("return") \
            .run(closed_over_aliased_type.graph)

        self.assertEqual(closed_over_aliased_type(), ColorClosedOver.RED.value)

        b = ColorClosedOver.RED

        @torch.jit.script
        def closed_over_aliased_value():
            return b.value

        FileCheck() \
            .check("prim::Constant[value={}]".format(b.value)) \
            .check_next("return") \
            .run(closed_over_aliased_value.graph)

        self.assertEqual(closed_over_aliased_value(), ColorClosedOver.RED.value)

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
        scripted = torch.jit.script(m)

        FileCheck() \
            .check("TestModule") \
            .check_next("ColorModuleAttr") \
            .check_same("prim::GetAttr[name=\"e\"]") \
            .check_next("prim::EnumValue") \
            .check_next("return") \
            .run(scripted.graph)

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
        scripted = torch.jit.script(m)

        FileCheck() \
            .check("TestModule") \
            .check_next("ColorForModuleReturn") \
            .check_same("prim::GetAttr[name=\"e\"]") \
            .check_next("return") \
            .run(scripted.graph)

        self.assertEqual(scripted(), ColorForModuleReturn.RED)


    def test_enum_iterate(self):
        global ColorForIterate

        class ColorForIterate(Enum):
            RED = 1
            GREEN = 2
            PURPLE = 3

        @torch.jit.script
        def iterate_enum(x: ColorForIterate):
            res: List[int] = []
            for e in ColorForIterate:
                if e != x:
                    res.append(e.value)
            return res

        FileCheck() \
            .check("Enum<__torch__.jit.test_enum.ColorForIterate>[]") \
            .check_same("ColorForIterate.RED") \
            .check_same("ColorForIterate.GREEN") \
            .check_same("ColorForIterate.PURPLE") \
            .run(str(iterate_enum.graph))

        # PURPLE always appear last because we follow Python's Enum definition order.
        self.assertEqual(iterate_enum(ColorForIterate.RED), [ColorForIterate.GREEN.value, ColorForIterate.PURPLE.value])
        self.assertEqual(iterate_enum(ColorForIterate.GREEN), [ColorForIterate.RED.value, ColorForIterate.PURPLE.value])


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
