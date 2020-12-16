import os
import sys

import torch
from typing import Any, List, Tuple

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


class InnerModuleNoIO(torch.nn.Module):
    def __init__(self, name):
        super(InnerModuleNoIO, self).__init__()
        self.name = name

    def forward(self):
        assert self.name == "inner_mod_name"


class OuterModuleNoIO(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(OuterModuleNoIO, self).__init__()
        self.name = name
        self.submodule = InnerModuleNoIO(submodule_name)

    def forward(self):
        assert self.name == "outer_mod_name"
        self.submodule()


class InnerModuleSingleIO(torch.nn.Module):
    def __init__(self, name):
        super(InnerModuleSingleIO, self).__init__()
        self.name = name

    def foo(self, input: str):
        return input

    def forward(self, input: str):
        input = input + "_inner_mod"
        input = self.foo(input)
        return input


class OuterModuleSingleIO(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(OuterModuleSingleIO, self).__init__()
        self.name = name
        self.submodule = InnerModuleSingleIO(submodule_name)

    def forward(self, input: str):
        input = input + "_outermod"
        return self.submodule(input)


class InnerModuleMultipleIO(torch.nn.Module):
    def __init__(self, name):
        super(InnerModuleMultipleIO, self).__init__()
        self.name = name

    def forward(self, input1: List[str], input2: str):
        input1.append(self.name)
        output2 = input2 + "_"
        return input1, output2


class OuterModuleMultipleIO(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(OuterModuleMultipleIO, self).__init__()
        self.name = name
        self.submodule = InnerModuleMultipleIO(submodule_name)

    def forward(self, input1: List[str], input2: str):
        input1.append(self.name)
        return self.submodule(input1, input2)


# Tests for JIT forward hooks and pre-hooks
class TestHooks(JitTestCase):
    def test_module_hook_and_pre_hook_no_IO(self):
        m = OuterModuleNoIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[None]):
            assert self.name == "outer_mod_name"

        def forward_hook(self, input: Tuple[None], output: None):
            assert self.name == "outer_mod_name"

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        self.checkModule(m, ())

        m2 = OuterModuleNoIO("outer_mod_name", "inner_mod_name")

        def pre_hook2(self, input: Tuple[None]):
            assert self.name == "inner_mod_name"

        def forward_hook2(self, input: Tuple[None], output: None):
            assert self.name == "inner_mod_name"

        m2.submodule.register_forward_pre_hook(pre_hook2)
        m2.submodule.register_forward_hook(forward_hook2)

        self.checkModule(m2, ())

    def test_module_hook_and_pre_hook_multiple_IO(self):
        m = OuterModuleMultipleIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[List[str], str]):
            assert self.name == "outer_mod_name"
            assert input[0][0] == "a"
            return ["pre_hook_overrid_name"], "pre_hook_override"

        def forward_hook(
            self, input: Tuple[List[str], str], output: Tuple[List[str], str]
        ):
            assert self.name == "outer_mod_name"
            assert (
                input[0][0] == "pre_hook_overrid_name"
            )  # what the pre_hook overrid instead of the original
            output2 = output[1] + "fh"
            return output[0], output2

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        self.checkModule(m, (["a"], "no_pre_hook"))

    def test_module_hook_and_pre_hook_multiple_IO_multiple_hooks(self):
        m = OuterModuleMultipleIO("outer_mod_name", "inner_mod_name")

        def pre_hook1(self, input: Tuple[List[str], str]):
            assert self.name == "outer_mod_name"
            assert input[0][0] == "a"
            return ["pre_hook_overrid_name"], "pre_hook_override"

        def pre_hook2(self, input: Tuple[List[str], str]):
            assert self.name == "outer_mod_name"
            assert input[0][0] == "pre_hook_overrid_name"
            return ["pre_hook_overrid_name2"], "pre_hook_override"

        def forward_hook1(
            self, input: Tuple[List[str], str], output: Tuple[List[str], str]
        ):
            assert self.name == "outer_mod_name"
            assert input[0][0] == "pre_hook_overrid_name2"
            output2 = output[1] + "fh1"
            return output[0], output2

        def forward_hook2(
            self, input: Tuple[List[str], str], output: Tuple[List[str], str]
        ):
            assert self.name == "outer_mod_name"
            assert input[0][0] == "pre_hook_overrid_name2"
            assert output[1] == "pre_hook_override_fh1"
            output2 = output[1] + "_fh2"
            return output[0], output2

        m.register_forward_pre_hook(pre_hook1)
        m.register_forward_pre_hook(pre_hook2)
        m.register_forward_hook(forward_hook1)
        m.register_forward_hook(forward_hook2)

        self.checkModule(m, (["a"], "no_pre_hook"))

    def test_module_forward_and_pre_hook_single_IO(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            assert input[0] == "a"
            return ("pre_hook_overrid_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_overrid_name",)
            output = output + "_fh"
            return output

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_module_forward_and_pre_hook_single_IO_same_hook_twice(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            return ("pre_hook_overrid_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            # note: 'output' of forward hook needs to not be wrapped in tuple
            # when there is a single element in the forward's return
            # this is to match eager's behavior
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_overrid_name",)
            output = output + "_fh"
            return output

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)
        m.register_forward_hook(forward_hook)

        scripted = torch.jit.script(m)

        self.checkModule(m, ("a",))

    def test_module_forward_and_pre_hook_single_IO_no_change(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            assert input[0] == "a"

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"
            assert input == ("a",)

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_module_forward_and_pre_hook_single_IO_multiple_hooks(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook1(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            assert input[0] == "a"
            return ("pre_hook_overrid_name1",)

        def pre_hook2(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            assert input[0] == "pre_hook_overrid_name1"
            return ("pre_hook_overrid_name2",)

        def forward_hook1(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_overrid_name2",)
            assert output == "pre_hook_overrid_name2_outermod_inner_mod"
            output = output + "_fh1"
            return output

        def forward_hook2(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_overrid_name2",)
            assert output == "pre_hook_overrid_name2_outermod_inner_mod_fh1"
            output = output + "_fh2"
            return output

        m.register_forward_pre_hook(pre_hook1)
        m.register_forward_pre_hook(pre_hook2)
        m.register_forward_hook(forward_hook1)
        m.register_forward_hook(forward_hook2)

        self.checkModule(m, ("a",))

    def test_submodule_hook_and_pre_hook_multiple_IO(self):
        m = OuterModuleMultipleIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[List[str], str]):
            assert self.name == "inner_mod_name"
            assert input[0][1] == "outer_mod_name"
            return ["pre_hook_overrid_name"], "pre_hook_override"

        def forward_hook(
            self, input: Tuple[List[str], str], output: Tuple[List[str], str]
        ):
            assert self.name == "inner_mod_name"
            assert input[0][0] == "pre_hook_overrid_name"
            output2 = output[1] + "fh"
            return output[0], output2

        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_hook(forward_hook)

        self.checkModule(m, (["a"], "no_pre_hook"))

    def test_submodule_hook_and_pre_hook_multiple_IO_multiple_hooks(self):
        m = OuterModuleMultipleIO("outer_mod_name", "inner_mod_name")

        def pre_hook1(self, input: Tuple[List[str], str]):
            assert self.name == "inner_mod_name"
            assert input[1] == "no_pre_hook"
            return ["pre_hook_overrid_name"], "pre_hook_override1"

        def pre_hook2(self, input: Tuple[List[str], str]):
            assert self.name == "inner_mod_name"
            assert input[1] == "pre_hook_override1"
            return ["pre_hook_overrid_name"], "pre_hook_override2"

        def forward_hook1(
            self, input: Tuple[List[str], str], output: Tuple[List[str], str]
        ):
            assert self.name == "inner_mod_name"
            assert input[1] == "pre_hook_override2"
            assert output[1] == "pre_hook_override2_"
            output2 = output[1] + "fh1"
            return output[0], output2

        def forward_hook2(
            self, input: Tuple[List[str], str], output: Tuple[List[str], str]
        ):
            assert self.name == "inner_mod_name"
            assert input[1] == "pre_hook_override2"
            assert output[1] == "pre_hook_override2_fh1"
            output2 = output[1] + "_fh2"
            return output[0], output2

        m.submodule.register_forward_pre_hook(pre_hook1)
        m.submodule.register_forward_pre_hook(pre_hook2)
        m.submodule.register_forward_hook(forward_hook1)
        m.submodule.register_forward_hook(forward_hook2)

        self.checkModule(m, (["a"], "no_pre_hook"))

    def test_submodule_forward_and_pre_hooks_single_IO(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"
            return ("pre_hook_overrid_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("pre_hook_overrid_name",)
            return output

        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_submodule_forward_and_pre_hook_single_IO_same_hook_twice(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            return ("pre_hook_overrid_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("pre_hook_overrid_name",)
            return output

        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_hook(forward_hook)
        m.submodule.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_submodule_forward_and_pre_hooks_single_IO_no_change(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("a_outermod",)

        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_submodule_forward_and_pre_hooks_single_IO_multiple_hooks(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook1(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"
            return ("pre_hook_overrid_name",)

        def pre_hook2(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "pre_hook_overrid_name"
            return ("pre_hook_overrid_name2",)

        def forward_hook1(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("pre_hook_overrid_name2",)
            assert output == "pre_hook_overrid_name2_inner_mod"
            return output + "_fwh1"

        def forward_hook2(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("pre_hook_overrid_name2",)
            assert output == "pre_hook_overrid_name2_inner_mod_fwh1"
            return output

        m.submodule.register_forward_pre_hook(pre_hook1)
        m.submodule.register_forward_pre_hook(pre_hook2)
        m.submodule.register_forward_hook(forward_hook1)
        m.submodule.register_forward_hook(forward_hook2)

        self.checkModule(m, (["a"]))

    def test_hook_method_name_collision(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def foo(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"
            return ("pre_hook_overrid_name",)

        m.submodule.register_forward_pre_hook(foo)

        err_msg = (
            r"Can't define hook: foo on class: .+ "
            "because a method with that name already exists."
        )
        with self.assertRaisesRegex(RuntimeError, err_msg,):
            torch.jit.script(m)

    # TODO: need to test error messages for incorrect schemas/signatures!!
