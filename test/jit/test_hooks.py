import os
import sys

import torch
from typing import List, Tuple

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


class OuterModuleSubmodForwardCall(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(OuterModuleSubmodForwardCall, self).__init__()
        self.name = name
        self.submodule = InnerModuleSingleIO(submodule_name)

    def forward(self, input: str):
        input = input + "_outermod"
        return self.submodule.forward(input)


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


class InnerModuleTupleSingleIO(torch.nn.Module):
    def __init__(self, name):
        super(InnerModuleTupleSingleIO, self).__init__()
        self.name = name

    def forward(self, input: Tuple[int]):
        input_access = input[0]
        return (1,)


class OuterModuleTupleSingleIO(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(OuterModuleTupleSingleIO, self).__init__()
        self.name = name
        self.submodule = InnerModuleTupleSingleIO(submodule_name)

    def forward(self, input: Tuple[int]):
        input_access = input[0]
        return self.submodule((1,))


class InnerModuleTupleDoubleIO(torch.nn.Module):
    def __init__(self, name):
        super(InnerModuleTupleDoubleIO, self).__init__()
        self.name = name

    def forward(self, input: Tuple[str, str]):
        input_access1 = input[0]
        input_access2 = input[1]
        return input


class OuterModuleTupleDoubleIO(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(OuterModuleTupleDoubleIO, self).__init__()
        self.name = name
        self.submodule = InnerModuleTupleDoubleIO(submodule_name)

    def forward(self, input: Tuple[str, str]):
        input_access1 = input[0]
        input_access2 = input[1]
        return self.submodule(("outer_mod", "outer_mod"))


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

    def test_submodule_hook_and_pre_hook_no_IO(self):
        m = OuterModuleNoIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[None]):
            assert self.name == "inner_mod_name"

        def forward_hook(self, input: Tuple[None], output: None):
            assert self.name == "inner_mod_name"

        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_hook(forward_hook)

        self.checkModule(m, ())

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
            return output, output

        def forward_hook2(self, input: Tuple[str], output: Tuple[str, str]):
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_overrid_name2",)
            assert output[0] == "pre_hook_overrid_name2_outermod_inner_mod_fh1"
            output = output[0] + "_fh2"
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
            return output[0], output2, output2

        def forward_hook2(
            self, input: Tuple[List[str], str], output: Tuple[List[str], str, str]
        ):
            assert self.name == "inner_mod_name"
            assert input[1] == "pre_hook_override2"
            assert output[1] == "pre_hook_override2_fh1"
            output2 = output[1] + "_fh2"
            return output[0], output2, output2

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

    def test_nested_tuple_IO(self):
        m = OuterModuleTupleDoubleIO("outer_mod_name", "inner_mod_name")

        def pre_hook_outermod(self, input: Tuple[Tuple[str, str]]):
            # 'return ("hello", "goodbye")' doesn't work with eager because
            # tuple is unpacked by eager when forward isn't expecting it
            return (("hello", "goodbye"),)

        def pre_hook_innermod(self, input: Tuple[Tuple[str, str]]):
            # 'return ("hey","howdy")' doesn't work with eager because
            # tuple unpacked by eager when forward isn't expecting it
            return (("hey", "howdy"),)

        def forward_hook_outermod(self, input: Tuple[Tuple[str, str]], output: str):
            return ("a", "b")

        def forward_hook_innermod(
            self, input: Tuple[Tuple[str, str]], output: Tuple[str, str]
        ):
            return "forward_inner_mod"

        m.register_forward_pre_hook(pre_hook_outermod)
        m.submodule.register_forward_pre_hook(pre_hook_innermod)
        m.register_forward_hook(forward_hook_outermod)
        m.submodule.register_forward_hook(forward_hook_innermod)

        self.checkModule(m, (("a", "b"),))

        m = OuterModuleTupleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook_outermod(self, input: Tuple[Tuple[int]]):
            # 'return (11,)' doesn't work with eager, inner tuple lost
            return ((11,),)

        def pre_hook_innermod(self, input: Tuple[Tuple[int]]):
            # 'return (22,)' doesn't work with eager, inner tuple lost
            return ((22,),)

        def forward_hook_outermod(self, input: Tuple[Tuple[int]], output: int):
            return (11,)

        def forward_hook_innermod(self, input: Tuple[Tuple[int]], output: Tuple[int]):
            return 22

        m.register_forward_pre_hook(pre_hook_outermod)
        m.submodule.register_forward_pre_hook(pre_hook_innermod)
        m.register_forward_hook(forward_hook_outermod)
        m.submodule.register_forward_hook(forward_hook_innermod)

        self.checkModule(m, ((3,),))

    def test_hook_method_name_collision(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def foo(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"
            return ("pre_hook_overrid_name",)

        m.submodule.register_forward_pre_hook(foo)

        err_msg = (
            "Can't define hook: foo on class: .+ "
            "because a method or hook with that name already exists."
        )
        with self.assertRaisesRegex(
            RuntimeError, err_msg,
        ):
            torch.jit.script(m)

    def test_submodule_forward_and_pre_hook_single_IO_no_tuple_returned(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook_a(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"
            return "pre_hook_overrid_name"

        def forward_hook_b(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("pre_hook_overrid_name",)
            output = output + "_fh"
            return output

        m.submodule.register_forward_pre_hook(pre_hook_a)
        m.submodule.register_forward_hook(forward_hook_b)

        self.checkModule(m, ("a",))

    def test_module_direct_forward_invocation(self):
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            return ("pre_hook_overrid_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_overrid_name",)
            output = output + "_fh"
            return output

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        m_scripted = torch.jit.script(m)

        self.assertEqual(m.forward("a"), m_scripted.forward("a"))
        self.assertNotEqual(m_scripted("a"), m_scripted.forward("a"))

    def test_submodule_direct_forward_invocation(self):
        m_submod_forward_call = OuterModuleSubmodForwardCall(
            "outer_mod_name", "inner_mod_name"
        )
        m_submod_call = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            return ("pre_hook_overrid_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert input == ("pre_hook_overrid_name",)
            return output + "_fh"

        m_submod_forward_call.submodule.register_forward_pre_hook(pre_hook)
        m_submod_forward_call.submodule.register_forward_hook(forward_hook)
        m_submod_call.submodule.register_forward_pre_hook(pre_hook)
        m_submod_call.submodule.register_forward_hook(forward_hook)

        m_submod_forward_call_scripted = torch.jit.script(m_submod_forward_call)
        m_submod_call_scripted = torch.jit.script(m_submod_call)

        self.assertEqual(
            m_submod_forward_call_scripted("a"), m_submod_forward_call("a")
        )
        self.assertNotEqual(
            m_submod_forward_call_scripted("a"), m_submod_call_scripted("a")
        )

    def test_hook_compilation_hint(self):
        # Tests if hook error message is printed out if erroring before schema check.
        # Useful for when user is scripting hooks while not aware of it
        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            assert input[4] == "a"  # out of bounds tuple range
            return ("pre_hook_overrid_name",)

        m.register_forward_pre_hook(pre_hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "This error occured while scripting the forward pre-hook 'pre_hook'",
        ):
            torch.jit.script(m)

    def test_wrong_pre_hook_signatures(self):
        # correct signature: pre_hook_c(self, input: Tuple[str])
        def pre_hook_wrong_input1(self, input: Tuple[None]):
            return ("hello",)

        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input1)

        with self.assertRaisesRegex(
            RuntimeError, "has the wrong inner types for the second tuple argument",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_input2(self, input: Tuple[str], input2: str):
            return ("hello",)

        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input2)

        with self.assertRaisesRegex(
            RuntimeError,
            "was expected to only have exactly 2 inputs but it had 3 inputs",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_input3(self, input: int):
            return ("hello",)

        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input3)

        with self.assertRaisesRegex(
            RuntimeError,
            "expected the second argument to be typed as a Tuple but"
            " found type: 'int' instead",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_output(self, input: Tuple[str]):
            return 1  # expecting Tuple[str], str, or None

        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_output)

        with self.assertRaisesRegex(
            RuntimeError, "returned the wrong type of: 'int'",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_tuple_return(self, input: Tuple[Tuple[int]]):
            return (11,)  # doesn't work with eager, inner tuple lost

        m = OuterModuleTupleSingleIO("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_tuple_return)

        with self.assertRaisesRegex(
            RuntimeError,
            "When forward has a single tuple input argument, "
            "the return needs to be 'None' or a nested tuple containing "
            r"forward's input tuple argument as in: 'Tuple\[Tuple\[int\]\]'",
        ):
            torch.jit.script(m)

    def test_wrong_hook_signatures(self):
        # correct signature:
        #   def forward_hook(self, input: Tuple[str], output: str)
        def forward_hook_wrong_input1(self, input: Tuple[str, str], output: str):
            return output

        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_input1)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong number of contained types for the "
            r"input argument's Tuple. Received type: 'Tuple\[str, str\]'",
        ):
            torch.jit.script(m)

        def forward_hook_wrong_input2(self, input: str, output: str):
            return output

        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_input2)

        with self.assertRaisesRegex(
            RuntimeError,
            "expected the second argument to be typed as a Tuple "
            "but found type: 'str' instead.",
        ):
            torch.jit.script(m)

        def forward_hook_wrong_input3(self, input: Tuple[None], output: str):
            return output

        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_input3)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong inner types for the second tuple"
            r" argument. Received type: 'Tuple\[None\]'",
        ):
            torch.jit.script(m)

        def forward_hook_wrong_output(self, input: Tuple[str], output: Tuple[str]):
            return output

        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_output)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong type for the output argument. Received"
            r" type: 'Tuple\[str\]'. Expected type: 'str'",
        ):
            torch.jit.script(m)

        def forward_hook_correct(self, input: Tuple[str], output: str):
            return (output,)

        def forward_hook_wrong_output_from_prev_hook(
            self, input: Tuple[str], output: str
        ):
            return output

        m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_correct)
        m.register_forward_hook(forward_hook_wrong_output_from_prev_hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong type for the output argument. "
            r"Received type: 'str'. Expected type: 'Tuple\[str\]'",
        ):
            torch.jit.script(m)
