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


class SubmoduleNoForwardInputs(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self):
        assert self.name == "inner_mod_name"


class ModuleNoForwardInputs(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleNoForwardInputs(submodule_name)

    def forward(self):
        self.submodule()


class SubmoduleForwardSingleInput(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def foo(self, input: str):
        return input

    def forward(self, input: str):
        input = input + "_inner_mod"
        input = self.foo(input)
        return input


class ModuleForwardSingleInput(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardSingleInput(submodule_name)

    def forward(self, input: str):
        input = input + "_outermod"
        return self.submodule(input)


class ModuleDirectFowardSubmodCall(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardSingleInput(submodule_name)

    def forward(self, input: str):
        input = input + "_outermod"
        return self.submodule.forward(input)


class SuboduleForwardMultipleInputs(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, input1: List[str], input2: str):
        input1.append(self.name)
        output2 = input2 + "_"
        return input1, output2


class ModuleForwardMultipleInputs(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        self.submodule = SuboduleForwardMultipleInputs(submodule_name)

    def forward(self, input1: List[str], input2: str):
        input1.append(self.name)
        return self.submodule(input1, input2)


class SubmoduleForwardTupleInput(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, input: Tuple[int]):
        input_access = input[0]
        return (1,)


class ModuleForwardTupleInput(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardTupleInput(submodule_name)

    def forward(self, input: Tuple[int]):
        input_access = input[0]
        return self.submodule((1,))


# Tests for JIT forward hooks and pre-hooks
class TestHooks(JitTestCase):
    def test_module_no_forward_input(self):
        # Test module level hooks with no forward input
        m = ModuleNoForwardInputs("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[None]):
            assert self.name == "outer_mod_name"

        def forward_hook(self, input: Tuple[None], output: None):
            assert self.name == "outer_mod_name"

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        self.checkModule(m, ())

    def test_submodule_no_forward_input(self):
        # Test submodule level hooks with no forward input
        m = ModuleNoForwardInputs("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[None]):
            assert self.name == "inner_mod_name"

        def forward_hook(self, input: Tuple[None], output: None):
            assert self.name == "inner_mod_name"

        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_hook(forward_hook)

        self.checkModule(m, ())

    def test_module_forward_multiple_inputs(self):
        # Test module level hooks with forward having multiple
        # inputs and returns
        m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[List[str], str]):
            assert self.name == "outer_mod_name"
            assert input[0][0] == "a"
            return ["pre_hook_override_name"], "pre_hook_override"

        def forward_hook(
            self, input: Tuple[List[str], str], output: Tuple[List[str], str]
        ):
            assert self.name == "outer_mod_name"
            assert input[0][0] == "pre_hook_override_name"
            output2 = output[1] + "fh"
            return output[0], output2

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        self.checkModule(m, (["a"], "no_pre_hook"))

    def test_module_multiple_hooks_multiple_inputs(self):
        # Test that module level hooks with multiple inputs execute
        # in correct order and pass correct information between each other
        m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

        def pre_hook1(self, input: Tuple[List[str], str]):
            assert self.name == "outer_mod_name"
            assert input[0][0] == "a"
            return ["pre_hook_override_name"], "pre_hook_override"

        def pre_hook2(self, input: Tuple[List[str], str]):
            assert self.name == "outer_mod_name"
            assert input[0][0] == "pre_hook_override_name"
            return ["pre_hook_override_name2"], "pre_hook_override"

        def forward_hook1(
            self, input: Tuple[List[str], str], output: Tuple[List[str], str]
        ):
            assert self.name == "outer_mod_name"
            assert input[0][0] == "pre_hook_override_name2"
            output2 = output[1] + "fh1"
            return output[0], output2

        def forward_hook2(
            self, input: Tuple[List[str], str], output: Tuple[List[str], str]
        ):
            assert self.name == "outer_mod_name"
            assert input[0][0] == "pre_hook_override_name2"
            assert output[1] == "pre_hook_override_fh1"
            output2 = output[1] + "_fh2"
            return output[0], output2

        m.register_forward_pre_hook(pre_hook1)
        m.register_forward_pre_hook(pre_hook2)
        m.register_forward_hook(forward_hook1)
        m.register_forward_hook(forward_hook2)

        self.checkModule(m, (["a"], "no_pre_hook"))

    def test_module_forward_single_input(self):
        # Test module level hooks work for forward with single input
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            assert input[0] == "a"
            return ("pre_hook_override_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_override_name",)
            output = output + "_fh"
            return output

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_module_same_hook_repeated(self):
        # Test modules can run same hook multiple times
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            return ("pre_hook_override_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_override_name",)
            output = output + "_fh"
            return output

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)
        m.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_module_hook_return_nothing(self):
        # Test module level hooks that reutrn nothing
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            assert input[0] == "a"

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"
            assert input == ("a",)

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_module_multiple_hooks_single_input(self):
        # Test modules can run multiple hooks with single input
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook1(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            assert input[0] == "a"
            return ("pre_hook_override_name1",)

        def pre_hook2(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            assert input[0] == "pre_hook_override_name1"
            return ("pre_hook_override_name2",)

        def forward_hook1(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_override_name2",)
            assert output == "pre_hook_override_name2_outermod_inner_mod"
            output = output + "_fh1"
            return output, output

        def forward_hook2(self, input: Tuple[str], output: Tuple[str, str]):
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_override_name2",)
            assert output[0] == "pre_hook_override_name2_outermod_inner_mod_fh1"
            output = output[0] + "_fh2"
            return output

        m.register_forward_pre_hook(pre_hook1)
        m.register_forward_pre_hook(pre_hook2)
        m.register_forward_hook(forward_hook1)
        m.register_forward_hook(forward_hook2)

        self.checkModule(m, ("a",))

    def test_submodule_forward_multiple_inputs(self):
        # Test submodules can run hooks that have multiple forward inputs
        m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[List[str], str]):
            assert self.name == "inner_mod_name"
            assert input[0][1] == "outer_mod_name"
            return ["pre_hook_override_name"], "pre_hook_override"

        def forward_hook(
            self, input: Tuple[List[str], str], output: Tuple[List[str], str]
        ):
            assert self.name == "inner_mod_name"
            assert input[0][0] == "pre_hook_override_name"
            output2 = output[1] + "fh"
            return output[0], output2

        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_hook(forward_hook)

        self.checkModule(m, (["a"], "no_pre_hook"))

    def test_submodule_multiple_hooks_multiple_inputs(self):
        # Test submodules can run multiple hooks with multiple
        # forward inputs
        m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

        def pre_hook1(self, input: Tuple[List[str], str]):
            assert self.name == "inner_mod_name"
            assert input[1] == "no_pre_hook"
            return ["pre_hook_override_name"], "pre_hook_override1"

        def pre_hook2(self, input: Tuple[List[str], str]):
            assert self.name == "inner_mod_name"
            assert input[1] == "pre_hook_override1"
            return ["pre_hook_override_name"], "pre_hook_override2"

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

    def test_submodule_forward_single_input(self):
        # Test submodules can run hooks with a single argument
        # passed to forward
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"
            return ("pre_hook_override_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("pre_hook_override_name",)
            return output

        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_submodule_same_hook_repeated(self):
        # Test submodules can run same hooks multiple times
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            return ("pre_hook_override_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("pre_hook_override_name",)
            return output

        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_hook(forward_hook)
        m.submodule.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_submodule_hook_return_nothing(self):
        # Test submodules can run hooks that return nothing
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("a_outermod",)

        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_submodule_multiple_hooks_single_input(self):
        # Test submodules can run multiple hooks that have a single input
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook1(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"
            return ("pre_hook_override_name",)

        def pre_hook2(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "pre_hook_override_name"
            return ("pre_hook_override_name2",)

        def forward_hook1(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("pre_hook_override_name2",)
            assert output == "pre_hook_override_name2_inner_mod"
            return output + "_fwh1"

        def forward_hook2(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("pre_hook_override_name2",)
            assert output == "pre_hook_override_name2_inner_mod_fwh1"
            return output

        m.submodule.register_forward_pre_hook(pre_hook1)
        m.submodule.register_forward_pre_hook(pre_hook2)
        m.submodule.register_forward_hook(forward_hook1)
        m.submodule.register_forward_hook(forward_hook2)

        self.checkModule(m, (["a"]))

    def test_forward_tuple_input(self):
        # Test case where forward is passed a single tuple for input.
        # This is different because eager always wraps pre-hook return arguments
        # in a tuple when the returned pre-hook result isn't a tuple
        # (to allow the result to be passed to another pre-hook if needed).
        # The eager behavior doesn't wrap the single tuple input pre-hook return in a
        # tuple as it should. To get consitent behavior between single tuple inputs and
        # the rest of the possible forward inputs, pre-hooks need to
        # wrap single tuple inputs returns in another tuple. This is
        # enforced by the schema checker.
        m = ModuleForwardTupleInput("outer_mod_name", "inner_mod_name")

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
        # Hooks can't have the same name as methods.
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def foo(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"
            return ("pre_hook_override_name",)

        m.submodule.register_forward_pre_hook(foo)

        with self.assertRaisesRegex(
            RuntimeError,
            "Can't define hook: foo on class: .+ "
            "because a method or hook with that name already exists.",
        ):
            torch.jit.script(m)

    def test_hook_hook_name_collision(self):
        # Test edge case of two hooks sharing name but not python definition
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def prehook(self, input: Tuple[str]):
            return "This is the first hook"

        m.submodule.register_forward_pre_hook(prehook)

        def prehook(self, input: Tuple[str]):
            return "This is the second hook"

        m.submodule.register_forward_pre_hook(prehook)

        with self.assertRaisesRegex(
            RuntimeError,
            "Pre-hook '.+' on .+ has at least two different python "
            "definitions. Please use unique names for all hooks.",
        ):
            torch.jit.script(m)

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def hook(self, input: Tuple[str], output: str):
            return "This is the first hook"

        m.submodule.register_forward_hook(hook)

        def hook(self, input: Tuple[str]):
            return "This is the second hook"

        m.submodule.register_forward_hook(hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "Hook '.+' on .+ has at least two different python "
            "definitions. Please use unique names for all hooks.",
        ):
            torch.jit.script(m)

    def test_submodule_forward_single_input_return_not_tupled(self):
        # Test to check that submodules can return modified inputs
        # that aren't wrapped in a tuple (to match eager behavior)
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"
            # return is wrapped in tuple in other test cases
            return "pre_hook_override_name"

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "inner_mod_name"
            assert input == ("pre_hook_override_name",)
            output = output + "_fh"
            return output

        m.submodule.register_forward_pre_hook(pre_hook)
        m.submodule.register_forward_hook(forward_hook)

        self.checkModule(m, ("a",))

    def test_module_direct_forward_invocation(self):
        # Test that hooks are only invoked when the module is
        # called directly and not when forward is called.
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            return ("pre_hook_override_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_override_name",)
            output = output + "_fh"
            return output

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        m_scripted = torch.jit.script(m)

        self.assertEqual(m.forward("a"), m_scripted.forward("a"))
        self.assertNotEqual(m_scripted("a"), m_scripted.forward("a"))

    def test_submodule_direct_forward_invocation(self):
        m_submod_forward_call = ModuleDirectFowardSubmodCall(
            "outer_mod_name", "inner_mod_name"
        )
        m_submod_call = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            return ("pre_hook_override_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert input == ("pre_hook_override_name",)
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
        # Useful for when user is scripting hooks while not aware of it.
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]):
            assert self.name == "outer_mod_name"
            assert input[4] == "a"  # out of bounds tuple range
            return ("pre_hook_override_name",)

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

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input1)

        with self.assertRaisesRegex(
            RuntimeError, "has the wrong inner types for the second tuple argument",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_input2(self, input: Tuple[str], input2: str):
            return ("hello",)

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input2)

        with self.assertRaisesRegex(
            RuntimeError,
            "was expected to only have exactly 2 inputs but it had 3 inputs",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_input3(self, input: int):
            return ("hello",)

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input3)

        with self.assertRaisesRegex(
            RuntimeError,
            "expected the second argument to be typed as a Tuple but"
            " found type: 'int' instead",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_output(self, input: Tuple[str]):
            return 1  # expecting Tuple[str], str, or None

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_output)

        with self.assertRaisesRegex(
            RuntimeError, "returned the wrong type of: 'int'",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_tuple_return(self, input: Tuple[Tuple[int]]):
            return (11,)  # doesn't work with eager, inner tuple lost

        m = ModuleForwardTupleInput("outer_mod_name", "inner_mod_name")
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

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_input1)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong number of contained types for the "
            r"input argument's Tuple. Received type: 'Tuple\[str, str\]'",
        ):
            torch.jit.script(m)

        def forward_hook_wrong_input2(self, input: str, output: str):
            return output

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_input2)

        with self.assertRaisesRegex(
            RuntimeError,
            "expected the second argument to be typed as a Tuple "
            "but found type: 'str' instead.",
        ):
            torch.jit.script(m)

        def forward_hook_wrong_input3(self, input: Tuple[None], output: str):
            return output

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_input3)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong inner types for the second tuple"
            r" argument. Received type: 'Tuple\[None\]'",
        ):
            torch.jit.script(m)

        def forward_hook_wrong_output(self, input: Tuple[str], output: Tuple[str]):
            return output

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
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

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_correct)
        m.register_forward_hook(forward_hook_wrong_output_from_prev_hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong type for the output argument. "
            r"Received type: 'str'. Expected type: 'Tuple\[str\]'",
        ):
            torch.jit.script(m)
