# Owner(s): ["oncall: jit"]

import os
import sys
import unittest
from typing import Tuple

import torch
from jit.test_hooks_modules import (
    create_forward_tuple_input,
    create_module_forward_multiple_inputs,
    create_module_forward_single_input,
    create_module_hook_return_nothing,
    create_module_multiple_hooks_multiple_inputs,
    create_module_multiple_hooks_single_input,
    create_module_no_forward_input,
    create_module_same_hook_repeated,
    create_submodule_forward_multiple_inputs,
    create_submodule_forward_single_input,
    create_submodule_forward_single_input_return_not_tupled,
    create_submodule_hook_return_nothing,
    create_submodule_multiple_hooks_multiple_inputs,
    create_submodule_multiple_hooks_single_input,
    create_submodule_no_forward_input,
    create_submodule_same_hook_repeated,
    create_submodule_to_call_directly_with_hooks,
    ModuleDirectforwardSubmodCall,
    ModuleForwardSingleInput,
    ModuleForwardTupleInput,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Tests for JIT forward hooks and pre-hooks
class TestHooks(JitTestCase):
    def test_module_no_forward_input(self):
        self.checkModule(create_module_no_forward_input(), ())

    def test_submodule_no_forward_input(self):
        self.checkModule(create_submodule_no_forward_input(), ())

    def test_module_forward_multiple_inputs(self):
        self.checkModule(
            create_module_forward_multiple_inputs(), (["a"], "no_pre_hook")
        )

    def test_module_multiple_hooks_multiple_inputs(self):
        self.checkModule(
            create_module_multiple_hooks_multiple_inputs(), (["a"], "no_pre_hook")
        )

    def test_module_forward_single_input(self):
        self.checkModule(create_module_forward_single_input(), ("a",))

    def test_module_same_hook_repeated(self):
        self.checkModule(create_module_same_hook_repeated(), ("a",))

    def test_module_hook_return_nothing(self):
        self.checkModule(create_module_hook_return_nothing(), ("a",))

    def test_module_multiple_hooks_single_input(self):
        self.checkModule(create_module_multiple_hooks_single_input(), ("a",))

    def test_submodule_forward_multiple_inputs(self):
        self.checkModule(
            create_submodule_forward_multiple_inputs(), (["a"], "no_pre_hook")
        )

    def test_submodule_multiple_hooks_multiple_inputs(self):
        self.checkModule(
            create_submodule_multiple_hooks_multiple_inputs(),
            (["a"], "no_pre_hook"),
        )

    def test_submodule_forward_single_input(self):
        self.checkModule(create_submodule_forward_single_input(), ("a",))

    def test_submodule_called_directly_with_hooks(self):
        module = create_submodule_to_call_directly_with_hooks()
        module_scripted = torch.jit.script(module)

        submodule = module.submodule
        scripted_submodule = module_scripted.submodule

        self.assertEqual(submodule("a"), scripted_submodule("a"))

    def test_submodule_same_hook_repeated(self):
        self.checkModule(create_submodule_same_hook_repeated(), ("a",))

    def test_submodule_hook_return_nothing(self):
        self.checkModule(create_submodule_hook_return_nothing(), ("a",))

    def test_submodule_multiple_hooks_single_input(self):
        self.checkModule(create_submodule_multiple_hooks_single_input(), (["a"]))

    def test_forward_tuple_input(self):
        self.checkModule(create_forward_tuple_input(), ((3,),))

    def test_submodule_forward_single_input_return_not_tupled(self):
        self.checkModule(
            create_submodule_forward_single_input_return_not_tupled(), ("a",)
        )

    def test_hook_method_name_collision(self):
        # Hooks can't have the same name as methods.
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def foo(self, input: Tuple[str]) -> Tuple[str]:
            assert self.name == "inner_mod_name"  # noqa: S101
            assert input[0] == "a_outermod"  # noqa: S101
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

        def prehook(self, input: Tuple[str]) -> Tuple[str]:
            return "This is the first hook"

        m.submodule.register_forward_pre_hook(prehook)

        def prehook(self, input: Tuple[str]) -> Tuple[str]:
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

    def test_module_direct_forward_invocation(self):
        # Test that hooks are only invoked when the module is
        # called directly and not when forward is called.
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
            return ("pre_hook_override_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"  # noqa: S101
            assert input == ("pre_hook_override_name",)  # noqa: S101
            output = output + "_fh"
            return output

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        m_scripted = torch.jit.script(m)

        self.assertEqual(m.forward("a"), m_scripted.forward("a"))
        self.assertNotEqual(m_scripted("a"), m_scripted.forward("a"))

    def test_submodule_direct_forward_invocation(self):
        m_submod_forward_call = ModuleDirectforwardSubmodCall(
            "outer_mod_name", "inner_mod_name"
        )
        m_submod_call = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
            return ("pre_hook_override_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            assert input == ("pre_hook_override_name",)  # noqa: S101
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

    # TODO: add this test back once figured out how to print error msg
    @unittest.skip
    def test_hook_compilation_hint(self):
        # Tests if hook error message is printed out if erroring after schema check.
        # Useful for when user is scripting hooks while not aware of it.
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
            assert self.name == "outer_mod_name"  # noqa: S101
            assert input[4] == "a"  # noqa: S101 out of bounds tuple range
            return ("pre_hook_override_name",)

        m.register_forward_pre_hook(pre_hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "This error occurred while scripting the forward pre-hook 'pre_hook'",
        ):
            torch.jit.script(m)

    def test_wrong_pre_hook_signatures(self):
        # correct signature: pre_hook_c(self, input: Tuple[str])
        def pre_hook_wrong_input1(self, input: Tuple[None]) -> Tuple[str]:
            return ("hello",)

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input1)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong inner types for the input tuple argument",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_input2(self, input: Tuple[str], input2: str) -> Tuple[str]:
            return ("hello",)

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input2)

        with self.assertRaisesRegex(
            RuntimeError,
            "was expected to only have exactly 2 inputs but it had 3 inputs",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_input3(self, input: int) -> Tuple[str]:
            return ("hello",)

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input3)

        with self.assertRaisesRegex(
            RuntimeError,
            "expected the input argument to be typed as a Tuple but"
            " found type: 'int' instead",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_output(self, input: Tuple[str]) -> int:
            return 1  # expecting Tuple[str], str, or None

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_output)

        with self.assertRaisesRegex(
            RuntimeError,
            "returned the wrong type of: 'int'",
        ):
            torch.jit.script(m)

        def pre_hook_no_output_annotation(self, input: Tuple[str]):
            return 1  # expecting Tuple[str], str, or None

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_no_output_annotation)

        with self.assertRaisesRegex(
            RuntimeError,
            "is missing a return annotation. Return annotations"
            " are required, please add one.",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_tuple_return(self, input: Tuple[Tuple[int]]) -> Tuple[int]:
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
            "expected the input argument to be typed as a Tuple "
            "but found type: 'str' instead.",
        ):
            torch.jit.script(m)

        def forward_hook_wrong_input3(self, input: Tuple[None], output: str):
            return output

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_input3)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong inner types for the input tuple"
            r" argument. Received type: 'Tuple\[NoneType\]'",
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


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
