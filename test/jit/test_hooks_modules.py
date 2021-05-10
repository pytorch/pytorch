import torch
from typing import List, Tuple


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


# Modules for JIT forward hook and pre-hooks python and cpp tests
def create_module_no_forward_input():
    # Use to test module level hooks with no forward input
    m = ModuleNoForwardInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[()]) -> None:
        assert self.name == "outer_mod_name"

    def forward_hook(self, input: Tuple[()], output: None):
        assert self.name == "outer_mod_name"

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_submodule_no_forward_input():
    # Use to test submodule level hooks with no forward input
    m = ModuleNoForwardInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[()]) -> None:
        assert self.name == "inner_mod_name"

    def forward_hook(self, input: Tuple[()], output: None):
        assert self.name == "inner_mod_name"

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_module_forward_multiple_inputs():
    # Use to test module level hooks with forward having multiple
    # inputs and returns
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "outer_mod_name"
        assert input[0][0] == "a"
        return ["pre_hook_override_name"], "pre_hook_override"

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        assert self.name == "outer_mod_name"
        assert input[0][0] == "pre_hook_override_name"
        output2 = output[1] + "fh"
        return output[0], output2

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_module_multiple_hooks_multiple_inputs():
    # Use to test that module level hooks with multiple inputs execute
    # in correct order and pass correct information between each other
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "outer_mod_name"
        assert input[0][0] == "a"
        return ["pre_hook_override_name"], "pre_hook_override"

    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
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

    return m


def create_module_forward_single_input():
    # Use to test module level hooks for forward with single input
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
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

    return m


def create_module_same_hook_repeated():
    # Use to test module can run same hook multiple times
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "outer_mod_name"
        input_change = input[0] + "_ph"
        return (input_change,)

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("a_ph_ph",)
        output = output + "_fh"
        return output

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_module_hook_return_nothing():
    # Use to test module level hooks that return nothing
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> None:
        assert self.name == "outer_mod_name"
        assert input[0] == "a"

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("a",)

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_module_multiple_hooks_single_input():
    # Use to test that modules can run multiple hooks with single input
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "outer_mod_name"
        assert input[0] == "a"
        return ("pre_hook_override_name1",)

    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
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

    return m


def create_submodule_forward_multiple_inputs():
    # Use to test that submodules can run hooks that have multiple forward inputs
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "inner_mod_name"
        assert input[0][1] == "outer_mod_name"
        return ["pre_hook_override_name"], "pre_hook_override"

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        assert self.name == "inner_mod_name"
        assert input[0][0] == "pre_hook_override_name"
        output2 = output[1] + "fh"
        return output[0], output2

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_multiple_hooks_multiple_inputs():
    # Use to test that submodules can run multiple hooks with multiple
    # forward inputs
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "inner_mod_name"
        assert input[1] == "no_pre_hook"
        return ["pre_hook_override_name"], "pre_hook_override1"

    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
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
        return output[0], output2

    m.submodule.register_forward_pre_hook(pre_hook1)
    m.submodule.register_forward_pre_hook(pre_hook2)
    m.submodule.register_forward_hook(forward_hook1)
    m.submodule.register_forward_hook(forward_hook2)

    return m


def create_submodule_forward_single_input():
    # Use to test that submodules can run hooks with a single argument
    # passed to forward
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"
        return ("pre_hook_override_name",)

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_override_name",)
        return output

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_to_call_directly_with_hooks():
    # Use to test that submodules have their hooks invoked when called
    # directly
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "inner_mod_name"
        return ("pre_hook_override_name",)

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_override_name",)
        return output + "_fh"

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_same_hook_repeated():
    # Use to test that submodules can run same hooks multiple times
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "inner_mod_name"
        changed = input[0] + "_ph"
        return (changed,)

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("a_outermod_ph_ph",)
        return output + "_fh"

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_hook_return_nothing():
    # Use to test that submodules can run hooks that return nothing
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> None:
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("a_outermod",)

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_multiple_hooks_single_input():
    # Use to test that submodules can run multiple hooks that have a single input
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"
        return ("pre_hook_override_name",)

    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
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

    return m


def create_forward_tuple_input():
    # Use to test case where forward is passed a single tuple for input.
    # This is different because eager always wraps pre-hook return arguments
    # in a tuple when the returned pre-hook result isn't a tuple
    # (to allow the result to be passed to another pre-hook if needed).
    # The eager behavior doesn't wrap the single tuple input pre-hook return in a
    # tuple as it should. To get consistent behavior between single tuple inputs and
    # the rest of the possible forward inputs, pre-hooks need to
    # wrap single tuple inputs returns in another tuple. This is
    # enforced by the schema checker.
    m = ModuleForwardTupleInput("outer_mod_name", "inner_mod_name")

    def pre_hook_outermod(self, input: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        # 'return (11,)' doesn't work with eager, inner tuple lost
        return ((11,),)

    def pre_hook_innermod(self, input: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
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

    return m


def create_submodule_forward_single_input_return_not_tupled():
    # Use to check that submodules can return modified inputs
    # that aren't wrapped in a tuple (to match eager behavior)
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> str:
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

    return m
