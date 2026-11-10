# Owner(s): ["oncall: jit"]

from typing import List, Tuple

import torch


# Register hooks during construction, before ScriptModule finalizes compilation.
class SubmoduleNoForwardInputs(torch.jit.ScriptModule):
    def __init__(self, name, *, forward_pre_hooks=(), forward_hooks=()):
        super().__init__()
        self.name = name
        self.define(
            "def forward(self):\n"
            "    if self.name != 'inner_mod_name':\n"
            "        raise AssertionError(\"Expected name 'inner_mod_name', got '\""
            ' + self.name + "\'")\n'
        )
        for hook in forward_pre_hooks:
            self.register_forward_pre_hook(hook)
        for hook in forward_hooks:
            self.register_forward_hook(hook)


class ModuleNoForwardInputs(torch.jit.ScriptModule):
    def __init__(
        self,
        name: str,
        submodule_name: str,
        *,
        forward_pre_hooks=(),
        forward_hooks=(),
        submodule_forward_pre_hooks=(),
        submodule_forward_hooks=(),
    ):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleNoForwardInputs(
            submodule_name,
            forward_pre_hooks=submodule_forward_pre_hooks,
            forward_hooks=submodule_forward_hooks,
        )
        self.define("def forward(self):\n    self.submodule()\n")
        for hook in forward_pre_hooks:
            self.register_forward_pre_hook(hook)
        for hook in forward_hooks:
            self.register_forward_hook(hook)


class SubmoduleForwardSingleInput(torch.jit.ScriptModule):
    def __init__(self, name, *, forward_pre_hooks=(), forward_hooks=()):
        super().__init__()
        self.name = name
        self.define("def foo(self, input: str):\n    return input\n")
        self.define(
            "def forward(self, input: str):\n"
            "    input = input + '_inner_mod'\n"
            "    input = self.foo(input)\n"
            "    return input\n"
        )
        for hook in forward_pre_hooks:
            self.register_forward_pre_hook(hook)
        for hook in forward_hooks:
            self.register_forward_hook(hook)


class ModuleForwardSingleInput(torch.jit.ScriptModule):
    def __init__(
        self,
        name: str,
        submodule_name: str,
        *,
        forward_pre_hooks=(),
        forward_hooks=(),
        submodule_forward_pre_hooks=(),
        submodule_forward_hooks=(),
    ):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardSingleInput(
            submodule_name,
            forward_pre_hooks=submodule_forward_pre_hooks,
            forward_hooks=submodule_forward_hooks,
        )
        self.define(
            "def forward(self, input: str):\n"
            "    input = input + '_outermod'\n"
            "    return self.submodule(input)\n"
        )
        for hook in forward_pre_hooks:
            self.register_forward_pre_hook(hook)
        for hook in forward_hooks:
            self.register_forward_hook(hook)


class ModuleDirectforwardSubmodCall(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardSingleInput(submodule_name)

    def forward(self, input: str):
        input = input + "_outermod"
        return self.submodule.forward(input)


class SuboduleForwardMultipleInputs(torch.jit.ScriptModule):
    def __init__(self, name, *, forward_pre_hooks=(), forward_hooks=()):
        super().__init__()
        self.name = name
        self.define(
            "def forward(self, input1: List[str], input2: str):\n"
            "    input1.append(self.name)\n"
            "    output2 = input2 + '_'\n"
            "    return (input1, output2)\n"
        )
        for hook in forward_pre_hooks:
            self.register_forward_pre_hook(hook)
        for hook in forward_hooks:
            self.register_forward_hook(hook)


class ModuleForwardMultipleInputs(torch.jit.ScriptModule):
    def __init__(
        self,
        name: str,
        submodule_name: str,
        *,
        forward_pre_hooks=(),
        forward_hooks=(),
        submodule_forward_pre_hooks=(),
        submodule_forward_hooks=(),
    ):
        super().__init__()
        self.name = name
        self.submodule = SuboduleForwardMultipleInputs(
            submodule_name,
            forward_pre_hooks=submodule_forward_pre_hooks,
            forward_hooks=submodule_forward_hooks,
        )
        self.define(
            "def forward(self, input1: List[str], input2: str):\n"
            "    input1.append(self.name)\n"
            "    return self.submodule(input1, input2)\n"
        )
        for hook in forward_pre_hooks:
            self.register_forward_pre_hook(hook)
        for hook in forward_hooks:
            self.register_forward_hook(hook)


class SubmoduleForwardTupleInput(torch.jit.ScriptModule):
    def __init__(self, name, *, forward_pre_hooks=(), forward_hooks=()):
        super().__init__()
        self.name = name
        self.define(
            "def forward(self, input: Tuple[int]):\n"
            "    input_access = input[0]\n"
            "    return (1,)\n"
        )
        for hook in forward_pre_hooks:
            self.register_forward_pre_hook(hook)
        for hook in forward_hooks:
            self.register_forward_hook(hook)


class ModuleForwardTupleInput(torch.jit.ScriptModule):
    def __init__(
        self,
        name: str,
        submodule_name: str,
        *,
        forward_pre_hooks=(),
        forward_hooks=(),
        submodule_forward_pre_hooks=(),
        submodule_forward_hooks=(),
    ):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardTupleInput(
            submodule_name,
            forward_pre_hooks=submodule_forward_pre_hooks,
            forward_hooks=submodule_forward_hooks,
        )
        self.define(
            "def forward(self, input: Tuple[int]):\n"
            "    input_access = input[0]\n"
            "    return self.submodule((1,))\n"
        )
        for hook in forward_pre_hooks:
            self.register_forward_pre_hook(hook)
        for hook in forward_hooks:
            self.register_forward_hook(hook)


# Modules for JIT forward hook and pre-hooks python and cpp tests
def create_module_no_forward_input():
    # Use to test module level hooks with no forward input

    def pre_hook(self, input: Tuple[()]) -> None:
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")

    def forward_hook(self, input: Tuple[()], output: None):
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")

    return ModuleNoForwardInputs(
        "outer_mod_name",
        "inner_mod_name",
        forward_pre_hooks=(pre_hook,),
        forward_hooks=(forward_hook,),
    )


def create_submodule_no_forward_input():
    # Use to test submodule level hooks with no forward input

    def pre_hook(self, input: Tuple[()]) -> None:
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")

    def forward_hook(self, input: Tuple[()], output: None):
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")

    return ModuleNoForwardInputs(
        "outer_mod_name",
        "inner_mod_name",
        submodule_forward_pre_hooks=(pre_hook,),
        submodule_forward_hooks=(forward_hook,),
    )


def create_module_forward_multiple_inputs():
    # Use to test module level hooks with forward having multiple
    # inputs and returns

    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "a":
            raise AssertionError(f"Expected input[0][0] 'a', got '{input[0][0]}'")
        return ["pre_hook_override_name"], "pre_hook_override"

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "pre_hook_override_name":
            raise AssertionError(
                f"Expected input[0][0] 'pre_hook_override_name', got '{input[0][0]}'"
            )
        output2 = output[1] + "fh"
        return output[0], output2

    return ModuleForwardMultipleInputs(
        "outer_mod_name",
        "inner_mod_name",
        forward_pre_hooks=(pre_hook,),
        forward_hooks=(forward_hook,),
    )


def create_module_multiple_hooks_multiple_inputs():
    # Use to test that module level hooks with multiple inputs execute
    # in correct order and pass correct information between each other

    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "a":
            raise AssertionError(f"Expected input[0][0] 'a', got '{input[0][0]}'")
        return ["pre_hook_override_name"], "pre_hook_override"

    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "pre_hook_override_name":
            raise AssertionError(
                f"Expected input[0][0] 'pre_hook_override_name', got '{input[0][0]}'"
            )
        return ["pre_hook_override_name2"], "pre_hook_override"

    def forward_hook1(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "pre_hook_override_name2":
            raise AssertionError(
                f"Expected input[0][0] 'pre_hook_override_name2', got '{input[0][0]}'"
            )
        output2 = output[1] + "fh1"
        return output[0], output2

    def forward_hook2(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "pre_hook_override_name2":
            raise AssertionError(
                f"Expected input[0][0] 'pre_hook_override_name2', got '{input[0][0]}'"
            )
        if output[1] != "pre_hook_override_fh1":
            raise AssertionError(
                f"Expected output[1] 'pre_hook_override_fh1', got '{output[1]}'"
            )
        output2 = output[1] + "_fh2"
        return output[0], output2

    return ModuleForwardMultipleInputs(
        "outer_mod_name",
        "inner_mod_name",
        forward_pre_hooks=(pre_hook1, pre_hook2),
        forward_hooks=(forward_hook1, forward_hook2),
    )


def create_module_forward_single_input():
    # Use to test module level hooks for forward with single input

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0] != "a":
            raise AssertionError(f"Expected input[0] 'a', got '{input[0]}'")
        return ("pre_hook_override_name",)

    def forward_hook(self, input: Tuple[str], output: str):
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name',), got {input}"
            )
        output = output + "_fh"
        return output

    return ModuleForwardSingleInput(
        "outer_mod_name",
        "inner_mod_name",
        forward_pre_hooks=(pre_hook,),
        forward_hooks=(forward_hook,),
    )


def create_module_same_hook_repeated():
    # Use to test module can run same hook multiple times

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        input_change = input[0] + "_ph"
        return (input_change,)

    def forward_hook(self, input: Tuple[str], output: str):
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input != ("a_ph_ph",):
            raise AssertionError(f"Expected input ('a_ph_ph',), got {input}")
        output = output + "_fh"
        return output

    return ModuleForwardSingleInput(
        "outer_mod_name",
        "inner_mod_name",
        forward_pre_hooks=(pre_hook, pre_hook),
        forward_hooks=(forward_hook, forward_hook),
    )


def create_module_hook_return_nothing():
    # Use to test module level hooks that return nothing

    def pre_hook(self, input: Tuple[str]) -> None:
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0] != "a":
            raise AssertionError(f"Expected input[0] 'a', got '{input[0]}'")

    def forward_hook(self, input: Tuple[str], output: str):
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input != ("a",):
            raise AssertionError(f"Expected input ('a',), got {input}")

    return ModuleForwardSingleInput(
        "outer_mod_name",
        "inner_mod_name",
        forward_pre_hooks=(pre_hook,),
        forward_hooks=(forward_hook,),
    )


def create_module_multiple_hooks_single_input():
    # Use to test that modules can run multiple hooks with single input

    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0] != "a":
            raise AssertionError(f"Expected input[0] 'a', got '{input[0]}'")
        return ("pre_hook_override_name1",)

    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0] != "pre_hook_override_name1":
            raise AssertionError(
                f"Expected input[0] 'pre_hook_override_name1', got '{input[0]}'"
            )
        return ("pre_hook_override_name2",)

    def forward_hook1(self, input: Tuple[str], output: str):
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name2",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name2',), got {input}"
            )
        if output != "pre_hook_override_name2_outermod_inner_mod":
            raise AssertionError(
                f"Expected output 'pre_hook_override_name2_outermod_inner_mod', got '{output}'"
            )
        output = output + "_fh1"
        return output, output

    def forward_hook2(self, input: Tuple[str], output: Tuple[str, str]):
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name2",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name2',), got {input}"
            )
        if output[0] != "pre_hook_override_name2_outermod_inner_mod_fh1":
            raise AssertionError(
                f"Expected output[0] 'pre_hook_override_name2_outermod_inner_mod_fh1', got '{output[0]}'"
            )
        output = output[0] + "_fh2"
        return output

    return ModuleForwardSingleInput(
        "outer_mod_name",
        "inner_mod_name",
        forward_pre_hooks=(pre_hook1, pre_hook2),
        forward_hooks=(forward_hook1, forward_hook2),
    )


def create_submodule_forward_multiple_inputs():
    # Use to test that submodules can run hooks that have multiple forward inputs

    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0][1] != "outer_mod_name":
            raise AssertionError(
                f"Expected input[0][1] 'outer_mod_name', got '{input[0][1]}'"
            )
        return ["pre_hook_override_name"], "pre_hook_override"

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0][0] != "pre_hook_override_name":
            raise AssertionError(
                f"Expected input[0][0] 'pre_hook_override_name', got '{input[0][0]}'"
            )
        output2 = output[1] + "fh"
        return output[0], output2

    return ModuleForwardMultipleInputs(
        "outer_mod_name",
        "inner_mod_name",
        submodule_forward_pre_hooks=(pre_hook,),
        submodule_forward_hooks=(forward_hook,),
    )


def create_submodule_multiple_hooks_multiple_inputs():
    # Use to test that submodules can run multiple hooks with multiple
    # forward inputs

    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[1] != "no_pre_hook":
            raise AssertionError(f"Expected input[1] 'no_pre_hook', got '{input[1]}'")
        return ["pre_hook_override_name"], "pre_hook_override1"

    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[1] != "pre_hook_override1":
            raise AssertionError(
                f"Expected input[1] 'pre_hook_override1', got '{input[1]}'"
            )
        return ["pre_hook_override_name"], "pre_hook_override2"

    def forward_hook1(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[1] != "pre_hook_override2":
            raise AssertionError(
                f"Expected input[1] 'pre_hook_override2', got '{input[1]}'"
            )
        if output[1] != "pre_hook_override2_":
            raise AssertionError(
                f"Expected output[1] 'pre_hook_override2_', got '{output[1]}'"
            )
        output2 = output[1] + "fh1"
        return output[0], output2, output2

    def forward_hook2(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str, str]
    ):
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[1] != "pre_hook_override2":
            raise AssertionError(
                f"Expected input[1] 'pre_hook_override2', got '{input[1]}'"
            )
        if output[1] != "pre_hook_override2_fh1":
            raise AssertionError(
                f"Expected output[1] 'pre_hook_override2_fh1', got '{output[1]}'"
            )
        output2 = output[1] + "_fh2"
        return output[0], output2

    return ModuleForwardMultipleInputs(
        "outer_mod_name",
        "inner_mod_name",
        submodule_forward_pre_hooks=(pre_hook1, pre_hook2),
        submodule_forward_hooks=(forward_hook1, forward_hook2),
    )


def create_submodule_forward_single_input():
    # Use to test that submodules can run hooks with a single argument
    # passed to forward

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0] != "a_outermod":
            raise AssertionError(f"Expected input[0] 'a_outermod', got '{input[0]}'")
        return ("pre_hook_override_name",)

    def forward_hook(self, input: Tuple[str], output: str):
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name',), got {input}"
            )
        return output

    return ModuleForwardSingleInput(
        "outer_mod_name",
        "inner_mod_name",
        submodule_forward_pre_hooks=(pre_hook,),
        submodule_forward_hooks=(forward_hook,),
    )


def create_submodule_to_call_directly_with_hooks():
    # Use to test that submodules have their hooks invoked when called
    # directly

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        return ("pre_hook_override_name",)

    def forward_hook(self, input: Tuple[str], output: str):
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name',), got {input}"
            )
        return output + "_fh"

    return ModuleForwardSingleInput(
        "outer_mod_name",
        "inner_mod_name",
        submodule_forward_pre_hooks=(pre_hook,),
        submodule_forward_hooks=(forward_hook,),
    )


def create_submodule_same_hook_repeated():
    # Use to test that submodules can run same hooks multiple times

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        changed = input[0] + "_ph"
        return (changed,)

    def forward_hook(self, input: Tuple[str], output: str):
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("a_outermod_ph_ph",):
            raise AssertionError(f"Expected input ('a_outermod_ph_ph',), got {input}")
        return output + "_fh"

    return ModuleForwardSingleInput(
        "outer_mod_name",
        "inner_mod_name",
        submodule_forward_pre_hooks=(pre_hook, pre_hook),
        submodule_forward_hooks=(forward_hook, forward_hook),
    )


def create_submodule_hook_return_nothing():
    # Use to test that submodules can run hooks that return nothing

    def pre_hook(self, input: Tuple[str]) -> None:
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0] != "a_outermod":
            raise AssertionError(f"Expected input[0] 'a_outermod', got '{input[0]}'")

    def forward_hook(self, input: Tuple[str], output: str):
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("a_outermod",):
            raise AssertionError(f"Expected input ('a_outermod',), got {input}")

    return ModuleForwardSingleInput(
        "outer_mod_name",
        "inner_mod_name",
        submodule_forward_pre_hooks=(pre_hook,),
        submodule_forward_hooks=(forward_hook,),
    )


def create_submodule_multiple_hooks_single_input():
    # Use to test that submodules can run multiple hooks that have a single input

    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0] != "a_outermod":
            raise AssertionError(f"Expected input[0] 'a_outermod', got '{input[0]}'")
        return ("pre_hook_override_name",)

    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0] != "pre_hook_override_name":
            raise AssertionError(
                f"Expected input[0] 'pre_hook_override_name', got '{input[0]}'"
            )
        return ("pre_hook_override_name2",)

    def forward_hook1(self, input: Tuple[str], output: str):
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name2",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name2',), got {input}"
            )
        if output != "pre_hook_override_name2_inner_mod":
            raise AssertionError(
                f"Expected output 'pre_hook_override_name2_inner_mod', got '{output}'"
            )
        return output + "_fwh1"

    def forward_hook2(self, input: Tuple[str], output: str):
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name2",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name2',), got {input}"
            )
        if output != "pre_hook_override_name2_inner_mod_fwh1":
            raise AssertionError(
                f"Expected output 'pre_hook_override_name2_inner_mod_fwh1', got '{output}'"
            )
        return output

    return ModuleForwardSingleInput(
        "outer_mod_name",
        "inner_mod_name",
        submodule_forward_pre_hooks=(pre_hook1, pre_hook2),
        submodule_forward_hooks=(forward_hook1, forward_hook2),
    )


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

    return ModuleForwardTupleInput(
        "outer_mod_name",
        "inner_mod_name",
        forward_pre_hooks=(pre_hook_outermod,),
        submodule_forward_pre_hooks=(pre_hook_innermod,),
        forward_hooks=(forward_hook_outermod,),
        submodule_forward_hooks=(forward_hook_innermod,),
    )


def create_submodule_forward_single_input_return_not_tupled():
    # Use to check that submodules can return modified inputs
    # that aren't wrapped in a tuple (to match eager behavior)

    def pre_hook(self, input: Tuple[str]) -> str:
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0] != "a_outermod":
            raise AssertionError(f"Expected input[0] 'a_outermod', got '{input[0]}'")
        # return is wrapped in tuple in other test cases
        return "pre_hook_override_name"

    def forward_hook(self, input: Tuple[str], output: str):
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name',), got {input}"
            )
        output = output + "_fh"
        return output

    return ModuleForwardSingleInput(
        "outer_mod_name",
        "inner_mod_name",
        submodule_forward_pre_hooks=(pre_hook,),
        submodule_forward_hooks=(forward_hook,),
    )


if __name__ == "__main__":
    raise RuntimeError(
        "This file is a collection of utils, it should be imported not executed directly"
    )
