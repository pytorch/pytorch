import argparse
import os
import sys
from typing import List, Tuple
import torch

# grab modules from test_jit_hooks.cpp
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from jit.test_hooks import *


# Tests for JIT forward hooks and pre-hooks
def test_module_no_forward_input():
    m = ModuleNoForwardInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[None]):
        assert self.name == "outer_mod_name"
        self.name = "outer_mod_name_changed_by_pre_hook"

    def forward_hook(self, input: Tuple[None], output: None):
        assert self.name == "outer_mod_name_changed_by_pre_hook"

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_module_no_forward_input" + ".pt")


def test_module_forward_multiple_inputs():
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[List[str], str]):
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

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_module_forward_multiple_inputs" + ".pt")


def test_module_multiple_hooks_multiple_inputs():
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

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_module_multiple_hooks_multiple_inputs" + ".pt")


def test_module_forward_single_input():
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook_(self, input: Tuple[str]):
        assert self.name == "outer_mod_name"
        assert input[0] == "a"
        return "pre_hook_override_name"

    def forward_hook_(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("pre_hook_override_name",)
        output = output + "_fh"
        return output

    m.register_forward_pre_hook(pre_hook_)
    m.register_forward_hook(forward_hook_)

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_module_forward_single_input" + ".pt")


def test_module_same_hook_repeated():
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]):
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

    scripted = torch.jit.script(m)

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_module_same_hook_repeated" + ".pt")


def test_module_hook_return_nothing():
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]):
        assert self.name == "outer_mod_name"
        assert input[0] == "a"

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("a",)

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_module_hook_return_nothing" + ".pt")


def test_module_multiple_hooks_single_input():
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
        return output[0] + "_fh2"

    m.register_forward_pre_hook(pre_hook1)
    m.register_forward_pre_hook(pre_hook2)
    m.register_forward_hook(forward_hook1)
    m.register_forward_hook(forward_hook2)

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_module_multiple_hooks_single_input" + ".pt")


def test_submodule_forward_multiple_inputs():
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[List[str], str]):
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

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_submodule_forward_multiple_inputs" + ".pt")


def test_submodule_multiple_hooks_multiple_inputs():
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
        return output[0], output2

    m.submodule.register_forward_pre_hook(pre_hook1)
    m.submodule.register_forward_pre_hook(pre_hook2)
    m.submodule.register_forward_hook(forward_hook1)
    m.submodule.register_forward_hook(forward_hook2)

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_submodule_multiple_hooks_multiple_inputs" + ".pt")


def test_submodule_forward_single_input():
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]):
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"
        return "pre_hook_override_name"

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_override_name",)
        return output

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    m_scripted = torch.jit.script(m)

    m_scripted.save(save_name + "test_submodule_forward_single_input" + ".pt")


def test_submodule_same_hook_repeated():
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]):
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

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_submodule_same_hook_repeated" + ".pt")


def test_submodule_hook_return_nothing():
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]):
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("a_outermod",)

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_submodule_hook_return_nothing" + ".pt")


def test_submodule_multiple_hooks_single_input():
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

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_submodule_multiple_hooks_single_input" + ".pt")


def test_forward_tuple_input():
    m = ModuleForwardTupleInput("outer_mod_name", "inner_mod_name")

    def pre_hook_outermod(self, input: Tuple[Tuple[int]]):
        # 'return (11,)' doesn't work with eager, inner tuple lost
        return ((11,),)

    def pre_hook_innermod(self, input: Tuple[Tuple[int]]):
        # 'return (22,)' doesn't work with eager, inner tuple lost
        return ((22,),)

    def forward_hook_outermod(self, input: Tuple[Tuple[int]], output: int):
        return 22

    def forward_hook_innermod(self, input: Tuple[Tuple[int]], output: Tuple[int]):
        return 33

    m.register_forward_pre_hook(pre_hook_outermod)
    m.submodule.register_forward_pre_hook(pre_hook_innermod)
    m.register_forward_hook(forward_hook_outermod)
    m.submodule.register_forward_hook(forward_hook_innermod)

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_forward_tuple_input" + ".pt")


def main():
    parser = argparse.ArgumentParser(
        description="Serialize a script modules with hooks attached"
    )
    parser.add_argument("--export-script-module-to", required=True)
    options = parser.parse_args()
    global save_name
    save_name = options.export_script_module_to + "_"

    test_submodule_forward_single_input()
    test_submodule_forward_multiple_inputs()
    test_submodule_multiple_hooks_single_input()
    test_submodule_multiple_hooks_multiple_inputs()
    test_submodule_hook_return_nothing()
    test_submodule_same_hook_repeated()

    test_module_forward_single_input()
    test_module_forward_multiple_inputs()
    test_module_multiple_hooks_single_input()
    test_module_multiple_hooks_multiple_inputs()
    test_module_hook_return_nothing()
    test_module_same_hook_repeated()

    test_module_no_forward_input()
    test_forward_tuple_input()

    print("OK: completed saving modules with hooks!")


if __name__ == "__main__":
    main()
