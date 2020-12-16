import argparse
import os.path
import sys
from typing import List, Tuple

import torch


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
        assert self.name == "outer_mod_name_changed_by_pre_hook"
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
def test_module_hook_and_pre_hook_no_IO():
    m = OuterModuleNoIO("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[None]):
        assert self.name == "outer_mod_name"
        self.name = "outer_mod_name_changed_by_pre_hook"

    def forward_hook(self, input: Tuple[None], output: None):
        assert self.name == "outer_mod_name_changed_by_pre_hook"

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    m_scripted = torch.jit.script(m)
    # note: if you run this module more than once it will fail
    m_scripted.save(save_name + "test_module_hook_and_pre_hook_no_IO" + ".pt")


def test_module_hook_and_pre_hook_multiple_IO():
    m = OuterModuleMultipleIO("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[List[str], str]):
        assert self.name == "outer_mod_name"
        assert input[0][0] == "a"
        return ["pre_hook_overrid_name"], "pre_hook_override"

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        assert self.name == "outer_mod_name"
        assert input[0][0] == "pre_hook_overrid_name"
        output2 = output[1] + "fh"
        return output[0], output2

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_module_hook_and_pre_hook_multiple_IO" + ".pt")


def test_module_hook_and_pre_hook_multiple_IO_multiple_hooks():
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

    m_scripted = torch.jit.script(m)
    m_scripted.save(
        save_name + "test_module_hook_and_pre_hook_multiple_IO_multiple_hooks" + ".pt"
    )


def test_module_forward_and_pre_hook_single_IO():
    m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

    def pre_hook_(self, input: Tuple[str]):
        assert self.name == "outer_mod_name"
        assert input[0] == "a"
        return ("pre_hook_overrid_name",)

    def forward_hook_(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("pre_hook_overrid_name",)
        output = output + "_fh"
        return output

    m.register_forward_pre_hook(pre_hook_)
    m.register_forward_hook(forward_hook_)

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_module_forward_and_pre_hook_single_IO" + ".pt")


def test_module_forward_and_pre_hook_single_IO_same_hook_twice():
    m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

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
    m_scripted.save(
        save_name + "test_module_forward_and_pre_hook_single_IO_same_hook_twice" + ".pt"
    )


def test_module_forward_and_pre_hook_single_IO_no_change():
    m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]):
        assert self.name == "outer_mod_name"
        assert input[0] == "a"

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("a",)

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    m_scripted = torch.jit.script(m)
    m_scripted.save(
        save_name + "test_module_forward_and_pre_hook_single_IO_no_change" + ".pt"
    )


def test_module_forward_and_pre_hook_single_IO_multiple_hooks():
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

    m_scripted = torch.jit.script(m)
    m_scripted.save(
        save_name + "test_module_forward_and_pre_hook_single_IO_multiple_hooks" + ".pt"
    )


def test_submodule_hook_and_pre_hook_multiple_IO():
    m = OuterModuleMultipleIO("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[List[str], str]):
        assert self.name == "inner_mod_name"
        assert input[0][1] == "outer_mod_name"
        return ["pre_hook_overrid_name"], "pre_hook_override"

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        assert self.name == "inner_mod_name"
        assert input[0][0] == "pre_hook_overrid_name"
        output2 = output[1] + "fh"
        return output[0], output2

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    m_scripted = torch.jit.script(m)
    m_scripted.save(save_name + "test_submodule_hook_and_pre_hook_multiple_IO" + ".pt")


def test_submodule_hook_and_pre_hook_multiple_IO_multiple_hooks():
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

    m_scripted = torch.jit.script(m)
    m_scripted.save(
        save_name
        + "test_submodule_hook_and_pre_hook_multiple_IO_multiple_hooks"
        + ".pt"
    )


def test_submodule_forward_and_pre_hooks_single_IO():
    m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

    def pre_hook_single(self, input: Tuple[str]):
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"
        return ("pre_hook_overrid_name",)

    def forward_hook_single(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_overrid_name",)
        return output

    m.submodule.register_forward_pre_hook(pre_hook_single)
    m.submodule.register_forward_hook(forward_hook_single)

    m_scripted = torch.jit.script(m)

    m_scripted.save(
        save_name + "test_submodule_forward_and_pre_hooks_single_IO" + ".pt"
    )


def test_submodule_forward_and_pre_hook_single_IO_same_hook_twice():
    m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

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
    m_scripted.save(
        save_name
        + "test_submodule_forward_and_pre_hook_single_IO_same_hook_twice"
        + ".pt"
    )


def test_submodule_forward_and_pre_hooks_single_IO_no_change():
    m = OuterModuleSingleIO("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]):
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("a_outermod",)

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    m_scripted = torch.jit.script(m)
    m_scripted.save(
        save_name + "test_submodule_forward_and_pre_hooks_single_IO_no_change" + ".pt"
    )


def test_submodule_forward_and_pre_hooks_single_IO_multiple_hooks():
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

    m_scripted = torch.jit.script(m)
    m_scripted.save(
        save_name
        + "test_submodule_forward_and_pre_hooks_single_IO_multiple_hooks"
        + ".pt"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Serialize a script module with custom ops"
    )
    parser.add_argument("--export-script-module-to", required=True)
    options = parser.parse_args()
    global save_name
    save_name = options.export_script_module_to + "_"

    test_submodule_forward_and_pre_hooks_single_IO_multiple_hooks()
    test_submodule_forward_and_pre_hooks_single_IO_no_change()
    test_submodule_forward_and_pre_hook_single_IO_same_hook_twice()
    test_submodule_forward_and_pre_hooks_single_IO()
    test_submodule_hook_and_pre_hook_multiple_IO_multiple_hooks()
    test_submodule_hook_and_pre_hook_multiple_IO()

    test_module_forward_and_pre_hook_single_IO_multiple_hooks()
    test_module_forward_and_pre_hook_single_IO_no_change()
    test_module_forward_and_pre_hook_single_IO_same_hook_twice()
    test_module_forward_and_pre_hook_single_IO()
    test_module_hook_and_pre_hook_multiple_IO_multiple_hooks()
    test_module_hook_and_pre_hook_multiple_IO()

    test_module_hook_and_pre_hook_no_IO()


if __name__ == "__main__":
    main()
