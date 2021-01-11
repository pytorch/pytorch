import argparse
import os
import sys
import torch

# grab modules from test_jit_hooks.cpp
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from jit.test_hooks_modules import *


# Tests for JIT forward hooks and pre-hooks
def test_module_no_forward_input():
    m_scripted = torch.jit.script(test_module_no_forward_input_model())
    m_scripted.save(save_name + "test_module_no_forward_input" + ".pt")


def test_module_forward_multiple_inputs():
    m_scripted = torch.jit.script(test_module_forward_multiple_inputs_model())
    m_scripted.save(save_name + "test_module_forward_multiple_inputs" + ".pt")


def test_module_multiple_hooks_multiple_inputs():
    m_scripted = torch.jit.script(test_module_multiple_hooks_multiple_inputs_model())
    m_scripted.save(save_name + "test_module_multiple_hooks_multiple_inputs" + ".pt")


def test_module_forward_single_input():
    m_scripted = torch.jit.script(test_module_forward_single_input_model())
    m_scripted.save(save_name + "test_module_forward_single_input" + ".pt")


def test_module_same_hook_repeated():
    m_scripted = torch.jit.script(test_module_same_hook_repeated_model())
    m_scripted.save(save_name + "test_module_same_hook_repeated" + ".pt")


def test_module_hook_return_nothing():
    m_scripted = torch.jit.script(test_module_hook_return_nothing_model())
    m_scripted.save(save_name + "test_module_hook_return_nothing" + ".pt")


def test_module_multiple_hooks_single_input():
    m_scripted = torch.jit.script(test_module_multiple_hooks_single_input_model())
    m_scripted.save(save_name + "test_module_multiple_hooks_single_input" + ".pt")


def test_submodule_forward_multiple_inputs():
    m_scripted = torch.jit.script(test_submodule_forward_multiple_inputs_model())
    m_scripted.save(save_name + "test_submodule_forward_multiple_inputs" + ".pt")


def test_submodule_multiple_hooks_multiple_inputs():
    m_scripted = torch.jit.script(test_submodule_multiple_hooks_multiple_inputs_model())
    m_scripted.save(save_name + "test_submodule_multiple_hooks_multiple_inputs" + ".pt")


def test_submodule_forward_single_input():
    m_scripted = torch.jit.script(test_submodule_forward_single_input_model())
    m_scripted.save(save_name + "test_submodule_forward_single_input" + ".pt")


def test_submodule_same_hook_repeated():
    m_scripted = torch.jit.script(test_submodule_same_hook_repeated_model())
    m_scripted.save(save_name + "test_submodule_same_hook_repeated" + ".pt")


def test_submodule_hook_return_nothing():
    m_scripted = torch.jit.script(test_submodule_hook_return_nothing_model())
    m_scripted.save(save_name + "test_submodule_hook_return_nothing" + ".pt")


def test_submodule_multiple_hooks_single_input():
    m_scripted = torch.jit.script(test_submodule_multiple_hooks_single_input_model())
    m_scripted.save(save_name + "test_submodule_multiple_hooks_single_input" + ".pt")


def test_forward_tuple_input():
    m_scripted = torch.jit.script(test_forward_tuple_input_model())
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
