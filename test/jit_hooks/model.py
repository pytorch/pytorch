import argparse
import os.path
import sys
from typing import List, Tuple

import torch

class Model(torch.jit.ScriptModule):
    def __init__(self):
        super(Model, self).__init__()
        self.p = "hi"

    @torch.jit.script_method
    def forward(self, input):
        return self.p 


class BasicModule(torch.nn.Module):
    def __init__(self, name):
        super(BasicModule, self).__init__()
        self.name = name

    def forward(self, input: List[str]):
        input.append(self.name)
        return input

class NestedModule(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(NestedModule, self).__init__()
        self.name = name
        self.submodule = BasicModule(submodule_name)

    def forward(self, input: List[str]):
        input.append(self.name)
        return self.submodule(input)

def pre_hook(self, input: Tuple[List[str]]):
    assert self.name == 'inner_mod_name'
    assert input[0][1] == 'outer_mod_name'
    return (['pre_hook_overrid_name'],)
    
def forward_hook(self, input: Tuple[List[str]], output: List[str]): #input is always a tuple for whatever reason? 
    assert self.name == 'inner_mod_name'
    assert input[0][0] == 'pre_hook_overrid_name' # what the pre_hook overrid instead of the original 
    return output.append('aaa')

def outside_func():
    print("hello!")

class BasicModuleOS(torch.nn.Module):
    def __init__(self, name):
        super(BasicModuleOS, self).__init__()
        self.name = name

    def other_method_sub(self):
        print("other_method!")

    def forward(self, input: List[str]):
        input.append(self.name)
        self.other_method_sub()
        return input

class NestedModuleOS(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(NestedModuleOS, self).__init__()
        self.name = name
        self.submodule = BasicModuleOS(submodule_name)
        self.outsideFunc = outside_func

    def other_method_top(self):
        print("other_method!")

    def forward(self, input: List[str]):
        input.append(self.name)
        self.other_method_top()
        return self.submodule(input)

def pre_hook_os(self, input: Tuple[List[str]]):
    assert self.name == 'outer_mod_name'
    return (['pre_hook_overrid_input'],)
    
def forward_hook_os(self, input: Tuple[List[str]], output: List[str]): #input is always a tuple for whatever reason? 
    assert self.name == 'outer_mod_name'
    print(input[0][0])
    assert input[0][0] == 'pre_hook_overrid_input' # what the pre_hook overrid instead of the original 
    return output + ["overrid"]

def main():
    parser = argparse.ArgumentParser(
        description="Serialize a script module with custom ops"
    )
    parser.add_argument("--export-script-module-to", required=True)
    options = parser.parse_args()
    
    modelOS = NestedModuleOS("outer_mod_name", "inner_mod_name") 
    pre_handle = modelOS.register_forward_pre_hook(pre_hook_os)
    forward_handle = modelOS.register_forward_hook(forward_hook_os)
    modelOS = torch.jit.script(modelOS)

    modelOS(["calling arg"])

    print(modelOS.graph)
    
    modelOS.save(options.export_script_module_to)


if __name__ == "__main__":
    main()
