import argparse
import os.path
import sys
from typing import List, Tuple

import torch


class InnerModuleSingleIO(torch.nn.Module):
    def __init__(self, name):
        super(InnerModuleSingleIO, self).__init__()
        self.name = name

    def forward(self, input: str):
        input = input + "_inner_mod"
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
        output2 = input2 + '_'
        return input1, output2

class OuterModuleMultipleIO(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(OuterModuleMultipleIO, self).__init__()
        self.name = name
        self.submodule = InnerModuleMultipleIO(submodule_name)

    def forward(self, input1: List[str], input2: str):
        input1.append(self.name)
        return self.submodule(input1, input2)

'''
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

def pre_hook_s(self, input: Tuple[List[str]]):
    assert self.name == 'inner_mod_name'
    return (['pre_hook_overrid_input'],)
    
def forward_hook_s(self, input: Tuple[List[str]], output: List[str]): #input is always a tuple for whatever reason? 
    assert self.name == 'inner_mod_name'
    print(input[0][0])
    assert input[0][0] == 'pre_hook_overrid_input' # what the pre_hook overrid instead of the original 
    return output + ["overrid"]

'''

def test_module_hook_and_pre_hook_double_IO(self):
        m = OuterModuleMultipleIO('outer_mod_name','inner_mod_name')

        def pre_hook(self, input: Tuple[List[str], str]):
            assert self.name == 'outer_mod_name'
            assert input[0][0] == 'a'
            return ['pre_hook_overrid_name'], 'pre_hook_override'
            
        def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
            assert self.name == 'outer_mod_name'
            assert input[0][0] == 'pre_hook_overrid_name' # what the pre_hook overrid instead of the original 
            output2 = output[1] + 'fh'
            return output[0], output2

        pre_handle = m.register_forward_pre_hook(pre_hook)
        forward_handle = m.register_forward_hook(forward_hook)

        m_scripted = torch.jit.script(m)
        m_scripted.save(options.export_script_module_to + "test_module_hook_and_pre_hook_double_IO")

def test_module_forward_and_pre_hooks_single_IO(self):
    m = OuterModuleSingleIO('outer_mod_name','inner_mod_name')

    def pre_hook(self, input: Tuple[str]):
        assert self.name == 'outer_mod_name'
        assert input[0] == 'a'
        return ('pre_hook_overrid_name',)
        
    def forward_hook(self, input: Tuple[str], output: str):
        # note: 'output' of forward hook needs to not be wrapped in tuple
        # when there is a single element in the forward's return 
        # this is to match eager's behavior 
        assert self.name == 'outer_mod_name'
        assert input == ('pre_hook_overrid_name',)
        output = output + "_fh"
        return output

    pre_handle = m.register_forward_pre_hook(pre_hook)
    forward_handle = m.register_forward_hook(forward_hook)

    self.checkModule(m, ('a',))

def test_submodule_hook_and_pre_hook_double_IO(self):
    m = OuterModuleMultipleIO('outer_mod_name','inner_mod_name')

    def pre_hook(self, input: Tuple[List[str], str]):
        assert self.name == 'inner_mod_name'
        assert input[0][1] == 'outer_mod_name'
        return ['pre_hook_overrid_name'], 'pre_hook_override'
        
    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        assert self.name == 'inner_mod_name'
        assert input[0][0] == 'pre_hook_overrid_name' # what the pre_hook overrid instead of the original 
        output2 = output[1] + 'fh'
        return output[0], output2

    pre_handle = m.submodule.register_forward_pre_hook(pre_hook)
    forward_handle = m.submodule.register_forward_hook(forward_hook)

    self.checkModule(m, (['a'], 'no_pre_hook'))

def test_submodule_forward_and_pre_hooks_single_IO(self):
    m = OuterModuleSingleIO('outer_mod_name','inner_mod_name')

    def pre_hook(self, input: Tuple[str]):
        assert self.name == 'inner_mod_name'
        assert input[0] == 'a_outermod'
        return ('pre_hook_overrid_name',)
        
    def forward_hook(self, input: Tuple[str], output: str):
        # note: 'output' of forward hook needs to not be wrapped in tuple
        # when there is a single element in the forward's return 
        # this is to match eager's behavior 
        assert self.name == 'inner_mod_name'
        assert input == ('pre_hook_overrid_name',)
        return output

    pre_handle = m.submodule.register_forward_pre_hook(pre_hook)
    forward_handle = m.submodule.register_forward_hook(forward_hook)

    self.checkModule(m, ('a',))

def test_submodule_forward_and_pre_hooks_single_IO_multiple_hooks(self):
    m = OuterModuleSingleIO('outer_mod_name','inner_mod_name')

    def pre_hook1(self, input: Tuple[str]):
        assert self.name == 'inner_mod_name'
        assert input[0] == 'a_outermod'
        return ('pre_hook_overrid_name',)

    def pre_hook2(self, input: Tuple[str]):
        assert self.name == 'inner_mod_name'
        assert input[0] == 'pre_hook_overrid_name'
        return ('pre_hook_overrid_name2',)
        
    def forward_hook(self, input: Tuple[str], output: str):
        # note: 'output' of forward hook needs to not be wrapped in tuple
        # when there is a single element in the forward's return 
        # this is to match eager's behavior 
        assert self.name == 'inner_mod_name'
        assert input == ('pre_hook_overrid_name2',)
        assert output == 'pre_hook_overrid_name2_inner_mod'
        return output


    pre_handle = m.submodule.register_forward_pre_hook(pre_hook1)
    pre_handle = m.submodule.register_forward_pre_hook(pre_hook2)
    forward_handle = m.submodule.register_forward_hook(forward_hook)

    print(torch.jit.script(m)('a'))

def main():
    parser = argparse.ArgumentParser(
        description="Serialize a script module with custom ops"
    )
    parser.add_argument("--export-script-module-to", required=True)
    options = parser.parse_args()
    global save_name
    save_name = options.export_script_module_to + "_"
    
    '''
    modelOS = NestedModuleOS("outer_mod_name", "inner_mod_name") 
    pre_handle = modelOS.register_forward_pre_hook(pre_hook_os)
    forward_handle = modelOS.register_forward_hook(forward_hook_os)
    modelOS = torch.jit.script(modelOS)
    modelOS(["calling arg"])
    print(modelOS.graph)
    modelOS.save(options.export_script_module_to)
    

    modelOS = NestedModuleOS("outer_mod_name", "inner_mod_name") 
    pre_handle = modelOS.submodule.register_forward_pre_hook(pre_hook_s)
    forward_handle = modelOS.submodule.register_forward_hook(forward_hook_s)
    print(f"eager result: {modelOS(['calling arg'])}")
    modelOS = torch.jit.script(modelOS)
    print(modelOS.graph)
    modelOS.save(options.export_script_module_to)
    

    m = OuterModuleSingleIO('outer_mod_name','inner_mod_name')

    def pre_hook1(self, input: Tuple[str]):
        assert self.name == 'inner_mod_name'
        assert input[0] == 'a_outermod'
        return ('pre_hook_overrid_name',)

    def pre_hook2(self, input: Tuple[str]):
        assert self.name == 'inner_mod_name'
        assert input[0] == 'pre_hook_overrid_name'
        return ('pre_hook_overrid_name2',)
        
    def forward_hook(self, input: Tuple[str], output: str):
        # note: 'output' of forward hook needs to not be wrapped in tuple
        # when there is a single element in the forward's return 
        # this is to match eager's behavior 
        assert self.name == 'inner_mod_name'
        assert input == ('pre_hook_overrid_name2',)
        assert output == 'pre_hook_overrid_name2_inner_mod'
        return output


    pre_handle = m.submodule.register_forward_pre_hook(pre_hook1)
    pre_handle = m.submodule.register_forward_pre_hook(pre_hook2)
    forward_handle = m.submodule.register_forward_hook(forward_hook)
    
    m = OuterModuleMultipleIO('outer_mod_name','inner_mod_name')
    

    def pre_hook(self, input: Tuple[List[str], str]):
        assert self.name == 'outer_mod_name'
        assert input[0][0] == 'a'
        
    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        assert self.name == 'outer_mod_name'
        assert input[0][0] == 'a' 

    pre_handle = m.register_forward_pre_hook(pre_hook)
    forward_handle = m.register_forward_hook(forward_hook)

    m_scripted = torch.jit.script(m)
    print(m_scripted.graph)
    m_scripted.save(options.export_script_module_to)
    '''
    m = OuterModuleSingleIO('outer_mod_name','inner_mod_name')

    def pre_hook(self, input: Tuple[str]):
        assert self.name == 'outer_mod_name'
        return ('pre_hook_overrid_name',)
        
    def forward_hook(self, input: Tuple[str], output: str):
        # note: 'output' of forward hook needs to not be wrapped in tuple
        # when there is a single element in the forward's return 
        # this is to match eager's behavior 
        assert self.name == 'outer_mod_name'
        assert input == ('pre_hook_overrid_name',)
        output = output + "_fh"
        return output

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)
    m.register_forward_hook(forward_hook)

    m_scripted = torch.jit.script(m)
    print(m_scripted.graph)
    m_scripted.save(options.export_script_module_to)

    m_scripted_loaded = torch.jit.load(options.export_script_module_to)
    print(m_scripted_loaded.graph) 
    m_scripted_loaded.save(options.export_script_module_to)

    print(m_scripted('a'))
    print(m_scripted_loaded('a'))



if __name__ == "__main__":
    main()
