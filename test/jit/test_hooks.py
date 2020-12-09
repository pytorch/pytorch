import os
import sys

import torch
from typing import Any, List, Tuple

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

class BasicModuleMultipleIO(torch.nn.Module):
    def __init__(self, name):
        super(BasicModuleMultipleIO, self).__init__()
        self.name = name

    def forward(self, input1: List[str], input2: str):
        input1.append(self.name)
        output2 = input2 + '_'
        return input1, output2

class NestedModuleMultipleIO(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(NestedModuleMultipleIO, self).__init__()
        self.name = name
        self.submodule = BasicModuleMultipleIO(submodule_name)

    def forward(self, input1: List[str], input2: str):
        input1.append(self.name)
        return self.submodule(input1, input2)

class DoubleInputsOutputs(torch.nn.Module):
    def __init__(self, name: str):
        super(DoubleInputsOutputs, self).__init__()
        self.name = name

    def forward(self, input1: List[str], input2: List[str]):
        input1.append(self.name)
        input2.append(self.name)
        return input1, input2

# Tests for JIT forward hooks and pre-hooks
class TestHooks(JitTestCase):

    

    def test_submodule_forward_and_pre_hooks_double_IO(self):
        m = NestedModuleMultipleIO('outer_mod_name','inner_mod_name')

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
        print(m(['a'], 'no_pre_hook'))

        scripted_m = torch.jit.script(m)
        print(scripted_m(['a'], 'no_pre_hook'))
        #self.assertEqual(scripted_m(['a'], 'no_pre_hook'), m(['a'], 'no_pre_hook'))

    def test_submodule_forward_and_pre_hooks_single_IO(self):
        m = NestedModule('outer_mod_name','inner_mod_name')

        def pre_hook(self, input: Tuple[List[str]]):
            assert self.name == 'inner_mod_name'
            assert input[0][1] == 'outer_mod_name'
            return ['pre_hook_overrid_name']
            
        def forward_hook(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            assert self.name == 'inner_mod_name'
            assert input[0][0] == 'pre_hook_overrid_name' # what the pre_hook overrid instead of the original 
            output.append('forward_hook_add')
            return output

        pre_handle = m.submodule.register_forward_pre_hook(pre_hook)
        forward_handle = m.submodule.register_forward_hook(forward_hook)
        print(m(['a']))


        scripted_m = torch.jit.script(m)
        print(scripted_m(['a']))
        #self.assertEqual(scripted_m(['a'], 'no_pre_hook'), m(['a'], 'no_pre_hook')

    
    '''
    def test_pre_hook_removal(self):
        m = BasicModule('basic')
        scripted_m = torch.jit.script(m)

        def pre_hook(self, input: Tuple[List[str]]):
            return ['1']

        pre_handle = m.register_forward_pre_hook(pre_hook)
        pre_handle_scripted = scripted_m.register_forward_pre_hook(pre_hook)
        
        self.assertEqual(['1','basic'], m(['a']))
        self.assertEqual(['1','basic'], scripted_m(['a']))

        pre_handle.remove()
        pre_hanlde_scripted.remove()        
        self.assertEqual(['a','basic'], m(['a']))
        self.assertEqual(['a','basic'], scripted_m(['a']))

    def test_pre_hook_replace_inputs(self):
        m = BasicModule('basic')
        scripted_m = torch.jit.script(m)

        def pre_hook(self, input: Tuple[List[str]]):
            return ['1']

        def pre_hook_tuple_return(self, input: Tuple[List[str]]):
            return (['1'])

        pre_handle = m.register_forward_pre_hook(pre_hook)
        pre_handle_scripted = scripted_m.register_forward_pre_hook(pre_hook)
        self.assertEqual(['1', 'basic'], m(['a']))
        self.assertEqual(['1', 'basic'], scripted_m(['a']))
        pre_handle.remove()
        pre_handle_scripted.remove()

        pre_handle = m.register_forward_pre_hook(pre_hook_tuple_return)
        pre_handle_scripted = scripted_m.register_forward_pre_hook(pre_hook_tuple_return)
        self.assertEqual(['1', 'basic'], m(['a']))
        self.assertEqual(['1', 'basic'], scripted_m(['a']))
        pre_handle.remove()
        pre_handle_scripted.remove()


    def test_pre_hook_access_module_attributes(self):
        m = BasicModule('basic')
        scripted_m = torch.jit.script(m)

        def pre_hook(self, input: Tuple[List[str]]):
            input[0].append(self.name)
            return input

        pre_handle = m.register_forward_pre_hook(pre_hook)
        pre_handle_scripted = scripted_m.register_forward_pre_hook(pre_hook)
        self.assertEqual(['basic', 'basic'], m([]))
        self.assertEqual(['basic', 'basic'], scripted_m([]))

    def test_pre_hook_multiple_hooks(self):
        m = BasicModule('basic')
        scripted_m = torch.jit.script(m)

        def pre_hook1(self, input: Tuple[List[str]]):
            input[0].append('1')
            return input
        
        def pre_hook2(self, input: Tuple[List[str]]):
            input[0].append('2')
            return input
        
        def pre_hook3(self, input: Tuple[List[str]]):
            input[0].append('3')
            return input

        pre_handle1 = m.register_forward_pre_hook(pre_hook1)
        pre_handle2 = m.register_forward_pre_hook(pre_hook2)
        pre_handle3 = m.register_forward_pre_hook(pre_hook3)
        pre_handle1_scripted = m.register_forward_pre_hook(pre_hook1)
        pre_handle2_scripted = m.register_forward_pre_hook(pre_hook2)
        pre_handle3_scripted = m.register_forward_pre_hook(pre_hook3)
        self.assertEqual(['1', '2', '3', 'basic'], m([]))
        self.assertEqual(['1', '2', '3', 'basic'], scripted_m([]))

        pre_handle2.remove()
        pre_handle2_scripted.remove()
        self.assertEqual(['1', '3', 'basic'], m([]))
        self.assertEqual(['1', '3', 'basic'], scripted_m([]))

    def test_pre_hook_submodule(self):
        m = NestedModule('outer_mod_name','inner_mod_name')
        scripted_m = torch.jit.script(m)

        def pre_hook(self, input: Tuple[List[str]]):
            input[0].append(self.name)
            return input
        
        pre_handle = m.submodule.register_forward_pre_hook(pre_hook)
        pre_handle_scripted = scripted_m.submodule.register_forward_pre_hook(pre_hook)
        self.assertEqual(['outer_mod_name', 'inner_mod_name', 'inner_mod_name'], m([]))
        self.assertEqual(['outer_mod_name', 'inner_mod_name', 'inner_mod_name'], scripted_m([]))
        
        pre_handle.remove()
        pre_handle_scripted.remove()
        self.assertEqual(['outer_mod_name', 'inner_mod_name'], m([]))
        self.assertEqual(['outer_mod_name', 'inner_mod_name'], scripted_m([]))


    def test_pre_hook_double_inputs(self):
        m = DoubleInputsOutputs('DoubleInputsOutputs')
        scripted_m = torch.jit.script(m)

        def pre_hook(self, input: Tuple[List[str], List[str]]):
            return (['1'],['2'])

        pre_handle = m.register_forward_pre_hook(pre_hook)
        pre_handle_scripted = scripted_m.register_forward_pre_hook(pre_hook)
        self.assertEqual((['1', 'DoubleInputsOutputs'], ['2', 'DoubleInputsOutputs']), m(['a'],['b']))
        self.assertEqual((['1', 'DoubleInputsOutputs'], ['2', 'DoubleInputsOutputs']), scripted_m(['a'],['b']))


    def test_forward_hook_removal(self):
        m = BasicModule('basic')
        scripted_m = torch.jit.script(m)

        def forward_hook(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            return ['1']

        forward_handle = m.register_forward_hook(forward_hook)
        forward_handle_scripted = scripted_m.register_forward_hook(forward_hook)
        self.assertEqual(['1'], m([]))
        self.assertEqual(['1'], scripted_m([]))

        forward_handle.remove()
        forward_handle_scripted.remove()
        self.assertEqual(['basic'], m([]))
        self.assertEqual(['basic'], scripted_m([]))

    def test_forward_hook_access_module_attributes(self):
        m = BasicModule('basic')
        scripted_m = torch.jit.script(m)

        def forward_hook(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            assert input[0][0] == 'a'
            assert self.name == 'basic'
            assert output[1] == 'basic' # TODO why isn't the output in a tuple?

        forward_handle = m.register_forward_hook(forward_hook)
        forward_handle_scripted = scripted_m.register_forward_hook(forward_hook)
        self.assertEqual(['a', 'basic'], m(['a']))
        self.assertEqual(['a', 'basic'], scripted_m(['a']))
    

    def test_forward_hook_replace_outputs(self):
        m = BasicModule('basic')
        scripted_m = torch.jit.script(m)

        def forward_hook(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            return ['1']

        def forward_hook_tuple_return(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            return (['1'])

        forward_handle = m.register_forward_hook(forward_hook)
        forward_handle_scripted = scripted_m.register_forward_hook(forward_hook)
        self.assertEqual(['1'], m([]))
        self.assertEqual(['1'], scripted_m([]))
        forward_handle.remove()
        forward_handle_scripted.remove()

        forward_handle = m.register_forward_hook(forward_hook_tuple_return)
        forward_handle_scripted = scripted_m.register_forward_hook(forward_hook_tuple_return)
        self.assertEqual(['1'], m([]))
        self.assertEqual(['1'], scripted_m([]))


    def test_forward_hook_multiple_hooks(self):
        m = BasicModule('basic')
        scripted_m = torch.jit.script(m)

        def forward_hook1(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            return output.append('1')
        
        def forward_hook2(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            return output.append('2')
        
        def forward_hook3(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            return output.append('3')

        forward_handle1 = m.register_forward_hook(forward_hook1)
        forward_handle2 = m.register_forward_hook(forward_hook2)
        forward_handle3 = m.register_forward_hook(forward_hook3)
        forward_handle1_scripted = scripted_m.register_forward_hook(forward_hook1)
        forward_handle2_scripted = scripted_m.register_forward_hook(forward_hook2)
        forward_handle3_scripted = scripted_m.register_forward_hook(forward_hook3)
        self.assertEqual(['basic', '1', '2', '3'], m([]))
        self.assertEqual(['basic', '1', '2', '3'], scripted_m([]))

        forward_handle2.remove()
        forward_handle2_scripted.remove()
        self.assertEqual(['basic', '1', '3'], m([]))
        self.assertEqual(['basic', '1', '3'], scripted_m([]))

    def test_forward_hook_multiple_hooks_submodules(self):
        m = NestedModule('outer_mod_name', 'inner_mod_name')
        #scripted_m = torch.jit.script(m)

        def forward_hook1(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            return output.append('1')
        
        def forward_hook2(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            return output.append('2')
        
        def forward_hook3(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            return output.append('3')

        forward_handle1 = m.submodule.register_forward_hook(forward_hook1)
        forward_handle2 = m.submodule.register_forward_hook(forward_hook2)
        forward_handle3 = m.submodule.register_forward_hook(forward_hook3)
        forward_handle1_scripted = scripted_m.submodule.register_forward_hook(forward_hook1)
        forward_handle2_scripted = scripted_m.submodule.register_forward_hook(forward_hook2)
        forward_handle3_scripted = scripted_m.submodule.register_forward_hook(forward_hook3)
        self.assertEqual(['outer_mod_name', 'inner_mod_name', '1', '2', '3'], m([]))
        self.assertEqual(['outer_mod_name', 'inner_mod_name', '1', '2', '3'], scripted_m([]))

        forward_handle2.remove()
        forward_handle2_scripted.remove()
        print(m([]))
        self.assertEqual(['outer_mod_name', 'inner_mod_name', '1', '3'], m([]))
        self.assertEqual(['outer_mod_name', 'inner_mod_name', '1', '3'], scripted_m([]))
        

    def test_forward_hook_submodule(self):
        m = NestedModule('outer_mod_name','inner_mod_name')
        scripted_m = torch.jit.script(m)

        def forward_hook(self, input: Tuple[List[str]], output: Tuple[List[str]]):
            return [self.name]
        
        forward_handle = m.submodule.register_forward_hook(forward_hook)
        forward_handle_scripted = scripted_m.submodule.register_forward_hook(forward_hook)
        self.assertEqual(['inner_mod_name'], m([]))
        self.assertEqual(['inner_mod_name'], scripted_m([]))
        
        forward_handle.remove()
        forward_handle_scripted.remove()
        self.assertEqual(['outer_mod_name', 'inner_mod_name'], m([]))
        self.assertEqual(['outer_mod_name', 'inner_mod_name'], scripted_m([]))

        
    def test_forward_hook_double_inputs_outputs(self):
        m = DoubleInputsOutputs('DoubleInputsOutputs')
        scripted_m = torch.jit.script(m)

        def forward_hook(self, input: Tuple[List[str], List[str]], output: Tuple[List[str], List[str]]):
            assert input[1][0] == 'b'
            return (['1'],['2'])

        forward_handle = m.register_forward_hook(forward_hook)
        forward_handle_scripted = scripted_m.register_forward_hook(forward_hook)
        self.assertEqual((['1'],['2']), m(['a'],['b']))
        self.assertEqual((['1'],['2']), scripted_m(['a'],['b']))

    def test_forward_and_pre_hooks(self):
        m = BasicModule('basic')
        scripted_m = torch.jit.script(m)

        def pre_hook(self, input: Tuple[List[str]]):
            return ['1']
            
        def forward_hook(self, input: Tuple[List[str], List[str]], output: Tuple[List[str], List[str]]):
            assert input[0][0] == '1'
            return ['2']

        pre_handle = m.register_forward_pre_hook(pre_hook)
        pre_handle_scripted = scripted_m.register_forward_pre_hook(pre_hook)
        forward_handle = m.register_forward_hook(forward_hook)
        forward_handle_scripted = scripted_m.register_forward_hook(forward_hook)

        self.assertEqual(['2'], m(['a']))
        self.assertEqual(['2'], scripted_m(['a']))

    def test_forward_and_pre_hooks_submodule(self):
        m = NestedModule('outer_mod_name','inner_mod_name')
        scripted_m = torch.jit.script(m)

        def pre_hook(self, input: Tuple[List[str]]):
            assert self.name == 'inner_mod_name'
            assert input[0][1] == 'outer_mod_name'
            return ['1']
            
        def forward_hook(self, input: Tuple[List[str], List[str]], output: Tuple[List[str], List[str]]):
            assert self.name == 'inner_mod_name'
            assert input[0][0] == '1'
            return ['2']

        pre_handle = m.submodule.register_forward_pre_hook(pre_hook)
        pre_handle_scripted = scripted_m.submodule.register_forward_pre_hook(pre_hook)
        forward_handle = m.submodule.register_forward_hook(forward_hook)
        forward_handle_scripted = scripted_m.submodule.register_forward_hook(forward_hook)

        self.assertEqual(['2'], m(['a']))
        self.assertEqual(['2'], scripted_m(['a']))
    '''


        
    
