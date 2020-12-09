# HOOK support currently
import torch
from collections import namedtuple
from typing import List, Tuple
'''
def freeHAHA(input: int = 2):
    print(input)
    print("haha2")


class BasicModule(torch.nn.Module):
    def __init__(self, name):
        super(BasicModule, self).__init__()
        self.name = name
        self.freeHAHA = freeHAHA
    
    @torch.jit.export
    def exportedHAHA(self, input: int = 3):
        print(self.name)
        print("exportHAHA")

    def methodHAHA(self):
        print(self.name)
        print("methodHAHA")

    def forward(self, input: List[str]):
        input.append(self.name)
        self.exportedHAHA(2)
        self.freeHAHA()
        self.methodHAHA()
        return input

class BasicMod(torch.nn.Module):
    def __init__(self, name: str):
        super(BasicMod, self).__init__()
        self.name = name
    
    def forward(self, input1: int, input2: int):
        #print(f"{input1}, {input2}")
        input1 = input1 + 1
        input2 = input2 + 1
        #print(f"{input1}, {input2}")
        return input1, input2

class TopMod(torch.nn.Module):
    def __init__(self, name: str):
        super(TopMod, self).__init__()
        self.name = name
        self.submod = BasicMod("basic_submod")

    def forward(self, input1: int, input2: int):
       input1 = input1 + 1
       input2 = input2 + 1 
       #input3, input4 = self.submod(input1, input2)
       return self.submod(input1, input2)

def pre_hook_submod(self, input: Tuple[int, int]):
    return (input[0] * 3, input[1] * 2)

def pre_hook_submod2(self, input: Tuple[int, int]):
    return (input[0] * 3, input[1] * 2)

def hook_submod(self, input: Tuple[int, int], output: Tuple[int, int]):
    return (output[0] * 10, 100)
    #return output[0] * 10

tm = TopMod("topmod")
tm.submod.register_forward_pre_hook(pre_hook_submod)
tm.submod.register_forward_pre_hook(pre_hook_submod2)
tm.submod.register_forward_hook(hook_submod)

scripted_tm = torch.jit.script(tm)
print(scripted_tm.graph)

print(f"eager: {tm(1,1)}")
print(f"TS: {scripted_tm(1,1)}")
'''

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


m = NestedModuleMultipleIO('outer_mod_name','inner_mod_name')

#scripted_no_hooks = torch.jit.script(NestedModuleMultipleIO('outer_mod_name','inner_mod_name'))
#print(f"single IO pre hooks: {scripted_no_hooks(['a'], 'no hooks!')}")

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

print("********** double IO submod hooks ************")
scripted_m = torch.jit.script(m)
print(f"eager:\n  {m(['a'], 'no_pre_hook')}")
print(f"scripted:\n  {scripted_m(['a'], 'no_pre_hook')}")
assert m(['a'], 'no_pre_hook') == scripted_m(['a'], 'no_pre_hook') 
print("********** end double IO submod hooks ************\n")


print("********** double IO no modifications submod hooks ************")
m = NestedModuleMultipleIO('outer_mod_name','inner_mod_name')
def pre_hook2(self, input: Tuple[List[str], str]):
    assert self.name == 'inner_mod_name'
    assert input[0][1] == 'outer_mod_name'
    
def forward_hook2(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
    assert self.name == 'inner_mod_name'
pre_handle = m.submodule.register_forward_pre_hook(pre_hook2)
forward_handle = m.submodule.register_forward_hook(forward_hook2)
scripted_m = torch.jit.script(m)
print(f"eager:\n  {m(['a'], 'no_pre_hook')}")
print(f"scripted:\n  {scripted_m(['a'], 'no_pre_hook')}")
assert m(['a'], 'no_pre_hook') == scripted_m(['a'], 'no_pre_hook') 
print("********** end double IO no modifications submod hooks ************\n")


# TODO check combination of no mods and mods 

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

m = NestedModule('outer_mod_name','inner_mod_name')

def pre_hook(self, input: Tuple[List[str]]):
    assert self.name == 'inner_mod_name'
    assert input[0][1] == 'outer_mod_name'
    return (['pre_hook_overrid_name'],)
    
def forward_hook(self, input: Tuple[List[str]], output: List[str]): #input is always a tuple for whatever reason? 
    assert self.name == 'inner_mod_name'
    assert input[0][0] == 'pre_hook_overrid_name' # what the pre_hook overrid instead of the original 
    return output.append('aaa')

scripted_no_hooks = torch.jit.script(NestedModule('outer_mod_name','inner_mod_name'))
print(f"single IO pre hooks: {scripted_no_hooks(['a'])}")

pre_handle = m.submodule.register_forward_pre_hook(pre_hook)
forward_handle = m.submodule.register_forward_hook(forward_hook)
print(m(['a']))

print("********** single IO submod hooks ************")
scripted_m = torch.jit.script(m)
print(f"eager:\n  {m(['a'])}")
print(f"scripted:\n  {scripted_m(['a'])}")
assert m(['a']) == scripted_m(['a']) 
print("********** end single IO submod hooks ************\n")

print("********** single IO no modifications submod hooks ************")
m = NestedModule('outer_mod_name','inner_mod_name')
def pre_hook3(self, input: Tuple[List[str]]):
    assert self.name == 'inner_mod_name'
    assert input[0][1] == 'outer_mod_name'
    
def forward_hook4(self, input: Tuple[List[str]], output: List[str]): #input is always a tuple for whatever reason? 
    assert self.name == 'inner_mod_name'

pre_handle = m.submodule.register_forward_pre_hook(pre_hook3)
forward_handle = m.submodule.register_forward_hook(forward_hook4)
scripted_m = torch.jit.script(m)
print(f"eager:\n  {m(['a'])}")
print(f"scripted:\n  {scripted_m(['a'])}")
assert m(['a']) == scripted_m(['a']) 
print("********** end single IO no modifications submod hooks ************\n")

class BasicModuleOS(torch.nn.Module):
    def __init__(self, name):
        super(BasicModuleOS, self).__init__()
        self.name = name

    def forward(self, input: List[str]):
        input.append(self.name)
        return input

class NestedModuleOS(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(NestedModuleOS, self).__init__()
        self.name = name
        self.submodule = BasicModuleOS(submodule_name)

    def forward(self, input: List[str]):
        input.append(self.name)
        return self.submodule(input)

def pre_hook6(self, input: Tuple[List[str]]):
    assert self.name == 'outer_mod_name'
    return (['pre_hook_overrid_name'],)
    
def forward_hook7(self, input: Tuple[List[str]], output: List[str]): #input is always a tuple for whatever reason? 
    assert self.name == 'outer_mod_name'
    assert input[0][0] == 'pre_hook_overrid_name' # what the pre_hook overrid instead of the original 
    return output + ["overrid"]

print("********** single IO no modifications mainmod hooks ************")
m = NestedModuleOS('outer_mod_name','inner_mod_name')
pre_handle = m.register_forward_pre_hook(pre_hook6)
forward_handle = m.register_forward_hook(forward_hook7)
print(f"eager: {m(['a'])}")
m_scripted = torch.jit.script(m)
print(f"scripted: {m_scripted(['a'])}")
assert m(['a']) == m_scripted(['a']) 
print("********** single IO no modifications mainmod hooks ************")

class BasicModuleMultipleIO_OS(torch.nn.Module):
    def __init__(self, name):
        super(BasicModuleMultipleIO_OS, self).__init__()
        self.name = name

    def forward(self, input1: List[str], input2: str):
        input1.append(self.name)
        output2 = input2 + '_'
        return input1, output2

class NestedModuleMultipleIO_OS(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(NestedModuleMultipleIO_OS, self).__init__()
        self.name = name
        self.submodule = BasicModuleMultipleIO_OS(submodule_name)

    def forward(self, input1: List[str], input2: str):
        input1.append(self.name)
        return self.submodule(input1, input2)

def pre_hook8(self, input: Tuple[List[str], str]):
    assert self.name == 'outer_mod_name'
    return ['pre_hook_overrid_name'], 'pre_hook_override'
    
def forward_hook9(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
    assert self.name == 'outer_mod_name'
    assert input[0][0] == 'pre_hook_overrid_name' # what the pre_hook overrid instead of the original 
    output2 = output[1] + 'fh'
    return output[0], output2

m = NestedModuleMultipleIO_OS('outer_mod_name','inner_mod_name')
pre_handle = m.register_forward_pre_hook(pre_hook8)
forward_handle = m.register_forward_hook(forward_hook9)

print("********** double IO mainmod hooks ************")
scripted_m = torch.jit.script(m)
print(f"eager:\n  {m(['a'], 'no_pre_hook')}")
print(f"scripted:\n  {scripted_m(['a'], 'no_pre_hook')}")
assert m(['a'], 'no_pre_hook') == scripted_m(['a'], 'no_pre_hook') 
print("********** end double IO mainmod hooks ************\n")



'''
Master Rewrite TODO:
- check multiple hooks compile
- rewrite get_method() class for CallMethod cleanup
- check that default arguments work 
- rewrite cu define in IR to not be so gacky
- rewrite ClassType get_methods again to not be so gacky 

Implement TODO:
- C++ useage
- error messages?
- test cases
- serialization

'''








