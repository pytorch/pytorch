import torch
from collections import namedtuple
from typing import List, Tuple

class InnerModule(torch.nn.Module):
    def __init__(self, name):
        super(InnerModule, self).__init__()
        self.name = name

    def forward(self, input: List[str]):
        input.append(self.name)
        return input

class OuterModule(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super(OuterModule, self).__init__()
        self.name = name
        self.submodule = InnerModule(submodule_name)

    def forward(self, input: List[str]):
        input.append(self.name)
        return self.submodule(input)

m = OuterModule('outer_mod_name','inner_mod_name')

def pre_hook(self, input: Tuple[List[str]]):
    return (['pre_hook_overrid_input'],)
    
def forward_hook_works_in_eager(self, input: Tuple[List[str]], output: List[str]):
    return ['aaa']
    #return output.append('aaa') # eager gets a List[str] so this is valid in eager, 
                                # TS is expecting a Tuple[List[str]] so it's not valid in TS
                    # TS error: Tried to access nonexistent attribute or method 'append' 
                    # of type 'Tuple[List[str]]'

def forward_hook_works_in_TS_and_eager(self, input: Tuple[List[str]], output: List[str]):
    output[0].append('aaa')
    return ['aaaaa']
    #return output[0].append('aaa') # eager gets a List[str] so this in not valid in eager, 
                                   # works with how TS will interper type of output
                    # eager Error: AttributeError: 'str' object has no attribute 'append'


# note: forward_hook output *is not in a tuple* if it is a single arg, is a tuple otherwise 
    


pre_handle = m.submodule.register_forward_pre_hook(pre_hook)
forward_handle = m.submodule.register_forward_hook(forward_hook_works_in_eager)

print(m(['a']))

scripted_m = torch.jit.script(m)
print(scripted_m(['a']))
