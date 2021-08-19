import torch
from typing import List, Union
import io

#def fn(flag: bool):
#    x: Union[List[bool], int, str] = [flag, True, False]
#    if isinstance(x, List[bool]):
#        if x[0] is True:
#            x[2] = True
#    return x

def fn():
    x: Union[List[torch.Tensor], int] = [torch.tensor(3)]
    #cond: bool = torch.jit.isinstance(x, List[torch.Tensor])   this fails rippp
    if torch.jit.isinstance(x, List[torch.Tensor]):
        x.append(torch.tensor(3))
    return x

scripted = torch.jit.script(fn)
print(scripted.graph)
print(scripted.code)
#print("\n\n\n")
#buffer1 = io.BytesIO()
#torch.jit.save(scripted, buffer1)
#buffer_copy = buffer1.getvalue()
#buffer2 = io.BytesIO(buffer_copy)
#loaded = torch.jit.load(buffer2)
#print(loaded.graph)


"""
goal:
%x: prim::ListConstruct = (%a)
%y: Union[List[torch.Tensor], int] = (%x)

or something like that



"""
