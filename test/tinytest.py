import torch
from typing import Union, Optional
import io

@torch.jit.script
def fn(flag: int) -> Union[str, int, None]:
    y: Union[int, str, None] = "foo"
    if flag == 0:
        x: Optional[Union[int, str]] = y
    elif flag == 1:
        x: Optional[Union[int, str]] = 1
    else:
        x: Optional[Union[int, str]] = None
    return x

print(fn.code)
buffer = io.BytesIO()
torch.jit.save(fn, buffer)
torch.jit.save(fn, "s.pt")
buffer_copy = buffer.getvalue()
buffer2 = io.BytesIO(buffer_copy)
l = torch.jit.load(buffer2)
print("\n\n\n")
print(type(l.forward))
print(l.code)


'''
ScriptMethod
/home/ansley/local/pytorch/torch/csrc/jit/serialization/python_print.cpp
"code",
[](Method& self) {
std::vector<at::IValue> constants;
PrintDepsTable deps;
PythonPrint pp(constants, deps);
pp.printMethod(self.function());
return pp.str();
})

'''

"""
def fn(flag: int) -> Union[int, NoneType, str]:
  if torch.eq(flag, 0):
    x : Optional[Union[int, str]] = "foo"
  else:
    if torch.eq(flag, 1):
      x0 : Optional[int] = 1
    else:
      x0 = None
    x = x0
  return x


(pytorch) [ansley@devvm1612.frc0 ~/local/pytorch] cat s/code/__torch__.py
class PlaceholderModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.PlaceholderModule,
    flag: int) -> Optional[Union[int, str]]:
    if torch.eq(flag, 0):
      x : Optional[Union[int, str]] = "foo"
    else:
      if torch.eq(flag, 1):
        x0 : Optional[int] = 1
      else:
        x0 = None
      x = x0
    return x


def forward(self,
    flag: int) -> Optional[Union[int, str]]:
  if torch.eq(flag, 0):
    x : Union[int, NoneType, str] = "foo"
  else:
    if torch.eq(flag, 1):
      x0 : Optional[int] = 1
    else:
      x0 = None
    x = x0
  return x




"""
