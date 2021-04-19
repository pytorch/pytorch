import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._C.key as key

HANDLED_FUNCTIONS = {}
class PythonTensor(torch.Tensor):
    def __init__(self, shape):
        self.value = torch.empty(shape)

    def __repr__(self):
        return f"PythonTensor({tuple(self.value.shape)})"

    def tensor(self):
        return self.value

    def __torch_function__(self, func, types, args=(), kwargs={}):
        print("torch_function: ", func)
        out = kwargs['val']
        if isinstance(out, torch.Tensor):
            return PythonTensor(out.shape)
        else:
            return out

import torchvision.models as models
x = PythonTensor((5))

class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(3, 3)

    def forward(self, x):
        return F.linear(x, torch.ones(3, 3))

model = Foo()
def f(x):
    out = torch.dot(x, torch.ones(5))
    out.backward()
    return out
    # out = (x*2).sum()
    # out.backward()
grad_x = key.addKey(x)
grad_x.requires_grad = True
f(grad_x)
# out = torch.vmap(f)(grad_x)
# print(torch.vmap(f)(key.addKey(x)))
# print(torch.jit.trace(torch.vmap(f),(x)))
# print(key.removeKey(out))
# import pdb; pdb.set_trace()