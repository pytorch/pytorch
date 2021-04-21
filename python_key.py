import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._C.key as key
import torch.fx as fx
torch._C._jit_override_can_fuse_on_cpu(True)
# import torch.autograd.functional
from types import FunctionType, CodeType

HANDLED_FUNCTIONS = {}
class PythonTensor(object):
    def __init__(self, out, proxy):
        if isinstance(out, torch.Tensor):
            self.value = torch.empty_like(out)
        else:
            self.value = torch.empty(out)
        self.proxy = proxy

    def __repr__(self):
        return f"PythonTensor({tuple(self.value.shape)})"

    def tensor(self):
        return self.value

    def __torch_function__(self, func, types, args=(), kwargs={}):
        namespace, func_name = func.split("::")
        func = getattr(getattr(torch.ops, namespace), func_name)
        outs = kwargs['val']
        rets = []
        proxy_args = [i.proxy if isinstance(i, PythonTensor) else i for i in args]
        out_proxy = func(*proxy_args)
        if len(outs) == 1 and isinstance(outs[0], torch.Tensor):
            return [PythonTensor(outs[0], out_proxy)]
        for idx, out in enumerate(outs):
            if isinstance(out, torch.Tensor):
                rets.append(PythonTensor(out, out_proxy[idx]))
            else:
                rets.append(out)
        return rets

import torchvision.models as models


class ModuleBackward(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        x_grad = key.addKey(PythonTensor((1, 3, 224, 224), x))
        x_grad.requires_grad = True
        out = self.mod(x_grad).sum()
        out.backward()
        return key.removeKey(x_grad.grad).proxy

def grad(f, inps):
    def f_grad(args):
        for idx in range(len(inps)):
            args[idx] = key.addKey(PythonTensor(inps[idx].shape, args[idx]))
            args[idx].requires_grad = True
        out = f(*args)
        out.backward()
        return [key.removeKey(args[idx].grad).proxy for idx in range(len(inps))]
    if len(inps) == 1:
        def f_out(x):
            return f_grad([x])
    elif len(inps) == 2:
        def f_out(a, b):
            return f_grad([a, b])
    return f_out

def f(a, b):
    c = a*b*a*b
    return (torch.sin(c)+ a).sum()
inps = (torch.randn(30000), torch.randn(30000))
grad_f = fx.symbolic_trace(grad(f, inps))

grad_f = torch.jit.trace(grad_f, inps)
print(grad_f.graph)

iters = 1000
for _ in range(3):
        grads = grad_f(*inps)
begin = time.time()
for _ in range(iters):
    grad_f(*inps)
print(time.time() - begin)

for inp in inps:
    inp.requires_grad = True
begin = time.time()
for _ in range(iters):
    f(*inps).backward()
print(time.time() - begin)
