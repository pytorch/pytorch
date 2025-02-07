import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses import FakeTensorMode


aten = torch.ops.aten


class PrintingMode(TorchDispatchMode):
  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    res = func(*args, **kwargs)
    if func == aten.convolution_backward.default:
        print(f"{func.__module__}.{func.__name__}({args}, {kwargs})")
        print(f"res={res}")
    return res


with FakeTensorMode():
    with PrintingMode():
        conv = nn.Conv2d(64, 64, 3, padding=1).train()
        x = torch.randn(1, 64, 32, 32)
        w = conv.weight
        b = conv.bias
        print(w.grad)
        res = torch.nn.functional.conv2d(x, w, b, padding=1)

        dres = torch.rand_like(res)
        res.backward(dres)
        print(w.grad)
