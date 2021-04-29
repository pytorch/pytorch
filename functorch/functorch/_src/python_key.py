import functools
import torch._C.key as key
from torch.fx import PythonTensor
import torch

class ModuleWrap(torch.nn.Module):
    def __init__(self, mod, inps):
        super().__init__()
        self.mod = mod
        self.inps = inps
        @functools.wraps(mod.forward)
        def forward_wrapped(self, *args):
            new_args = []
            for inp, arg in zip(inps, args):
                if isinstance(inp, torch.Tensor):
                    new_arg = key.addKey(PythonTensor(inp.shape, arg))
                else:
                    new_arg = inp
                new_args.append(new_arg)
            out = self.mod(*new_args)
            return key.removeKey(out).proxy

        type(self).forward = forward_wrapped

def key_wrap(f, inps):
    @functools.wraps(f)
    def wrapped(*args):
        new_args = []
        for inp, arg in zip(inps, args):
            if isinstance(inp, torch.Tensor):
                new_arg = key.addKey(PythonTensor(inp.shape, arg))
            else:
                new_arg = inp
            new_args.append(new_arg)
        out = f(*new_args)
        if key.hasKey(out):
            return key.removeKey(out).proxy
        else:
            return out
    return wrapped