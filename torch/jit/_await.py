import torch

from torch.utils import set_module
from torch.jit._builtins import _register_builtin
from torch._jit_internal import Await

set_module(Await, "torch.jit")

def _awaitable(func, *args, **kwargs):
    return torch._C._awaitable(func, *args, **kwargs)

def _awaitable_wait(aw):
    return torch._C._awaitable_wait(aw)

def _awaitable_nowait(o):
    return torch._C._awaitable_nowait(o)


_register_builtin(_awaitable_wait, "aten::awaitable_wait")
_register_builtin(_awaitable_nowait, "aten::awaitable_nowait")
