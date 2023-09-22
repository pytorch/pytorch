import torch
from torch._jit_internal import _Await
from torch.jit._builtins import _register_builtin

from torch.utils import set_module

set_module(_Await, "torch.jit")


def _awaitable(func, *args, **kwargs):
    r"""
    Creates Await object that will call specified functioni with specified args,
    when it is requested for the result.
    """
    return torch._C._awaitable(func, *args, **kwargs)


def _awaitable_wait(aw):
    r"""
    Requests await the result of execution, if Await is not completed yet,
    the func will be called immediately.
    """
    return torch._C._awaitable_wait(aw)


def _awaitable_nowait(o):
    r"""
    Creates completed Await with specified result.
    """
    return torch._C._awaitable_nowait(o)


_register_builtin(_awaitable_wait, "prim::awaitable_wait")
_register_builtin(_awaitable_nowait, "prim::awaitable_nowait")
