import torch
import torch.utils._pytree as pytree
from torch._ops import PyOperator
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
)

"""
Experimental implementation of JAX-like while_loop operator.
"""
while_loop = PyOperator("while_loop")

@while_loop.py_impl(DispatchKey.AutogradCUDA)
@while_loop.py_impl(DispatchKey.AutogradCPU)
def while_loop_autograd(cond_fun, body_fun, init_val):
    # TODO: support autograd
    flat_operands, _ = pytree.tree_flatten([cond_fun, body_fun, init_val])
    assert all([not f.requires_grad for f in flat_operands
                if isinstance(f, torch.Tensor)])

    _ = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.AutogradCPU))
    return while_loop(cond_fun, body_fun, init_val)


while_loop.fallthrough(DispatchKey.ADInplaceOrView)
while_loop.fallthrough(DispatchKey.BackendSelect)

@while_loop.py_impl(DispatchKey.CUDA)
@while_loop.py_impl(DispatchKey.CPU)
def while_loop_cpu(cond_fun, body_fun, init_val):
    mode = _get_current_dispatch_mode()
    assert (mode is None), "Mode should never be enabled for CPU/CUDA key"
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val
