import torch
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from functorch.experimental.ops import PyOperator
from torch.utils._pytree import tree_flatten

"""
We're going to define a `cond` operation.
In order to do this, we need implementations for each of the dispatch keys.
"""
from contextlib import contextmanager

@contextmanager
def suspend_mode(mode):
    torch._C._set_torch_dispatch_mode(None)
    yield
    torch._C._set_torch_dispatch_mode(mode)


def cond_dense(pred, true_fn, false_fn, operands):
    mode = torch._C._get_torch_dispatch_mode()
    if mode:
        with suspend_mode(mode):
            args = (pred, true_fn, false_fn, operands)
            return mode.__torch_dispatch__(cond, None, args, {})
    try:
        if pred:
            return true_fn(operands)
        else:
            return false_fn(operands)
    except Exception as e:
        # Do something proper here, someday
        print("Exception", e)



def cond_autograd(pred, true_fn, false_fn, *operands):
    # TODO: support autograd
    flat_operands, _ = tree_flatten((true_fn, false_fn) + operands)
    assert all([not f.requires_grad for f in flat_operands
                if isinstance(f, torch.Tensor)])

    guard = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.AutogradCPU))
    return cond(pred, true_fn, false_fn, *operands)


def cond_adinplaceorview(*args, **kwargs):
    guard = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.ADInplaceOrView))
    return cond(*args, **kwargs)


def python_fallback(op):
    def inner(*args, **kwargs):
        mode = torch._C._get_torch_dispatch_mode()

        if mode:
            with suspend_mode(mode):
                return mode.__torch_dispatch__(op, None, args, kwargs)
        else:
            return cond_dense(*args)

    return inner


cond = PyOperator('cond')
cond.impl(DispatchKey.CPU, cond_dense, True)
cond.impl(DispatchKey.AutogradCPU, cond_autograd)
cond.fallthrough(DispatchKey.ADInplaceOrView)
cond.fallthrough(DispatchKey.BackendSelect)

cond.impl(DispatchKey.Python, python_fallback(cond))
cond.fallthrough(DispatchKey.PythonTLSSnapshot)
