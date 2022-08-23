import torch
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from functorch.experimental.ops import PyOperator, fallthrough_fn
from torch.utils._pytree import tree_flatten
from torch.fx.experimental.proxy_tensor import get_isolated_graphmodule, get_proxy_slot
import torch.utils._pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
import random
import string

"""
We're going to define a `cond` operation.
In order to do this, we need implementations for each of the dispatch keys.
"""
from contextlib import contextmanager

# TODO(voz): Move out somewhere else once other py dispatched ops need it
@contextmanager
def suspend_mode(mode):
    assert(mode is not None), "Cannot suspend None mode"
    assert(isinstance(mode, TorchDispatchMode)), f"Unexpected mode type {mode.__class__}"
    torch._C._set_torch_dispatch_mode(None)
    try:
        yield
    finally:
        torch._C._set_torch_dispatch_mode(mode)

@contextmanager
def enable_mode(mode):
    curr_mode = torch._C._get_torch_dispatch_mode()
    torch._C._set_torch_dispatch_mode(mode)
    try:
        yield
    finally:
        torch._C._set_torch_dispatch_mode(curr_mode)


def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands):
    def _unwrap_proxy(e):
        return get_proxy_slot(e, proxy_mode.tracer, e, lambda e: e.proxy )

    assert isinstance(operands, list), "Cond operands must be a list of tensors"
    assert all(isinstance(o, torch.Tensor) for o in operands), "Cond operands must be a list of tensors"

    true_graph = get_isolated_graphmodule(true_fn, operands, {})
    false_graph = get_isolated_graphmodule(false_fn, operands, {})

    # There are probably better ways - I know that create_arg has some self incrementing name
    # magic to it, but since we explicitly have to get the name for register_module,
    # I was not sure how to do that. This kinda simulates it.
    next_name = None
    i = 0
    while not next_name:
        candidate = f"true_graph_{i}"
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate

    true_name = next_name
    false_name = f"false_graph_{i}"
    assert(not hasattr(proxy_mode.tracer.root, false_name))

    proxy_mode.tracer.root.register_module(true_name, true_graph)
    proxy_mode.tracer.root.register_module(false_name, false_graph)

    # with no_dispatch():
    # This is not amazing.
    # However, if we have nested operators that have a call_function
    # in their graph that is not a torch op (ex: see conditional below, nested cond)
    # we cannot get metadata for it from just looking at out vars.
    # The reason is that an operation on the output of such an op is not
    # evalauted as a torch.Tensor.
    # So we execute the real true and false fn here and compare metadata
    inp_ops = [o for o in operands]
    true_result = true_graph(operands)
    false_result = false_graph(operands)
    def recursive_compare_same(a, b):
        assert(type(a) == type(b))
        if isinstance(a, torch.Tensor):
            assert(a.dtype == b.dtype)
            assert(a.size() == b.size())
            assert(a.stride() == b.stride())
            assert(a.device == b.device)
        elif isinstance(a, (list, tuple)):
            assert(len(a) == len(b))
            for i in range(0, len(a)):
                recursive_compare_same(a[i], b[i])

    recursive_compare_same(true_result, false_result)

    args = (pred, true_graph, false_graph, [operands])

    proxy_args = pytree.tree_map(_unwrap_proxy, args)

    return proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, {},
                                            name="conditional")


def cond_dense(pred, true_fn, false_fn, operands):
    mode = torch._C._get_torch_dispatch_mode()
    assert (mode is None), "Mode should never be enabled for CPU key"
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)


def cond_autograd(pred, true_fn, false_fn, *operands):
    # TODO: support autograd
    flat_operands, _ = tree_flatten([true_fn, false_fn] + [operands])
    assert all([not f.requires_grad for f in flat_operands
                if isinstance(f, torch.Tensor)])

    guard = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.AutogradCPU))
    return cond(pred, true_fn, false_fn, *operands)


def python_fallback(op):
    def inner(pred, true_fn, false_fn, operands):
        mode = torch._C._get_torch_dispatch_mode()
        assert (mode is not None), "Mode should always be enabled for python fallback key"
        with suspend_mode(mode):
            res = trace_cond(mode, op, pred, true_fn, false_fn, operands)
        return res

    return inner


cond = PyOperator('cond')
cond.impl(DispatchKey.CPU, cond_dense)
cond.impl(DispatchKey.Python, python_fallback(cond))
cond.impl(DispatchKey.PythonTLSSnapshot, fallthrough_fn)
cond.impl(DispatchKey.AutogradCPU, cond_autograd)
cond.impl(DispatchKey.ADInplaceOrView, fallthrough_fn)
cond.impl(DispatchKey.BackendSelect, fallthrough_fn)
