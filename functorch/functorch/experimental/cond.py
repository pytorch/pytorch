import torch
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from functorch.experimental.ops import PyOperator
from torch.utils._pytree import tree_flatten
from torch.fx.experimental.proxy_tensor import ProxyTensor, get_isolated_graphmodule
import torch.utils._pytree as pytree
from torch.dispatch.dispatcher import dispatcher_singleton

"""
We're going to define a `cond` operation.
In order to do this, we need implementations for each of the dispatch keys.
"""
from contextlib import contextmanager

# TODO(voz): Move out somewhere else once other py dispatched ops need it
@contextmanager
def suspend_mode(mode):
    torch._C._set_torch_dispatch_mode(None)
    try:
        yield
    finally:
        torch._C._set_torch_dispatch_mode(mode)


def trace_cond(proxy_mode, func_overload, args, kwargs=None):
    assert kwargs is None or not kwargs
    pred, true_fn, false_fn, operands = args
    # We only get to this step if we are (1) in tracing (2) In the right dispatch step
    # As this op is recursively re-entrant, we need to reset the record
    # TODO(voz): Make this feel better?
    dispatcher_singleton.reset_dispatch_record()

    if isinstance(operands, ProxyTensor):
        operands = [operands]  # Little hack because * on a single ProxyTensor unpacks it
    else:
        operands = operands

    true_graph = get_isolated_graphmodule(true_fn, operands, {})
    false_graph = get_isolated_graphmodule(false_fn, operands, {})
    true_name = "true_graph"
    false_name = "false_graph"
    proxy_mode.tracer.root.register_module(true_name, true_graph)
    proxy_mode.tracer.root.register_module(false_name, false_graph)

    if isinstance(operands, ProxyTensor):
        operands = [operands]  # Prevent unwanted unpacking

    args = (pred, true_graph, false_graph, operands)

    def _unwrap_proxy(e):
        return e.proxy if isinstance(e, ProxyTensor) else e

    proxy_args = pytree.tree_map(_unwrap_proxy, args)

    # Does this need random slug appended so as not to collide?
    proxy_res = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, kwargs,
                                            name="conditional")

    return proxy_res

def cond_dense(pred, true_fn, false_fn, operands):
    mode = torch._C._get_torch_dispatch_mode()
    if mode:
        # Back to dispatch
        return cond(pred, true_fn, false_fn, *operands)
    try:
        # We only get to this step if we are (1) NOT in tracing (2) In the right dispatch step
        # As this op is recursively re-entrant, we need to reset the record
        # TODO(voz): Make this feel better?
        dispatcher_singleton.reset_dispatch_record()
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

def cond_backendselect(*args, **kwargs):
    guard = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.BackendSelect))
    return cond(*args, **kwargs)

def python_fallback(op):
    def inner(*args, **kwargs):
        mode = torch._C._get_torch_dispatch_mode()
        if mode:
            with suspend_mode(mode):
                res = trace_cond(mode, op, args, kwargs)
                return res
        else:
            # Unlikely to ever get here / something went wrong
            return cond(*args, **kwargs)

    return inner


cond = PyOperator('cond')
cond.impl(DispatchKey.CPU, cond_dense)

cond.impl(DispatchKey.Python, python_fallback(cond))
cond.fallthrough(DispatchKey.PythonTLSSnapshot)

cond.impl(DispatchKey.AutogradCPU, cond_autograd)
cond.impl(DispatchKey.ADInplaceOrView, cond_adinplaceorview)
cond.fallthrough(DispatchKey.BackendSelect)
