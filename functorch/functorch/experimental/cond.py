import torch
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from functorch.experimental.ops import PyOperator
from torch.utils._pytree import tree_flatten
from torch.fx.experimental.proxy_tensor import ProxyTensor, get_isolated_graphmodule, disable_proxy_modes_tracing
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
    assert(mode is not None), "Cannot suspend None mode"
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


def trace_cond(proxy_mode, func_overload, args, kwargs=None):
    assert kwargs is None or not kwargs
    pred, true_fn, false_fn, operands = args

    def _unwrap_proxy(e):
        return e.proxy if isinstance(e, ProxyTensor) else e

    def _extract_elements(e):
        return e.elem if isinstance(e, ProxyTensor) else e

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

    true_args = []
    for node in true_graph.graph.nodes:
        if node.op == 'output':
            true_args.extend(*node.args)
    
    false_args = []
    for node in false_graph.graph.nodes:
        if node.op == 'output':
            false_args.extend(*node.args)

    assert(len(true_args) == len(false_args))
    for i in range(len(true_args)):
        t_arg_meta = true_args[i]
        f_arg_meta = false_args[i]
        # WIP don't look at this yet 
        # assert(t_arg_meta.meta == f_arg_meta.meta)

    if isinstance(operands, ProxyTensor):
        operands = [operands]  # Prevent unwanted unpacking

    args = (pred, true_graph, false_graph, operands)

    proxy_args = pytree.tree_map(_unwrap_proxy, args)

    elements = pytree.tree_map(_extract_elements, args)

    # Does this need random slug appended so as not to collide?
    proxy_res = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, kwargs,
                                            name="conditional")
    return proxy_res


def cond_dense(pred, true_fn, false_fn, *operands):
    mode = torch._C._get_torch_dispatch_mode()
    assert (mode is None), "Mode should never be enabled for CPU key"
    try:
        if pred:
            return true_fn(*operands)
        else:
            return false_fn(*operands)
    except Exception as e:
        # Do something proper here, someday
        print("Exception:", e)
        raise e


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
            return cond(args, kwargs)
    return inner


cond = PyOperator('cond')
cond.impl(DispatchKey.CPU, cond_dense)
cond.impl(DispatchKey.Python, python_fallback(cond))
cond.fallthrough(DispatchKey.PythonTLSSnapshot)
cond.impl(DispatchKey.AutogradCPU, cond_autograd)
cond.fallthrough(DispatchKey.ADInplaceOrView)
cond.fallthrough(DispatchKey.BackendSelect)
