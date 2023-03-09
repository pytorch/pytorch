import torch
import torch.utils._pytree as pytree
import itertools
from functools import partial
from torch._ops import PyOperator
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)

from torch.fx.experimental.proxy_tensor import (
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
    unwrap_proxy,
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

# Required by make_fx tracing "symbolic" mode
# We cannot directly call fallthrough here due to issue #89037.
@while_loop.py_impl(DispatchKey.PythonDispatcher)
def cond_python_dispatcher(*args):
    _ = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.PythonDispatcher))
    return while_loop(*args)
while_loop.fallthrough(DispatchKey.PythonTLSSnapshot)

@while_loop.py_impl(ProxyTorchDispatchMode)
def while_loop_proxy_torch_dispatch_mode(cond_fun, body_fun, init_val):
    mode = _get_current_dispatch_mode()
    assert (mode is not None), "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        res = trace_while_loop(mode, while_loop, cond_fun, body_fun, init_val)
    return res

def get_next_name(root, prefix):
    next_name = None
    for i in itertools.count():
        candidate = f"{prefix}_{i}"
        if not hasattr(root, candidate):
            next_name = candidate
            break
    return next_name

def trace_while_loop(proxy_mode, func_overload, cond_fun, body_fun, init_val):
    assert pytree.tree_all(lambda v: isinstance(v, torch.Tensor), init_val), "init_val must be a pytree of tensors"
    cond_graph = make_fx(cond_fun)(init_val)
    body_graph = make_fx(body_fun)(init_val)

    root = proxy_mode.tracer.root
    cond_name = get_next_name(root, "while_loop_cond_graph")
    body_name = get_next_name(root, "while_loop_body_graph")
    root.register_module(cond_name, cond_graph)
    root.register_module(body_name, body_graph)

    args = (cond_graph, body_graph, init_val)

    proxy_args = pytree.tree_map(partial(unwrap_proxy, proxy_mode), args)

    out_proxy = proxy_mode.tracer.create_proxy("call_function", func_overload, proxy_args, {}, name="while_loop")

    # NB: we cannot track the init_val due to the data dependent nature of while_loop.
    # We assume the output has the same pytree structure and the leaves of pytree have
    # the same meta data as init_val.
    out = pytree.tree_map(lambda fake_t: torch.empty_like(fake_t), init_val)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)
