import functools

import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._higher_order_ops.utils import autograd_not_implemented

from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode


def invoke(fn, grad):
    return invoke_op(fn, grad)


invoke_op = HigherOrderOperator("invoke")


@invoke_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(fn, grad):
    mode = _get_current_dispatch_mode()
    if isinstance(fn, functools.partial):
        fn.__name__ = fn.func.__name__  # type: ignore[attr-defined]
    grad = torch._functorch.aot_autograd.from_fun(grad)
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (grad,))
    out_proxy = mode.tracer.create_proxy(
        "call_function", fn, proxy_args, {}, name="invocation"
    )
    grad = fn(grad)
    grad = track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    return torch._functorch.aot_autograd.to_fun(grad)


@invoke_op.py_impl(FakeTensorMode)
def inner_fake(fn, grad):
    return fn(grad)


@invoke_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def invoke_op_dense(fn, grad):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return fn(grad)


invoke_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(invoke_op, deferred_error=True)
)


@invoke_op.py_impl(DispatchKey.Functionalize)
def invoke_functionalized(fn, grad):
    mode = _get_current_dispatch_mode()
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        return invoke_op(fn, grad)


# TODO(voz): Make this automatic for keys, this is very ugly atm
invoke_op.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
invoke_op.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
invoke_op.fallthrough(DispatchKey.ADInplaceOrView)
invoke_op.fallthrough(DispatchKey.BackendSelect)
invoke_op.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
invoke_op.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
