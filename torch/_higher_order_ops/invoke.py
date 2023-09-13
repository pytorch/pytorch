import functools

import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._higher_order_ops.utils import autograd_not_implemented

from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode


# Invoke is a higher order op meant for both invoking a bound hook,
# and for registering it as a call_function in the backward graph.
# This allows us to re-enter dynamo during compiled autograd to trace (or graph break)
# the hook as needed. This, in turn, means we can support hooks in backward with complex python
# state mutation. If we were to not do this, the hooks would get inlined into their composing aten ops,
# and we would lose the python state mutation.
def invoke(fn, grad, reenter):
    return invoke_op(fn, grad, reenter)


invoke_op = HigherOrderOperator("invoke")


def dynamo_interceding_fn_wrapper(grad, *, fn):
    # This wrapper intercepts calls to fn, and calls the real fn via invoke
    # However, as reenter is set to false, the call_function created during trace
    # will point to the actual fn, and not to this function.
    return invoke_op(fn, grad, reenter=False)


@invoke_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(fn, grad, reenter):
    mode = _get_current_dispatch_mode()
    if isinstance(fn, functools.partial):
        fn.__name__ = fn.func.__name__  # type: ignore[attr-defined]
    original_fn = fn
    if reenter:
        # If the reenter flag is set, we wrap the original fn in dynamo_interceding_fn_wrapper
        # and write that to the graph. This produces an aot_autograd graph during backwards that
        # points to dynamo_interceding_fn_wrapper. Then, during compiled autograd, we use
        # dynamo_interceding_fn_wrapper to invoke the original fn under dynamo. The actual
        # dynamo part of dynamo_interceding_fn_wrapper happens during compiled autograd.
        fn = functools.partial(dynamo_interceding_fn_wrapper, fn=fn)
        fn.__name__ = fn.func.__name__
    grad = torch._functorch.aot_autograd.from_fun(grad)
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (grad,))
    out_proxy = mode.tracer.create_proxy(
        "call_function", fn, proxy_args, {}, name="invocation"
    )
    grad = original_fn(grad)
    grad = track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    return torch._functorch.aot_autograd.to_fun(grad)


@invoke_op.py_impl(FakeTensorMode)
def inner_fake(fn, grad, reenter):
    return fn(grad)


@invoke_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def invoke_op_dense(fn, grad, reenter):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return fn(grad)


invoke_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(invoke_op, deferred_error=True)
)


@invoke_op.py_impl(DispatchKey.Functionalize)
def invoke_functionalized(fn, grad, reenter):
    mode = _get_current_dispatch_mode()
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        return invoke_op(fn, grad, reenter)


# TODO(voz): Make this automatic for keys, this is very ugly atm
invoke_op.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
invoke_op.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
invoke_op.fallthrough(DispatchKey.ADInplaceOrView)
invoke_op.fallthrough(DispatchKey.BackendSelect)
invoke_op.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
invoke_op.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
