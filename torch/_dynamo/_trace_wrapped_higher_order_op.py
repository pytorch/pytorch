import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._higher_order_ops.utils import autograd_not_implemented

from torch._ops import HigherOrderOperator
from torch.utils._python_dispatch import _get_current_dispatch_mode

from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch._subclasses import FakeTensorMode
import functools

# Trace wrapped is a higher order op meant for both invoking a bound function,
# and for registering it as a call_function in the backward graph.
# This allows us to re-enter dynamo during compiled autograd to trace (or graph break)
# the functions as needed. This, in turn, means we can support functions in backward with complex python
# state mutation. If we were to not do this, the functions would get inlined into their composing aten ops,
# and we would lose the python state mutation.
def _trace_wrapped(*args, fn):
    return _trace_wrapped_op(*args, fn=fn)

_trace_wrapped_op = HigherOrderOperator("_trace_wrapped")


def self_invoke(*args, fn):
    # This wrapper intercepts calls to fn, and calls the real fn via _trace_wrapped
    # Dynamo unpacks this higher order op into calling the wrapped fn
    return _trace_wrapped_op(*args, fn=fn)


@_trace_wrapped_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(mode, *args, fn):
    import torch._functorch.aot_autograd

    assert len(args) == 1, breakpoint()
    grad = args[0]
    assert isinstance(grad, torch.Tensor)

    if isinstance(fn, functools.partial):
        fn.__name__ = fn.func.__name__  # type: ignore[attr-defined]
    original_fn = fn
    # We've implemented a higher-order operator that remains consistent in proxy tensor tracing.
    # However, Dynamo is aware and traces this into its genuine functionality.
    # The operator's purpose is to facilitate invoking non-traceable functions
    # and embedding them directly into the graph. Essentially, this transforms function
    # calls into "leaf modules" as per traditional FX terminology.
    # Note: Instead of naming it "allow_in_graph", we opted for a different name since "allow_in_graph"
    # might imply that it's traceable, whereas this function is intrinsically non-traceable.
    # Note2: I hate this name
    fn = functools.partial(self_invoke, fn=fn)
    fn.__name__ = fn.func.__name__

    is_func = torch._is_functional_tensor(grad)
    if is_func:
        grad = torch._functorch.aot_autograd.from_fun(grad)

    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (grad,))
    out_proxy = mode.tracer.create_proxy(
        "call_function", fn, proxy_args, {}, name="invocation"
    )
    grad = track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)

    # We have a little shortcut here, wherein we DO NOT yet run a meta func, and so
    # we take on an assumption that input and output meta matches. As such, we must introduce
    # a runtime assert
    proxy_args = pytree.tree_map(
        mode.tracer.unwrap_proxy, (grad, grad.size(), grad.stride(), grad.dtype)
    )
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        torch._functional_assert_tensor_metadata,
        proxy_args,
        {},
        name="assert",
    )
    grad = track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    return (
        pytree.tree_map(torch._functorch.aot_autograd.to_fun, grad) if is_func else grad
    )


@_trace_wrapped_op.py_impl(FakeTensorMode)
def inner_fake(*args, fn):
    raise RuntimeError("This op should never be invoked here")


@_trace_wrapped_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _trace_wrapped_op_dense(*args, fn):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return fn(*args)


_trace_wrapped_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(_trace_wrapped_op, deferred_error=True)
)


@_trace_wrapped_op.py_impl(DispatchKey.Functionalize)
def _trace_wrapped_functionalized(*args, fn):
    mode = _get_current_dispatch_mode()
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        return _trace_wrapped_op(*args, fn=fn)


# TODO(voz): Make this automatic for keys, this is very ugly atm
_trace_wrapped_op.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
_trace_wrapped_op.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
_trace_wrapped_op.fallthrough(DispatchKey.ADInplaceOrView)
_trace_wrapped_op.fallthrough(DispatchKey.BackendSelect)
_trace_wrapped_op.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
_trace_wrapped_op.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
