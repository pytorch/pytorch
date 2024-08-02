# mypy: allow-untyped-defs
import torch
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented

from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch.fx.experimental._backward_state import BackwardState

from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode
from torch.utils._pytree import tree_map_only


__all__ = ["trace_wrapped"]


# trace_wrapped(*args, fn) is equivalent to fn(*args), but with a twist:
# if you make_fx trace through this call, we will not actually trace into fn; instead,
# we will directly insert it as a call_function to fn in the graph.
# (Unlike make_fx, Dynamo WILL inline into fn.)
# You can think of this as a one off allow_in_graph equivalent for proxy tensor tracing.
#
# Because proxy tensor tracing does not actually run the function, there are
# requirements on the behavior of fn. We are still figuring it out, but here is the current state:
#
# 1) fn SHOULD only take a single argument, which must be a tensor
# 2) fn MUST return a new tensor with the same metadata as the original tensor
#    (e.g., zeros_like(input) is a permissible implementation of fn).
#    This is verified via an extra assert that is inserted into the traced graph.
# 3) fn MAY have side effects, but it MAY NOT perform metadata mutation on other tensors
#    participating in proxy tensor tracing (it MAY mutate other tensors, it MAY mutate Python state)
# These requirements stem from the requirement that we need to continue performing proxy tensor tracing,
# which assumes accurate fake tensor metadata, without actually running fn.
# In the future, we may allow for a "meta" function associated with fn to allow for more interesting input-output patterns.
#
# Note that tensors / Python state are allowed to be mutated.
# This is relaxed constraint is not always sound, but it is sound for backward tracing with fake
# tensors as it takes place in AOTAutograd, as the backward pass is guaranteed not to depend on concrete
# tensor values (via fake tensor) or Python state (because the autograd engine doesn't depend on Python).
#
# The intended use case for this function is to allow AOTAutograd to defer complex
# backward hooks to compiled autograd. AOTAutograd performs a make_fx trace which preserves
# the function call as is in the graph, and only when we Dynamo through the backward graph in
# compiled autograd do we inline into the function.


def trace_wrapped(*args, **kwargs):
    with torch.no_grad():
        return _trace_wrapped_op(*args, **kwargs)


# TODO(jansel): need to ensure this does not get DCEed
_trace_wrapped_op = HigherOrderOperator("trace_wrapped")


def _assert_meta(grad, size, stride, dtype):
    assert grad.size() == size, "size mismatch"
    assert grad.stride() == stride, "stride mismatch"
    assert grad.dtype == dtype, "dtype mismatch"
    return grad


@_trace_wrapped_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(mode, *args, bw_state=None, **kwargs):
    def self_invoke(*args, **dyn_kwargs):
        with torch.no_grad():
            return _trace_wrapped_op(*args, **dyn_kwargs, **kwargs)

    def unwrap_proxies(x):
        if isinstance(x, torch.Tensor):
            return mode.tracer.unwrap_proxy(x)
        if isinstance(x, (list, tuple)):
            return type(x)(map(unwrap_proxies, x))
        if x is None:
            return None
        raise AssertionError(f"unhandled type: {type(x)}")

    proxy_kwargs = {}
    if bw_state is not None:
        assert isinstance(bw_state, BackwardState) and bw_state.proxy is not None
        proxy_kwargs["bw_state"] = bw_state.proxy
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        self_invoke,
        unwrap_proxies(args),
        proxy_kwargs,
        name="trace_wrapped",
    )

    if args[0] is None:
        grad = args[1]  # module backward hooks
    else:
        grad = args[0]  # other backward hooks
    grad = tree_map_only(torch.Tensor, torch.empty_like, grad)
    track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    return grad


@_trace_wrapped_op.py_impl(FakeTensorMode)
def inner_fake(*args, **kwargs):
    raise RuntimeError("This op should never be invoked here")


@_trace_wrapped_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _trace_wrapped_op_dense(*args, fn, **kwargs):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return fn(*args, **kwargs)


_trace_wrapped_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(_trace_wrapped_op, deferred_error=True)
)


@_trace_wrapped_op.py_functionalize_impl
def _trace_wrapped_functionalized(ctx, *args, **kwargs):
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        return ctx.wrap_tensors(_trace_wrapped_op(*unwrapped_args, **kwargs))
