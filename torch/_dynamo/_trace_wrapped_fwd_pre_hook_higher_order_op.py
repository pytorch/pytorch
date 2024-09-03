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


__all__ = ["trace_wrapped_fwd_pre_hook"]

def trace_wrapped_fwd_pre_hook(*args, **kwargs):
    return _trace_wrapped_fwd_pre_hook_op(*args, **kwargs)


class TraceWrappedFwdPreHook(HigherOrderOperator):
    def __init__(self):
        super().__init__("trace_wrapped_fwd_pre_hook")

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


_trace_wrapped_fwd_pre_hook_op = TraceWrappedFwdPreHook()


@_trace_wrapped_fwd_pre_hook_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(mode, *args, bw_state=None, **kwargs):
    def self_invoke(*args, **dyn_kwargs):
        return _trace_wrapped_fwd_pre_hook_op(*args, **dyn_kwargs, **kwargs)

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
    unwrap_proxies_args = unwrap_proxies(args)
    new_args_proxy = mode.tracer.create_proxy(
        "call_function",
        self_invoke,
        unwrap_proxies_args,
        proxy_kwargs,
        name="trace_wrapped_fwd_pre_hook",
    )

    print(f"fwd_pre_hook: inner_trace: args: {args}")
    new_args = args[1]
    track_tensor_tree(new_args, new_args_proxy, constant=None, tracer=mode.tracer)
    return new_args


@_trace_wrapped_fwd_pre_hook_op.py_impl(FakeTensorMode)
def inner_fake(*args, **kwargs):
    print(f"fwd_pre_hook: inner_fake: args: {args}")
    return args[2]  # args[0] is hook, args[1] is module, args[2] is args tuple


@_trace_wrapped_fwd_pre_hook_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _trace_wrapped_fwd_pre_hook_op_dense(*args, fn, **kwargs):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return fn(*args, **kwargs)


_trace_wrapped_fwd_pre_hook_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(_trace_wrapped_fwd_pre_hook_op, deferred_error=True)
)


@_trace_wrapped_fwd_pre_hook_op.py_functionalize_impl
def _trace_wrapped_fwd_pre_hook_functionalized(ctx, *args, **kwargs):
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        return ctx.wrap_tensors(_trace_wrapped_fwd_pre_hook_op(*unwrapped_args, **kwargs))
