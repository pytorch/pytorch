from contextlib import contextmanager

import torch
from torch._C import DispatchKey  # @manual
from torch._functorch._aot_autograd.utils import KNOWN_TYPES
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.fx.node import has_side_effect
from torch.utils import _pytree as pytree

# The call_torchbind operator represents a method invocation on a torchbind
# object. The calling convention is:
#   call_torchbind(self: ScriptObject, method_name: str, *method_args, **method_kwargs)
# We do not expect users to write this operator directly. Instead it will be
# emitted by Dynamo when tracing encounters a torchbind object.
call_torchbind = HigherOrderOperator("call_torchbind")

# Register this operator as side-effectful with FX.
# TODO: this is not really sufficient. While passes (hopefully) check
# Node.is_impure() and make good decisions, we also assume we can execute the
# graph as many times as we want without changing behavior, which is NOT true of
# ops that mutate torchbind object state.
has_side_effect(call_torchbind)

_orig_scriptmethod_call = torch.ScriptMethod.__call__


def torchbind_method_redispatch(self, *args, **kwargs):
    if isinstance(self.raw_owner, torch.ScriptObject):
        return call_torchbind(self.raw_owner, self.name, *args, **kwargs)
    return _orig_scriptmethod_call(self, *args, **kwargs)


@contextmanager
def enable_torchbind_tracing():
    """Context manager that acts as a feature flag to enable torchbind tracing
    behavior. Once torchbind tracing has been stabilized, we can remove this and
    turn it always on.
    """
    try:
        KNOWN_TYPES.append(torch.ScriptObject)
        torch.ScriptMethod.__call__ = torchbind_method_redispatch  # type: ignore[method-assign]
        yield
    finally:
        assert (
            KNOWN_TYPES.pop() is torch.ScriptObject
        ), "Someone else messed with KNOWN_TYPES during tracing, exploding."
        torch.ScriptMethod.__call__ = _orig_scriptmethod_call  # type: ignore[method-assign]


@call_torchbind.py_impl(DispatchKey.CompositeExplicitAutograd)
def call_torchbind_impl(obj, method, *args, **kwargs):
    return _orig_scriptmethod_call(getattr(obj, method), *args, **kwargs)


@call_torchbind.py_impl(ProxyTorchDispatchMode)
def inner(mode, *args, **kwargs):
    if mode.enable_tracing:
        proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
        proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)

        out_proxy = mode.tracer.create_proxy(
            "call_function",
            call_torchbind,
            proxy_args,
            proxy_kwargs,
        )
        out = call_torchbind_impl(*args, **kwargs)

        return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    else:
        return call_torchbind(*args, **kwargs)


# TODO: currently we just run the C++ implementation with fake tensors.
# But we should make it possible to register a fake torchbind implementation.
@call_torchbind.py_impl(FakeTensorMode)
def call_torchbind_fake(mode, *args, **kwargs):
    with mode:
        return call_torchbind_impl(*args, **kwargs)


call_torchbind.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(call_torchbind, deferred_error=True)
)


@call_torchbind.py_functionalize_impl
def call_torchbind_func(ctx, *args, **kwargs):
    args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        return ctx.wrap_tensors(call_torchbind(*args, **kwargs))
