# mypy: allow-untyped-defs
import logging
from contextlib import contextmanager

import torch
from torch._C import DispatchKey  # @manual
from torch._functorch._aot_autograd.utils import KNOWN_TYPES
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._library.fake_class_registry import _ns_and_class_name, FakeScriptObject
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.fx.node import has_side_effect
from torch.utils import _pytree as pytree


log = logging.getLogger(__name__)

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
    if isinstance(obj, torch.ScriptObject):
        return _orig_scriptmethod_call(getattr(obj, method), *args, **kwargs)
    elif isinstance(obj, FakeScriptObject):
        return getattr(obj.wrapped_obj, method)(*args, **kwargs)
    else:
        raise RuntimeError(f"Unsupported first arg type {type(obj)} for call_torchbind")


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
        out = call_torchbind(*args, **kwargs)

        obj, method, *rest_args = args
        if isinstance(obj, torch.ScriptObject):
            ns, class_name = _ns_and_class_name(
                obj._type().qualified_name()  # type: ignore[attr-defined]
            )
            log.warning(
                "Tracing torchbind method %s.%s with real ScriptObject. This may"
                " cause the original object being mutated. If this is not intended,"
                ' You can register a fake class with torch._library.register_fake_class("%s::%s").',
                class_name,
                method,
                ns,
                class_name,
            )

        ret = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
        if "val" not in out_proxy.node.meta:
            assert out is None or isinstance(
                out, (int, float, bool)
            ), "Currently, only these constant dtypes are supported to be returned from torchbind methods."
            out_proxy.node.meta["val"] = out
        return ret
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
    from torch._higher_order_ops.effects import handle_effects

    return handle_effects(
        ctx.mode._allow_token_discovery, ctx.mode._tokens, call_torchbind, args, kwargs
    )
