import logging
from contextlib import contextmanager

import torch
from torch._C import DispatchKey  # @manual
from torch._functorch._aot_autograd.utils import KNOWN_TYPES
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._library.abstract_impl_class import _ns_and_class_name, FakeScriptObject
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


def _need_python_dispatch():
    curr_stack_len = torch._C._len_torch_dispatch_stack()
    # The check for Python in the exclude set is so we properly respect `with no_dispatch()`
    # calls inside of a mode.
    return curr_stack_len > 0 and not torch._C._dispatch_tls_is_dispatch_key_excluded(
        DispatchKey.Python
    )


def _is_torch_bind_operator(op, args, kwargs):
    return any(
        isinstance(arg, FakeScriptObject) for arg in list(args) + list(kwargs.values())
    )


def _can_skip_cpp_dispatcher(op, args, kwargs):
    return _need_python_dispatch() or torch._ops.need_handle_pre_dispatch()


def _create_skipping_cpp_hanlder(op_overload):
    if torch._ops.need_handle_pre_dispatch():
        return torch._ops.create_predispatch_handler(op_overload)
    elif _need_python_dispatch():

        def handler(*args, **kwargs):
            with torch.utils._python_dispatch._pop_mode_temporarily() as curr_mode:
                return torch._ops._mannually_invoke_dispatch_mode_in_python(
                    curr_mode, op_overload, args, kwargs
                )

        return handler
    else:
        raise RuntimeError(
            "Only support skippping cpp dispatcher by jumping to either pre-dispatch or Python"
        )


# OpOverloadPackt can be called directly. We directly dispatch to
# the default implementation for now to skp C++ dispatcher.
# We could handle other variants when needed.
def OpOverloadPacket_torchbind_redispatch(orig_call):
    def wrapped(self_, *args, **kwargs):
        if _is_torch_bind_operator(self_, args, kwargs):
            return self_.default(*args, **kwargs)
        return orig_call(self_, *args, **kwargs)

    return wrapped


# We want to skip C++ dispatcher for torchbind operator when there're dispatch mode on the
# stack. The reason is that the abstract class (similar to fake tensor for tensor) of torch bind object
# resides in Python and we cannot pass the Python object created from it to C++ dispatcher
# due to schema mismatch.
#
# Implementation-wise, we need to handle two cases: If there are dispatch keys on pre-dispatch mode stack,
# we'll let pre-dispatch handle the operator. If pre-dispatch mode is off but there are
# dispatcher modes on the torch dispatch stack, we pop mode from the stack and directly dispatch there.
# Otherwise, we'll fallback to the original call.
def OpOverload_torchbind_redispatch(orig_call):
    def wrapped(self_, *args, **kwargs):
        if _is_torch_bind_operator(self_, args, kwargs):
            if _can_skip_cpp_dispatcher(self_, args, kwargs):
                return _create_skipping_cpp_hanlder(self_)(*args, **kwargs)
            else:
                raise RuntimeError(
                    f"Some inputs of operartor {self_} are abstract, which indicates"
                    f" we're under tracing/exporting but"
                    " there's no torch_dispatch mode on the stack. This is likely"
                    " caused by creating abstract custom classes without tracing."
                )
        return orig_call(self_, *args, **kwargs)

    return wrapped


@contextmanager
def enable_torchbind_tracing():
    """Context manager that acts as a feature flag to enable torchbind tracing
    behavior. Once torchbind tracing has been stabilized, we can remove this and
    turn it always on.
    """
    prior_OpOverloadPacket_call = torch._ops.OpOverloadPacket.__call__
    prior_OpOverload_call = torch._ops.OpOverload.__call__
    try:
        KNOWN_TYPES.append(torch.ScriptObject)
        torch.ScriptMethod.__call__ = torchbind_method_redispatch  # type: ignore[method-assign]
        torch._ops.OpOverloadPacket.__call__ = OpOverloadPacket_torchbind_redispatch(prior_OpOverloadPacket_call)  # type: ignore[method-assign]
        torch._ops.OpOverload.__call__ = OpOverload_torchbind_redispatch(prior_OpOverload_call)  # type: ignore[method-assign]
        yield
    finally:
        assert (
            KNOWN_TYPES.pop() is torch.ScriptObject
        ), "Someone else messed with KNOWN_TYPES during tracing, exploding."
        torch.ScriptMethod.__call__ = _orig_scriptmethod_call  # type: ignore[method-assign]
        torch._ops.OpOverloadPacket.__call__ = prior_OpOverloadPacket_call  # type: ignore[method-assign]
        torch._ops.OpOverload.__call__ = prior_OpOverload_call  # type: ignore[method-assign]


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
                ' You can register a fake class with torch._library.impl_abstract_class("%s::%s").',
                class_name,
                method,
                ns,
                class_name,
            )

        return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    else:
        return call_torchbind(*args, **kwargs)


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
