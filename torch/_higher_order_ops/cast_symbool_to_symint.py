import torch

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode

from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.fx.experimental.symbolic_shapes import create_symint_from_symbool_guardless

cast_symbool_to_symint = HigherOrderOperator("cast_symbool_to_symint")


@cast_symbool_to_symint.py_impl(DispatchKey.CompositeExplicitAutograd)
def cast_dense(maybe_symbool: bool):
    assert isinstance(maybe_symbool, (bool, torch.SymBool))
    return create_symint_from_symbool_guardless(maybe_symbool)


@cast_symbool_to_symint.py_impl(FakeTensorMode)
def cast_fake(mode, symbool):
    return cast_symbool_to_symint(symbool)


@cast_symbool_to_symint.py_functionalize_impl
def cast_functionalize(ctx, symbool):
    with ctx.redispatch_to_next():
        return cast_symbool_to_symint(symbool)


@cast_symbool_to_symint.py_impl(ProxyTorchDispatchMode)
def trace_cast(proxy_mode, maybe_symbool):
    out = create_symint_from_symbool_guardless(maybe_symbool)

    proxy_symbool = proxy_mode.tracer.unwrap_proxy(maybe_symbool)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        cast_symbool_to_symint,
        (proxy_symbool,),
        {},
        name="cast_symbool_to_symint",
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)
