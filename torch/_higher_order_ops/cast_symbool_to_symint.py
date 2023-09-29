import sympy

import torch

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode

from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree

cast_symbool_to_symint = HigherOrderOperator("cast_symbool_to_symint")


def convert_symbool_to_int_with_hint(maybe_symbool):
    if isinstance(maybe_symbool, bool):
        return int(maybe_symbool)

    return int(maybe_symbool.node.require_hint())


def create_symint_guardless_no_proxy(maybe_symbool):
    if isinstance(maybe_symbool, bool):
        return int(maybe_symbool)

    int_sym = sympy.Piecewise((1, maybe_symbool.node.expr), (0, True))
    return maybe_symbool.node.shape_env.create_symintnode(
        int_sym, hint=int(maybe_symbool.node.require_hint())
    )


@cast_symbool_to_symint.py_impl(DispatchKey.CompositeExplicitAutograd)
def cast_dense(maybe_symbool: bool):
    assert isinstance(maybe_symbool, (bool, torch.SymBool))
    return create_symint_guardless_no_proxy(maybe_symbool)


@cast_symbool_to_symint.py_impl(FakeTensorMode)
def cast_fake(mode, symbool):
    return cast_symbool_to_symint(symbool)


@cast_symbool_to_symint.py_functionalize_impl
def cast_functionalize(ctx, symbool):
    with ctx.redispatch_to_next():
        return cast_symbool_to_symint(symbool)


@cast_symbool_to_symint.py_impl(ProxyTorchDispatchMode)
def trace_cast(proxy_mode, maybe_symbool):
    out = create_symint_guardless_no_proxy(maybe_symbool)

    proxy_symbool = proxy_mode.tracer.unwrap_proxy(maybe_symbool)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        cast_symbool_to_symint,
        (proxy_symbool,),
        {},
        name="cast_symbool_to_symint",
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


cast_symbool_to_symint.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
cast_symbool_to_symint.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
cast_symbool_to_symint.fallthrough(DispatchKey.ADInplaceOrView)
cast_symbool_to_symint.fallthrough(DispatchKey.BackendSelect)
cast_symbool_to_symint.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
cast_symbool_to_symint.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
