
import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)
from torch._C import DispatchKey
from torch._functorch.eager_transforms import (
    _unwrap_all_tensors_from_functional,
    _wrap_all_tensors_to_functional,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND


out_dtype = HigherOrderOperator("out_dtype")
out_dtype.fallthrough(DispatchKey.PythonDispatcher)
out_dtype.fallthrough(DispatchKey.PythonTLSSnapshot)
out_dtype.fallthrough(DispatchKey.ADInplaceOrView)
out_dtype.fallthrough(DispatchKey.BackendSelect)
out_dtype.fallthrough(DispatchKey.AutocastCPU)


def trace_out_dtype(proxy_mode, func_overload, op, output_dtype, *args):
    if not isinstance(op, torch._ops.OpOverload):
        raise ValueError("out_dtype's first argument must be an OpOverload")

    with disable_proxy_modes_tracing():
        casted_args = pytree.tree_map_only(
            torch.Tensor, lambda arg: arg.to(dtype=output_dtype), args
        )
        out = op(*casted_args)

    node_args = (op, output_dtype, *args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="out_dtype"
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@out_dtype.py_impl(DispatchKey.CompositeExplicitAutograd)
def out_dtype_dense(
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args
):
    if not isinstance(op, torch._ops.OpOverload):
        raise ValueError("out_dtype's first argument must be an OpOverload")

    flat_inputs = pytree.tree_flatten(args)[0] + [torch.ones(1, dtype=output_dtype)]
    promote_dtype: torch.dtype = elementwise_dtypes(
        *flat_inputs,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )[0]

    casted_args = pytree.tree_map_only(
        torch.Tensor, lambda arg: arg.to(dtype=promote_dtype), args
    )
    res = op(*casted_args).to(dtype=output_dtype)
    return res


@out_dtype.py_impl(DispatchKey.Autograd)
def out_dtype_autograd(
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args
):
    if not isinstance(op, torch._ops.OpOverload):
        raise ValueError("out_dtype's first argument must be an OpOverload")

    # TODO: support autograd
    flat_operands, _ = pytree.tree_flatten(args)
    assert all(
        not f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)
    ), "Autograd is not supported for out_dtype"

    _ = torch._C.ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
    )
    return out_dtype(op, output_dtype, *args)


@out_dtype.py_impl(ProxyTorchDispatchMode)
def out_dtype_proxy(
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args
):
    if not isinstance(op, torch._ops.OpOverload):
        raise ValueError("out_dtype's first argument must be an OpOverload")

    mode = _get_current_dispatch_mode()
    assert (mode is not None), "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        if mode.enable_tracing:
            return trace_out_dtype(mode, out_dtype, op, output_dtype, *args)
        else:
            return out_dtype(op, output_dtype, *args)


@out_dtype.py_impl(FakeTensorMode)
def out_dtype_fake_tensor_mode(
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args
):
    if not isinstance(op, torch._ops.OpOverload):
        raise ValueError("out_dtype's first argument must be an OpOverload")
    return out_dtype_dense(op, output_dtype, *args)


@out_dtype.py_impl(torch._C.DispatchKey.Functionalize)
def out_dtype_func1(op, output_dtype, *args):
    if not isinstance(op, torch._ops.OpOverload):
        raise ValueError("out_dtype's first argument must be an OpOverload")
    if op._schema.is_mutable:
        raise ValueError("out_dtype's first argument needs to be a functional operator")

    reapply_views = torch._C._functionalization_reapply_views_tls()
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_args = tuple(
        _unwrap_all_tensors_from_functional(arg, reapply_views=reapply_views)
        for arg in args
    )
    # pyre-ignore
    guard = torch._C.ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
    )
    try:
        res = out_dtype(op, output_dtype, *unwrapped_args)
        return _wrap_all_tensors_to_functional(res, level=0)
    finally:
        del guard


@out_dtype.py_impl(torch._C._functorch.TransformType.Functionalize)
def out_dtype_func2(interpreter, op, output_dtype, *args):
    reapply_views = interpreter.functionalize_add_back_views()
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_args = tuple(
        _unwrap_all_tensors_from_functional(arg, reapply_views=reapply_views)
        for arg in args
    )

    with interpreter.lower():
        res = out_dtype(op, output_dtype, *unwrapped_args)
        return _wrap_all_tensors_to_functional(res, level=interpreter.level())
