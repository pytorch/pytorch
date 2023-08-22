
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
from torch._C import DispatchKey, _ExcludeDispatchKeyGuard, DispatchKeySet
from torch._functorch.eager_transforms import (
    _unwrap_all_tensors_from_functional,
    _wrap_all_tensors_to_functional,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._higher_order_ops.utils import autograd_not_implemented

# TODO to figure out a more generic approach
ALLOWABLE_OPS = [
    torch.ops.aten.mm.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
]


class OutDtypeOperator(HigherOrderOperator):
    """
    The out_dtype operator takes an existing ATen functional operator, an
    `out_dtype` argument, and arguments to the original operator, and executes
    the original operator and returns a Tensor with the `out_dtype` precision.
    This operator does not mandate a compute precision so it allows the
    representation to not be opinionated about the exact implementation.

    The general implementation for all operators will be the following:
        1. Promote inputs dtypes based on default PyTorch dtype promotion rules,
            using the dtypes of all input Tensors/Scalars and the `out_dtype`
            arugument.
        2. Execute the operator
        3. Cast the output to `out_dtype`
    """


    def __init__(self):
        super().__init__("out_dtype")
        # TODO(ydwu4): Subclassing HigherOrderOperator causes __module__ to
        # become different (torch._higher_order_ops.out_dtype) which will result
        # in torch.fx to record the op incorrectly in the graph.
        self.__module__ = "torch.ops.higher_order"

    def __call__(self, op, output_dtype, *args):
        if not isinstance(op, torch._ops.OpOverload):
            raise ValueError("out_dtype's first argument must be an OpOverload")
        if op._schema.is_mutable:
            raise ValueError("out_dtype's first argument needs to be a functional operator")
        if not(
            len(op._schema.returns) == 1 and
            isinstance(op._schema.returns[0].type, torch.TensorType)
        ):
            raise ValueError(
                "out_dtype's can only apply to ops that return a single tensor"
                f"Instead got {[r.type for r in op._schema.returns]}"
            )

        if op not in ALLOWABLE_OPS:
            raise ValueError(
                f"out_dtype only allows the following operators: {ALLOWABLE_OPS}."
            )

        res = super().__call__(op, output_dtype, *args)

        return res


out_dtype = OutDtypeOperator()
out_dtype.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
out_dtype.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
out_dtype.fallthrough(DispatchKey.ADInplaceOrView)  # type: ignore[attr-defined]
out_dtype.fallthrough(DispatchKey.BackendSelect)  # type: ignore[attr-defined]
out_dtype.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]


def trace_out_dtype(proxy_mode, func_overload, op, output_dtype, *args):
    with disable_proxy_modes_tracing():
        # This is a simplified implementation of this operator just for tracing.
        # Actual implementation may also first promote the arguments
        out = op(*args).to(dtype=output_dtype)

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
    if (
        len(args)==2 and
        isinstance(args[0], torch.Tensor) and
        isinstance(args[1], torch.Tensor) and
        args[0].dtype == torch.int8 and
        args[1].dtype == torch.int8 and
        args[0].is_cuda and
        args[1].is_cuda and
        op == torch.ops.aten.mm.default and
        output_dtype == torch.int32
    ):
        return torch._int_mm(*args)
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


out_dtype.py_impl(DispatchKey.Autograd)(autograd_not_implemented(out_dtype, deferred_error=True))


@out_dtype.py_impl(ProxyTorchDispatchMode)
def out_dtype_proxy(
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args
):
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
    return out_dtype_dense(op, output_dtype, *args)


@out_dtype.py_impl(DispatchKey.Functionalize)
def out_dtype_func1(op, output_dtype, *args):
    reapply_views = torch._C._functionalization_reapply_views_tls()
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_args = tuple(
        _unwrap_all_tensors_from_functional(arg, reapply_views=reapply_views)
        for arg in args
    )
    # pyre-ignore
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        res = out_dtype(op, output_dtype, *unwrapped_args)
    return _wrap_all_tensors_to_functional(res, level=0)


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
