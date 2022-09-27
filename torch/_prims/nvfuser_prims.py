# Module for defining "primitive" operations executable by the nvFuser. This
# list exists to decouple main set of primitives from the ones that provide a
# lowering of the op to nvFuser’s Python interface. Mostly torch.ops.nvprims is
# a subset of the primitives in torch.ops.prims, but some additional primitives
# can be added in the future for the corresponding higher-level torch/aten
# functions.

from functools import wraps
from typing import Any, Dict, Optional

import torch

from torch._prims.context import NvfuserPrimsMode, TorchRefsMode
from torch._prims.vjp import vjp_implementations as _vjp_impls
from torch._prims_common import (
    DimsSequenceType,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    getnvFuserDtype,
    make_contiguous_strides_for,
    ShapeType,
    TensorLikeType,
)

from torch._prims_common.wrappers import (
    backwards_not_supported,
    elementwise_type_promotion_wrapper,
)
from torch.utils._mode_utils import autodispatch_below_autograd
from torch.utils._pytree import tree_flatten, tree_unflatten

nvprim_namespace = "nvprims"
nvprim = torch.library.Library(nvprim_namespace, "DEF")
nvprim_impl = torch.library.Library(
    nvprim_namespace, "IMPL", "CompositeExplicitAutograd"
)
nvprim_implicit_impl = torch.library.Library(
    nvprim_namespace, "IMPL", "CompositeImplicitAutograd"
)
nvprim_autograd_impl = torch.library.Library(nvprim_namespace, "IMPL", "Autograd")
nvprim_meta_impl = torch.library.Library(nvprim_namespace, "IMPL", "Meta")

nvprim_names = [
    "abs",
    "acos",
    "asin",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "bitwise_not",
    "ceil",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "floor",
    "imag",
    "isfinite",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "real",
    "reciprocal",
    "neg",
    "round",
    "rsqrt",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "transpose",
    "trunc",
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "div",
    "eq",
    "fmod",
    "ge",
    "gt",
    "le",
    "lt",
    "mul",
    "ne",
    "pow",
    "remainder",
    "sub",
    "squeeze",
    "view_of",
    "broadcast_in_dim",
    "where",
    "convert_element_type",
    "sum",
    "var",
    "amax",
    "amin",
]

_nvfuser_impls: Dict[str, Any] = {}

_nvfuser_unary_ops = {
    "abs",
    "acos",
    "asin",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "bitwise_not",
    "ceil",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "floor",
    "imag",
    "isfinite",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "reciprocal",
    "neg",
    "real",
    "round",
    "rsqrt",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
}


def _assert_nvfuser_op_exists(fname: str):
    try:
        from torch._C._nvfuser import FusionDefinition as fd  # type: ignore[import]

        assert getattr(fd.Operators, fname)
    except ImportError:
        # Not all PyTorch builds have nvfuser
        pass


for fname in _nvfuser_unary_ops:
    exec(
        f"""
# Ensure that the nvfuser implementation exists
_assert_nvfuser_op_exists("{fname}")

def _{fname}_nvfuser(fd, a):
    return fd.ops.{fname}(a)  # type: ignore[attr-defined]

_nvfuser_impls["{fname}"] = _{fname}_nvfuser
"""
    )

_nvfuser_binary_ops = {
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "div",
    "eq",
    "fmod",
    "ge",
    "gt",
    "le",
    "lt",
    "mul",
    "ne",
    "pow",
    "remainder",
    "sub",
}

for fname in _nvfuser_binary_ops:
    exec(
        f"""
# Ensure that the nvfuser implementation exists
_assert_nvfuser_op_exists("{fname}")

def _{fname}_nvfuser(fd, a, b):
    return fd.ops.{fname}(a, b)  # type: ignore[attr-defined]

_nvfuser_impls["{fname}"] = _{fname}_nvfuser
"""
    )

_nvfuser_ternary_ops = {
    "where",
}

for fname in _nvfuser_ternary_ops:
    exec(
        f"""
# Ensure that the nvfuser implementation exists
_assert_nvfuser_op_exists("{fname}")

def _{fname}_nvfuser(fd, a, b, c):
    return fd.ops.{fname}(a, b, c)  # type: ignore[attr-defined]

_nvfuser_impls["{fname}"] = _{fname}_nvfuser
"""
    )


def _broadcast_in_dim_nvfuser(
    fd: Any,
    a: TensorLikeType,
    shape: ShapeType,
    broadcast_dimensions: ShapeType,
):
    return fd.ops.broadcast_in_dim(a, shape, broadcast_dimensions)  # type: ignore[attr-defined]


def _convert_element_type_nvfuser(fd: Any, a: TensorLikeType, dtype: torch.dtype):
    nvfuser_dtype = getnvFuserDtype(dtype)
    return fd.ops.cast(a, nvfuser_dtype)  # type: ignore[attr-defined]


def _transpose_nvfuser(fd, a, permutation):
    return fd.ops.permute(a, permutation)  # type: ignore[attr-defined]


def _squeeze_nvfuser(fd, a, a_shape, dimensions):
    for idx in reversed(sorted(dimensions)):
        a = fd.ops.squeeze(a, a_shape, idx)
        a_shape = a_shape[:idx] + a_shape[idx + 1 :]
    return a


def _view_of_nvfuser(fd, a):
    return fd.ops.set(a)


def _sum_nvfuser(
    fd: Any,
    a: TensorLikeType,
    dims: DimsSequenceType,
):
    keep_dims = False
    output_dtype = torch._C._nvfuser.DataType.Null
    return fd.ops.sum(a, dims, keep_dims, output_dtype)


def _var_nvfuser(
    fd: Any,
    a: TensorLikeType,
    dims: DimsSequenceType,
    *,
    correction: int,
):
    keep_dims = False
    return fd.ops.var(a, dims, correction, keep_dims)


def _var_mean_nvfuser(
    fd: Any,
    a: TensorLikeType,
    dims: DimsSequenceType,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: int,
):
    # Unbiased arg shouldn't be set when this function is called
    assert unbiased is None
    # Ignore keepdim arg, because currently it's automatically converted into nvfuser's symbolic scalar
    # keepdim is handled by the reference implementation
    keepdim = False
    return fd.ops.var_mean(a, dims, correction, keepdim)


def _rand_like_nvfuser(fd: Any, a: TensorLikeType):
    return fd.ops.rand_like(a)


def _amax_nvfuser(
    fd: Any,
    a: TensorLikeType,
    dims: DimsSequenceType,
):
    keep_dims = False
    return fd.ops.max(a, dims, keep_dims)


def _amin_nvfuser(
    fd: Any,
    a: TensorLikeType,
    dims: DimsSequenceType,
):
    keep_dims = False
    return fd.ops.min(a, dims, keep_dims)


_nvfuser_impls["broadcast_in_dim"] = _broadcast_in_dim_nvfuser
_nvfuser_impls["convert_element_type"] = _convert_element_type_nvfuser
_nvfuser_impls["transpose"] = _transpose_nvfuser
_nvfuser_impls["squeeze"] = _squeeze_nvfuser
_nvfuser_impls["view_of"] = _view_of_nvfuser
_nvfuser_impls["rand_like"] = _rand_like_nvfuser
_nvfuser_impls["sum"] = _sum_nvfuser
_nvfuser_impls["var"] = _var_nvfuser
_nvfuser_impls["var_mean"] = _var_mean_nvfuser
_nvfuser_impls["amax"] = _amax_nvfuser
_nvfuser_impls["amin"] = _amin_nvfuser


def _register_vjp(prim, vjp_impl):
    class PrimFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, args_spec, *flat_args):
            # torch.autograd.Function can't accept keyword arguments
            args, kwargs = tree_unflatten(flat_args, args_spec)  # type: ignore[arg-type]
            with autodispatch_below_autograd():
                out = prim(*args, **kwargs)

            out_packed = (out,) if isinstance(out, torch.Tensor) else out
            ctx.nout = len(out_packed)

            # Only tensors can be saved for backward
            args_tensors = tuple(t for t in args if isinstance(t, torch.Tensor))
            ctx.ntensorargs = len(args_tensors)
            ctx.save_for_backward(*out_packed, *args_tensors)

            ctx.args = args
            ctx.kwargs = kwargs
            return out

        @staticmethod
        def backward(ctx, *bw_args):
            if vjp_impl is None:
                raise RuntimeError(f"backwards not supported on prim {prim.name}")

            fw_tensorargs = iter(
                ctx.saved_tensors[ctx.nout : ctx.nout + ctx.ntensorargs]
            )
            fw_args = [
                a if not isinstance(a, torch.Tensor) else next(fw_tensorargs)
                for a in ctx.args
            ]
            fw_out = ctx.saved_tensors[: ctx.nout]

            with NvfuserPrimsMode(), TorchRefsMode():
                vjp_result = vjp_impl(*bw_args, *fw_out, *fw_args, **ctx.kwargs)
            vjp_result = (
                (vjp_result,) if isinstance(vjp_result, torch.Tensor) else vjp_result
            )

            flat_args_kwargs, _ = tree_flatten((ctx.args, ctx.kwargs))
            assert len(vjp_result) == len(flat_args_kwargs)

            # Replace the output with None for each non-tensor argument
            vjp_result = tuple(
                None if not isinstance(a, torch.Tensor) else t
                for a, t in zip(flat_args_kwargs, vjp_result)
            )
            return None, *vjp_result

    @wraps(prim)
    def _autograd_impl(*args, **kwargs):
        flat_args, args_spec = tree_flatten((args, kwargs))
        return PrimFunction.apply(args_spec, *flat_args)

    return _autograd_impl


def register_rand_like():
    name = "rand_like"

    nvprim.define(
        "rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, "
        + "Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"
    )

    def _meta_rand_like(
        self,
        *,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
        memory_format=None,
    ):
        strides = make_contiguous_strides_for(self.shape)
        return torch._prims.TensorMeta(
            shape=self.shape,
            strides=strides,
            dtype=dtype,
            device=device,
        )

    def _prim_impl(
        self,
        *,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
        memory_format=None,
    ):
        return torch.rand_like(
            self,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            memory_format=memory_format,
        )

    nvprim_impl.impl(name, _prim_impl)
    nvprim_meta_impl.impl(name, _meta_rand_like)

    prim_packet = getattr(torch.ops.nvprims, name)
    prim = prim_packet.default

    nvprim_autograd_impl.impl(name, backwards_not_supported(prim))

    for p in (prim_packet, prim):
        p.__doc__ = "Computes rand_like"
        p.impl_nvfuser = _nvfuser_impls["rand_like"]
        p.return_type = torch._prims_common.RETURN_TYPE.NEW  # type: ignore[attr-defined]


def register_var_mean():
    """This function is used to register the var_mean function in torch.ops.nvprims module."""
    name = "var_mean.main"

    # This overload must be default for correct dispatching of var_mean(Tensor, bool)
    # It's registered as CompositeImplicit function.
    nvprim.define("var_mean(Tensor inp, bool unbiased) -> (Tensor, Tensor)")

    # This signature tries to combine several overloads of the torch.var_mean function into one overload.
    # It's registered as CompositeImplicit function.
    nvprim.define(
        "var_mean.all_args(Tensor inp, int[1]? dim=None, bool? unbiased=None, bool keepdim=False, *, int? correction=None)"
        + " -> (Tensor, Tensor)"
    )

    # This signature is the primitive recorded on the trace for nvFuser.
    nvprim.define(
        f"{name}(Tensor inp, int[] dims, *, int correction) -> (Tensor, Tensor)"
    )

    # This function is used for device="meta" Tensors.
    def _meta_var_mean(inp, dims, *, correction):
        if torch._prims_common.is_complex_dtype(inp.dtype):
            output_dtype = torch._prims_common.corresponding_real_dtype(inp.dtype)
        else:
            output_dtype = inp.dtype
        var = torch._prims._reduction_meta(inp, dims, output_dtype=output_dtype)
        mean = torch._prims._reduction_meta(inp, dims, output_dtype=inp.dtype)
        return (var, mean)

    # This function is used under _AutoDispatchBelowAutograd context
    def _prim_impl(inp, dims, *, correction):
        return torch.var_mean(inp, dims, correction=correction)

    nvprim_impl.impl(name, _prim_impl)
    nvprim_meta_impl.impl(name, _meta_var_mean)

    prim_packet = torch.ops.nvprims.var_mean
    prim = prim_packet.main

    def _unbiased_overload_impl(inp, unbiased):
        return prim_packet.all_args(inp, dim=None, unbiased=unbiased)

    nvprim_implicit_impl.impl("var_mean", _unbiased_overload_impl)

    @elementwise_type_promotion_wrapper(
        type_promoting_args=("a",),
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    )
    def _var_mean_ref(a, dim=None, unbiased=None, keepdim=False, *, correction=None):
        correction = torch._prims_common.set_correction(unbiased, correction)
        # reduces over all dimensions if dim=() is passed
        if dim == () or dim == []:
            dim = None
        dim = torch._prims_common.reduction_dims(a.shape, dim)

        # For complex tensors eager computes the variance as the sum of variances of
        # the real and imaginary parts
        # TODO: Creating a complex tensor from real and imaginary parts is not supported
        if torch._prims_common.is_complex_dtype(a.dtype):
            raise NotImplementedError("Complex tensors are not supported")

        var_mean = prim(a, dim, correction=correction)

        if keepdim:
            output_shape = [a.shape[i] if i not in dim else 1 for i in range(a.ndim)]
            broadcast_dims = [i for i in range(a.ndim) if i not in dim]
            var, mean = var_mean
            var = torch.ops.nvprims.broadcast_in_dim(var, output_shape, broadcast_dims)
            mean = torch.ops.nvprims.broadcast_in_dim(
                mean, output_shape, broadcast_dims
            )
            var_mean = (var, mean)
        return var_mean

    def _var_mean_ref_nvprims_mode(
        a, dim=None, unbiased=None, keepdim=False, *, correction=None
    ):
        # This wrapper is needed to convert prims calls inside
        # elementwise_type_promotion_wrapper to nvprims calls
        with NvfuserPrimsMode():
            return _var_mean_ref(a, dim, unbiased, keepdim, correction=correction)

    nvprim_implicit_impl.impl("var_mean.all_args", _var_mean_ref_nvprims_mode)
    nvprim_autograd_impl.impl(name, _register_vjp(prim, _vjp_impls["var_mean"]))

    for p in (prim_packet, prim):
        p.__doc__ = "Computes the variance and mean of x over the list of dimensions specified in the dim argument"
        p.impl_nvfuser = _nvfuser_impls["var_mean"]
        p.return_type = torch._prims_common.RETURN_TYPE.NEW  # type: ignore[attr-defined]


def register_nvprims():
    """Registers all nvFuser primitives in the torch.ops.nvprims module."""
    register_var_mean()
    register_rand_like()

    for name in nvprim_names:
        main_prim = getattr(torch.ops.prims, name)

        nvprim.define(main_prim.schema)
        nvprim_impl.impl(name, main_prim.prim_impl)
        nvprim_meta_impl.impl(name, main_prim.prim_meta_impl)

        prim_packet = getattr(torch.ops.nvprims, name)
        prim = prim_packet.default

        # if name=="abs":
        #     vjp_impl = _vjp_impls["abs"]
        # else:
        #     vjp_impl = None
        vjp_impl = _vjp_impls.get(name, None)
        # nvprim_autograd_impl.impl(name, backwards_not_supported(prim))
        nvprim_autograd_impl.impl(name, _register_vjp(prim, vjp_impl))

        for p in (prim_packet, prim):
            p.__doc__ = main_prim.__doc__
            p.impl_nvfuser = _nvfuser_impls[name]
            p.return_type = main_prim.return_type  # type: ignore[attr-defined]
