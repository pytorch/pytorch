# Module for defining "primitive" operations executable by the nvFuser. This
# list exists to decouple main set of primitives from the ones that provide a
# lowering of the op to nvFuser’s Python interface. Mostly torch.ops.nvprims is
# a subset of the primitives in torch.ops.prims, but some additional primitives
# can be added in the future for the corresponding higher-level torch/aten
# functions.

import math
from functools import wraps
from typing import Any, Callable, Dict, Optional, Sequence

import torch
import torch._prims_common as utils

from torch._prims.context import NvfuserPrimsMode, TorchRefsMode
from torch._prims_common import (
    DimsSequenceType,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    getnvFuserDtype,
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
_nvfuser_impls["squeeze"] = _squeeze_nvfuser
_nvfuser_impls["view_of"] = _view_of_nvfuser
_nvfuser_impls["sum"] = _sum_nvfuser
_nvfuser_impls["var"] = _var_nvfuser
_nvfuser_impls["var_mean"] = _var_mean_nvfuser
_nvfuser_impls["amax"] = _amax_nvfuser
_nvfuser_impls["amin"] = _amin_nvfuser


prims = torch.ops.prims


# add dimensions in canonicalized, sorted order
# so that the tensor has the appropriate rank.
def _unsqueeze_dims(a, dims, rank):
    dims = utils.canonicalize_dims(rank, dims)
    for dim in sorted(dims):
        a = torch._refs.unsqueeze(a, dim)
    return a


def _expand(a, *shape):
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = tuple(shape[0])
    utils.check(
        len(shape) >= len(a.shape),
        lambda: "expand: the requested shape has too few dimensions!",
    )
    offset = len(shape) - len(a.shape)
    shape_ = list(shape)
    for idx, x in enumerate(a.shape):
        offset_idx = idx + offset
        requested_length = shape[offset_idx]
        utils.check(
            requested_length == x or x == 1 or requested_length == -1,
            lambda: f"expand: attempting to expand a dimension of length {x}!",
        )

        shape_[offset_idx] = requested_length if requested_length != -1 else x
    utils.validate_shape(shape_)
    return prims.broadcast_in_dim(
        a, shape_, tuple(range(offset, len(a.shape) + offset))
    )


def _dim_size(a, dims):
    dims = utils.canonicalize_dims(a.ndim, dims)
    reduction_size = 1
    for idx, size in enumerate(a.size()):
        if idx in dims:
            reduction_size *= size
    return reduction_size


def _restore_reduced_dims(a, dims, shape):
    if a.size() == shape:
        return a
    unsqueezed_a = _unsqueeze_dims(a, dims, len(shape))
    return _expand(unsqueezed_a, shape)


# Reference: https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml#L1109-L1115
def _amax_amin_vjp(grad, result, self, dims, keepdim: Optional[bool] = None):
    expanded_grad = _restore_reduced_dims(grad, dims, self.shape)
    expanded_result = _restore_reduced_dims(result, dims, self.shape)
    mask = torch.eq(expanded_result, self)

    num_extra_none_results = len(dims)
    if keepdim is not None:
        num_extra_none_results += 1

    if num_extra_none_results == 0:
        return (expanded_grad / torch.sum(mask, dims, keepdim=True)) * mask
    else:
        # Return None for each dim and for keepdim argument.
        return (expanded_grad / torch.sum(mask, dims, keepdim=True)) * mask, *(
            None,
        ) * num_extra_none_results


def _sum_vjp(grad, result, self, dims):
    # Return None for each dim.
    return _restore_reduced_dims(grad, dims, self.shape), *(None,) * len(dims)


def _mean_vjp(grad, self, dims):
    mean_local_grad = 1.0 / _dim_size(self, dims)
    return _restore_reduced_dims(grad, dims, self.shape) * mean_local_grad


def _var_vjp(grad, result, self, dims, correction):
    var_reduction_size = _dim_size(self, dims)
    var_reduction_size -= correction
    constant = 2.0 / var_reduction_size

    # expand grad and mean tensors to self tensor size
    expanded_grad = _restore_reduced_dims(grad, dims, self.shape)
    mean = torch._refs.mean(self, dims, keepdim=True)
    expanded_mean = torch._refs.broadcast_to(mean, self.shape)
    var_local_grad = constant * prims.sub(self, expanded_mean)
    # Return None for each dim and for correction argument.
    return expanded_grad * var_local_grad, *(None,) * (len(dims) + 1)


def _broadcast_in_dim_vjp(grad, result, self, shape, broadcast_dimensions):
    # TODO: implement prims.sum_to and nvprims.sum_to
    pre_expand_shape = [
        1 if i not in broadcast_dimensions else x for i, x in enumerate(shape)
    ]
    grad = grad.sum_to_size(pre_expand_shape)
    squeeze_dims = [i for i in range(0, grad.ndim) if i not in broadcast_dimensions]
    grad = prims.squeeze(grad, squeeze_dims)
    return grad, *(None,) * (len(shape) + len(broadcast_dimensions))


def _squeeze_vjp(grad, result, self, dims):
    return _unsqueeze_dims(grad, dims, self.ndim), *(None,) * len(dims)


def _var_mean_vjp(grad_var, grad_mean, var, mean, self, dims, correction):
    grad = _var_vjp(grad_var, var, self, dims, correction)[0] + _mean_vjp(
        grad_mean, self, dims
    )
    return grad, *(None,) * (len(dims) + 1)


_vjp_impls: Dict[str, Any] = {
    "abs": lambda grad, result, self: prims.mul(grad, prims.sign(self)),
    "acos": lambda grad, result, self: prims.mul(
        grad, prims.neg(prims.rsqrt(prims.sub(1, prims.pow(self, 2))))
    ),
    "add": lambda grad, result, self, other: (grad, grad),
    "amax": _amax_amin_vjp,
    "amin": _amax_amin_vjp,
    "asin": lambda grad, result, self: prims.mul(
        grad, prims.rsqrt(prims.sub(1, prims.pow(self, 2)))
    ),
    "atan": lambda grad, result, self: prims.mul(
        grad, prims.reciprocal(prims.add(1, prims.pow(self, 2)))
    ),
    "atan2": lambda grad, result, self, other: (
        prims.mul(
            grad, prims.div(other, prims.add(prims.pow(self, 2), prims.pow(other, 2)))
        ),
        prims.mul(
            grad,
            prims.div(
                prims.neg(self), prims.add(prims.pow(self, 2), prims.pow(other, 2))
            ),
        ),
    ),
    "atanh": lambda grad, result, self: prims.mul(
        grad, prims.reciprocal(prims.sub(1, prims.pow(self, 2)))
    ),
    "bitwise_and": None,  # Only integers supported
    "bitwise_not": None,  # Only integers supported
    "bitwise_or": None,  # Only integers supported
    "bitwise_xor": None,  # Only integers supported
    "broadcast_in_dim": _broadcast_in_dim_vjp,
    "ceil": lambda grad, result, self: prims.mul(grad, 0),
    "convert_element_type": lambda grad, result, self, dtype: (
        prims.convert_element_type(grad, self.dtype),
        None,
    ),
    "cos": lambda grad, result, self: prims.mul(grad, prims.neg(prims.sin(self))),
    "cosh": lambda grad, result, self: prims.mul(grad, prims.sinh(self)),
    "div": lambda grad, result, self, other: (
        prims.div(grad, other),
        prims.mul(prims.mul(prims.neg(grad), self), prims.pow(other, -2)),
    ),
    "eq": None,
    "erf": lambda grad, result, self: prims.mul(
        grad,
        prims.mul(2 / math.sqrt(math.pi), prims.exp(prims.neg(prims.pow(self, 2)))),
    ),
    "erfc": lambda grad, result, self: prims.mul(
        grad,
        prims.mul(-2 / math.sqrt(math.pi), prims.exp(prims.neg(prims.pow(self, 2)))),
    ),
    "exp": lambda grad, result, self: prims.mul(grad, result),
    "expm1": lambda grad, result, self: prims.mul(grad, prims.add(result, 1)),
    "floor": lambda grad, result, self: prims.mul(grad, 0),
    "fmod": lambda grad, result, self, other: (
        grad,
        prims.mul(prims.neg(grad), prims.trunc(prims.div(self, other))),
    ),
    "ge": None,  # Output is not differentiable
    "gt": None,  # Output is not differentiable
    "imag": None,  # TODO
    "isfinite": None,  # Output is not differentiable
    "le": None,  # Output is not differentiable
    "lgamma": lambda grad, result, self: prims.mul(grad, prims.digamma(self)),
    "log": lambda grad, result, self: prims.div(grad, self),
    "log10": lambda grad, result, self: prims.div(grad, prims.mul(self, math.log(10))),
    "log1p": lambda grad, result, self: prims.div(grad, prims.add(self, 1)),
    "log2": lambda grad, result, self: prims.div(grad, prims.mul(self, math.log(2))),
    "lt": None,  # Output is not differentiable
    "mul": lambda grad, result, self, other: (
        prims.mul(grad, other),
        prims.mul(grad, self),
    ),
    "ne": None,  # Output is not differentiable
    "neg": lambda grad, result, self: prims.neg(grad),
    "pow": lambda grad, result, self, other: (
        prims.mul(grad, prims.mul(other, prims.pow(self, prims.sub(other, 1)))),
        prims.mul(grad, prims.mul(prims.log(self), result)),
    ),
    "real": None,  # TODO
    "reciprocal": lambda grad, result, self: prims.mul(
        grad, prims.neg(prims.pow(result, 2))
    ),
    "remainder": lambda grad, result, self, other: (
        grad,
        prims.neg(prims.mul(grad, prims.floor(prims.div(self, other)))),
    ),
    "round": lambda grad, result, self: prims.mul(grad, 0),
    "rsqrt": lambda grad, result, self: prims.mul(
        grad, prims.mul(-0.5, prims.div(result, self))
    ),
    "sign": lambda grad, result, self: prims.mul(grad, 0),
    "sin": lambda grad, result, self: prims.mul(grad, prims.cos(self)),
    "sinh": lambda grad, result, self: prims.mul(grad, prims.cosh(self)),
    "sqrt": lambda grad, result, self: prims.mul(grad, prims.div(0.5, result)),
    "squeeze": _squeeze_vjp,
    "sub": lambda grad, result, self, other: (grad, prims.neg(grad)),
    "sum": _sum_vjp,
    "tan": lambda grad, result, self: prims.mul(
        grad, prims.add(1, prims.pow(result, 2))
    ),
    "tanh": lambda grad, result, self: prims.mul(
        grad, prims.sub(1, prims.pow(result, 2))
    ),
    "trunc": lambda grad, result, self: prims.mul(grad, 0),
    "var": _var_vjp,
    "var_mean": _var_mean_vjp,
    "view_of": lambda grad, result, self: prims.view_of(grad),
    "where": lambda grad, result, condition, self, other: (
        None,
        prims.where(condition, grad, 0),
        prims.where(condition, 0, grad),
    ),
}


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

            # TODO: use save for backward to save the args
            # Only tensors can be saved for backward
            args_tensors = tuple(t for t in args if isinstance(t, torch.Tensor))
            ctx.ntensorargs = len(args_tensors)
            ctx.save_for_backward(*out_packed, *args_tensors)

            ctx.args = args
            ctx.kwargs = kwargs
            # ctx.save_for_backward(*out_packed)
            return out

        @staticmethod
        def backward(ctx, *bw_args):
            if vjp_impl is None:
                raise RuntimeError(f"backwards not supported on prim {prim.name}")

            # print(f"calling backward for prim {prim}")

            # TODO: use save for backward to save the args
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

            # print(f"vjp_result: {vjp_result}")
            # print(f"fw_args: {fw_args}")
            # print len
            flat_args_kwargs, _ = tree_flatten((ctx.args, ctx.kwargs))
            # print(f"len(vjp_result): {len(vjp_result)}")
            # print(f"len(flat_args_kwargs): {len(flat_args_kwargs)}")
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
