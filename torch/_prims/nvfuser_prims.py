# Module for defining "primitive" operations executable by the nvFuser. This
# list exists to decouple main set of primitives from the ones that provide a
# lowering of the op to nvFuser’s Python interface. Mostly torch.ops.nvprims is
# a subset of the primitives in torch.ops.prims, but some additional primitives
# can be added in the future for the corresponding higher-level torch/aten
# functions.

from typing import Any, Dict

import torch

from torch._prims_common import (
    DimsSequenceType,
    getnvFuserDtype,
    ShapeType,
    TensorLikeType,
)

from torch._prims_common.wrappers import backwards_not_supported

nvprim_namespace = "nvprims"
nvprim = torch.library.Library(nvprim_namespace, "DEF")
nvprim_impl = torch.library.Library(
    nvprim_namespace, "IMPL", "CompositeExplicitAutograd"
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
_nvfuser_impls["sum"] = _sum_nvfuser
_nvfuser_impls["var"] = _var_nvfuser
_nvfuser_impls["amax"] = _amax_nvfuser
_nvfuser_impls["amin"] = _amin_nvfuser


def register_nvprims():
    """Registers all nvFuser primitives in the torch.ops.nvprims module."""
    for name in nvprim_names:
        main_prim = getattr(torch.ops.prims, name)

        nvprim.define(main_prim.schema)
        nvprim_impl.impl(name, main_prim.prim_impl)
        nvprim_meta_impl.impl(name, main_prim.prim_meta_impl)

        prim_packet = getattr(torch.ops.nvprims, name)
        prim = prim_packet.default

        nvprim_autograd_impl.impl(name, backwards_not_supported(prim))

        for p in (prim_packet, prim):
            p.__doc__ = main_prim.__doc__
            p.impl_nvfuser = _nvfuser_impls[name]
            p.return_type = main_prim.return_type  # type: ignore[attr-defined]
