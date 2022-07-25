# Module for defining "primitive" operations executable by the nvFuser.
# This list exists to decouple main set of primitives from the ones that
# provide a lowering of the op to nvFuser’s Python interface.
# Mostly torch.ops.nvprims is a subset of the primitives in torch.ops.prims,
# but some additional primitives can be added in the future for the corresponding
# higher-level torch/aten functions.

import torch

from torch.utils._pytree import tree_flatten, tree_unflatten

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


def _autograd_fn(prim):
    class BackwardsNotSupported(torch.autograd.Function):
        @staticmethod
        def forward(ctx, args_spec, *flat_args):
            args, kwargs = tree_unflatten(flat_args, args_spec)  # type: ignore[arg-type]
            g = torch._C._AutoDispatchBelowAutograd()
            try:
                return prim(*args, **kwargs)
            finally:
                del g

        @staticmethod
        def backward(ctx, *args):
            raise RuntimeError("backwards not supported on prim")

    def _autograd_impl(*args, **kwargs):
        flat_args, args_spec = tree_flatten((args, kwargs))
        return BackwardsNotSupported.apply(args_spec, *flat_args)

    return _autograd_impl


def register_nvprims():
    """Registers all nvFuser primitives in the torch.ops.nvprims module."""
    for name in nvprim_names:
        main_prim = getattr(torch.ops.prims, name)

        nvprim.define(main_prim.schema)
        nvprim_impl.impl(name, main_prim.prim_impl)
        nvprim_meta_impl.impl(name, main_prim.prim_meta_impl)

        _prim_packet = getattr(torch.ops.nvprims, name)
        _prim = _prim_packet.default

        nvprim_autograd_impl.impl(name, _autograd_fn(_prim))

        for p in (_prim_packet, _prim):
            p.__doc__ = main_prim.__doc__
            p.impl_nvfuser = main_prim.impl_nvfuser  # type: ignore[attr-defined]
            p.return_type = main_prim.return_type  # type: ignore[attr-defined]
