# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import logging
import operator
from typing import Optional, TYPE_CHECKING

import torch
from torch import _prims, Tensor


if TYPE_CHECKING:
    from collections.abc import Sequence


log = logging.getLogger(__name__)


def make_prim(
    schema: str,
    impl_aten,
    return_type=_prims.RETURN_TYPE.NEW,
    doc: str = "",
    tags: Optional[Sequence[torch.Tag]] = None,
):
    if isinstance(return_type, tuple):

        def meta(*args, **kwargs):
            return tuple(_prims.TensorMeta(o) for o in impl_aten(*args, **kwargs))

    else:

        def meta(*args, **kwargs):
            return _prims.TensorMeta(impl_aten(*args, **kwargs))

    return _prims._make_prim(
        schema=schema,
        return_type=return_type,
        meta=meta,
        impl_aten=impl_aten,
        doc=doc,
        tags=tags,
    )


def eager_force_stride(input_tensor: Tensor, stride) -> Tensor:
    if input_tensor.stride() == stride:
        return input_tensor
    new_tensor = input_tensor.clone().as_strided(
        input_tensor.shape,
        stride,
    )
    new_tensor.copy_(input_tensor)
    return new_tensor


def eager_prepare_softmax(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    amax = torch.amax(x, dim, keepdim=True)
    return amax, torch.sum(torch.exp(x - amax), dim, keepdim=True)


# Custom prims used for handling randomness
seed = make_prim(
    "inductor_seed(Device device) -> Tensor",
    lambda device: torch.randint(2**63 - 1, [], device=device),
    doc="create a fresh seed (one per call) for use with inductor_rand",
    tags=(torch.Tag.nondeterministic_seeded,),
)
seeds = make_prim(
    "inductor_seeds(int count, Device device) -> Tensor",
    lambda count, device: torch.randint(2**63 - 1, [count], device=device),
    doc="Horizontal fusion of many inductor_seed() calls",
    tags=(torch.Tag.nondeterministic_seeded,),
)
lookup_seed = make_prim(
    # if inductor_lookup_seed changes, update partitioners.py
    "inductor_lookup_seed(Tensor seeds, int index) -> Tensor",
    lambda seeds, index: seeds[index].clone(),
    doc="Extract a single seed from the result of inductor_seeds()",
)
# inductor_random() doesn't accept a dtype.
# instead, its lowering always burns in float32, and conversions to a different type
# are explicit in the graph. We therefore need this impl (used during tracing) to hardcoded
# the dtype, so it always faithfully produces a float32 tensor during tracing,
# even if the default dtype is set to something else.
random = make_prim(
    "inductor_random(SymInt[] size, Tensor seed, str mode) -> Tensor",
    lambda size, seed, mode: getattr(torch, mode)(
        size, device=seed.device, dtype=torch.float32
    ),
    doc="torch.rand()/torch.randn() using backend-specific RNG that can be fused",
)
randint = make_prim(
    "inductor_randint(SymInt low, SymInt high, SymInt[] size, Tensor seed) -> Tensor",
    lambda low, high, size, seed: torch.randint(low, high, size, device=seed.device),
    doc="torch.randint() using backend-specific RNG that can be fused",
)
force_stride_order = make_prim(
    "inductor_force_stride_order(Tensor input, SymInt[] stride) -> Tensor",
    eager_force_stride,
    doc="Force the stride order for input tensor. No-op if the input tensor already has the stride. Do a copy otherwise",
)
_unsafe_index_put_ = make_prim(
    "_unsafe_index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)",
    lambda self, indices, values, accumulate=False: torch.ops.aten.index_put_(
        self, indices, values, accumulate
    ),
    doc="Unsafe index_put_ (doesn't issue device asserts)",
)
fma = make_prim(
    "fma(Tensor a, Tensor b, Tensor c) -> Tensor",
    lambda a, b, c: (a * b) + c,
    doc="Fused multiply add: fma(a, b, c) -> (a * b) + c without rounding after the multiplication",
    tags=(torch.Tag.pointwise,),
)
prepare_softmax_online = make_prim(
    "prepare_softmax_online(Tensor a, int dim) -> (Tensor, Tensor)",
    eager_prepare_softmax,
    return_type=(_prims.RETURN_TYPE.NEW, _prims.RETURN_TYPE.NEW),
    doc="Prepare the softmax by computing the max and sum.",
)


def _flattened_index_to_nd(indices, width):
    import sympy

    from torch.utils._sympy.functions import FloorDiv

    dim = len(width)

    if dim == 1:
        return [indices]
    elif dim >= 2:
        m = functools.reduce(operator.mul, width[1:])
        if isinstance(indices, sympy.Expr) or isinstance(m, sympy.Expr):
            ih = FloorDiv(indices, m)
        else:
            ih = indices // m
        indices_new = indices - (ih * m)
        return [ih, *_flattened_index_to_nd(indices_new, width[1:])]
    else:
        raise ValueError(f"Unknown dim: {dim}")


def _flatten_index(indices, width):
    result = indices[0]
    for d in range(1, len(indices)):
        result = width[d] * result + indices[d]
    return result


def _low_memory_max_pool_with_offsets_aten(
    self,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    dim = len(kernel_size)
    if dim == 2:
        vals, indices = torch.ops.aten.max_pool2d_with_indices(
            self, kernel_size, stride, padding, dilation, ceil_mode
        )
    else:
        vals, indices = torch.ops.aten.max_pool3d_with_indices(
            self, kernel_size, stride, padding, dilation, ceil_mode
        )

    idhw = _flattened_index_to_nd(indices, self.shape[-dim:])

    dhw_inc = []

    for d in range(dim):
        bh_shape = [1] * self.ndim
        bh_shape[-dim + d] = -1
        bh = torch.arange(
            indices.shape[-dim + d], dtype=torch.int64, device=self.device
        ).view(bh_shape)
        hbase = bh * stride[d] - padding[d]
        h_inc = (idhw[d] - hbase) // dilation[d]
        dhw_inc.append(h_inc)

    offsets = _flatten_index(dhw_inc, kernel_size)

    return vals, offsets.to(torch.int8)


def _low_memory_max_pool_offsets_to_indices_aten(
    offsets,
    kernel_size,
    input_size,
    stride,
    padding,
    dilation,
):
    dim = len(kernel_size)
    offsets = offsets.to(torch.int64)
    dhw_inc = _flattened_index_to_nd(offsets, kernel_size)

    idhw = []
    for d in range(dim):
        bh_shape = [1] * offsets.ndim
        bh_shape[-dim + d] = -1
        bh = torch.arange(
            offsets.shape[-dim + d], dtype=torch.int64, device=offsets.device
        ).view(bh_shape)
        hbase = bh * stride[d] - padding[d]
        idhw.append(hbase + dhw_inc[d] * dilation[d])

    return _flatten_index(idhw, input_size)


_low_memory_max_pool_with_offsets = make_prim(
    "_low_memory_max_pool_with_offsets(Tensor self, SymInt[] kernel_size, SymInt[] stride,  SymInt[] padding, SymInt[] dilation, bool ceil_mode) -> (Tensor, Tensor)",  # noqa: B950
    _low_memory_max_pool_with_offsets_aten,
    return_type=(_prims.RETURN_TYPE.NEW, _prims.RETURN_TYPE.NEW),
    doc="Instead of returning indices, returns indices offsets.",
)

_low_memory_max_pool_offsets_to_indices = make_prim(
    "_low_memory_max_pool_offsets_to_indices(Tensor self, SymInt[] kernel_size, SymInt[] input_size, SymInt[] stride, SymInt[] padding, SymInt[] dilation) -> Tensor",  # noqa: B950
    _low_memory_max_pool_offsets_to_indices_aten,
    doc="Convert small int offsets to regular indices.",
)


def _cvt_e8m0_rceil_aten(inp: Tensor) -> Tensor:
    """
    Convert float to e8m0 format with ceiling rounding and satfinite semantics.

    e8m0 format: 8-bit biased exponent (bias=127), no mantissa.
    For MX format scaling, this extracts the exponent with ceiling rounding.
    Uses satfinite semantics: inf is saturated to 254 (max finite e8m0).
    Accepts float32, float16, or bfloat16 (upcasted to float32 internally).
    """
    if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            f"cvt_e8m0_rceil requires float32, float16, or bfloat16 input, got {inp.dtype}"
        )
    if inp.dtype != torch.float32:
        inp = inp.to(torch.float32)
    inp_bits = inp.view(torch.int32)
    biased_exp = (inp_bits >> 23) & 0xFF
    mantissa = inp_bits & 0x7FFFFF
    needs_round_up = mantissa != 0
    e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
    # satfinite: clamp to max finite e8m0 value (254), not 255 (inf/nan)
    e8m0_biased = torch.clamp(e8m0_biased, 0, 254)
    return e8m0_biased.to(torch.uint8)


cvt_e8m0_rceil = make_prim(
    "inductor_cvt_e8m0_rceil(Tensor input) -> Tensor",
    _cvt_e8m0_rceil_aten,
    doc="Convert float to e8m0 with ceiling rounding. Uses PTX cvt.rp.satfinite.ue8m0x2.f32 on SM100+.",
)
