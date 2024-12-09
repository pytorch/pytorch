# mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch
from torch import _prims, Tensor


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
    lambda seeds, index: seeds[index],
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
)


def _low_memory_max_pool2d_with_offsets_aten(
    self,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    vals, indices = torch.ops.aten.max_pool2d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode
    )

    input_width = self.shape[-1]
    kernel_width = kernel_size[1]

    bh_shape = [1] * self.ndim
    bh_shape[-2] = -1
    bh = torch.arange(indices.shape[-2], dtype=torch.int64, device=self.device).view(
        bh_shape
    )

    bw_shape = [1] * self.ndim
    bw_shape[-1] = -1
    bw = torch.arange(indices.shape[-1], dtype=torch.int64, device=self.device).view(
        bw_shape
    )

    hbase = bh * stride[0] - padding[0]
    wbase = bw * stride[1] - padding[1]

    ih = indices // input_width
    iw = indices - (ih * input_width)

    h_inc = ih - hbase
    w_inc = iw - wbase

    offsets = h_inc * kernel_width + w_inc

    return vals, offsets.to(torch.int8)


def _low_memory_max_pool2d_offsets_to_indices_aten(
    offsets, kernel_width, input_width, stride, padding
):
    offsets = offsets.to(torch.int64)
    h_inc = offsets // kernel_width
    w_inc = offsets - (h_inc * kernel_width)

    bh_shape = [1] * offsets.ndim
    bh_shape[-2] = -1
    bh = torch.arange(offsets.shape[-2], dtype=torch.int64, device=offsets.device).view(
        bh_shape
    )

    bw_shape = [1] * offsets.ndim
    bw_shape[-1] = -1
    bw = torch.arange(offsets.shape[-1], dtype=torch.int64, device=offsets.device).view(
        bw_shape
    )

    hbase = bh * stride[0] - padding[0]
    wbase = bw * stride[1] - padding[1]

    ih = hbase + h_inc
    iw = wbase + w_inc
    return ih * input_width + iw


_low_memory_max_pool2d_with_offsets = make_prim(
    "_low_memory_max_pool2d_with_offsets(Tensor self, SymInt[2] kernel_size, SymInt[2] stride,  SymInt[2] padding, SymInt[2] dilation, bool ceil_mode) -> (Tensor, Tensor)",  # noqa: B950
    _low_memory_max_pool2d_with_offsets_aten,
    return_type=(_prims.RETURN_TYPE.NEW, _prims.RETURN_TYPE.NEW),
    doc="Instead of returning indices, returns indices offsets.",
)

_low_memory_max_pool2d_offsets_to_indices = make_prim(
    "_low_memory_max_pool2d_offsets_to_indices(Tensor self, SymInt kernel_w, SymInt input_w, SymInt[2] stride, SymInt[2] padding) -> Tensor",  # noqa: B950
    _low_memory_max_pool2d_offsets_to_indices_aten,
    doc="Convert small int offsets to regular indices.",
)
