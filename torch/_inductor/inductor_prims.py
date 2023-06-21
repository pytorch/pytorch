import logging

import torch
from torch import _prims
from torch._prims_common import RETURN_TYPE

log = logging.getLogger(__name__)


def make_prim(
    schema,
    impl_aten,
    return_type=_prims.RETURN_TYPE.NEW,
    doc="",
    tags=None,
):
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


def eager_force_stride(input_tensor, stride):
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
    doc="Horizontally fusion of many inductor_seed() calls",
    tags=(torch.Tag.nondeterministic_seeded,),
)
lookup_seed = make_prim(
    # if inductor_lookup_seed changes, update partitioners.py
    "inductor_lookup_seed(Tensor seeds, int index) -> Tensor",
    lambda seeds, index: seeds[index],
    doc="Extract a single seed from the result of inductor_seeds()",
)
random = make_prim(
    "inductor_random(SymInt[] size, Tensor seed, str mode) -> Tensor",
    lambda size, seed, mode: getattr(torch, mode)(size, device=seed.device),
    doc="torch.rand()/torch.randn() using backend-specific RNG that can be fused",
)
randint = make_prim(
    "inductor_randint(SymInt low, SymInt high, SymInt[] size, Tensor seed) -> Tensor",
    lambda low, high, size, seed: torch.randint(low, high, size, device=seed.device),
    doc="torch.randint() using backend-specific RNG that can be fused",
)
force_stride_order = make_prim(
    "inductor_force_stride_order(Tensor input, SymInt[] stride) -> Tensor",
    lambda input_tensor, stride: eager_force_stride(input_tensor, stride),
    doc="Force the stride order for input tensor. No-op if the input tensor already has the stride. Do a copy otherwise",


def inductor_bucket_index_impl(values, offsets):
    assert values.dtype == torch.int32 or values.dtype == torch.int64
    assert offsets.dim() == 1
    result = torch.zeros_like(values)
    for idx in range(len(offsets) - 1, 0, -1):
        result[(values >= offsets[idx]) & (result == 0)] = idx
    return result


def inductor_bucket_index_meta(values, offsets):
    return torch.empty_like(values, memory_format=torch.preserve_format)


bucket_index = _prims._make_prim(
    schema="inductor_bucket_index(Tensor values, Tensor offsets) -> Tensor",
    meta=inductor_bucket_index_meta,
    impl_aten=inductor_bucket_index_impl,
    return_type=RETURN_TYPE.NEW,
    doc="""
    For a tensor `offsets` representing boundaries of different buckets,
    this will elementwise map each value into the corresponding bucket.
    For example, if offsets = [0, 2, 5, 9] (representing buckets [0, 2), [2, 5), [5, 9))
    then values = [0, 1, 2, 3, 4, 5, 6, 7, 8] will map to [0, 0, 1, 1, 1, 2, 2, 2, 2].
    """,
)
