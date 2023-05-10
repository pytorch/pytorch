import logging

import torch
from torch import _prims

log = logging.getLogger(__name__)


def make_prim(
    schema,
    impl_aten,
    return_type=_prims.RETURN_TYPE.NEW,
    doc="",
):
    def meta(*args, **kwargs):
        return _prims.TensorMeta(impl_aten(*args, **kwargs))

    return _prims._make_prim(
        schema=schema,
        return_type=return_type,
        meta=meta,
        impl_aten=impl_aten,
        doc=doc,
    )


# Custom prims used for handling randomness
seed = make_prim(
    "inductor_seed(Device device) -> Tensor",
    lambda device: torch.randint(2**63 - 1, [], device=device),
    doc="create a fresh seed (one per call) for use with inductor_rand",
)
seeds = make_prim(
    "inductor_seeds(int count, Device device) -> Tensor",
    lambda count, device: torch.randint(2**63 - 1, [count], device=device),
    doc="Horizontally fusion of many inductor_seed() calls",
)
lookup_seed = make_prim(
    # if inductor_lookup_seed changes, update partitioners.py
    "inductor_lookup_seed(Tensor(a) seeds, int index) -> Tensor(a)",
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
