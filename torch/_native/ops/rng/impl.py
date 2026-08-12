"""aten distribution overrides: uniform_ and normal_ on CUDA.

These are the RNG members of the NULLARY family (no input tensor; every element's value
comes from the generator), so they reuse the same "the output carries shape/device/layout"
shape as fill_ / arange -- what they add is generator state, drawn through
Generator.philox_state (see _common).

uniform_ / normal_ are the ops with an explicit `CUDA:` dispatch; torch.rand / randn /
rand_like / randn_like are CompositeExplicitAutograd wrappers that allocate and then call
these, so overriding the two in-place ops serves the whole family.

Bit-exactness with aten is verified, not assumed: values AND the generator's offset
advancement match aten's distribution_nullary_kernel exactly for both distributions across
sizes (see test_rng_dsl.py). That matters beyond this call -- a wrong reservation would
desync every subsequent random op even if this one's values looked fine.
"""

import torch

from ... import cutedsl_utils as cu
from ...utils import capability as cap
from ...utils.lazy import LazyModule


if False:  # TYPE_CHECKING-style guard without the import cost
    from . import cutedsl_kernels as CK
else:
    CK = LazyModule("torch._native.ops.rng.cutedsl_kernels")


# fp32 only for now: aten picks its unroll_factor from
# sizeof(dist return)/sizeof(accscalar_t), which is 4 for float32 (float4/float) but 2 for
# float64 (double2/double) and needs curand's *_double box-muller. Halves go through fp32
# accumulation plus a narrowing store, a third layout. Each is a separate exactness proof,
# so they are added one at a time rather than assumed.
_DTYPES = (torch.float32,)


def _serveable(self) -> bool:
    # A default generator only: an explicit `generator=` argument means a non-default
    # engine whose state we are not reserving from, so decline it to aten.
    return (
        isinstance(self, torch.Tensor)
        and self.dtype in _DTYPES
        and self.is_contiguous()
        and self.numel() > 0
        and not self.is_neg()
        and not self.is_conj()
        and cap.dlpack_offset_ok(self)
        and not cap.is_traced(self)
        and cap.device_ok(self)
        and cap.on_current_device(self)
    )


def _uniform_cond(self, from_=0.0, to=1.0, *, generator=None):
    if generator is not None:
        return False
    return _serveable(self)


def _uniform_impl(self, from_=0.0, to=1.0, *, generator=None):
    # aten: value = rand * (to - from) + from in opmath_t, then the (0,1] -> [0,1) bound
    # reversal that maps an exact `to` back to `from` (pytorch#16706) -- the kernel needs
    # `to` itself for that comparison, hence the third argument.
    return CK.fill_random(
        self, "uniform", float(to) - float(from_), float(from_), float(to)
    )


def _normal_cond(self, mean=0.0, std=1.0, *, generator=None):
    if generator is not None:
        return False
    if std < 0.0:
        return False  # aten raises; let it
    return _serveable(self)


def _normal_impl(self, mean=0.0, std=1.0, *, generator=None):
    # aten: value = z * std + mean, z from curand_normal4's box-muller.
    return CK.fill_random(self, "normal", float(std), float(mean))


def register_rng_overrides() -> None:
    cu.register_op_override(
        "aten", "uniform_", "CUDA", cond=_uniform_cond, impl=_uniform_impl
    )
    cu.register_op_override(
        "aten", "normal_", "CUDA", cond=_normal_cond, impl=_normal_impl
    )
