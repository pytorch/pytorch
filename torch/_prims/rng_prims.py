from typing import Optional, Tuple

import torch
from torch import _prims

from torch._prims_common import CUDARngStateHelper, make_contiguous_strides_for
from torch._prims_common.wrappers import backwards_not_supported
from torch.types import _device, _dtype

rngprim_namespace = "rngprims"
rngprim = torch.library.Library(rngprim_namespace, "DEF")
rngprim_impl = torch.library.Library(
    rngprim_namespace, "IMPL", "CompositeExplicitAutograd"
)
rngprim_autograd_impl = torch.library.Library(rngprim_namespace, "IMPL", "Autograd")
rngprim_meta_impl = torch.library.Library(rngprim_namespace, "IMPL", "Meta")


def throw_on_non_cuda(device):
    raise RuntimeError(
        f"You are trying to functionalize a {device.type} RNG operator but {device.type} does not "
        f"use Philox/counter-based RNG. Therefore, functionalizing a {device.type} RNG operator is "
        "not supported. We are discussing the possibility of a Philox-based RNG implementation for CPU."
    )


def register_philox_rand():
    name = "philox_rand"
    schema = "philox_rand(int[] size, Tensor seed, Tensor offset, int[]? stride, Device? device=None, ScalarType? dtype=None) -> Tensor"  # noqa: B950

    rngprim.define(schema)

    def _philox_rand_meta(
        shape: torch.Size,
        seed: torch.Tensor,
        offset: torch.Tensor,
        stride: Optional[Tuple[int, ...]],
        device: _device,
        dtype: _dtype,
    ):
        # stride arg will be useful for distributed usecase. Currently, its unused.
        assert stride is None
        stride = make_contiguous_strides_for(shape)
        return _prims.TensorMeta(
            shape=shape, strides=stride, dtype=dtype, device=device
        )

    def _philox_rand(
        shape: torch.Size,
        seed: torch.Tensor,
        offset: torch.Tensor,
        stride: Optional[Tuple[int, ...]],
        device: _device,
        dtype: _dtype,
    ):
        # stride arg will be useful for distributed usecase. Currently, its unused.
        assert stride is None
        if device.type == "cpu":
            devices = []
        else:
            devices = [device]

        if device.type != "cuda":
            raise throw_on_non_cuda(device)

        with torch.random.fork_rng(devices):
            CUDARngStateHelper.set_torch_state_tensor(seed, offset)
            return torch.rand(shape, device=device, dtype=dtype)

    rngprim_impl.impl(name, _philox_rand)
    rngprim_meta_impl.impl(name, _philox_rand_meta)

    prim_packet = getattr(torch._ops.ops.rngprims, name)
    prim = prim_packet.default
    prim._tags = (torch.Tag.nondeterministic_seeded,)  # type: ignore[attr-defined]

    rngprim_autograd_impl.impl(name, backwards_not_supported(prim))

    for p in (prim_packet, prim):
        p.__doc__ = "Philox based stateless rand operator"
        p.return_type = torch._prims_common.RETURN_TYPE.NEW  # type: ignore[attr-defined]

        p.schema = schema
        p.prim_meta_impl = _philox_rand_meta
        p.impl_aten = _philox_rand


def register_rng_prims():
    register_philox_rand()
