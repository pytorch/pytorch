from typing import Tuple

import torch
from torch import _prims
from torch._decomp.decompositions_for_rng import RNGStateHelper, throw_on_cpu
from torch.types import _device, _dtype


def _philox_rand(
    shape: torch.Size,
    seed: torch.Tensor,
    offset: torch.Tensor,
    stride: Tuple[int, ...],
    device: _device,
    dtype: _dtype,
):
    # stride arg will be useful for distributed usecase. Currently, its unused.
    if device.type == "cpu":
        devices = []
    else:
        devices = [device]

    if device.type == "cpu":
        raise throw_on_cpu()
    devices = [device]

    with torch.random.fork_rng(devices):
        RNGStateHelper.set_torch_state_tensor(seed, offset)
        return torch.rand(shape, device=device, dtype=dtype)


def _philox_rand_meta(
    shape: torch.Size,
    seed: torch.Tensor,
    offset: torch.Tensor,
    stride: Tuple[int, ...],
    device: _device,
    dtype: _dtype,
):
    return _prims.TensorMeta(shape=shape, strides=stride, dtype=dtype, device=device)


def register_rng_prims():
    _prims._make_prim(
        schema="philox_rand(int[] size, Tensor seed, Tensor offset, int[] stride, Device? device=None, ScalarType? dtype=None) -> Tensor",  # noqa: B950
        return_type=_prims.RETURN_TYPE.NEW,
        meta=_philox_rand_meta,
        impl_aten=_philox_rand,
        tags=(torch.Tag.nondeterministic_seeded,),  # type: ignore[attr-defined]
        doc="",
    )
