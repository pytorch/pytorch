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


def register_rng_prim(name, schema, impl_aten, impl_meta, doc, tags=None):
    rngprim.define(schema)
    rngprim_impl.impl(name, impl_aten)
    rngprim_meta_impl.impl(name, impl_meta)

    prim_packet = getattr(torch._ops.ops.rngprims, name)
    prim = prim_packet.default
    if tags:
        prim._tags = tags

    rngprim_autograd_impl.impl(name, backwards_not_supported(prim))

    for p in (prim_packet, prim):
        p.__doc__ = doc
        p.return_type = torch._prims_common.RETURN_TYPE.NEW  # type: ignore[attr-defined]

        p.schema = schema
        p.impl_aten = impl_aten
        p.prim_meta_impl = impl_meta


def register_philox_rand():
    name = "philox_rand"
    schema = "philox_rand(SymInt[] size, Tensor seed, Tensor offset, int[]? stride, Device? device=None, ScalarType? dtype=None) -> Tensor"  # noqa: B950

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

    register_rng_prim(
        name=name,
        schema=schema,
        impl_aten=_philox_rand,
        impl_meta=_philox_rand_meta,
        doc="Philox based stateless rand operator",
        tags=(torch.Tag.nondeterministic_seeded,),  # type: ignore[attr-defined]
    )


def register_phiilox_rand_offset():
    name = "philox_rand_offset"
    schema = "philox_rand_offset(SymInt[] size) -> Tensor"  # noqa: B950

    def _philox_rand_offset_meta(
        shape: torch.Size,
    ):
        return _prims.TensorLike(torch.tensor(0, dtype=torch.int64))

    def _philox_rand_offset(
        shape: torch.Size,
    ):
        # For impl, look at the function calc_execution_policy in the file
        # aten/src/ATen/native/cuda/DistributionTemplates.h. The impl was copied at
        # commit hash 72aa0667bd16707d50eb8fa337092a1f5d11dfb6

        numel_scalar = 1
        for dim_size in shape:
            numel_scalar *= dim_size
        numel = torch.scalar_tensor(numel_scalar, dtype=torch.int64)

        block_size = 256
        unroll = 4
        curand4_engine_calls = 4
        device_property = torch.cuda.get_device_properties(torch.cuda.current_device())
        blocks_per_sm = device_property.max_threads_per_multi_processor // block_size
        grid_size = (numel + block_size - 1) // block_size
        grid_size = min(
            grid_size, device_property.multi_processor_count * blocks_per_sm
        )
        offset = (
            (numel - 1) // (block_size * grid_size * unroll) + 1
        ) * curand4_engine_calls
        return offset

    register_rng_prim(
        name=name,
        schema=schema,
        impl_aten=_philox_rand_offset,
        impl_meta=_philox_rand_offset_meta,
        doc="Computes offset for the philox_rand operator",
    )


def register_rng_prims():
    register_philox_rand()
    register_phiilox_rand_offset()
