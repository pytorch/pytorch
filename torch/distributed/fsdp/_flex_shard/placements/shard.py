# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.tensor import Shard as DTensorShard

from ..placement_contract import (
    BucketParamStorageLayout,
    BucketStorageLayout,
    GlobalLayout,
    Placement,
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementReduceGradResult,
    PlacementUnshardResult,
)
from ..utils import (
    _get_single_placement,
    _record_comm_if_eager,
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
    _record_function_if_eager,
)
from .utils import foreach_copy_, reduce_grads_on_one_rank, reduce_scatter_grads


if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.distributed.device_mesh import DeviceMesh

    from ..bucket_storage import ParamInfo, PlacementFn
    from ..placement_contract import GradientReduction


class Shard(Placement):
    """Symmetric sharding — parameter split along dim across all ranks.

    Parameters sharded on different dims share one bucket. Ranks hold
    ``torch.chunk`` chunks (full chunks, at most one short chunk, then empty
    ones), each stored compactly at the start of a slot of the first chunk's
    size, so that the ranks' slots line up in the collectives. ``Shard(i)``
    chunks with ``i > 0`` are reordered to dim 0 for the reduce-scatter, uneven
    ones padded there, and the all-gather copies uneven ones out row by row. A
    bucket copies its parameters directly between their layouts and the
    collective buffers with FSDP's native collective copies, as FSDP2's
    ``with_native_copy`` functions do: ``fsdp::_all_gather_copy_out_`` from
    pytorch/pytorch#197204 and ``fsdp::chunk_cat_mixed_dtype`` with
    ``num_leading_dims`` from pytorch/pytorch#200179; a bucket with uneven
    ``Shard(i)``, ``i > 0``, parameters copies each parameter separately.
    """

    # Finish copies the gathered values into a refill's persistent buffers;
    # run gathers into a separate buffer.
    refills_persistent_buffers_in_finish = True

    @dataclass(frozen=True)
    class _ReduceGradLayout:
        padded_sizes: list[torch.Size]

    @dataclass(frozen=True)
    class _PaddedUnshardLayout:
        padded_local_numels: tuple[int, ...]
        # Per-param mode, for buckets whose params differ in dtype: each
        # param's slot (offset, size) in a rank's chunk of the gather buffer,
        # in elements of slot_dtype (uint8 when the unsharded dtypes differ)
        slots: tuple[tuple[int, int], ...] | None = None
        slot_dtype: torch.dtype | None = None

    @dataclass(frozen=True)
    class _UnshardState:
        infos: list[ParamInfo]
        world_size: int
        pg: Any
        debug_fqn: str | None
        padded_layout: Shard._PaddedUnshardLayout
        # A 1-rank unshard without all-gather: finish copies the local shards.
        local_copy_out: bool = False
        # An FSDP2 AllGather comm (e.g. symmetric memory) that allocated the
        # gather buffer and runs the collective, or None for all_gather_single
        comm: Any = None

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
        layout: Shard._ReduceGradLayout
        rank: int
        world_size: int
        pg: Any
        debug_fqn: str | None
        comm: Any = None

    def __init__(self, dim: int = 0):
        self.dim = dim

    @override
    def _api_placement(self) -> DTensorShard:
        return DTensorShard(self.dim)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Shard):
            return False
        return self.dim == other.dim

    def __hash__(self) -> int:
        return hash((type(self), self.dim))

    def __repr__(self) -> str:
        return f"Shard({self.dim})"

    @override
    def bucket_compatibility_key(self) -> object:
        return type(self)

    def _check_shardable(self, fqn: str, shape: torch.Size) -> None:
        if not 0 <= self.dim < len(shape):
            raise ValueError(
                f"{self!r} is invalid for parameter {fqn!r} with shape {tuple(shape)}."
            )

    @override
    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        dim_size = global_shape[self.dim]
        chunk = (dim_size + world_size - 1) // world_size
        start = chunk * rank
        local_dim = min(dim_size, start + chunk) - min(dim_size, start)
        shape = list(global_shape)
        shape[self.dim] = local_dim
        return torch.Size(shape)

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        chunks = list(torch.chunk(param, world_size, dim=self.dim))
        empty_shape = list(param.shape)
        empty_shape[self.dim] = 0
        while len(chunks) < world_size:
            chunks.append(param.new_empty(empty_shape))
        return chunks[rank]

    @override
    def global_layout(
        self, info: ParamInfo, rank: int, world_size: int
    ) -> GlobalLayout:
        dim_size = info.global_shape[self.dim]
        chunk_size = (dim_size + world_size - 1) // world_size
        global_offset = [0] * len(info.global_shape)
        global_offset[self.dim] = min(chunk_size * rank, dim_size)
        has_data = info.local_numel > 0
        return GlobalLayout(
            global_shape=tuple(info.global_shape),
            global_offsets=(tuple(global_offset),) if has_data else (),
            local_offsets=((0,) * len(info.local_shape),) if has_data else (),
            local_sizes=(tuple(info.local_shape),) if has_data else (),
        )

    @override
    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout | None:
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        param_layouts: dict[str, BucketParamStorageLayout] = {}
        byte_offset = 0
        for fqn, param in named_params:
            placement = _get_single_placement(param_placements[fqn])
            if not isinstance(placement, Shard):
                raise AssertionError(f"Expected Shard for {fqn!r}, got {placement!r}")
            placement._check_shardable(fqn, param.shape)
            local_shape = placement.compute_local_shape(param.shape, rank, world_size)
            local_numel = local_shape.numel()
            # Rank 0 has the largest chunk; reserve its capacity on every rank.
            padded_numel = placement.compute_local_numel(param.shape, 0, world_size)
            storage_nbytes = padded_numel * param.dtype.itemsize
            param_layouts[fqn] = BucketParamStorageLayout(
                local_shape=local_shape,
                local_numel=local_numel,
                byte_offset=byte_offset,
                storage_nbytes=storage_nbytes,
            )
            byte_offset += storage_nbytes

        return BucketStorageLayout(
            param_layouts=param_layouts,
            total_bytes=byte_offset,
        )

    @override
    def copy_param_to_storage(
        self,
        byte_storage: torch.Tensor,
        info: ParamInfo,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        if param.device.type != "meta":
            data_nbytes = info.local_numel * info.dtype.itemsize
            padding_nbytes = info.storage_nbytes - data_nbytes
            if padding_nbytes > 0:
                byte_storage[
                    info.byte_offset + data_nbytes : info.byte_offset
                    + info.storage_nbytes
                ].zero_()
        super().copy_param_to_storage(
            byte_storage,
            info,
            param,
            rank,
            world_size,
        )

    def _padded_unshard_layout(
        self,
        infos: list[ParamInfo],
    ) -> Shard._PaddedUnshardLayout:
        itemsize = infos[0].dtype.itemsize
        padded_local_numels = tuple(info.storage_nbytes // itemsize for info in infos)
        return Shard._PaddedUnshardLayout(
            padded_local_numels=padded_local_numels,
        )

    def _get_padded_bucket_storage_view(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        layout: Shard._PaddedUnshardLayout,
    ) -> torch.Tensor:
        """Return the padded bucket-storage alias used as all-gather input."""
        itemsize = tensors[0].dtype.itemsize
        bucket_byte_offset = infos[0].byte_offset

        return torch.as_strided(
            tensors[0],
            (sum(layout.padded_local_numels),),
            (1,),
            storage_offset=bucket_byte_offset // itemsize,
        )

    def _copy_out_unshard(
        self,
        gathered: torch.Tensor,
        infos: list[ParamInfo],
        world_size: int,
        layout: Shard._PaddedUnshardLayout,
        persistent_buffers: list[torch.Tensor] | None = None,
        local_copy_out: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return full params and the padded buffers backing them.

        ``persistent_buffers`` are padded buffers from a previous persistent unshard
        to write into instead of allocating. Such a refill returns no full params:
        the caller keeps the ones from the first unshard, which view these buffers.
        """
        full_params: list[torch.Tensor] = []
        padded_full_params: list[torch.Tensor] = []
        for idx, (info, padded_local_numel) in enumerate(
            zip(infos, layout.padded_local_numels, strict=True)
        ):
            padded_full_param = (
                persistent_buffers[idx]
                if persistent_buffers is not None
                else torch.empty(
                    _full_param_numel(info, padded_local_numel, world_size),
                    dtype=info.unsharded_dtype,
                    device=gathered.device,
                )
            )
            padded_full_params.append(padded_full_param)
            if persistent_buffers is None:
                full_params.append(
                    padded_full_param[: info.global_numel].view(info.global_shape)
                )

        if local_copy_out:
            # The local shards are the full params: cast each straight into
            # its buffer, as FSDP2 initializes unsharded params on a 1-rank
            # group.
            foreach_copy_(
                padded_full_params,
                list(gathered.split(list(layout.padded_local_numels))),
            )
            return full_params, padded_full_params
        if any(_has_uneven_chunks(info, world_size) for info in infos):
            slots, offset = [], 0
            for numel in layout.padded_local_numels:
                slots.append((offset, numel))
                offset += numel
            _copy_out_rank_chunks(
                padded_full_params, gathered.view(world_size, -1), slots, infos
            )
            return full_params, padded_full_params
        # One copy places every rank's chunk: each padded param is a
        # [leading size, world size, rest] view of its full layout.
        torch.ops.fsdp._all_gather_copy_out_(
            padded_full_params,
            gathered,
            list(layout.padded_local_numels),
            [math.prod(info.global_shape[: info.placement.dim]) for info in infos],
            world_size,
        )
        return full_params, padded_full_params

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
        *,
        comm: Any = None,
        device: torch.device | None = None,
    ) -> PlacementPreparedUnshard:
        """Prepare buffers for the bucket all-gather unshard. ``comm`` replaces
        the all-gather (FSDP2's custom ``AllGather``); ``device`` is where to
        gather local shards stored elsewhere (CPU offload)."""
        return self._prepare_unshard_bucket(
            tensors,
            infos,
            mesh,
            debug_fqn,
            skip_single_rank_all_gather=True,
            comm=comm,
            device=device,
        )

    def _prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
        *,
        skip_single_rank_all_gather: bool,
        comm: Any = None,
        device: torch.device | None = None,
    ) -> PlacementPreparedUnshard:
        """Prepare the send and gather buffers, or on a 1-rank mesh only the
        local storage when ``skip_single_rank_all_gather``. MixedBucket, which
        all-gathers every group in one collective, needs the buffers."""
        ws = mesh.size()
        dtype = infos[0].unsharded_dtype
        # The collective's device, which differs from the storage's for CPU
        # offload
        device = tensors[0].device if device is None else device
        if any(
            info.unsharded_dtype != dtype or info.dtype != infos[0].dtype
            for info in infos
        ):
            return self._prepare_unshard_bucket_bytes(
                tensors,
                infos,
                mesh,
                debug_fqn,
                skip_single_rank_all_gather=skip_single_rank_all_gather,
                comm=comm,
                device=device,
            )

        with _record_copy_in_if_eager():
            padded_layout = self._padded_unshard_layout(infos)
            local_storage = self._get_padded_bucket_storage_view(
                tensors,
                infos,
                padded_layout,
            )
            numel = local_storage.numel()
            if ws == 1 and skip_single_rank_all_gather:
                # As FSDP2 on a 1-rank group: no all-gather and no gather
                # buffer. Finish copies the local shards into the unsharded
                # params, which live on the compute device under CPU offload.
                return PlacementPreparedUnshard(
                    placement=self,
                    buffers=[local_storage.to(device, non_blocking=True)],
                    placement_state=Shard._UnshardState(
                        infos=infos,
                        world_size=ws,
                        pg=mesh.get_group(),
                        debug_fqn=debug_fqn,
                        padded_layout=padded_layout,
                        local_copy_out=True,
                    ),
                )
            if comm is None:
                gather_buf = torch.empty(ws * numel, dtype=dtype, device=device)
            else:
                gather_buf = comm.allocate((ws * numel,), dtype=dtype, device=device)
            if (
                local_storage.dtype == dtype
                and local_storage.device == device
                and comm is None
            ):
                send_buf = local_storage
            else:
                # Cast straight into this rank's chunk of the gather buffer and
                # all-gather in place, as FSDP2's copy-in does, rather than
                # into a send buffer that would live until finish. Custom comms
                # (e.g. symmetric memory) also all-gather in place.
                send_buf = gather_buf.narrow(0, mesh.get_local_rank() * numel, numel)
                if numel > 0 and local_storage.device != device:
                    send_buf.copy_(local_storage, non_blocking=True)
                elif numel > 0:
                    foreach_copy_([send_buf], [local_storage])

        return PlacementPreparedUnshard(
            placement=self,
            buffers=[send_buf, gather_buf],
            placement_state=Shard._UnshardState(
                infos=infos,
                world_size=ws,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                padded_layout=padded_layout,
                comm=comm,
            ),
        )

    def _prepare_unshard_bucket_bytes(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
        *,
        skip_single_rank_all_gather: bool,
        comm: Any = None,
        device: torch.device | None = None,
    ) -> PlacementPreparedUnshard:
        """Prepare a bucket whose params differ in dtype, as FSDP2 does: each
        param is copied into its own slot of the gather buffer, which is uint8
        when the unsharded dtypes differ and the unsharded dtype otherwise."""
        ws = mesh.size()
        device = tensors[0].device if device is None else device
        padded_local_numels = tuple(
            info.storage_nbytes // info.dtype.itemsize for info in infos
        )
        locals_ = [
            torch.as_strided(
                tensor,
                (numel,),
                (1,),
                storage_offset=info.byte_offset // info.dtype.itemsize,
            )
            for tensor, info, numel in zip(
                tensors, infos, padded_local_numels, strict=True
            )
        ]
        unsharded_dtypes = {info.unsharded_dtype for info in infos}
        slot_dtype = (
            next(iter(unsharded_dtypes)) if len(unsharded_dtypes) == 1 else torch.uint8
        )
        # Slots are packed back to back, as in FSDP2's all-gather input
        slots = []
        offset = 0
        for info, numel in zip(infos, padded_local_numels, strict=True):
            size = numel * info.unsharded_dtype.itemsize // slot_dtype.itemsize
            slots.append((offset, size))
            offset += size
        chunk_numel = offset
        layout = Shard._PaddedUnshardLayout(
            padded_local_numels=padded_local_numels,
            slots=tuple(slots),
            slot_dtype=slot_dtype,
        )
        state = Shard._UnshardState(
            infos=infos,
            world_size=ws,
            pg=mesh.get_group(),
            debug_fqn=debug_fqn,
            padded_layout=layout,
            local_copy_out=ws == 1 and skip_single_rank_all_gather,
            comm=comm,
        )
        if state.local_copy_out:
            # CPU offload: the unsharded params live on the compute device
            locals_ = [t.to(device, non_blocking=True) for t in locals_]
            return PlacementPreparedUnshard(
                placement=self, buffers=locals_, placement_state=state
            )
        with _record_copy_in_if_eager():
            size = (ws * chunk_numel,)
            gather_buf = (
                torch.empty(size, dtype=slot_dtype, device=device)
                if comm is None
                else comm.allocate(size, dtype=slot_dtype, device=device)
            )
            send_buf = gather_buf.narrow(
                0, mesh.get_local_rank() * chunk_numel, chunk_numel
            )
            dsts, srcs = [], []
            for (slot_offset, slot_size), info, src in zip(
                slots, infos, locals_, strict=True
            ):
                if slot_size == 0:
                    continue
                if src.device != device:  # CPU offload
                    src = src.to(device, non_blocking=True)
                if slot_dtype == torch.uint8:
                    # Byte slots: cast into a temporary, as FSDP2's copy-in does
                    src = src.to(info.unsharded_dtype).view(torch.uint8)
                dsts.append(send_buf.narrow(0, slot_offset, slot_size))
                srcs.append(src)
            if dsts:
                # Casts to the slot dtype as it copies
                foreach_copy_(dsts, srcs)
        return PlacementPreparedUnshard(
            placement=self, buffers=[send_buf, gather_buf], placement_state=state
        )

    @override
    def run_prepared_unshard(
        self, prepared: PlacementPreparedUnshard, *, async_op: bool = False
    ) -> dist.Work | None:
        """Launch the prepared bucket all-gather."""
        if not isinstance(prepared.placement_state, Shard._UnshardState):
            raise AssertionError(
                "Expected Shard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        if prepared.placement_state.local_copy_out:
            return None
        send_buf = prepared.buffers[0]
        with _record_comm_if_eager(
            "FlexShard::all_gather",
            prepared.placement_state.debug_fqn,
        ):
            if (comm := prepared.placement_state.comm) is not None:
                return comm(
                    output_tensor=prepared.buffers[1],
                    input_tensor=send_buf,
                    group=prepared.placement_state.pg,
                    async_op=async_op,
                )
            return dist.all_gather_single(
                prepared.buffers[1],
                send_buf,
                group=prepared.placement_state.pg,
                async_op=async_op,
            )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Finish the prepared bucket all-gather and assemble full parameters."""
        if not isinstance(prepared.placement_state, Shard._UnshardState):
            raise AssertionError(
                "Expected Shard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        with _record_copy_out_if_eager():
            state = prepared.placement_state
            if state.padded_layout.slots is not None:
                full_params, padded_full_params = self._copy_out_unshard_bytes(
                    prepared, state
                )
                return PlacementUnshardResult(
                    full_params=full_params,
                    persistent_buffers=padded_full_params
                    if prepared.persistent
                    else [],
                )
            full_params, padded_full_params = self._copy_out_unshard(
                prepared.buffers[0 if state.local_copy_out else 1],
                state.infos,
                state.world_size,
                state.padded_layout,
                persistent_buffers=prepared.persistent_buffers,
                local_copy_out=state.local_copy_out,
            )

        return PlacementUnshardResult(
            full_params=full_params,
            persistent_buffers=padded_full_params if prepared.persistent else [],
        )

    def _copy_out_unshard_bytes(
        self, prepared: PlacementPreparedUnshard, state: Shard._UnshardState
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        layout = state.padded_layout
        persistent_buffers = prepared.persistent_buffers
        device = prepared.buffers[0].device
        full_params: list[torch.Tensor] = []
        padded_full_params: list[torch.Tensor] = []
        for idx, (info, numel) in enumerate(
            zip(state.infos, layout.padded_local_numels, strict=True)
        ):
            padded_full_param = (
                persistent_buffers[idx]
                if persistent_buffers is not None
                else torch.empty(
                    _full_param_numel(info, numel, state.world_size),
                    dtype=info.unsharded_dtype,
                    device=device,
                )
            )
            padded_full_params.append(padded_full_param)
            if persistent_buffers is None:
                full_params.append(
                    padded_full_param[: info.global_numel].view(info.global_shape)
                )
        if state.local_copy_out:
            foreach_copy_(padded_full_params, list(prepared.buffers))
            return full_params, padded_full_params
        # One copy places every rank's chunk. Byte views keep its rank
        # interleaving: a param's chunk is [leading size, rest] elements, i.e.
        # [leading size, rest * itemsize] bytes.
        slot_dtype = layout.slot_dtype
        outputs = [
            p if slot_dtype != torch.uint8 else p.view(torch.uint8)
            for p in padded_full_params
        ]
        slots = layout.slots
        if slots is None:
            raise AssertionError("The native copy-out needs the slots of its layout")
        if any(_has_uneven_chunks(info, state.world_size) for info in state.infos):
            _copy_out_rank_chunks(
                outputs,
                prepared.buffers[1].view(state.world_size, -1),
                list(slots),
                state.infos,
            )
            return full_params, padded_full_params
        keep = [i for i, (_, size) in enumerate(slots) if size]
        if keep:
            torch.ops.fsdp._all_gather_copy_out_(
                [outputs[i] for i in keep],
                prepared.buffers[1],
                [slots[i][1] for i in keep],
                [
                    math.prod(state.infos[i].global_shape[: _shard_dim(state.infos[i])])
                    for i in keep
                ],
                state.world_size,
            )
        return full_params, padded_full_params

    def _pack_reduce_scatter_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        world_size: int,
        comm: Any = None,
        copy_in_fn: Any = None,
    ) -> tuple[torch.Tensor, Shard._ReduceGradLayout]:
        shard_dims = [_shard_dim(info) for info in infos]
        copy_in_tensors, num_leading_dims, padded_sizes = (
            _reduce_scatter_copy_in_layout(tensors, shard_dims, world_size)
        )
        size = sum(s.numel() for s in padded_sizes)
        dtype = infos[0].grad_reduce_dtype
        device = tensors[0].device
        send_buf = (
            torch.empty(size, dtype=dtype, device=device)
            if comm is None
            else comm.allocate((size,), dtype=dtype, device=device)
        )
        if copy_in_fn is not None:
            # FSDP2's set_reduce_scatter_input_fn
            copy_in_fn(tensors, shard_dims=shard_dims, world_size=world_size)(send_buf)
        else:
            _reduce_scatter_copy_in(
                copy_in_tensors, num_leading_dims, world_size, send_buf
            )
        return send_buf, Shard._ReduceGradLayout(padded_sizes)

    def _unpack_reduce_scatter_grad(
        self,
        recv_buf: torch.Tensor,
        infos: list[ParamInfo],
        layout: Shard._ReduceGradLayout,
        rank: int,
        world_size: int,
    ) -> list[torch.Tensor]:
        results: list[torch.Tensor] = []
        flat_offset = 0
        for info, padded_size in zip(infos, layout.padded_sizes, strict=True):
            dim = _shard_dim(info)
            local_shape = info.placement.compute_local_shape(
                info.global_shape, rank, world_size
            )
            if dim != 0 and info.global_shape[dim] % world_size:
                # The shard arrives padded on its shard dim, as the copy-in
                # padded it: the padded strides view its data, which the
                # runtime copies compact after the post-reduce steps (see
                # begin_reduce_grad)
                padded_shape = list(info.global_shape)
                padded_shape[dim] = -(-padded_shape[dim] // world_size)
                shard = torch.as_strided(
                    recv_buf,
                    local_shape,
                    make_contiguous_strides_for(padded_shape),
                    storage_offset=recv_buf.storage_offset() + flat_offset,
                )
            else:
                shard = (
                    recv_buf.narrow(0, flat_offset, local_shape.numel())
                    .view(local_shape)
                    .contiguous()
                )
            results.append(shard)
            flat_offset += padded_size.numel() // world_size
        return results

    @override
    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
        *,
        comm: Any = None,
        copy_in_fn: Any = None,
    ) -> PlacementPreparedReduceGrad:
        """Pack full gradients for bucket reduce-scatter. ``comm`` replaces the
        reduce-scatter (FSDP2's custom ``ReduceScatter``) and ``copy_in_fn`` the
        copy-in (FSDP2's ``set_reduce_scatter_input_fn``)."""
        ws = mesh.size()
        with _record_function_if_eager("FlexShard::reduce_scatter_copy_in", debug_fqn):
            send_buf, layout = self._pack_reduce_scatter_grad(
                tensors, infos, ws, comm, copy_in_fn
            )
            if send_buf.numel() % ws != 0:
                raise AssertionError(
                    f"Packed reduce-scatter buffer has {send_buf.numel()} elements, "
                    f"which is not divisible by world size {ws}."
                )
        # As FSDP2: free the full grads once copied in, before the
        # reduce-scatter allocates its output
        tensors.clear()
        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[send_buf],
            placement_state=Shard._ReduceGradState(
                infos=infos,
                layout=layout,
                rank=mesh.get_local_rank(),
                world_size=ws,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                comm=comm,
            ),
        )

    @override
    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
        reduction: GradientReduction,
    ) -> PlacementReduceGradResult:
        """Reduce a prepared gradient request using reduce-scatter."""
        if not isinstance(prepared.placement_state, Shard._ReduceGradState):
            raise AssertionError(
                "Expected Shard._ReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        send_buf = prepared.buffers[0]
        if state.world_size == 1:
            # As FSDP2 on a 1-rank group, no reduce-scatter; unlike FSDP2, no
            # output buffer either: the packed grads are this rank's shards.
            reduce_grads_on_one_rank(send_buf, reduction)
            recv_buf, buffers = send_buf, []
        else:
            size = send_buf.numel() // state.world_size
            recv_buf = (
                torch.empty(size, dtype=send_buf.dtype, device=send_buf.device)
                if state.comm is None
                else state.comm.allocate(
                    (size,), dtype=send_buf.dtype, device=send_buf.device
                )
            )
            with _record_comm_if_eager(
                "FlexShard::post_backward_reduce", state.debug_fqn
            ):
                reduce_scatter_grads(
                    recv_buf, send_buf, reduction, state.pg, comm=state.comm
                )
            buffers = [recv_buf]
        with _record_function_if_eager(
            "FlexShard::reduce_scatter_copy_out", state.debug_fqn
        ):
            sharded_grads = self._unpack_reduce_scatter_grad(
                recv_buf, state.infos, state.layout, state.rank, state.world_size
            )
        return PlacementReduceGradResult(sharded_grads, buffers, flat_output=recv_buf)


def _reduce_scatter_copy_in_layout(
    tensors: list[torch.Tensor], shard_dims: list[int], world_size: int
) -> tuple[list[torch.Tensor], list[int], list[torch.Size]]:
    """Return the grads to copy in, each grad's number of dims before its shard
    dim that the copy-in keeps in each chunk (0 to chunk along dim 0), and each
    grad's padded size in the reduce-scatter input."""
    num_leading_dims = [0] * len(tensors)
    if world_size > 1:
        tensors = list(tensors)
        for idx, (tensor, shard_dim) in enumerate(
            zip(tensors, shard_dims, strict=True)
        ):
            if shard_dim == 0:
                continue
            if tensor.size(shard_dim) % world_size:
                # Each rank's chunk, zero-padded to the first chunk's size,
                # along dim 0, as FSDP2's copy-in does for uneven shards
                tensors[idx] = _pad_and_reorder_by_rank(tensor, shard_dim, world_size)
            elif tensor.is_contiguous():
                num_leading_dims[idx] = shard_dim
            else:
                # As FSDP2 does, reorder the grad along dim 0 so that each
                # rank's chunk is contiguous.
                tensors[idx] = torch.cat(
                    torch.chunk(tensor, world_size, dim=shard_dim), dim=0
                )
    padded_sizes: list[torch.Size] = []
    for tensor, leading_dims in zip(tensors, num_leading_dims, strict=True):
        padded_size = list(tensor.shape)
        if leading_dims == 0:
            padded_size[0] = (
                (tensor.size(0) + world_size - 1) // world_size
            ) * world_size
        padded_sizes.append(torch.Size(padded_size))
    return tensors, num_leading_dims, padded_sizes


def _has_uneven_chunks(info: ParamInfo, world_size: int) -> bool:
    """Whether ``info``'s parameter splits unevenly on a nonzero shard dim, so
    that the ranks' compact chunks are not rows of one padded layout."""
    dim = _shard_dim(info)
    return dim != 0 and info.global_shape[dim] % world_size != 0


def _full_param_numel(info: ParamInfo, padded_local_numel: int, world_size: int) -> int:
    """The elements of the buffer behind a full param: every rank's slot, or,
    for uneven chunks on a nonzero dim, only the param, which then needs no
    padding inside its rows."""
    if _has_uneven_chunks(info, world_size):
        return info.global_numel
    return world_size * padded_local_numel


def _copy_out_rank_chunks(
    outputs: list[torch.Tensor],
    rank_major: torch.Tensor,
    slots: list[tuple[int, int]],
    infos: list[ParamInfo],
) -> None:
    """Copy each param's chunks out of ``rank_major``, whose row r holds rank
    r's slots, into its output (both in the gather buffer's dtype, bytes for a
    uint8 buffer), one param at a time: the native copy-out reads every slot,
    while uneven chunks on a nonzero dim are not rows of one padded layout.

    For example, a (2, 5) param on Shard(1) over 4 ranks stores chunks of 2, 2,
    1 and 0 columns compactly, written here as p13 for P[1, 3] and .. for
    padding. One copy moves the full chunks and one the short chunk:

        rank_major (its slot)      output
        rank 0   p00 p01 p10 p11   p00 p01 p02 p03 p04
        rank 1   p02 p03 p12 p13   p10 p11 p12 p13 p14
        rank 2   p04 p14 ..  ..
        rank 3   ..  ..  ..  ..
    """
    world_size = rank_major.size(0)
    dsts: list[torch.Tensor] = []
    srcs: list[torch.Tensor] = []
    for output, (offset, size), info in zip(outputs, slots, infos, strict=True):
        if size == 0 or output.numel() == 0:
            continue
        chunks = rank_major[:, offset : offset + size]
        dim = _shard_dim(info)
        shape = info.global_shape
        outer = math.prod(shape[:dim])
        if not _has_uneven_chunks(info, world_size):
            # Each rank's chunk is [outer, size / outer], as for the native copy
            row = size // outer
            dsts.append(output[: world_size * size].view(outer, world_size, row))
            srcs.append(chunks.view(world_size, outer, row).transpose(0, 1))
            continue
        dim_size = shape[dim]
        chunk_size = -(-dim_size // world_size)
        num_full, short_size = divmod(dim_size, chunk_size)
        # Output elements per index of the shard dim
        inner = output.numel() // (outer * dim_size)
        rows = output.view(outer, dim_size, inner)
        if num_full:
            full = chunks[:num_full, : outer * chunk_size * inner]
            dsts.append(
                rows[:, : num_full * chunk_size].view(
                    outer, num_full, chunk_size, inner
                )
            )
            srcs.append(full.view(num_full, outer, chunk_size, inner).transpose(0, 1))
        if short_size:
            short = chunks[num_full, : outer * short_size * inner]
            dsts.append(rows[:, num_full * chunk_size :])
            srcs.append(short.view(outer, short_size, inner))
    if dsts:
        torch._foreach_copy_(dsts, srcs)


def _shard_dim(info: ParamInfo) -> int:
    """The shard dim of a parameter of a ``Shard`` bucket."""
    placement = info.placement
    if not isinstance(placement, Shard):
        raise AssertionError(f"Expected Shard for {info.fqn}, got {placement!r}")
    return placement.dim


def _pad_and_reorder_by_rank(
    grad: torch.Tensor, shard_dim: int, world_size: int
) -> torch.Tensor:
    """
    Returns grad's chunks on shard_dim, each zero-padded to the first chunk's
    size and stacked along dim 0, in one pass over grad. For example, a (2, 5)
    gradient on Shard(1) over 4 ranks gives the ranks 2, 2, 1 and 0 columns:

        grad                    shards (4, 2, 2), returned as (8, 2)
        g00 g01 g02 g03 g04     rank 0    rank 1    rank 2    rank 3
        g10 g11 g12 g13 g14     g00 g01   g02 g03   g04 0     0 0
                                g10 g11   g12 g13   g14 0     0 0
    """
    size = grad.shape[shard_dim]
    chunk_size = math.ceil(size / world_size)
    # Full chunks come first, then at most one short chunk, then empty ones
    num_full, short_size = divmod(size, chunk_size)
    shard_shape = (*grad.shape[:shard_dim], chunk_size, *grad.shape[shard_dim + 1 :])
    shards = grad.new_zeros(world_size, *shard_shape)
    # One copy moves the full chunks to a leading rank dim, a second the short chunk
    full = grad.narrow(shard_dim, 0, num_full * chunk_size)
    full = full.unflatten(shard_dim, (num_full, chunk_size)).movedim(shard_dim, 0)
    shards[:num_full].copy_(full)
    if short_size:
        short_chunk = grad.narrow(shard_dim, num_full * chunk_size, short_size)
        shards[num_full].narrow(shard_dim, 0, short_size).copy_(short_chunk)
    return shards.flatten(0, 1)


def _reduce_scatter_copy_in(
    tensors: list[torch.Tensor],
    num_leading_dims: list[int],
    world_size: int,
    out: torch.Tensor,
) -> None:
    # Grads may differ in dtype (per-parameter grad_dtype). The copy-in casts
    # each to the reduce dtype as it copies, in fused kernels for CUDA bf16 or
    # fp16 into fp32.
    torch.ops.fsdp.chunk_cat_mixed_dtype(
        tensors,
        dim=0,
        num_chunks=world_size,
        out=out.view(world_size, -1),
        num_leading_dims=num_leading_dims,
    )


def reduce_scatter_input_fn(
    unsharded_grads: list[torch.Tensor], *, shard_dims: list[int], world_size: int
):
    """The ``Shard`` placement's reduce-scatter copy-in in the form of an FSDP2
    reduce-scatter input function: it returns a function that copies the grads
    into a given reduce-scatter input."""
    tensors, num_leading_dims, _ = _reduce_scatter_copy_in_layout(
        unsharded_grads, shard_dims, world_size
    )

    def copy_in(reduce_scatter_input: torch.Tensor) -> None:
        _reduce_scatter_copy_in(
            tensors, num_leading_dims, world_size, reduce_scatter_input
        )

    return copy_in


def _engine_placements(
    param_placements: Mapping[str, tuple[Placement | DTensorShard, ...]],
) -> dict[str, tuple[Placement, ...]]:
    """``placement_fn``'s placements, with DTensor's ``Shard(dim)`` as the
    engine's ``Shard(dim)``."""
    return {
        fqn: tuple(
            Shard(placement.dim) if isinstance(placement, DTensorShard) else placement
            for placement in placements
        )
        for fqn, placements in param_placements.items()
    }


def make_shard_placement_fn(dim: int = 0) -> PlacementFn:
    """Return a placement function assigning DTensor's ``Shard(dim)`` per parameter."""

    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[DTensorShard]]:
        del mesh
        return {fqn: (DTensorShard(dim),) for fqn, _ in named_params}

    return placement_fn


def per_param_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[DTensorShard]]:
    """DTensor's ``Shard(0)`` per parameter (FSDP2-style)."""
    del mesh
    return {fqn: (DTensorShard(0),) for fqn, _ in named_params}


__all__ = [
    "make_shard_placement_fn",
    "per_param_placements",
    "Shard",
]
