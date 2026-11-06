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
    from torch.distributed.device_mesh import DeviceMesh

    from ..bucket_storage import ParamInfo, PlacementFn
    from ..placement_contract import GradientReduction


class Shard(Placement):
    """Symmetric sharding — parameter split along dim across all ranks.

    Parameters sharded on different dims share one bucket. As in FSDP2, a
    ``Shard(i)`` parameter with ``i > 0`` must split evenly; its chunks are
    reordered to dim 0 around the collectives, and only ``Shard(0)`` pads uneven
    chunks. A bucket copies its parameters directly between their layouts and
    the collective buffers with FSDP's native collective copies, as FSDP2's
    ``with_native_copy`` functions do: ``fsdp::_all_gather_copy_out_`` from
    pytorch/pytorch#197204 and ``fsdp::chunk_cat_mixed_dtype`` with
    ``num_leading_dims`` from pytorch/pytorch#200179.
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

    @dataclass(frozen=True)
    class _UnshardState:
        infos: list[ParamInfo]
        world_size: int
        pg: Any
        debug_fqn: str | None
        padded_layout: Shard._PaddedUnshardLayout
        # A 1-rank unshard without all-gather: finish copies the local shards.
        local_copy_out: bool = False

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
        layout: Shard._ReduceGradLayout
        rank: int
        world_size: int
        pg: Any
        debug_fqn: str | None

    def __init__(self, dim: int = 0):
        self.dim = dim

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

    def _check_shardable(self, fqn: str, shape: torch.Size, world_size: int) -> None:
        if not 0 <= self.dim < len(shape):
            raise ValueError(
                f"{self!r} is invalid for parameter {fqn!r} with shape {tuple(shape)}."
            )
        if self.dim > 0 and shape[self.dim] % world_size != 0:
            raise NotImplementedError(
                f"{self!r} requires parameter {fqn!r} with shape {tuple(shape)} to "
                f"split evenly over {world_size} ranks, as in FSDP2; only Shard(0) "
                "supports uneven sharding."
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
            placement._check_shardable(fqn, param.shape, world_size)
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
                    world_size * padded_local_numel,
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
    ) -> PlacementPreparedUnshard:
        """Prepare buffers for the bucket all-gather unshard."""
        return self._prepare_unshard_bucket(
            tensors, infos, mesh, debug_fqn, skip_single_rank_all_gather=True
        )

    def _prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
        *,
        skip_single_rank_all_gather: bool,
    ) -> PlacementPreparedUnshard:
        """Prepare the send and gather buffers, or on a 1-rank mesh only the
        local storage when ``skip_single_rank_all_gather``. MixedBucket, which
        all-gathers every group in one collective, needs the buffers."""
        ws = mesh.size()
        dtype = infos[0].unsharded_dtype
        device = tensors[0].device
        if any(info.unsharded_dtype != dtype for info in infos):
            raise ValueError("Shard requires one unsharded dtype per bucket.")

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
                # params.
                return PlacementPreparedUnshard(
                    placement=self,
                    buffers=[local_storage],
                    placement_state=Shard._UnshardState(
                        infos=infos,
                        world_size=ws,
                        pg=mesh.get_group(),
                        debug_fqn=debug_fqn,
                        padded_layout=padded_layout,
                        local_copy_out=True,
                    ),
                )
            gather_buf = torch.empty(ws * numel, dtype=dtype, device=device)
            if local_storage.dtype == dtype:
                send_buf = local_storage
            else:
                # Cast straight into this rank's chunk of the gather buffer and
                # all-gather in place, as FSDP2's copy-in does, rather than
                # into a send buffer that would live until finish.
                send_buf = gather_buf.narrow(0, mesh.get_local_rank() * numel, numel)
                if numel > 0:
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
            ),
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

    def _pack_reduce_scatter_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        world_size: int,
    ) -> tuple[torch.Tensor, Shard._ReduceGradLayout]:
        dtype = infos[0].grad_reduce_dtype
        device = tensors[0].device
        # Per grad: the number of dims before its shard dim, which the copy-in
        # keeps in each chunk, or 0 to chunk along dim 0.
        num_leading_dims = [0] * len(tensors)
        if world_size > 1:
            tensors = list(tensors)
            for idx, (tensor, info) in enumerate(zip(tensors, infos, strict=True)):
                if (shard_dim := info.placement.dim) == 0:
                    continue
                if tensor.is_contiguous():
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

        input_numel = sum(s.numel() for s in padded_sizes)
        send_buf = torch.empty(input_numel, dtype=dtype, device=device)
        send_buf_2d = send_buf.view(world_size, -1)
        # Grads may differ in dtype (per-parameter grad_dtype). The copy-in
        # casts each to the reduce dtype as it copies, in fused kernels for CUDA
        # bf16 or fp16 into fp32.
        torch.ops.fsdp.chunk_cat_mixed_dtype(
            tensors,
            dim=0,
            num_chunks=world_size,
            out=send_buf_2d,
            num_leading_dims=num_leading_dims,
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
            local_shape = info.placement.compute_local_shape(
                info.global_shape, rank, world_size
            )
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
    ) -> PlacementPreparedReduceGrad:
        """Pack full gradients for bucket reduce-scatter."""
        ws = mesh.size()
        with _record_function_if_eager("FlexShard::reduce_scatter_copy_in", debug_fqn):
            send_buf, layout = self._pack_reduce_scatter_grad(tensors, infos, ws)
            if send_buf.numel() % ws != 0:
                raise AssertionError(
                    f"Packed reduce-scatter buffer has {send_buf.numel()} elements, "
                    f"which is not divisible by world size {ws}."
                )
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
            recv_buf = torch.empty(
                send_buf.numel() // state.world_size,
                dtype=send_buf.dtype,
                device=send_buf.device,
            )
            with _record_comm_if_eager(
                "FlexShard::post_backward_reduce", state.debug_fqn
            ):
                reduce_scatter_grads(recv_buf, send_buf, reduction, state.pg)
            buffers = [recv_buf]
        with _record_function_if_eager(
            "FlexShard::reduce_scatter_copy_out", state.debug_fqn
        ):
            sharded_grads = self._unpack_reduce_scatter_grad(
                recv_buf, state.infos, state.layout, state.rank, state.world_size
            )
        return PlacementReduceGradResult(sharded_grads, buffers)


def make_shard_placement_fn(dim: int = 0) -> PlacementFn:
    """Return a placement function assigning ``Shard(dim)`` per parameter."""

    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        del mesh
        return {fqn: (Shard(dim),) for fqn, _ in named_params}

    return placement_fn


def per_param_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    """Shard(0) per parameter (FSDP2-style)."""
    return make_shard_placement_fn(0)(named_params, mesh)


__all__ = [
    "make_shard_placement_fn",
    "per_param_placements",
    "Shard",
]
