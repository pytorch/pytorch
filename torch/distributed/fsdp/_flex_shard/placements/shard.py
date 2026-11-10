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

from ..all_gather_layout import (
    AllGatherInputMetadata,
    AllGatherResult,
    foreach_all_gather_copy_out,
)
from ..bucket_runtime import _PLAIN_REPRESENTATION
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
from .shard_all_gather import ShardAllGather
from .utils import foreach_copy_, reduce_grads_on_one_rank, reduce_scatter_grads


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..all_gather_layout import AllGatherLayout
    from ..bucket_runtime import ParamRepresentation
    from ..bucket_storage import ParamInfo, PlacementFn
    from ..placement_contract import GradientReduction
    from .shard_all_gather import ParamAllGather


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

    The all-gather gathers each parameter's padded local shard, or the inputs
    that its representation gives (FSDP2's all-gather extensions), into
    outputs that back its unsharded tensor (see ``ShardAllGather``). A custom
    ``AllGatherLayout`` packs and finalizes them instead of the native copies.
    """

    # Finish copies the gathered values into a refill's persistent buffers;
    # run gathers into a separate buffer.
    refills_persistent_buffers_in_finish = True

    @dataclass(frozen=True)
    class _ReduceGradLayout:
        padded_sizes: list[torch.Size]

    @dataclass
    class _UnshardState:
        infos: list[ParamInfo]
        world_size: int
        pg: Any
        debug_fqn: str | None
        device: torch.device
        all_gather: ShardAllGather
        # Whether the bucket's all-gather keeps the outputs across unshards;
        # otherwise a refill's outputs are its persistent buffers
        bucket_keeps_outputs: bool
        # Whether each param all-gathers its padded local shard, rather than
        # inputs from its representation
        local_shards: list[bool]
        # Each param's inputs' dtypes and elements as all-gathered
        param_dtypes: list[list[torch.dtype]]
        param_numels: list[list[int]]
        # Each input's elements in a rank's chunk of the gather buffer (bytes
        # for a uint8 buffer) and its outer size (see _AllGatherOutputLayout)
        split_sizes: list[int]
        outer_sizes: list[int]
        # Whether FSDP's native copies run instead of the layout's (see
        # AllGatherLayout._uses_native_copies)
        native: bool
        # The layout that finalizes this unshard, and its metadata
        layout: AllGatherLayout
        output_metadata: object | None = None
        # An FSDP2 AllGather comm (e.g. symmetric memory) that allocated the
        # gather buffer and runs the collective, or None for all_gather_single
        comm: Any = None
        # A 1-rank unshard without all-gather: finish copies these inputs, on
        # the compute device, into the outputs
        single_rank_inputs: list[list[torch.Tensor]] | None = None

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

    @override
    def new_bucket_state(
        self, infos: list[ParamInfo], representation: ParamRepresentation
    ) -> ShardAllGather:
        return ShardAllGather(representation, len(infos))

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
        bucket_state: ShardAllGather | None = None,
    ) -> PlacementPreparedUnshard:
        """Prepare buffers for the bucket all-gather unshard. ``comm`` replaces
        the all-gather (FSDP2's custom ``AllGather``); ``device`` is where to
        gather local shards stored elsewhere (CPU offload); ``bucket_state`` is
        the bucket's all-gather (see ``new_bucket_state``)."""
        return self._prepare_unshard_bucket(
            tensors,
            infos,
            mesh,
            debug_fqn,
            skip_single_rank_all_gather=True,
            comm=comm,
            device=device,
            bucket_state=bucket_state,
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
        bucket_state: ShardAllGather | None = None,
    ) -> PlacementPreparedUnshard:
        """Prepare the send and gather buffers, or on a 1-rank mesh only the
        inputs when ``skip_single_rank_all_gather``. MixedBucket, which
        all-gathers every group in one collective, needs the buffers. Without
        ``bucket_state`` (a traced unshard, a MixedBucket group), the unshard
        keeps its own all-gather, whose outputs are the persistent buffers."""
        world_size, rank = mesh.size(), mesh.get_local_rank()
        # The collective's device, which differs from the storage's for CPU
        # offload
        device = tensors[0].device if device is None else device
        all_gather = bucket_state or ShardAllGather(_PLAIN_REPRESENTATION, len(infos))
        layout = all_gather.layout
        with _record_copy_in_if_eager():
            # Each param's flattened inputs from its representation, or None
            # for its padded local shard
            param_inputs = _record_all_gather_inputs(all_gather, infos, world_size)
            params = all_gather.params
            param_dtypes = [
                [info.unsharded_dtype] if inputs is None else [t.dtype for t in inputs]
                for inputs, info in zip(param_inputs, infos, strict=True)
            ]
            param_numels = [
                [info.storage_nbytes // info.dtype.itemsize]
                if inputs is None
                else [t.numel() for t in inputs]
                for inputs, info in zip(param_inputs, infos, strict=True)
            ]
            single_rank = world_size == 1 and skip_single_rank_all_gather
            dtypes = {dtype for input_dtypes in param_dtypes for dtype in input_dtypes}
            if not dtypes and not single_rank:
                raise ValueError(
                    "cannot all-gather a parameter group without any payloads"
                )
            dtype = dtypes.pop() if len(dtypes) == 1 else torch.uint8
            split_sizes = [
                numel * input_dtype.itemsize // dtype.itemsize
                for numels, input_dtypes in zip(param_numels, param_dtypes)
                for numel, input_dtype in zip(numels, input_dtypes)
            ]
            state = Shard._UnshardState(
                infos=infos,
                world_size=world_size,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                device=device,
                all_gather=all_gather,
                bucket_keeps_outputs=bucket_state is not None,
                local_shards=[inputs is None for inputs in param_inputs],
                param_dtypes=param_dtypes,
                param_numels=param_numels,
                split_sizes=split_sizes,
                outer_sizes=[
                    output_layout.outer_size
                    for param in params
                    for output_layout in param.output_layouts
                ],
                # The native copies read uneven Shard(i), i > 0, chunks stored
                # compactly, as local shards are, not padded inside their rows
                native=layout._uses_native_copies
                and not any(
                    param.drops_row_padding and inputs is not None
                    for param, inputs in zip(params, param_inputs, strict=True)
                ),
                layout=layout,
                comm=comm,
            )
            if single_rank:
                # As FSDP2 on a 1-rank group: no all-gather and no gather
                # buffer. Finish copies the inputs into the outputs, which live
                # on the compute device under CPU offload.
                local = (
                    _packed_local_shards(tensors, infos)
                    if all(inputs is None for inputs in param_inputs)
                    else None
                )
                if local is not None:
                    local = local.to(device, non_blocking=True)
                    buffers = [local]
                    state.single_rank_inputs = [
                        [t] for t in local.split([n for (n,) in param_numels])
                    ]
                else:
                    state.single_rank_inputs = [
                        [
                            t.to(device, non_blocking=True)
                            for t in (
                                [_local_shard_slot(tensor, info)]
                                if inputs is None
                                else inputs
                            )
                        ]
                        for inputs, tensor, info in zip(
                            param_inputs, tensors, infos, strict=True
                        )
                    ]
                    buffers = [t for ts in state.single_rank_inputs for t in ts]
                return PlacementPreparedUnshard(
                    placement=self, buffers=buffers, placement_state=state
                )
            numel = sum(split_sizes)
            copy_in = None
            if not state.native:
                # Copies, since copy_in and the default copy-out read these lists
                copy_in, state.layout, state.output_metadata = layout.prepare(
                    AllGatherInputMetadata(
                        input_split_sizes=list(split_sizes),
                        input_outer_sizes=list(state.outer_sizes),
                        input_numel=numel,
                        world_size=world_size,
                        dtype=dtype,
                        device=device,
                    )
                )
            size = (world_size * numel,)
            gather_buf = (
                torch.empty(size, dtype=dtype, device=device)
                if comm is None
                else comm.allocate(size, dtype=dtype, device=device)
            )
            if copy_in is None:
                send_buf = _native_copy_in(
                    tensors,
                    infos,
                    param_inputs,
                    split_sizes,
                    gather_buf.narrow(0, rank * numel, numel),
                    gather_in_place=comm is not None,
                )
            else:
                inputs = [
                    t.view(torch.uint8) if dtype == torch.uint8 else t
                    for inputs, tensor, info in zip(
                        param_inputs, tensors, infos, strict=True
                    )
                    for t in (
                        [_padded_input(tensor, info, world_size, device)]
                        if inputs is None
                        else inputs
                    )
                ]
                send_buf, gather_buf = copy_in(
                    inputs, gather_buf, state.split_sizes, numel, rank
                )
        return PlacementPreparedUnshard(
            placement=self, buffers=[send_buf, gather_buf], placement_state=state
        )

    @override
    def run_prepared_unshard(
        self, prepared: PlacementPreparedUnshard, *, async_op: bool = False
    ) -> dist.Work | None:
        """Launch the prepared bucket all-gather."""
        state = prepared.placement_state
        if not isinstance(state, Shard._UnshardState):
            raise AssertionError(
                f"Expected Shard._UnshardState, got {type(state).__name__}"
            )
        if state.single_rank_inputs is not None:
            return None
        send_buf, gather_buf = prepared.buffers
        with _record_comm_if_eager("FlexShard::all_gather", state.debug_fqn):
            if state.comm is None:
                return dist.all_gather_single(
                    gather_buf, send_buf, group=state.pg, async_op=async_op
                )
            kwargs = {}
            if state.layout.comm_type is not None:
                kwargs["output_metadata"] = state.output_metadata
            return state.comm(
                output_tensor=gather_buf,
                input_tensor=send_buf,
                group=state.pg,
                async_op=async_op,
                **kwargs,
            )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Finish the prepared bucket all-gather into the params' outputs and
        return the unsharded params built from them."""
        state = prepared.placement_state
        if not isinstance(state, Shard._UnshardState):
            raise AssertionError(
                f"Expected Shard._UnshardState, got {type(state).__name__}"
            )
        all_gather, params = state.all_gather, state.all_gather.params
        # A refill's persistent buffers
        refilled = prepared.persistent_buffers
        if refilled is not None and not state.bucket_keeps_outputs:
            all_gather.adopt_outputs(refilled)
        with _record_copy_out_if_eager():
            if state.single_rank_inputs is not None:
                _copy_single_rank_inputs(
                    params, state.single_rank_inputs, state.param_dtypes, state.device
                )
            elif state.native:
                # A refill's outputs are persistent buffers, which the runtime
                # re-allocated, but for those that a representation released
                # (releases_all_gather_outputs) before their sizes were recorded
                if refilled is None or all_gather.representation.defines_all_gathers():
                    for param, numels, dtypes in zip(
                        params, state.param_numels, state.param_dtypes, strict=True
                    ):
                        param.init_all_gather_outputs(
                            numels, dtypes, state.world_size, state.device
                        )
                        param.alloc_all_gather_outputs()
                _native_copy_out(
                    prepared.buffers[1],
                    params,
                    state.local_shards,
                    state.infos,
                    state.split_sizes,
                    state.outer_sizes,
                    state.world_size,
                )
            else:
                result = AllGatherResult(
                    prepared.buffers[1],
                    None,
                    None,
                    state.param_dtypes,
                    state.param_numels,
                    state.split_sizes,
                    state.outer_sizes,
                    state.layout,
                    state.output_metadata,
                    prepared.buffers[0],
                )
                try:
                    all_gather.layout_buffers = foreach_all_gather_copy_out(
                        result, params, state.pg, all_gather.layout_buffers
                    )
                except ValueError:
                    # Free what a rejected refill re-allocated, as FSDP2 does
                    if not all_gather.keep_storage:
                        if all_gather.layout_buffers is not None:
                            all_gather.layout_buffers.free()
                        for param in params:
                            param.free_all_gather_outputs()
                    raise
            full_params = _unsharded_tensors(
                all_gather, state.infos, refill=refilled is not None
            )
        if refilled is not None:
            return PlacementUnshardResult(full_params=[], persistent_buffers=refilled)
        persistent: dict[int, torch.Tensor] = {}
        if prepared.persistent:
            for param in params:
                unpadded = (
                    [] if param.unpadded_output is None else [param.unpadded_output]
                )
                for tensor in (
                    *param.all_gather_outputs,
                    *unpadded,
                    *param.inner_tensors,
                ):
                    persistent.setdefault(id(tensor), tensor)
            if all_gather.layout_buffers is not None:
                for tensor in all_gather.layout_buffers.tensors:
                    persistent.setdefault(id(tensor), tensor)
        return PlacementUnshardResult(
            full_params=full_params, persistent_buffers=list(persistent.values())
        )

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


def _copy_out_rank_chunks(
    outputs: list[torch.Tensor],
    rank_major: torch.Tensor,
    split_sizes: list[int],
    outer_sizes: list[int],
    uneven_infos: list[ParamInfo | None],
) -> None:
    """Copy each output's chunks out of ``rank_major``, whose row r holds rank
    r's chunk, one output at a time (both in the gather buffer's dtype, bytes
    for a uint8 buffer). Each rank's chunk is [outer size, rest], as for the
    native copy, filling a prefix of a larger output, except for an output in
    ``uneven_infos``: its param splits unevenly on a nonzero dim, so that its
    ranks' compact chunks are not rows of one padded layout.

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
    offset = 0
    for output, size, outer, info in zip(
        outputs, split_sizes, outer_sizes, uneven_infos, strict=True
    ):
        chunks = rank_major[:, offset : offset + size]
        offset += size
        if size == 0 or output.numel() == 0:
            continue
        if info is None:
            row = size // outer
            dsts.append(output[: world_size * size].view(outer, world_size, row))
            srcs.append(chunks.view(world_size, outer, row).transpose(0, 1))
            continue
        dim = _shard_dim(info)
        shape = info.global_shape
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


def _record_all_gather_inputs(
    all_gather: ShardAllGather, infos: list[ParamInfo], world_size: int
) -> list[list[torch.Tensor] | None]:
    """Record each param's all-gather inputs for this unshard and return them
    flattened: those its representation gives, or None for its padded local
    shard."""
    representation = all_gather.representation
    custom_layout = not all_gather.layout._rank_major_output_only
    if not representation.defines_all_gathers():
        # A local shard's output layout depends only on these
        key = (world_size, custom_layout)
        if all_gather.local_shards_key != key:
            for param, info in zip(all_gather.params, infos, strict=True):
                param.name = info.fqn
                param.use_local_shard(
                    shard_world_size=world_size,
                    shard_dim=_shard_dim(info),
                    global_shape=info.global_shape,
                    custom_layout=custom_layout,
                )
            all_gather.local_shards_key = key
        return [None] * len(infos)
    all_gather.local_shards_key = None
    param_inputs: list[list[torch.Tensor] | None] = []
    for index, (param, info) in enumerate(zip(all_gather.params, infos, strict=True)):
        param.name = info.fqn
        inputs = representation.all_gather_inputs(index)
        sharding = {
            "shard_world_size": world_size,
            "shard_dim": _shard_dim(info),
            "global_shape": info.global_shape,
            "custom_layout": custom_layout,
        }
        if inputs is None:
            param.use_local_shard(**sharding)
            param_inputs.append(None)
        else:
            param_inputs.append(param.flatten_inputs(inputs, **sharding))
    return param_inputs


def _local_shard_slot(tensor: torch.Tensor, info: ParamInfo) -> torch.Tensor:
    """The param's slot of the bucket storage that ``tensor``, its local shard,
    views: the local shard first, then zeros (see ``Shard``)."""
    itemsize = info.dtype.itemsize
    return torch.as_strided(
        tensor,
        (info.storage_nbytes // itemsize,),
        (1,),
        storage_offset=info.byte_offset // itemsize,
    )


def _packed_local_shards(
    tensors: list[torch.Tensor], infos: list[ParamInfo]
) -> torch.Tensor | None:
    """The slots of every param as one flat view of the bucket storage, if they
    lie back to back in one dtype."""
    first = infos[0]
    end = first.byte_offset
    for info in infos:
        if info.dtype != first.dtype or info.byte_offset != end:
            return None
        end += info.storage_nbytes
    itemsize = first.dtype.itemsize
    return torch.as_strided(
        tensors[0],
        ((end - first.byte_offset) // itemsize,),
        (1,),
        storage_offset=first.byte_offset // itemsize,
    )


def _padded_input(
    tensor: torch.Tensor, info: ParamInfo, world_size: int, device: torch.device
) -> torch.Tensor:
    """The param's padded local shard as FSDP2 all-gathers it, which an
    ``AllGatherLayout`` receives: on ``device``, in the unsharded dtype, and an
    uneven ``Shard(i)``, ``i > 0``, chunk padded at the end of each row rather
    than stored compactly."""
    slot = _local_shard_slot(tensor, info)
    if slot.device != device:  # CPU offload
        slot = slot.to(device, non_blocking=True)
    if not _has_uneven_chunks(info, world_size):
        return slot.to(info.unsharded_dtype)
    dim = _shard_dim(info)
    padded_size = list(info.global_shape)
    padded_size[dim] = -(-padded_size[dim] // world_size)
    padded = slot.new_zeros(padded_size, dtype=info.unsharded_dtype)
    padded.narrow(dim, 0, info.local_shape[dim]).copy_(
        slot[: info.local_numel].view(info.local_shape)
    )
    return padded.view(-1)


def _native_copy_in(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    param_inputs: list[list[torch.Tensor] | None],
    split_sizes: list[int],
    send_buf: torch.Tensor,
    *,
    gather_in_place: bool,
) -> torch.Tensor:
    """Copy the inputs into ``send_buf``, this rank's chunk of the gather
    buffer, and return the collective's input: the bucket storage itself if it
    holds every local shard back to back in the gather dtype, unless a custom
    comm gathers in place, else ``send_buf``, which the collective gathers in
    place, as FSDP2's copy-in does. Local shards are cast to their unsharded
    dtypes as they are copied."""
    if all(inputs is None for inputs in param_inputs) and all(
        info.unsharded_dtype == send_buf.dtype for info in infos
    ):
        local = _packed_local_shards(tensors, infos)
        if local is not None:
            if (
                local.dtype == send_buf.dtype
                and local.device == send_buf.device
                and not gather_in_place
            ):
                return local
            if send_buf.numel() > 0 and local.device != send_buf.device:
                send_buf.copy_(local, non_blocking=True)  # CPU offload
            elif send_buf.numel() > 0:
                foreach_copy_([send_buf], [local])
            return send_buf
    dsts, srcs = [], []
    offset = 0
    sizes = iter(split_sizes)
    for inputs, tensor, info in zip(param_inputs, tensors, infos, strict=True):
        for src in [_local_shard_slot(tensor, info)] if inputs is None else inputs:
            size = next(sizes)
            if size:
                if src.device != send_buf.device:  # CPU offload
                    src = src.to(send_buf.device, non_blocking=True)
                if send_buf.dtype == torch.uint8:
                    if inputs is None:
                        # Cast into a temporary, as FSDP2's copy-in does
                        src = src.to(info.unsharded_dtype)
                    src = src.view(torch.uint8)
                dsts.append(send_buf.narrow(0, offset, size))
                srcs.append(src)
            offset += size
    # Casts to the gather dtype as it copies
    foreach_copy_(dsts, srcs)
    return send_buf


def _native_copy_out(
    gathered: torch.Tensor,
    params: list[ParamAllGather],
    local_shards: list[bool],
    infos: list[ParamInfo],
    split_sizes: list[int],
    outer_sizes: list[int],
    world_size: int,
) -> None:
    """Copy the rank-major ``gathered`` output into the params' outputs with
    one ``fsdp::_all_gather_copy_out_``, or one output at a time when an output
    does not take every rank's whole chunk: the compactly stored chunks of an
    uneven ``Shard(i)``, ``i > 0``, local shard, or an extension's smaller
    payload, which fills a prefix of its output."""
    if gathered.numel() == 0:
        return
    byte_views = gathered.dtype == torch.uint8
    outputs: list[torch.Tensor] = []
    uneven: list[ParamInfo | None] = []
    one_copy = True
    for param, local_shard, info in zip(params, local_shards, infos, strict=True):
        compact = local_shard and param.drops_row_padding
        for output in param.all_gather_outputs:
            if byte_views:
                output = output.view(torch.uint8)
            if compact or (
                not local_shard
                and output.numel() != world_size * split_sizes[len(outputs)]
            ):
                one_copy = False
            outputs.append(output)
            uneven.append(info if compact else None)
    if not one_copy:
        _copy_out_rank_chunks(
            outputs, gathered.view(world_size, -1), split_sizes, outer_sizes, uneven
        )
        return
    keep = [i for i, size in enumerate(split_sizes) if size]
    if keep:
        # One copy places every rank's chunk: each output is a [outer size,
        # world size, rest] view of its full layout
        torch.ops.fsdp._all_gather_copy_out_(
            [outputs[i] for i in keep],
            gathered,
            [split_sizes[i] for i in keep],
            [outer_sizes[i] for i in keep],
            world_size,
        )


def _copy_single_rank_inputs(
    params: list[ParamAllGather],
    param_inputs: list[list[torch.Tensor]],
    param_dtypes: list[list[torch.dtype]],
    device: torch.device,
) -> None:
    """As FSDP2 on one rank: copy the inputs into the outputs, casting local
    shards to their unsharded dtypes."""
    dsts, srcs = [], []
    for param, inputs, dtypes in zip(params, param_inputs, param_dtypes, strict=True):
        param.init_all_gather_outputs([t.numel() for t in inputs], dtypes, 1, device)
        param.alloc_all_gather_outputs()
        for output, inp in zip(param.all_gather_outputs, inputs, strict=True):
            if output.dtype == inp.dtype and output.numel() != inp.numel():
                output = output[: inp.numel()]
            elif inp.dtype == torch.uint8 != output.dtype:
                output = output.view(torch.uint8)
            dsts.append(output)
            srcs.append(inp)
    foreach_copy_(dsts, srcs)


def _unsharded_tensors(
    all_gather: ShardAllGather, infos: list[ParamInfo], *, refill: bool
) -> list[torch.Tensor]:
    """Return each param's unsharded tensor: its output, or the tensor that its
    representation builds from its outputs. A refill fills the outputs behind
    the tensors of the first unshard, which the representation may rebuild in
    place (FSDP2's ``fsdp_post_all_gather(..., out=...)``), and returns none."""
    representation = all_gather.representation
    defines_all_gathers = representation.defines_all_gathers()
    if refill and not defines_all_gathers:
        return []
    tensors: list[torch.Tensor] = []
    for index, (param, info) in enumerate(zip(all_gather.params, infos, strict=True)):
        outputs = param.unsharded_outputs
        built = (
            representation.post_all_gather(index, outputs, refill)
            if defines_all_gathers
            else None
        )
        if refill:
            continue
        if built is None:
            if len(outputs) != 1:
                raise AssertionError(
                    f"Expected 1 all_gather_output, got {len(outputs)}"
                )
            param.inner_tensors = []
            tensors.append(outputs[0][: info.global_numel].view(info.global_shape))
            continue
        tensor, inner_tensors = built[0], list(built[1])
        if param.outputs_view_layout_buffers:
            # Inner tensors that view the layout buffers are the bucket's to free
            ptrs = {t.untyped_storage().data_ptr() for t in param.all_gather_outputs}
            inner_tensors = [
                t for t in inner_tensors if t.untyped_storage().data_ptr() not in ptrs
            ]
        param.inner_tensors = inner_tensors
        # The representation rebuilds this param in place on refills, so it
        # need not view the persistent buffers, and the runtime takes it as is
        unsharded_param = nn.Parameter(
            representation.unsharded_param(
                index,
                torch.as_strided(
                    tensor,
                    info.global_shape,
                    info.global_stride,
                    storage_offset=tensor.storage_offset(),
                ),
            ),
            requires_grad=info.requires_grad,
        )
        for name, value in info.param_attrs.items():
            setattr(unsharded_param, name, value)
        unsharded_param.grad_dtype = info.unsharded_grad_dtype
        tensors.append(unsharded_param)
    if defines_all_gathers:
        for index, param in enumerate(all_gather.params):
            if representation.releases_all_gather_outputs(index):
                param.free_all_gather_outputs()
    return tensors


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
