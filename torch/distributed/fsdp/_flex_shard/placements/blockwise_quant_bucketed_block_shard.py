# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized all-gather on bucketed block sharding with blockwise scales.

``BlockwiseQuantBucketedBlockShard`` cuts a bucket into contiguous
``shard_block_rows`` row units over the original dense parameters. Every rank
owns complete block rows, so it can quantize its dense shard locally -- no
cross-rank ``amax``. The all-gather moves temporary **quantized data + scales**
buffers, and the gathered weight is **bit-identical** to "all-gather bf16, then
block-quantize".

Subclasses define the quantization format: the local quantizer, the number and
size of scale tensors per row chunk, and the order in which gathered per-rank
scale chunks form a parameter's global scales.

Stacked weights such as ``[num_linears, out_dim, in_dim]`` are planned and
quantized as their flattened ``[num_linears * out_dim, in_dim]`` matrix, the
operand stacked linears multiply by. A block row may therefore straddle two
stacked matrices.

The master weight stays dense bf16/fp32-sharded with no padding in sharded
storage. The gathered data and scales are compacted into one buffer that backs
the unsharded weights; in eager mode the bucket runtime keeps it as a persistent
buffer, freed on reshard and refilled on the next unshard.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, TYPE_CHECKING
from typing_extensions import override

import torch
import torch.distributed as dist
import torch.nn as nn

from ..placement_contract import (
    BucketParamStorageLayout,
    BucketStorageLayout,
    GlobalLayout,
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)
from ..utils import (
    _record_comm_if_eager,
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
)
from .block_shard import BucketedBlockShard
from .utils import foreach_copy_


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..bucket_storage import ParamInfo
    from ..placement_contract import Placement

_VEC_COPY_DTYPE = torch.int32
_VEC_COPY_NBYTES = _VEC_COPY_DTYPE.itemsize
# CUDA's vectorized copy paths -- ``CatArrayBatchedCopy_vectorized`` for
# ``torch.cat`` and ``TensorIterator``'s vectorized elementwise kernels for
# ``copy_`` -- require 16-byte-aligned source and destination pointers. Cat also
# requires each copied slice to contain a multiple of 16 bytes. Padding rank
# segments and scale starts satisfies the pointer requirement and lets common
# model shapes with naturally aligned payload lengths use the vectorized path.
_VEC_ALIGN_BYTES = 16


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _align_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _validate_block_size(block_size: int) -> None:
    if not isinstance(block_size, int):
        raise TypeError(f"block_size must be an int, got {type(block_size).__name__}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")


def _validate_2d_non_empty(shape: torch.Size, what: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"{what} expects 2D, got {tuple(shape)}")
    out_dim, in_dim = shape
    if out_dim <= 0 or in_dim <= 0:
        raise ValueError(f"{what} requires non-empty 2D shape, got {tuple(shape)}")
    return out_dim, in_dim


def _matrix_shape(shape: torch.Size) -> torch.Size:
    """Return the matrix a stacked weight is quantized as."""
    return torch.Size((math.prod(shape[:-1]), shape[-1]))


def _unravel_index(index: int, shape: Sequence[int]) -> tuple[int, ...]:
    coords = []
    for size in reversed(shape):
        index, coord = divmod(index, size)
        coords.append(coord)
    return tuple(reversed(coords))


def _scale_shape(shape: torch.Size, block_size: int) -> tuple[int, int]:
    _validate_block_size(block_size)
    out_dim, in_dim = _validate_2d_non_empty(shape, "blockwise scale")
    return (_ceil_div(out_dim, block_size), _ceil_div(in_dim, block_size))


def _padded_shape(shape: torch.Size, block_size: int) -> tuple[int, int]:
    scale_rows, scale_cols = _scale_shape(shape, block_size)
    return scale_rows * block_size, scale_cols * block_size


def _pad_2d_to_block_shape(
    tensor: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    _validate_block_size(block_size)
    out_dim, in_dim = _validate_2d_non_empty(tensor.shape, "blockwise tensor")
    padded_out, padded_in = _padded_shape(tensor.shape, block_size)
    if out_dim == padded_out and in_dim == padded_in:
        return tensor.contiguous()
    padded = torch.zeros(
        (padded_out, padded_in),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:out_dim, :in_dim].copy_(tensor)
    return padded


def _callable_name(value: Any) -> str:
    return getattr(value, "__qualname__", type(value).__qualname__)


class _ScaleRegion(NamedTuple):
    """Bytes ``[offset, offset + nbytes)`` of one rank's gathered payload."""

    rank: int
    offset: int
    nbytes: int


class BlockwiseQuantBucketedBlockShard(BucketedBlockShard):
    """Bucketed block shard that all-gathers blockwise-quantized weights.

    The bucket planner cuts the original dense parameters into contiguous
    ``shard_block_rows``-row units. Persistent shards stay dense and contain
    only real parameter rows; temporary unshard buffers hold one-byte quantized
    data, the scales grouped by scale tensor index, and tail padding for the
    all-gather.

    Subclasses implement ``_chunk_scale_numels``, ``_quantize_local_chunk``,
    ``_param_scale_view``, and ``_make_quantized_weight``, and may override
    ``_scale_regions`` for scale tensors whose per-rank chunks do not
    concatenate flat.
    """

    # One-byte quantized data dtype and the dtype of every scale tensor.
    qdata_dtype: torch.dtype
    scale_dtype: torch.dtype
    num_scale_tensors: int = 1

    @dataclass(frozen=True)
    class _BlockRowParam:
        """Per-parameter block-row span used by the bucket partition planner."""

        fqn: str
        global_shape: torch.Size
        param_offset: int

        @property
        def matrix_shape(self) -> torch.Size:
            return _matrix_shape(self.global_shape)

    @dataclass(frozen=True)
    class _BlockRowUnit:
        """One indivisible row-block unit, including a possible partial tail."""

        fqn: str
        row_start: int
        row_end: int
        dense_numel: int
        cost_bytes: int

    @dataclass(frozen=True)
    class _QuantChunk:
        """One rank-owned contiguous row range for a single parameter."""

        rank: int
        fqn: str
        row_start: int
        row_end: int
        global_dense_offset: int
        rank_dense_offset: int
        qdata_offset: int

    @dataclass(frozen=True)
    class _QuantPackChunk:
        """Static copy-in plan for one rank-local parameter chunk."""

        param_index: int
        fqn: str
        local_shape: torch.Size
        qdata_numel: int
        scale_numels: tuple[int, ...]

    @dataclass(frozen=True)
    class _QuantRankPackPlan:
        """Static copy-in plan for one rank.

        The scales follow the aligned data grouped by scale tensor index, and
        each group holds the rank's chunks in parameter order.
        """

        chunks: tuple[BlockwiseQuantBucketedBlockShard._QuantPackChunk, ...]
        qdata_numel: int
        scale_byte_offset: int
        packed_nbytes: int
        qdata_chunk_numels: tuple[int, ...]
        scale_chunk_nbytes: tuple[int, ...]

    @dataclass(frozen=True)
    class _QuantBucketMetadata:
        """Quantized communication layout derived from dense rank intervals."""

        chunks: tuple[BlockwiseQuantBucketedBlockShard._QuantChunk, ...]
        rank_pack_plans: tuple[BlockwiseQuantBucketedBlockShard._QuantRankPackPlan, ...]
        rank_qdata_numels: tuple[int, ...]
        rank_scale_numels: tuple[int, ...]
        rank_scale_byte_offsets: tuple[int, ...]
        rank_packed_nbytes: tuple[int, ...]
        packed_send_nbytes_per_rank: int
        # Per scale tensor index; see ``_scale_regions``.
        scale_regions_by_index: tuple[tuple[_ScaleRegion, ...], ...]

    @dataclass(frozen=True)
    class _QuantMetadataPlan:
        """Immutable metadata retaining its identity-keyed bucket layout."""

        bucket_layout: Any
        metadata: BlockwiseQuantBucketedBlockShard._QuantBucketMetadata

    @dataclass(frozen=True)
    class _QuantUnshardState:
        """Prepared-unshard state needed to all-gather and compact rank rows."""

        infos: list[ParamInfo]
        pg: Any
        debug_fqn: str | None
        metadata: BlockwiseQuantBucketedBlockShard._QuantBucketMetadata

    @dataclass(frozen=True)
    class _CopyOutPlan:
        """Rank-row payload regions of one bucket, with device copy descriptors.

        ``regions`` holds ``(rank, row byte offset, nbytes, out byte offset)``.
        ``descriptors`` is keyed by device, row stride and copy unit, in bytes.
        """

        metadata: BlockwiseQuantBucketedBlockShard._QuantBucketMetadata
        regions: tuple[tuple[int, int, int, int], ...]
        vec_aligned: bool
        descriptors: dict[tuple[torch.device, int, int], Any]

    @dataclass(frozen=True)
    class _LocalPackedSourceState:
        """Source identity and view state for one cached local shard."""

        storage: Any
        version: int
        storage_offset: int
        shape: torch.Size
        stride: tuple[int, ...]
        dtype: torch.dtype
        is_conj: bool
        is_neg: bool

    @dataclass(frozen=True)
    class _LocalPackedInfoState:
        """ParamInfo state that can change the quantized payload."""

        info: ParamInfo
        unsharded_dtype: torch.dtype

    @dataclass(frozen=True)
    class _LocalPackedCacheEntry:
        """Versioned local quantized send payload for one bucket and rank."""

        info_states: tuple[BlockwiseQuantBucketedBlockShard._LocalPackedInfoState, ...]
        source_states: tuple[
            BlockwiseQuantBucketedBlockShard._LocalPackedSourceState, ...
        ]
        metadata: BlockwiseQuantBucketedBlockShard._QuantBucketMetadata
        packed: torch.Tensor

    def __init__(
        self,
        *,
        world_size: int,
        shard_block_rows: int,
        qdata_dtype: torch.dtype,
    ) -> None:
        if world_size <= 0:
            raise ValueError(
                f"{type(self).__name__} world_size must be positive, got {world_size}."
            )
        super().__init__(blocks_per_rank=(1,) * world_size)
        _validate_block_size(shard_block_rows)
        if qdata_dtype.itemsize != 1:
            raise ValueError(
                f"{type(self).__name__} requires a one-byte qdata_dtype, got "
                f"{qdata_dtype}."
            )
        self.qdata_dtype = qdata_dtype
        self.shard_block_rows = shard_block_rows
        self._metadata_plans: dict[
            int,
            BlockwiseQuantBucketedBlockShard._QuantMetadataPlan,
        ] = {}
        self._local_packed_cache: dict[
            tuple[int, int, torch.device, Any],
            BlockwiseQuantBucketedBlockShard._LocalPackedCacheEntry,
        ] = {}
        self._copy_out_plans: dict[
            tuple[int, int, bool],
            BlockwiseQuantBucketedBlockShard._CopyOutPlan,
        ] = {}

    @property
    def world_size(self) -> int:
        return len(self.blocks_per_rank)

    @property
    def _scale_itemsize(self) -> int:
        return self.scale_dtype.itemsize

    # ------------------------------------------------------------------
    # Format hooks
    # ------------------------------------------------------------------

    def _chunk_scale_numels(self, rows: int, in_dim: int) -> tuple[int, ...]:
        """Element count of each scale tensor for a chunk starting on a unit edge."""
        raise NotImplementedError

    def _quantize_local_chunk(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Quantize one dense row chunk into row-major data and its scale tensors."""
        raise NotImplementedError

    def _scale_regions(
        self,
        scale_index: int,
        rank_pack_plans: tuple[_QuantRankPackPlan, ...],
    ) -> tuple[_ScaleRegion, ...]:
        """Regions that compact scale tensor ``scale_index`` of every parameter.

        Concatenating the regions yields those scale tensors in parameter
        order. By default each rank contributes its whole ``scale_index``
        group, so rank chunks concatenate flat into each parameter's scale
        tensor.
        """
        itemsize = self._scale_itemsize
        regions: list[_ScaleRegion] = []
        for rank, plan in enumerate(rank_pack_plans):
            start = plan.scale_byte_offset + itemsize * sum(
                sum(chunk.scale_numels[:scale_index]) for chunk in plan.chunks
            )
            nbytes = itemsize * sum(
                chunk.scale_numels[scale_index] for chunk in plan.chunks
            )
            regions.append(_ScaleRegion(rank, start, nbytes))
        return tuple(regions)

    def _param_scale_view(
        self,
        scale_index: int,
        scale: torch.Tensor,
        matrix_shape: torch.Size,
    ) -> torch.Tensor:
        """View one of a parameter's compacted scale tensors in its consumer shape."""
        raise NotImplementedError

    def _make_quantized_weight(
        self,
        qdata: torch.Tensor,
        scales: tuple[torch.Tensor, ...],
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor:
        """Construct the consumer tensor from gathered data and scales."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Block-row planner
    # ------------------------------------------------------------------

    @staticmethod
    def _can_partition_units(
        unit_costs: list[int],
        num_partitions: int,
        capacity: int,
    ) -> bool:
        partitions = 1
        current = 0
        for unit_cost in unit_costs:
            if unit_cost > capacity:
                return False
            if current + unit_cost <= capacity:
                current += unit_cost
            else:
                partitions += 1
                current = unit_cost
        return partitions <= num_partitions

    @staticmethod
    def _min_partition_capacity(
        unit_costs: list[int],
        num_partitions: int,
    ) -> int:
        if num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive, got {num_partitions}.")
        if not unit_costs:
            return 0
        low = max(
            max(unit_costs),
            (sum(unit_costs) + num_partitions - 1) // num_partitions,
        )
        high = sum(unit_costs)
        while low < high:
            mid = (low + high) // 2
            if BlockwiseQuantBucketedBlockShard._can_partition_units(
                unit_costs,
                num_partitions,
                mid,
            ):
                high = mid
            else:
                low = mid + 1
        return low

    @staticmethod
    def _rank_unit_cuts(
        unit_costs: list[int],
        world_size: int,
        capacity: int,
    ) -> tuple[int, ...]:
        cuts = [0]
        current = 0
        for unit_idx, unit_cost in enumerate(unit_costs):
            if current > 0 and current + unit_cost > capacity:
                cuts.append(unit_idx)
                current = 0
            current += unit_cost
        cuts.append(len(unit_costs))

        while len(cuts) - 1 < world_size:
            split_at = None
            for idx, (start, end) in enumerate(zip(cuts, cuts[1:])):
                if end - start > 1:
                    split_at = idx + 1
                    cuts.insert(split_at, start + 1)
                    break
            if split_at is None:
                cuts.append(cuts[-1])
        if len(cuts) - 1 > world_size:
            raise AssertionError(
                "Quantized block-row planner produced more rank intervals than ranks."
            )
        return tuple(cuts)

    def _validate_param_shape(self, fqn: str, shape: torch.Size) -> None:
        if len(shape) < 2:
            raise ValueError(
                f"{type(self).__name__} only supports 2D or stacked 2D weights, "
                f"but {fqn!r} has shape {tuple(shape)}."
            )
        if 0 in shape:
            raise ValueError(
                f"{type(self).__name__} requires non-empty weights, "
                f"but {fqn!r} has shape {tuple(shape)}."
            )

    def _build_block_row_units(
        self,
        params: list[_BlockRowParam],
    ) -> list[_BlockRowUnit]:
        qdata_itemsize = self.qdata_dtype.itemsize
        units: list[BlockwiseQuantBucketedBlockShard._BlockRowUnit] = []
        for param in params:
            out_dim, in_dim = param.matrix_shape
            for row_start in range(0, out_dim, self.shard_block_rows):
                row_end = min(row_start + self.shard_block_rows, out_dim)
                dense_numel = (row_end - row_start) * in_dim
                scale_numel = sum(self._chunk_scale_numels(row_end - row_start, in_dim))
                units.append(
                    BlockwiseQuantBucketedBlockShard._BlockRowUnit(
                        fqn=param.fqn,
                        row_start=row_start,
                        row_end=row_end,
                        dense_numel=dense_numel,
                        cost_bytes=dense_numel * qdata_itemsize
                        + scale_numel * self._scale_itemsize,
                    )
                )
        return units

    def _block_row_params_from_named(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
    ) -> tuple[list[_BlockRowParam], list[_BlockRowUnit]]:
        dtype = named_params[0][1].dtype
        params: list[BlockwiseQuantBucketedBlockShard._BlockRowParam] = []
        param_offset = 0
        for fqn, param in named_params:
            if param.dtype != dtype:
                raise ValueError(
                    f"{type(self).__name__} requires one dtype per bucket: "
                    f"{named_params[0][0]!r} uses {dtype} but {fqn!r} uses "
                    f"{param.dtype}."
                )
            placements = param_placements[fqn]
            if placements != (self,):
                raise ValueError(
                    f"{type(self).__name__} requires the same placement "
                    f"instance for every parameter in a bucket; {fqn!r} uses "
                    f"{placements!r}."
                )
            self._validate_param_shape(fqn, param.shape)
            params.append(
                BlockwiseQuantBucketedBlockShard._BlockRowParam(
                    fqn=fqn,
                    global_shape=param.shape,
                    param_offset=param_offset,
                )
            )
            param_offset += param.numel()
        return params, self._build_block_row_units(params)

    def _block_row_params_from_infos(
        self,
        infos: list[ParamInfo],
    ) -> tuple[list[_BlockRowParam], list[_BlockRowUnit]]:
        bucket_layout = self._bucket_layout(infos[0])
        params: list[BlockwiseQuantBucketedBlockShard._BlockRowParam] = []
        for info in infos:
            self._validate_param_shape(info.fqn, info.global_shape)
            param_layout = bucket_layout.param_layouts[info.fqn]
            params.append(
                BlockwiseQuantBucketedBlockShard._BlockRowParam(
                    fqn=info.fqn,
                    global_shape=info.global_shape,
                    param_offset=param_layout.param_offset,
                )
            )
        return params, self._build_block_row_units(params)

    def _build_quant_metadata(
        self,
        params: list[_BlockRowParam],
        units: list[_BlockRowUnit],
        unit_cuts: tuple[int, ...],
    ) -> _QuantBucketMetadata:
        params_by_fqn = {param.fqn: param for param in params}
        param_indices_by_fqn = {param.fqn: index for index, param in enumerate(params)}
        scale_itemsize = self._scale_itemsize
        chunks: list[BlockwiseQuantBucketedBlockShard._QuantChunk] = []
        rank_pack_plans: list[BlockwiseQuantBucketedBlockShard._QuantRankPackPlan] = []
        rank_qdata_numels: list[int] = []
        rank_scale_numels: list[int] = []
        for rank, (unit_start, unit_end) in enumerate(
            zip(unit_cuts[:-1], unit_cuts[1:], strict=True)
        ):
            rank_dense_offset = 0
            qdata_offset = 0
            rank_units = units[unit_start:unit_end]
            rank_pack_chunks: list[
                BlockwiseQuantBucketedBlockShard._QuantPackChunk
            ] = []
            idx = 0
            while idx < len(rank_units):
                first = rank_units[idx]
                last = first
                idx += 1
                while idx < len(rank_units) and rank_units[idx].fqn == first.fqn:
                    last = rank_units[idx]
                    idx += 1
                param = params_by_fqn[first.fqn]
                _, in_dim = param.matrix_shape
                row_start = first.row_start
                row_end = last.row_end
                chunk_dense_numel = (row_end - row_start) * in_dim
                chunks.append(
                    BlockwiseQuantBucketedBlockShard._QuantChunk(
                        rank=rank,
                        fqn=param.fqn,
                        row_start=row_start,
                        row_end=row_end,
                        global_dense_offset=param.param_offset + row_start * in_dim,
                        rank_dense_offset=rank_dense_offset,
                        qdata_offset=qdata_offset,
                    )
                )
                rank_pack_chunks.append(
                    BlockwiseQuantBucketedBlockShard._QuantPackChunk(
                        param_index=param_indices_by_fqn[param.fqn],
                        fqn=param.fqn,
                        local_shape=torch.Size((row_end - row_start, in_dim)),
                        qdata_numel=chunk_dense_numel,
                        scale_numels=self._chunk_scale_numels(
                            row_end - row_start,
                            in_dim,
                        ),
                    )
                )
                rank_dense_offset += chunk_dense_numel
                qdata_offset += chunk_dense_numel
            scale_numel = sum(sum(chunk.scale_numels) for chunk in rank_pack_chunks)
            rank_qdata_numels.append(qdata_offset)
            rank_scale_numels.append(scale_numel)
            rank_pack_plans.append(
                BlockwiseQuantBucketedBlockShard._QuantRankPackPlan(
                    chunks=tuple(rank_pack_chunks),
                    qdata_numel=qdata_offset,
                    scale_byte_offset=_align_up(qdata_offset, _VEC_ALIGN_BYTES),
                    packed_nbytes=_align_up(qdata_offset, _VEC_ALIGN_BYTES)
                    + scale_numel * scale_itemsize,
                    qdata_chunk_numels=tuple(
                        chunk.qdata_numel for chunk in rank_pack_chunks
                    ),
                    scale_chunk_nbytes=tuple(
                        chunk.scale_numels[scale_index] * scale_itemsize
                        for scale_index in range(self.num_scale_tensors)
                        for chunk in rank_pack_chunks
                    ),
                )
            )
        rank_scale_byte_offsets = tuple(
            _align_up(qdata_numel, _VEC_ALIGN_BYTES)
            for qdata_numel in rank_qdata_numels
        )
        rank_packed_nbytes = tuple(
            scale_byte_offset + scale_numel * scale_itemsize
            for scale_byte_offset, scale_numel in zip(
                rank_scale_byte_offsets,
                rank_scale_numels,
                strict=True,
            )
        )
        return BlockwiseQuantBucketedBlockShard._QuantBucketMetadata(
            chunks=tuple(chunks),
            rank_pack_plans=tuple(rank_pack_plans),
            rank_qdata_numels=tuple(rank_qdata_numels),
            rank_scale_numels=tuple(rank_scale_numels),
            rank_scale_byte_offsets=rank_scale_byte_offsets,
            rank_packed_nbytes=rank_packed_nbytes,
            packed_send_nbytes_per_rank=_align_up(
                max(rank_packed_nbytes, default=0),
                _VEC_ALIGN_BYTES,
            ),
            scale_regions_by_index=tuple(
                _merge_adjacent_regions(
                    self._scale_regions(scale_index, tuple(rank_pack_plans))
                )
                for scale_index in range(self.num_scale_tensors)
            ),
        )

    def _quant_metadata_from_infos(
        self,
        infos: list[ParamInfo],
    ) -> _QuantBucketMetadata:
        bucket_layout = self._bucket_layout(infos[0])
        plan = self._metadata_plans.get(id(bucket_layout))
        if plan is not None and plan.bucket_layout is bucket_layout:
            return plan.metadata

        params, units = self._block_row_params_from_infos(infos)
        unit_cuts = self._unit_cuts_from_rank_layout(
            units,
            bucket_layout.rank_offsets,
            bucket_layout.rank_numels,
        )
        metadata = self._build_quant_metadata(params, units, unit_cuts)
        self._metadata_plans[id(bucket_layout)] = self._QuantMetadataPlan(
            bucket_layout=bucket_layout,
            metadata=metadata,
        )
        return metadata

    def _unit_cuts_from_rank_layout(
        self,
        units: list[_BlockRowUnit],
        rank_offsets: tuple[int, ...],
        rank_numels: tuple[int, ...],
    ) -> tuple[int, ...]:
        unit_dense_offsets = [0]
        for unit in units:
            unit_dense_offsets.append(unit_dense_offsets[-1] + unit.dense_numel)
        offset_to_unit = {offset: idx for idx, offset in enumerate(unit_dense_offsets)}
        rank_ends = tuple(
            start + numel
            for start, numel in zip(rank_offsets, rank_numels, strict=True)
        )
        boundaries = (*rank_offsets, rank_ends[-1])
        if any(offset not in offset_to_unit for offset in boundaries):
            raise AssertionError(
                "Quantized block-row rank layout is not aligned to row-block units."
            )
        return tuple(offset_to_unit[offset] for offset in boundaries)

    def _chunk_numel(
        self,
        chunk: _QuantChunk,
        in_dim: int,
    ) -> int:
        return (chunk.row_end - chunk.row_start) * in_dim

    # ------------------------------------------------------------------
    # Local quantize + pack, with a forward/backward payload cache
    # ------------------------------------------------------------------

    @staticmethod
    def _local_packed_source_state(
        tensor: torch.Tensor,
    ) -> _LocalPackedSourceState:
        return BlockwiseQuantBucketedBlockShard._LocalPackedSourceState(
            # Keep the storage alive so allocator address reuse cannot turn a
            # replacement tensor into a false cache hit.
            storage=tensor.untyped_storage(),
            version=tensor._version,
            storage_offset=tensor.storage_offset(),
            shape=tensor.shape,
            stride=tensor.stride(),
            dtype=tensor.dtype,
            is_conj=tensor.is_conj(),
            is_neg=tensor.is_neg(),
        )

    @staticmethod
    def _local_packed_source_matches(
        tensor: torch.Tensor,
        state: _LocalPackedSourceState,
    ) -> bool:
        return (
            tensor._version == state.version
            and tensor.untyped_storage() is state.storage
            and tensor.storage_offset() == state.storage_offset
            and tensor.shape == state.shape
            and tensor.stride() == state.stride
            and tensor.dtype == state.dtype
            and tensor.is_conj() == state.is_conj
            and tensor.is_neg() == state.is_neg
        )

    @staticmethod
    def _local_packed_info_state(info: ParamInfo) -> _LocalPackedInfoState:
        return BlockwiseQuantBucketedBlockShard._LocalPackedInfoState(
            info=info,
            unsharded_dtype=info.unsharded_dtype,
        )

    @staticmethod
    def _local_packed_info_matches(
        info: ParamInfo,
        state: _LocalPackedInfoState,
    ) -> bool:
        return info is state.info and info.unsharded_dtype == state.unsharded_dtype

    @staticmethod
    def _local_packed_cache_context(
        tensors: list[torch.Tensor],
    ) -> tuple[torch.device, Any]:
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        stream = torch.cuda.current_stream(device)
        return device, stream

    def _pack_local_quantized_chunks(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        rank: int,
        metadata: _QuantBucketMetadata,
    ) -> torch.Tensor:
        if len(tensors) != len(infos):
            raise ValueError(
                "Quantized copy-in requires one tensor per ParamInfo, got "
                f"{len(tensors)} tensors and {len(infos)} infos."
            )
        pack_plan = metadata.rank_pack_plans[rank]
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        local_packed = torch.empty(
            metadata.packed_send_nbytes_per_rank,
            dtype=torch.uint8,
            device=device,
        )
        qdata_end = pack_plan.qdata_numel
        scale_start = pack_plan.scale_byte_offset
        payload_end = pack_plan.packed_nbytes
        if qdata_end < scale_start:
            local_packed[qdata_end:scale_start].zero_()
        if payload_end < local_packed.numel():
            local_packed[payload_end:].zero_()
        local_qdata_bytes = local_packed[:qdata_end]
        local_scale_bytes = local_packed[scale_start:payload_end]
        local_chunks: list[
            tuple[BlockwiseQuantBucketedBlockShard._QuantPackChunk, torch.Tensor]
        ] = []
        staging_dsts: list[torch.Tensor] = []
        staging_srcs: list[torch.Tensor] = []
        for chunk in pack_plan.chunks:
            info = infos[chunk.param_index]
            if info.fqn != chunk.fqn:
                raise AssertionError(
                    "Quantized pack plan parameter order changed: expected "
                    f"{chunk.fqn!r} at index {chunk.param_index}, got {info.fqn!r}."
                )
            tensor = tensors[chunk.param_index].reshape(chunk.local_shape)
            if tensor.numel() != chunk.qdata_numel:
                raise AssertionError(
                    "Quantized local dense shard does not match planned chunk for "
                    f"{chunk.fqn!r}: expected {chunk.qdata_numel} elements, got "
                    f"{tensor.numel()}."
                )
            quant_dtype = info.unsharded_dtype
            if tensor.is_contiguous() and tensor.dtype == quant_dtype:
                quant_input = tensor
            else:
                quant_input = torch.empty(
                    tensor.shape,
                    dtype=quant_dtype,
                    device=tensor.device,
                )
                if tensor.numel() > 0:
                    staging_dsts.append(quant_input)
                    staging_srcs.append(tensor)
            local_chunks.append((chunk, quant_input))

        foreach_copy_(staging_dsts, staging_srcs)

        qdata_srcs: list[torch.Tensor] = []
        scale_srcs_by_index: list[list[torch.Tensor]] = [
            [] for _ in range(self.num_scale_tensors)
        ]
        for chunk, quant_input in local_chunks:
            quant, scales = self._quantize_local_chunk(quant_input)
            if quant.numel() != chunk.qdata_numel:
                raise AssertionError(
                    "Quantizer returned the wrong data size for "
                    f"{chunk.fqn!r}: expected {chunk.qdata_numel}, got "
                    f"{quant.numel()}."
                )
            if len(scales) != len(chunk.scale_numels):
                raise AssertionError(
                    f"Quantizer returned {len(scales)} scale tensors for "
                    f"{chunk.fqn!r}, expected {len(chunk.scale_numels)}."
                )
            for scale_index, (scale, scale_numel) in enumerate(
                zip(scales, chunk.scale_numels)
            ):
                if scale.numel() != scale_numel:
                    raise AssertionError(
                        "Quantizer returned the wrong scale size for "
                        f"{chunk.fqn!r}: expected {scale_numel}, got "
                        f"{scale.numel()}."
                    )
                scale_srcs_by_index[scale_index].append(
                    scale.reshape(-1).view(torch.uint8)
                )
            qdata_srcs.append(quant.reshape(-1).view(torch.uint8))
        scale_srcs = [src for srcs in scale_srcs_by_index for src in srcs]

        if len(qdata_srcs) == 1 and len(scale_srcs) == 1:
            packed_dsts = [local_qdata_bytes, local_scale_bytes]
        elif qdata_srcs:
            packed_dsts = [
                *torch.split(local_qdata_bytes, pack_plan.qdata_chunk_numels),
                *torch.split(local_scale_bytes, pack_plan.scale_chunk_nbytes),
            ]
        else:
            packed_dsts = []
        foreach_copy_(packed_dsts, [*qdata_srcs, *scale_srcs])
        return local_packed

    def _cached_local_quantized_chunks(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        rank: int,
    ) -> tuple[torch.Tensor, _QuantBucketMetadata]:
        """Cache one forward payload for its backward re-gather.

        Some fused optimizers mutate parameters without advancing ``_version``.
        Forward therefore always refreshes the entry; backward may consume that
        entry once when every secondary source and execution-context check agrees.
        """
        metadata = self._quant_metadata_from_infos(infos)
        if torch.compiler.is_compiling():
            return (
                self._pack_local_quantized_chunks(tensors, infos, rank, metadata),
                metadata,
            )

        device, stream = self._local_packed_cache_context(tensors)
        cache_prefix = (rank, id(metadata))
        cache_key = (*cache_prefix, device, stream)
        if not _is_backward_graph_task():
            packed = self._pack_local_quantized_chunks(
                tensors,
                infos,
                rank,
                metadata,
            )
            for existing_key in tuple(self._local_packed_cache):
                if existing_key[:2] == cache_prefix:
                    del self._local_packed_cache[existing_key]
            self._local_packed_cache[cache_key] = self._LocalPackedCacheEntry(
                info_states=tuple(
                    self._local_packed_info_state(info) for info in infos
                ),
                source_states=tuple(
                    self._local_packed_source_state(tensor) for tensor in tensors
                ),
                metadata=metadata,
                packed=packed,
            )
            return packed, metadata

        cached = self._local_packed_cache.pop(cache_key, None)
        info_matches = (
            cached is not None
            and cached.metadata is metadata
            and len(cached.info_states) == len(infos)
            and all(
                self._local_packed_info_matches(info, state)
                for info, state in zip(infos, cached.info_states, strict=True)
            )
        )
        if (
            cached is not None
            and info_matches
            and len(cached.source_states) == len(tensors)
            and all(
                self._local_packed_source_matches(tensor, state)
                for tensor, state in zip(
                    tensors,
                    cached.source_states,
                    strict=True,
                )
            )
        ):
            return cached.packed, metadata

        return (
            self._pack_local_quantized_chunks(tensors, infos, rank, metadata),
            metadata,
        )

    # ------------------------------------------------------------------
    # Placement contract
    # ------------------------------------------------------------------

    @override
    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout:
        from ..bucket_storage import BucketLayout, BucketParamLayout

        if not named_params:
            return BucketStorageLayout(param_layouts={}, total_bytes=0)
        world_size = mesh.size()
        self._validate_world_size(world_size)

        rank = mesh.get_local_rank()
        dtype = named_params[0][1].dtype
        params, units = self._block_row_params_from_named(
            named_params,
            param_placements,
        )
        unit_costs = [unit.cost_bytes for unit in units]
        unit_dense_numels = [unit.dense_numel for unit in units]

        max_rank_cost_nbytes = self._min_partition_capacity(unit_costs, world_size)
        unit_cuts = self._rank_unit_cuts(
            unit_costs,
            world_size,
            max_rank_cost_nbytes,
        )
        unit_dense_offsets = [0]
        for unit_dense_numel in unit_dense_numels:
            unit_dense_offsets.append(unit_dense_offsets[-1] + unit_dense_numel)

        rank_offsets = tuple(unit_dense_offsets[cut] for cut in unit_cuts[:-1])
        rank_ends = tuple(unit_dense_offsets[cut] for cut in unit_cuts[1:])
        rank_numels = tuple(
            end - start for start, end in zip(rank_offsets, rank_ends, strict=True)
        )
        metadata = self._build_quant_metadata(params, units, unit_cuts)
        local_chunks = {
            chunk.fqn: chunk for chunk in metadata.chunks if chunk.rank == rank
        }

        bucket_param_layouts = {}
        storage_layouts: dict[str, BucketParamStorageLayout] = {}
        for param in params:
            chunk = local_chunks.get(param.fqn)
            # Local shards are matrix row ranges; stacked dims stay as size-1
            # dims so the shard keeps the parameter's rank.
            stack_ones = (1,) * (len(param.global_shape) - 2)
            in_dim = param.global_shape[-1]
            if chunk is None:
                local_shape = torch.Size([*stack_ones, 0, in_dim])
                local_numel = 0
                byte_offset = 0
                local_global_offset = param.param_offset
            else:
                local_shape = torch.Size(
                    [*stack_ones, chunk.row_end - chunk.row_start, in_dim]
                )
                local_numel = self._chunk_numel(chunk, in_dim)
                byte_offset = chunk.rank_dense_offset * dtype.itemsize
                local_global_offset = chunk.global_dense_offset
            bucket_param_layouts[param.fqn] = BucketParamLayout(
                param_offset=param.param_offset,
                local_global_offset=local_global_offset,
            )
            storage_layouts[param.fqn] = BucketParamStorageLayout(
                local_shape=local_shape,
                local_numel=local_numel,
                byte_offset=byte_offset,
                storage_nbytes=local_numel * dtype.itemsize,
            )

        bucket_layout = BucketLayout(
            global_numel=unit_dense_offsets[-1],
            local_numel=rank_numels[rank],
            rank_offsets=rank_offsets,
            rank_numels=rank_numels,
            param_layouts=bucket_param_layouts,
        )
        self._metadata_plans[id(bucket_layout)] = self._QuantMetadataPlan(
            bucket_layout=bucket_layout,
            metadata=metadata,
        )
        for fqn, layout in storage_layouts.items():
            storage_layouts[fqn] = BucketParamStorageLayout(
                local_shape=layout.local_shape,
                local_numel=layout.local_numel,
                byte_offset=layout.byte_offset,
                storage_nbytes=layout.storage_nbytes,
                bucket_layout=bucket_layout,
            )

        return BucketStorageLayout(
            param_layouts=storage_layouts,
            total_bytes=rank_numels[rank] * dtype.itemsize,
        )

    @override
    def global_layout(
        self, info: ParamInfo, rank: int, world_size: int
    ) -> GlobalLayout:
        """Split this rank's matrix row range into one box per stacked matrix."""
        global_shape = tuple(info.global_shape)
        if info.local_numel == 0:
            return GlobalLayout(global_shape, (), (), ())
        *stack_shape, out_dim, in_dim = global_shape
        stack_zeros = (0,) * len(stack_shape)
        stack_ones = (1,) * len(stack_shape)
        param_layout = self._param_layout(info)
        row_start = (
            param_layout.local_global_offset - param_layout.param_offset
        ) // in_dim
        row_end = row_start + info.local_numel // in_dim
        global_offsets = []
        local_offsets = []
        local_sizes = []
        for matrix in range(row_start // out_dim, _ceil_div(row_end, out_dim)):
            matrix_row = matrix * out_dim
            start = max(row_start, matrix_row)
            end = min(row_end, matrix_row + out_dim)
            global_offsets.append(
                (*_unravel_index(matrix, stack_shape), start - matrix_row, 0)
            )
            local_offsets.append((*stack_zeros, start - row_start, 0))
            local_sizes.append((*stack_ones, end - start, in_dim))
        return GlobalLayout(
            global_shape=global_shape,
            global_offsets=tuple(global_offsets),
            local_offsets=tuple(local_offsets),
            local_sizes=tuple(local_sizes),
        )

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Quantize dense local shards into temporary data + scale send buffers."""
        prepared = self._prepare_local_unshard_payload(
            tensors,
            infos,
            mesh,
            debug_fqn,
        )
        local = prepared.buffers[0]
        gathered = torch.empty(
            mesh.size() * local.numel(),
            dtype=local.dtype,
            device=local.device,
        )
        prepared.buffers.append(gathered)
        return prepared

    def _prepare_local_unshard_payload(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Prepare the local send payload without allocating gather output."""
        rank = mesh.get_local_rank()
        with _record_copy_in_if_eager():
            local_packed, metadata = self._cached_local_quantized_chunks(
                tensors,
                infos,
                rank,
            )

        return PlacementPreparedUnshard(
            placement=self,
            buffers=[local_packed],
            placement_state=BlockwiseQuantBucketedBlockShard._QuantUnshardState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                metadata=metadata,
            ),
        )

    def _unshard_state(self, prepared: PlacementPreparedUnshard) -> _QuantUnshardState:
        state = prepared.placement_state
        if not isinstance(state, BlockwiseQuantBucketedBlockShard._QuantUnshardState):
            raise AssertionError(
                f"Expected {type(self).__name__}._QuantUnshardState, "
                f"got {type(state).__name__}"
            )
        return state

    @override
    def run_prepared_unshard(
        self, prepared: PlacementPreparedUnshard, *, async_op: bool = False
    ) -> dist.Work | None:
        """All-gather packed quantized data and scale bytes in one collective.

        Each rank's payload is padded to the same size, so the collective writes
        directly into one flat output tensor. Reshard-after-forward frees the
        compacted quantized weight after forward and re-gathers it before
        backward. The local packed payload is reused while its dense source
        shards remain unchanged. Data and scale bit patterns are transported
        as uint8, making the collective dtype-agnostic.
        """
        state = self._unshard_state(prepared)
        local, gathered = prepared.buffers
        with _record_comm_if_eager("FlexShard::all_gather", state.debug_fqn):
            return dist.all_gather_into_tensor(
                output_tensor=gathered,
                input_tensor=local,
                group=state.pg,
                async_op=async_op,
            )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Slice each param from this placement's contiguous gathered buffer."""
        state = self._unshard_state(prepared)
        rank_rows = prepared.buffers[1].view(
            len(state.metadata.rank_qdata_numels),
            prepared.buffers[0].numel(),
        )
        return self._finish_unshard_from_rank_rows(
            prepared,
            rank_rows,
        )

    def _finish_unshard_from_rank_rows(
        self,
        prepared: PlacementPreparedUnshard,
        rank_rows: torch.Tensor,
    ) -> PlacementUnshardResult:
        """Unpack directly from rank-major rows."""
        state = self._unshard_state(prepared)
        expected_shape = (
            len(state.metadata.rank_qdata_numels),
            state.metadata.packed_send_nbytes_per_rank,
        )
        if tuple(rank_rows.shape) != expected_shape:
            raise ValueError(
                f"Expected gathered rank rows with shape {expected_shape}, "
                f"got {tuple(rank_rows.shape)}."
            )
        if rank_rows.dtype != torch.uint8:
            raise ValueError(
                "Packed quantized gathered rank rows must use uint8 transport, "
                f"got {rank_rows.dtype}."
            )
        return self._finish_quant_unshard_from_rank_rows(
            prepared,
            rank_rows,
            state,
        )

    def _finish_quant_unshard_from_rank_rows(
        self,
        prepared: PlacementPreparedUnshard,
        rank_rows: torch.Tensor,
        state: _QuantUnshardState,
    ) -> PlacementUnshardResult:
        metadata = state.metadata
        qdata_nbytes = sum(metadata.rank_qdata_numels)
        # Each scale tensor index's group holds exactly the bytes its regions
        # gather.
        scale_nbytes_by_index = [
            sum(region.nbytes for region in regions)
            for regions in metadata.scale_regions_by_index
        ]
        scale_byte_offset = _align_up(qdata_nbytes, _VEC_ALIGN_BYTES)
        # Persistent unshards keep ``compact`` (data + scales) as the storage
        # backing the returned quantized weights.
        compact = (
            prepared.persistent_buffers[0]
            if prepared.persistent_buffers is not None
            else torch.empty(
                scale_byte_offset + sum(scale_nbytes_by_index),
                dtype=torch.uint8,
                device=rank_rows.device,
            )
        )
        qdata_bytes = compact[:qdata_nbytes]
        scale_bytes = compact[scale_byte_offset:]

        with _record_copy_out_if_eager():
            # Rank cuts partition one globally ordered block-row stream. Joining
            # each rank's valid regions therefore restores parameter order without
            # scattering every chunk separately.
            if not self._copy_out_rank_rows_triton(metadata, rank_rows, compact):
                self._copy_out_rank_rows_cat(
                    metadata,
                    rank_rows,
                    compact,
                    qdata_nbytes=qdata_nbytes,
                    scale_byte_offset=scale_byte_offset,
                )

        qdata_flat = qdata_bytes.view(self.qdata_dtype)
        flat_scales_by_index = [
            index_scale_bytes.view(self.scale_dtype)
            for index_scale_bytes in torch.split(scale_bytes, scale_nbytes_by_index)
        ]
        matrix_shapes = [_matrix_shape(info.global_shape) for info in state.infos]
        param_scale_numels = [
            self._chunk_scale_numels(*matrix_shape) for matrix_shape in matrix_shapes
        ]
        # ``torch.split`` raises unless the parameters exactly tile each buffer,
        # so a format's scale regions must gather exactly its parameters' scales.
        qdata_params = torch.split(
            qdata_flat, [info.global_numel for info in state.infos]
        )
        param_scales_by_index = [
            torch.split(flat, [numels[scale_index] for numels in param_scale_numels])
            for scale_index, flat in enumerate(flat_scales_by_index)
        ]
        full_params: list[torch.Tensor] = []
        for param_index, (info, qdata, matrix_shape) in enumerate(
            zip(state.infos, qdata_params, matrix_shapes, strict=True)
        ):
            scales = tuple(
                self._param_scale_view(
                    scale_index, param_scales[param_index], matrix_shape
                )
                for scale_index, param_scales in enumerate(param_scales_by_index)
            )
            full_params.append(
                self._make_quantized_weight_checked(
                    qdata.view(info.global_shape),
                    scales,
                    orig_dtype=info.unsharded_dtype,
                    requires_grad=info.requires_grad,
                )
            )
        return PlacementUnshardResult(
            full_params=full_params,
            persistent_buffers=[compact] if prepared.persistent else [],
        )

    def _copy_out_plan(
        self,
        metadata: _QuantBucketMetadata,
        *,
        data_elem_nbytes: int,
        with_scales: bool,
    ) -> _CopyOutPlan:
        key = (id(metadata), data_elem_nbytes, with_scales)
        plan = self._copy_out_plans.get(key)
        if plan is not None and plan.metadata is metadata:
            return plan

        regions: list[tuple[int, int, int, int]] = []
        out_offset = 0
        for rank, qdata_numel in enumerate(metadata.rank_qdata_numels):
            nbytes = qdata_numel * data_elem_nbytes
            regions.append((rank, 0, nbytes, out_offset))
            out_offset += nbytes
        if with_scales:
            out_offset = _align_up(out_offset, _VEC_ALIGN_BYTES)
            for scale_regions in metadata.scale_regions_by_index:
                for region in scale_regions:
                    regions.append(
                        (region.rank, region.offset, region.nbytes, out_offset)
                    )
                    out_offset += region.nbytes
        nonempty_regions = tuple(region for region in regions if region[2] > 0)
        plan = BlockwiseQuantBucketedBlockShard._CopyOutPlan(
            metadata=metadata,
            regions=nonempty_regions,
            vec_aligned=all(
                value % _VEC_COPY_NBYTES == 0
                for region in nonempty_regions
                for value in region[1:]
            ),
            descriptors={},
        )
        self._copy_out_plans[key] = plan
        return plan

    def _copy_out_rank_rows_triton(
        self,
        metadata: _QuantBucketMetadata,
        rank_rows: torch.Tensor,
        out: torch.Tensor,
        *,
        data_elem_nbytes: int = 1,
        with_scales: bool = True,
    ) -> bool:
        """Join every rank's payload regions into ``out`` with one kernel.

        ``rank_rows`` and ``out`` are uint8. Data regions land in rank order at
        the start of ``out``. With ``with_scales``, the scale regions follow
        from the 16-byte aligned offset after the data, grouped by scale tensor
        index (see ``_scale_regions``). The copy plan is built once per bucket
        layout and row stride, so a gather costs one launch instead of a view
        per rank and region. Returns False when the caller should fall back to
        cat: while compiling or off CUDA.
        """
        if torch.compiler.is_compiling() or rank_rows.device.type != "cuda":
            return False
        plan = self._copy_out_plan(
            metadata,
            data_elem_nbytes=data_elem_nbytes,
            with_scales=with_scales,
        )
        row_stride_bytes = rank_rows.stride(0)
        # The physical row storage, including any mixed-bucket padding between
        # rows: offsets in the plan are relative to the first row's start.
        source = rank_rows.as_strided(
            ((rank_rows.size(0) - 1) * row_stride_bytes + rank_rows.size(1),),
            (1,),
        )
        if (
            plan.vec_aligned
            and _is_vec_copy_viewable(rank_rows)
            and _is_vec_copy_viewable(out)
        ):
            source = source.view(_VEC_COPY_DTYPE)
            out = out.view(_VEC_COPY_DTYPE)
        unit_nbytes = source.element_size()
        descriptor_key = (rank_rows.device, row_stride_bytes, unit_nbytes)
        descriptor = plan.descriptors.get(descriptor_key)
        if descriptor is None:
            from ._copy_kernels import SegmentCopyDescriptor

            descriptor = SegmentCopyDescriptor.create(
                src_offsets=[
                    (rank * row_stride_bytes + row_offset) // unit_nbytes
                    for rank, row_offset, _, _ in plan.regions
                ],
                numels=[nbytes // unit_nbytes for _, _, nbytes, _ in plan.regions],
                dst_offsets=[
                    out_offset // unit_nbytes for _, _, _, out_offset in plan.regions
                ],
                device=rank_rows.device,
            )
            plan.descriptors[descriptor_key] = descriptor

        from ._copy_kernels import copy_segments_triton

        copy_segments_triton(source, out, descriptor)
        return True

    def _copy_out_rank_rows_cat(
        self,
        metadata: _QuantBucketMetadata,
        rank_rows: torch.Tensor,
        compact: torch.Tensor,
        *,
        qdata_nbytes: int,
        scale_byte_offset: int,
    ) -> None:
        """Cat fallback of ``_copy_out_rank_rows_triton`` for compile and CPU."""
        qdata_bytes = compact[:qdata_nbytes]
        scale_bytes = compact[scale_byte_offset:]
        row_stride_bytes = rank_rows.stride(0)
        qdata_storage_regions = tuple(
            (rank * row_stride_bytes, rank_qdata_numel)
            for rank, rank_qdata_numel in enumerate(metadata.rank_qdata_numels)
        )
        # Gather grouped by scale tensor index so each group compacts to one
        # contiguous, parameter-ordered region; the format orders each group's
        # regions.
        scale_storage_regions = tuple(
            (region.rank * row_stride_bytes + region.offset, region.nbytes)
            for regions in metadata.scale_regions_by_index
            for region in regions
        )
        # View the physical row storage once, including any mixed-bucket padding
        # between rows, so cat inputs do not need intermediate per-rank views.
        # Retain byte cats for unaligned or odd inputs.
        use_vec_copies = _is_vec_copy_viewable(rank_rows) and _is_vec_copy_viewable(
            compact
        )
        regions_and_outs = (
            (qdata_storage_regions, qdata_bytes),
            (scale_storage_regions, scale_bytes),
        )
        if use_vec_copies:
            rank_storage_bytes = rank_rows.as_strided(
                ((rank_rows.size(0) - 1) * row_stride_bytes + rank_rows.size(1),),
                (1,),
            )
            rank_storage_copy_units = rank_storage_bytes.view(_VEC_COPY_DTYPE)
            for regions, out in regions_and_outs:
                if all(
                    offset % _VEC_COPY_NBYTES == 0 and length % _VEC_COPY_NBYTES == 0
                    for offset, length in regions
                ):
                    _cat_byte_regions_from_flat_storage(
                        regions,
                        rank_storage_copy_units,
                        out.view(_VEC_COPY_DTYPE),
                        copy_unit_nbytes=_VEC_COPY_NBYTES,
                    )
                else:
                    _cat_byte_regions_from_flat_storage(
                        regions,
                        rank_storage_bytes,
                        out,
                    )
        else:
            rank_byte_rows = rank_rows.unbind(0)
            for regions, out in regions_and_outs:
                _cat_byte_regions_from_rank_rows(
                    regions,
                    rank_byte_rows,
                    out,
                    row_stride_bytes=row_stride_bytes,
                )

    def _make_quantized_weight_checked(
        self,
        qdata: torch.Tensor,
        scales: tuple[torch.Tensor, ...],
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor:
        weight = self._make_quantized_weight(
            qdata,
            scales,
            orig_dtype=orig_dtype,
            requires_grad=requires_grad,
        )
        if not isinstance(weight, torch.Tensor):
            raise TypeError(
                "_make_quantized_weight must return a torch.Tensor, got "
                f"{type(weight).__name__}."
            )
        return weight


def _merge_adjacent_regions(
    regions: tuple[_ScaleRegion, ...],
) -> tuple[_ScaleRegion, ...]:
    """Join each region that continues the previous one in the same rank."""
    merged: list[_ScaleRegion] = []
    for region in regions:
        if region.nbytes == 0:
            continue
        if merged:
            prev = merged[-1]
            if prev.rank == region.rank and prev.offset + prev.nbytes == region.offset:
                merged[-1] = prev._replace(nbytes=prev.nbytes + region.nbytes)
                continue
        merged.append(region)
    return tuple(merged)


def _cat_byte_regions_from_flat_storage(
    byte_regions: tuple[tuple[int, int], ...],
    source: torch.Tensor,
    out: torch.Tensor,
    *,
    copy_unit_nbytes: int = 1,
) -> None:
    torch.cat(
        [
            source.narrow(
                0,
                byte_offset // copy_unit_nbytes,
                byte_length // copy_unit_nbytes,
            )
            for byte_offset, byte_length in byte_regions
            if byte_length > 0
        ],
        out=out,
    )


def _cat_byte_regions_from_rank_rows(
    byte_regions: tuple[tuple[int, int], ...],
    rows: tuple[torch.Tensor, ...],
    out: torch.Tensor,
    *,
    row_stride_bytes: int,
) -> None:
    torch.cat(
        [
            rows[byte_offset // row_stride_bytes].narrow(
                0,
                byte_offset % row_stride_bytes,
                byte_length,
            )
            for byte_offset, byte_length in byte_regions
            if byte_length > 0
        ],
        out=out,
    )


def _is_vec_copy_viewable(tensor: torch.Tensor) -> bool:
    """Return whether a uint8 tensor satisfies PyTorch's dtype-view contract."""
    if torch.compiler.is_compiling():
        return False
    return (
        tensor.dtype == torch.uint8
        and tensor.ndim > 0
        and tensor.stride(-1) == 1
        and tensor.size(-1) % _VEC_COPY_NBYTES == 0
        and tensor.storage_offset() % _VEC_COPY_NBYTES == 0
        and all(stride % _VEC_COPY_NBYTES == 0 for stride in tensor.stride()[:-1])
    )


def _is_backward_graph_task() -> bool:
    return torch._C._current_graph_task_id() >= 0


__all__ = [
    "BlockwiseQuantBucketedBlockShard",
]
