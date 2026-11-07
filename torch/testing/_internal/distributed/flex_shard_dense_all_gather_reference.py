# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dense all-gather reference for the blockwise FP8 placement tests.

It checks that the FP8 all-gather trains bitwise the same as all-gathering
bf16 and quantizing afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import torch
from torch.distributed.fsdp._flex_shard.placement_contract import (
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)
from torch.distributed.fsdp._flex_shard.placements.blockwise_quant_bucketed_block_shard import (
    _align_up,
    _matrix_shape,
    _scale_shape,
    _VEC_ALIGN_BYTES,
    BlockwiseQuantBucketedBlockShard,
)
from torch.distributed.fsdp._flex_shard.placements.fp8_bucketed_block_shard import (
    _quantize_dense_weight_to_blockwise_fp8,
    Fp8BucketedBlockShard,
)
from torch.distributed.fsdp._flex_shard.placements.utils import foreach_copy_
from torch.distributed.fsdp._flex_shard.utils import (
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
)


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp._flex_shard.bucket_storage import ParamInfo


class DenseAllGatherReferencePlacement(Fp8BucketedBlockShard):
    """All-gather dense shards using the FP8 placement's exact shard layout.

    It inherits the FP8 placement's block-row planner, storage layout, and grad
    reduce-scatter, so sharding, initialization and grads are identical. Only
    the unshard differs: each rank sends its dense rows, and the full
    parameters are quantized after the all-gather into the FP8 placement's
    compact buffer, which persistent unshards keep.
    """

    def __repr__(self) -> str:
        return (
            "DenseAllGatherReferencePlacement("
            f"world_size={self.world_size}, "
            f"block_size={self.block_size})"
        )

    @override
    def _prepare_local_unshard_payload(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        rank = mesh.get_local_rank()
        metadata = self._quant_metadata_from_infos(infos)
        pack_plan = metadata.rank_pack_plans[rank]
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        # Each rank sends its dense rows in the order its FP8 bytes would take,
        # padded to the largest rank's payload.
        local_dense = torch.empty(
            _align_up(max(metadata.rank_qdata_numels), _VEC_ALIGN_BYTES),
            dtype=infos[0].unsharded_dtype,
            device=device,
        )
        local_dense[pack_plan.qdata_numel :].zero_()
        dsts, srcs = [], []
        offset = 0
        for chunk in pack_plan.chunks:
            dsts.append(local_dense[offset : offset + chunk.qdata_numel])
            srcs.append(tensors[chunk.param_index].reshape(-1))
            offset += chunk.qdata_numel
        with _record_copy_in_if_eager():
            foreach_copy_(dsts, srcs)
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[local_dense],
            placement_state=BlockwiseQuantBucketedBlockShard._QuantUnshardState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                metadata=metadata,
            ),
        )

    @override
    def _finish_unshard_from_rank_rows(
        self,
        prepared: PlacementPreparedUnshard,
        rank_rows: torch.Tensor,
    ) -> PlacementUnshardResult:
        state = self._unshard_state(prepared)
        metadata = state.metadata
        dense_dtype = state.infos[0].unsharded_dtype
        if rank_rows.dtype != dense_dtype:
            raise ValueError(
                f"Expected gathered dense rank rows of dtype {dense_dtype}, "
                f"got {rank_rows.dtype}."
            )
        # The FP8 placement's compact layout: fp8 data, then aligned scales.
        qdata_nbytes = sum(metadata.rank_qdata_numels)
        scale_nbytes = sum(metadata.rank_scale_numels) * self.scale_dtype.itemsize
        scale_byte_offset = _align_up(qdata_nbytes, _VEC_ALIGN_BYTES)
        compact = (
            prepared.persistent_buffers[0]
            if prepared.persistent_buffers is not None
            else torch.empty(
                scale_byte_offset + scale_nbytes,
                dtype=torch.uint8,
                device=rank_rows.device,
            )
        )
        qdata_flat = compact[:qdata_nbytes].view(self.qdata_dtype)
        scale_flat = compact[scale_byte_offset:].view(self.scale_dtype)
        with _record_copy_out_if_eager():
            # Rank cuts partition one parameter-ordered block-row stream, so
            # joining the ranks' rows restores the full parameters.
            dense_flat = rank_rows.new_empty(qdata_nbytes)
            if not self._copy_out_rank_rows_triton(
                metadata,
                rank_rows.view(torch.uint8),
                dense_flat.view(torch.uint8),
                data_elem_nbytes=rank_rows.element_size(),
                with_scales=False,
            ):
                torch.cat(
                    [
                        rank_rows[rank, :numel]
                        for rank, numel in enumerate(metadata.rank_qdata_numels)
                    ],
                    out=dense_flat,
                )
            full_params: list[torch.Tensor] = []
            offset = 0
            scale_offset = 0
            for info in state.infos:
                matrix_shape = _matrix_shape(info.global_shape)
                scale_shape = _scale_shape(matrix_shape, self.block_size)
                scale_numel = scale_shape[0] * scale_shape[1]
                fp8_data, recip_scale = _quantize_dense_weight_to_blockwise_fp8(
                    dense_flat[offset : offset + info.global_numel].view(matrix_shape),
                    self.block_size,
                    self.qdata_dtype,
                )
                qdata_view = qdata_flat[offset : offset + info.global_numel]
                scale_view = scale_flat[scale_offset : scale_offset + scale_numel]
                qdata_view.view(matrix_shape).copy_(fp8_data)
                scale_view.view(scale_shape).copy_(recip_scale)
                full_params.append(
                    self._make_quantized_weight(
                        qdata_view.view(info.global_shape),
                        (scale_view.view(scale_shape),),
                        orig_dtype=info.unsharded_dtype,
                        requires_grad=info.requires_grad,
                    )
                )
                offset += info.global_numel
                scale_offset += scale_numel
        if offset != qdata_flat.numel() or scale_offset != scale_flat.numel():
            raise AssertionError("Dense reference bucket metadata is inconsistent.")
        return PlacementUnshardResult(
            full_params=full_params,
            persistent_buffers=[compact] if prepared.persistent else [],
        )
