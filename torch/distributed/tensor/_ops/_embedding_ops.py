# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from dataclasses import dataclass, field
from typing import cast, Optional

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementList,
    StrategyType,
)
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


aten = torch.ops.aten


@dataclass
class MaskBuffer:
    data: Optional[torch.Tensor] = None
    # refcount allows shared usage of the MaskBuffer, as long as all users have the same data
    refcount: int = 0

    def materialize_mask(self, mask):
        if self.refcount == 0:
            self.data = mask
        else:
            assert self.data is not None
            if not torch.equal(self.data, mask):
                raise RuntimeError(
                    "MaskBuffer has been materialized with conflicting data"
                )
        self.refcount += 1

    def release_mask(self):
        if self.refcount == 0 or self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")
        self.refcount -= 1
        if self.refcount == 0:
            self.data = None

    def apply_mask(self, tensor):
        if self.refcount == 0 or self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")

        # NOTE: _MaskPartial is being used by the embedding op and the gather op.
        # For gather, the mask has the same dimension as the output tensor, whereas
        # the output of the embedding op has an additional dimension compare to the input,
        # hence the output masking logic below having two different cases.
        if tensor.ndim == self.data.ndim:
            tensor[self.data] = 0.0
        else:
            tensor[self.data, :] = 0.0


@dataclass(frozen=True)
class _MaskPartial(Partial):
    """
    A partial mask placement devised for rowwise sharded embedding op, where we need
    to mask and adjust the indices to the local embedding shard, embedding masking
    is a special type of the Partial placement

    NOTE: the lifecycle of this MaskPartial placement follows the corresponding DTensor
    lifecycle, i.e. the indices_mask would only be alive during the lifetime of the DTensor.
    """

    mask_buffer: MaskBuffer = field(default_factory=MaskBuffer)

    # required fields for computing the local offset and deriving the mask
    offset_shape: Optional[torch.Size] = None
    offset_dim: int = 0

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # override parent logic to perform partial mask for embedding
        num_chunks = mesh.size(mesh_dim)
        # get local shard size and offset on the embedding_dim
        assert self.offset_shape is not None, (
            "offset_shape needs to be set for _MaskPartial"
        )
        local_shard_size, local_offset_on_dim = Shard._local_shard_size_and_offset(
            self.offset_shape[self.offset_dim],
            num_chunks,
            mesh.get_local_rank(mesh_dim),
        )
        # Build the input mask and save it for the current partial placement
        # this is so that the output of embedding op can reuse the same partial
        # placement saved mask to perform mask + reduction
        mask = (tensor < local_offset_on_dim) | (
            tensor >= local_offset_on_dim + local_shard_size
        )
        # mask the input tensor
        masked_tensor = tensor.clone() - local_offset_on_dim
        masked_tensor[mask] = 0
        # materialize the mask buffer to be used for reduction
        self.mask_buffer.materialize_mask(mask)
        return masked_tensor

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # by the time we ned reduction, we should have already saved the mask
        assert self.mask_buffer.data is not None

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # perform sum reduction
        return funcol.all_reduce(
            tensor, reduceOp=self.reduce_op, group=(mesh, mesh_dim)
        )

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # by the time we ned reduction, we should have already saved the mask
        assert self.mask_buffer.data is not None

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # call reduce_shard_tensor of the shard_spec.
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MaskPartial):
            return False

        # if either data is not None, we invalidate the sharding cache, as this indicates
        # the current MaskPartial placement is still in use and should not be used for cache hit.
        if self.mask_buffer.data is not None or other.mask_buffer.data is not None:
            return False

        return (
            self.reduce_op == other.reduce_op
            and self.offset_shape == other.offset_shape
            and self.offset_dim == other.offset_dim
        )

    def __hash__(self) -> int:
        return 1 + hash(
            (
                self.reduce_op,
                self.offset_shape,
                self.offset_dim,
            )
        )

    def __repr__(self) -> str:
        """
        machine readable representation of the MaskPartial placement
        """
        return f"_MaskPartial(offset_shape={self.offset_shape}, offset_dim={self.offset_dim})"

    def __str__(self) -> str:
        """
        human readable representation of the MaskPartial placement
        """
        return "MaskP"


@register_op_strategy(aten.embedding.default)
def embedding_strategy(op_schema: OpSchema) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    """
    weight_strategy = cast(OpStrategy, op_schema.args_schema[0])
    indices_strategy = cast(OpStrategy, op_schema.args_schema[1])
    mesh = op_schema.get_mesh_from_args()

    weight_shape = weight_strategy.shape
    indices_shape = indices_strategy.shape
    output_emd_dim = len(indices_shape)

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, weight, input_indices]
    # first we always have replicate all for inputs and output
    all_replicate: PlacementList = [Replicate()] * 3
    single_mesh_dim_strategies.append(all_replicate)

    # colwise sharding, output shard on last dim, weight shard on dim 1, input replicate
    colwise_sharding: PlacementList = [Shard(output_emd_dim), Shard(1), Replicate()]
    single_mesh_dim_strategies.append(colwise_sharding)

    # rowwise sharding, output is embedding partial, weight shard on dim 0, input accepts embedding partial
    embedding_partial_placement = _MaskPartial(offset_shape=weight_shape, offset_dim=0)

    # NOTE we want to reuse the same mask partial placement so that we can reuse the same mask that generates
    # from the input indices and use it for output reduction
    rowwise_sharding: PlacementList = [
        embedding_partial_placement,
        Shard(0),
        embedding_partial_placement,
    ]
    single_mesh_dim_strategies.append(rowwise_sharding)

    # batch dim sharding, weight replicated, input can shard on any dim, output follows input
    for input_dim in range(len(indices_shape)):
        batch_sharding: PlacementList = [
            Shard(input_dim),
            Replicate(),
            Shard(input_dim),
        ]
        single_mesh_dim_strategies.append(batch_sharding)

    return expand_to_full_mesh_op_strategy(mesh, op_schema, single_mesh_dim_strategies)


@register_op_strategy(aten.embedding_dense_backward.default)
def embedding_dense_backward_strategy(op_schema: OpSchema) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    """
    grad_out_strategy = cast(OpStrategy, op_schema.args_schema[0])
    indices_strategy = cast(OpStrategy, op_schema.args_schema[1])
    mesh = op_schema.get_mesh_from_args()

    grad_out_shape = grad_out_strategy.shape
    indices_shape = indices_strategy.shape
    grad_out_ndim = len(grad_out_shape)

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, weight, input_indices]
    # first we always have replicate all for inputs and output
    all_replicate: PlacementList = [Replicate()] * 3
    single_mesh_dim_strategies.append(all_replicate)

    # colwise sharding backward, grad_out shard on last dim, input replicate,
    # weight grad shard colwise
    colwise_sharding: PlacementList = [Shard(1), Shard(grad_out_ndim - 1), Replicate()]
    single_mesh_dim_strategies.append(colwise_sharding)

    # batch dim sharding, weight replicated, grad_out/input have same sharding
    # that can shard on any dim, weight grad partial
    for input_dim in range(len(indices_shape)):
        batch_sharding: PlacementList = [Partial(), Shard(input_dim), Shard(input_dim)]
        single_mesh_dim_strategies.append(batch_sharding)

    # grad_out partial, input replicate, weight grad keep partial
    partial_sharding: PlacementList = [Partial(), Partial(), Replicate()]
    single_mesh_dim_strategies.append(partial_sharding)

    return expand_to_full_mesh_op_strategy(mesh, op_schema, single_mesh_dim_strategies)
