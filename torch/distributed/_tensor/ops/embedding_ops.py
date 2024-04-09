# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
import itertools
from dataclasses import dataclass, field
from typing import cast, List, Optional

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor.op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    StrategyType,
)
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    is_tensor_shardable,
    register_op_strategy,
)

from torch.distributed._tensor.placement_types import (
    _Partial,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)

from torch.distributed.device_mesh import DeviceMesh

aten = torch.ops.aten


@dataclass
class MaskBuffer:
    data: Optional[torch.Tensor] = None

    def materialize_mask(self, mask):
        if self.data is not None:
            raise RuntimeError("MaskBuffer has already been materialized")
        self.data = mask

    def release_mask(self):
        # TODO: evaluate if we need to release the mask buffer or the buffer
        # can just have the same lifetime as the _Partial placement
        if self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")
        self.data = None

    def apply_mask(self, tensor):
        if self.data is None:
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
class _MaskPartial(_Partial):
    """
    A partial mask placement devised for rowwise sharded embedding op, where we need
    to mask and adjust the indices to the local embedding shard, embedding masking
    is a special type of the Partial placement

    NOTE: the lifecycle of this MaskPartial placement follows the corresponding DTensor
    lifecycle, i.e. the indices_mask would only be alive during the lifetime of the DTensor.
    """

    logical_dim_size: int = -1
    mask_buffer: MaskBuffer = field(default_factory=MaskBuffer)

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # override parent logic to perform partial mask for embedding
        num_chunks = mesh.size(mesh_dim)
        # get local shard size and offset on the embedding_dim
        local_shard_size, local_offset_on_dim = Shard._local_shard_size_on_dim(
            self.logical_dim_size,
            num_chunks,
            mesh.get_local_rank(mesh_dim),
            return_offset=True,
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
            tensor, reduceOp=self.reduce_op.name, group=(mesh, mesh_dim)
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
            and self.logical_dim_size == other.logical_dim_size
        )

    def __hash__(self) -> int:
        return 1 + hash(
            (self.logical_dim_size, id(self.mask_buffer.data), self.reduce_op)
        )

    def __repr__(self) -> str:
        """
        machine readable representation of the MaskPartial placement
        """
        return f"_MaskPartial(logical_dim_size={self.logical_dim_size})"

    def __str__(self) -> str:
        """
        human readable representation of the MaskPartial placement
        """
        return "MaskP"


@register_op_strategy(aten.embedding.default)
def embedding_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    # TODO: implement rowwise sharding
    """
    weight_strategy = cast(OpStrategy, op_schema.args_schema[0])
    indices_strategy = cast(OpStrategy, op_schema.args_schema[1])

    weight_shape = weight_strategy.output_shape
    indices_shape = indices_strategy.output_shape
    output_emd_dim = len(indices_shape)

    all_mesh_dim_strategies = []

    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # placement list stores placements of [output, weight, input_indices]
        # first we always have replicate all for inputs and output
        all_replicate: List[Placement] = [Replicate()] * 3
        single_mesh_dim_strategies.append(all_replicate)

        # colwise sharding, output shard on last dim, weight shard on dim 1, input replicate
        colwise_sharding = [Shard(output_emd_dim), Shard(1), Replicate()]
        single_mesh_dim_strategies.append(colwise_sharding)

        # rowwise sharding, output is embedding partial, weight shard on dim 0, input accepts embedding partial
        embedding_partial_placement = _MaskPartial(logical_dim_size=weight_shape[0])

        # NOTE we want to reuse the same mask partial placement so that we can reuse the same mask that generates
        # from the input indices and use it for output reduction
        rowwise_sharding = [
            embedding_partial_placement,
            Shard(0),
            embedding_partial_placement,
        ]
        single_mesh_dim_strategies.append(rowwise_sharding)

        # batch dim sharding, weight replicated, input can shard on any dim, output follows input
        for input_dim in range(len(indices_shape)):
            batch_sharding = [Shard(input_dim), Replicate(), Shard(input_dim)]
            single_mesh_dim_strategies.append(batch_sharding)

        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = []
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        if is_tensor_shardable(weight_shape, spec_list[1]) and is_tensor_shardable(
            indices_shape, spec_list[2]
        ):
            # only add to the strategy list when both weight and indices are shardable
            weight_spec, indices_spec = spec_list[1:]
            redistribute_cost = [
                generate_redistribute_costs(weight_strategy, weight_spec),
                generate_redistribute_costs(indices_strategy, indices_spec),
            ]
            strat = PlacementStrategy(
                output_specs=spec_list[0],
                input_specs=spec_list[1:],
                redistribute_cost=redistribute_cost,
            )
            all_strategies.append(strat)

    return OpStrategy(all_strategies)


@register_op_strategy(aten.embedding_dense_backward.default)
def embedding_dense_backward_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    # TODO: implement rowwise sharding backward
    """
    grad_out_strategy = cast(OpStrategy, op_schema.args_schema[0])
    indices_strategy = cast(OpStrategy, op_schema.args_schema[1])

    grad_out_shape = grad_out_strategy.output_shape
    indices_shape = indices_strategy.output_shape
    grad_out_ndim = len(grad_out_shape)

    all_mesh_dim_strategies = []

    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # placement list stores placements of [output, weight, input_indices]
        # first we always have replicate all for inputs and output
        all_replicate: List[Placement] = [Replicate()] * 3
        single_mesh_dim_strategies.append(all_replicate)

        # colwise sharding backward, grad_out shard on last dim, input replicate,
        # weight grad shard colwise
        colwise_sharding = [Shard(1), Shard(grad_out_ndim - 1), Replicate()]
        single_mesh_dim_strategies.append(colwise_sharding)

        # batch dim sharding, weight replicated, grad_out/input have same sharding
        # that can shard on any dim, weight grad partial
        for input_dim in range(len(indices_shape)):
            batch_sharding = [_Partial(), Shard(input_dim), Shard(input_dim)]
            single_mesh_dim_strategies.append(batch_sharding)

        # grad_out partial, input replicate, weight grad keep partial
        partial_sharding = [_Partial(), _Partial(), Replicate()]
        single_mesh_dim_strategies.append(partial_sharding)

        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = []
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        if is_tensor_shardable(grad_out_shape, spec_list[1]) and is_tensor_shardable(
            indices_shape, spec_list[2]
        ):
            # only add to the strategy list when both grad_out and indices are shardable
            grad_out_spec, indices_spec = spec_list[1:]
            redistribute_cost = [
                generate_redistribute_costs(grad_out_strategy, grad_out_spec),
                generate_redistribute_costs(indices_strategy, indices_spec),
            ]
            strat = PlacementStrategy(
                output_specs=spec_list[0],
                input_specs=spec_list[1:],
                redistribute_cost=redistribute_cost,
            )
            all_strategies.append(strat)

    return OpStrategy(all_strategies)
