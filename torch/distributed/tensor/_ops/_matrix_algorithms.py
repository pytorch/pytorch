# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_algorithm import (
    not_implemented,
    register_op_algorithm_impl,
    register_op_algorithm_selector,
)
from torch.distributed.tensor._op_schema import OpAlgorithm, OpInfo, OpSchema, OpSpec
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard
from torch.fx.experimental.symbolic_shapes import guard_or_false


aten = torch.ops.aten

_ONESIDED_MM_RHS_SHARD0 = "onesided_mm_rhs_shard0"


def _is_shard(placement: Placement, dim: int) -> bool:
    return isinstance(placement, Shard) and placement.dim == dim


def _is_replicate(placement: Placement) -> bool:
    return isinstance(placement, Replicate)


def _has_single_placement(spec: DTensorSpec) -> bool:
    return spec.mesh.ndim == 1 and len(spec.placements) == 1


@register_op_algorithm_selector(aten.mm.default)
def _select_mm_algorithm(
    op_schema: OpSchema,
    selected_strategy: OpSpec,
) -> OpAlgorithm | None:
    if len(op_schema.args_spec) != 2 or selected_strategy.input_specs is None:
        return None

    lhs_spec, rhs_spec = op_schema.args_spec
    desired_lhs_spec, desired_rhs_spec = selected_strategy.input_specs
    output_spec = selected_strategy.output_spec
    if not isinstance(output_spec, DTensorSpec):
        return None

    specs = (lhs_spec, rhs_spec, desired_lhs_spec, desired_rhs_spec, output_spec)
    if any(not _has_single_placement(spec) for spec in specs):
        return None
    if lhs_spec.ndim != 2 or rhs_spec.ndim != 2:
        return None

    if not (
        _is_shard(lhs_spec.placements[0], 0)
        and _is_shard(rhs_spec.placements[0], 0)
        and _is_shard(desired_lhs_spec.placements[0], 0)
        and _is_replicate(desired_rhs_spec.placements[0])
        and _is_shard(output_spec.placements[0], 0)
    ):
        return None

    mesh_size = rhs_spec.mesh.size(0)
    if not guard_or_false(rhs_spec.shape[0] % mesh_size == 0):
        return None

    return OpAlgorithm(_ONESIDED_MM_RHS_SHARD0)


@register_op_algorithm_impl(_ONESIDED_MM_RHS_SHARD0)
def _run_onesided_mm_rhs_shard0(
    op_info: OpInfo,
    algorithm: OpAlgorithm,
) -> object:
    if len(op_info.local_args) != 2 or len(op_info.flat_args_schema) < 2:
        return not_implemented
    lhs, rhs = op_info.local_args
    lhs_spec, rhs_spec = op_info.flat_args_schema[:2]
    if not (
        isinstance(lhs, torch.Tensor)
        and isinstance(rhs, torch.Tensor)
        and isinstance(lhs_spec, DTensorSpec)
        and isinstance(rhs_spec, DTensorSpec)
    ):
        return not_implemented
    if torch.is_grad_enabled() and (lhs.requires_grad or rhs.requires_grad):
        return not_implemented
    if lhs.device.type != "cuda" or rhs.device.type != "cuda":
        return not_implemented
    if lhs.device != rhs.device or lhs.layout != torch.strided or rhs.layout != torch.strided:
        return not_implemented
    if not lhs.is_contiguous() or not rhs.is_contiguous():
        return not_implemented
    if not symm_mem.is_symm_mem_tensor(rhs):
        return not_implemented

    group = op_info.compute_mesh.get_group(algorithm.mesh_dim)
    group_size = dist.get_world_size(group)
    group_rank = dist.get_rank(group)
    if rhs_spec.shape[0] % group_size != 0:
        return not_implemented

    shard_k = rhs_spec.shape[0] // group_size
    if rhs.size(0) != shard_k or lhs.size(1) != rhs_spec.shape[0]:
        return not_implemented

    out = lhs.new_zeros((lhs.size(0), rhs.size(1)))
    remote_rhs = torch.empty_like(rhs)
    for peer in range(group_size):
        rhs_shard = (
            rhs
            if peer == group_rank
            else symm_mem.get(remote_rhs, rhs, group, peer=peer)
        )
        lhs_slice = lhs.narrow(1, peer * shard_k, shard_k)
        out.addmm_(lhs_slice, rhs_shard)

    return out
