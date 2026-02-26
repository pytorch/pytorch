import operator
from functools import reduce
from typing import cast, Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch.distributed.tensor._op_schema import OutputSharding
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)


# Mapping from argmin/argmax ops to their corresponding value ops (min/max)
_ARGMINMAX_REDUCTION_OPS = {
    torch.ops.aten.argmax.default: torch.max,
    torch.ops.aten.argmin.default: torch.min,
}


def _get_output_sharding(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> OutputSharding:
    """Get the output sharding for the given op."""
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    return output_sharding


def _prep_arguments(
    op_call_repr: str,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None,
) -> tuple[
    torch.Tensor,
    torch.Size,
    "torch.distributed.device_mesh.DeviceMesh",
    tuple[Placement, ...],
    Optional[int],
    bool,
]:
    """
    Prepare arguments for nonlinear reduction ops.

    Returns:
        local_tensor: The local tensor to operate on
        global_shape: The global shape of the DTensor
        device_mesh: The device mesh
        placements: The placements tuple
        dim: The reduction dimension (can be None)
        keepdim: Whether to keep the reduced dimension
    """
    input_dtensor = cast(dtensor.DTensor, args[0])
    dim: Optional[int] = None
    keepdim: bool = False

    if not isinstance(input_dtensor, dtensor.DTensor):
        raise NotImplementedError
    if len(args) > 1:
        dim = cast(int, args[1])
    if len(args) > 2:
        keepdim = cast(bool, args[2])
    if kwargs:
        if "dim" in kwargs:
            dim = cast(int, kwargs["dim"])
        if "keepdim" in kwargs:
            keepdim = cast(bool, kwargs["keepdim"])
    device_mesh = input_dtensor.device_mesh
    placements = input_dtensor.placements

    # check for partial placements and handle it as a replicate.
    if any(isinstance(p, Partial) for p in placements):
        target_placements = [
            Replicate() if isinstance(p, Partial) else p for p in placements
        ]
        input_dtensor = input_dtensor.redistribute(
            device_mesh=device_mesh, placements=target_placements
        )
        placements = input_dtensor.placements
    local_tensor = input_dtensor.to_local()
    global_shape = input_dtensor.shape

    return local_tensor, global_shape, device_mesh, placements, dim, keepdim


def _get_expected_shape(
    local_tensor: torch.Tensor, dim: Optional[int], keepdim: bool
) -> torch.Size:
    """Compute the expected output shape after reduction."""
    input_shape = list(local_tensor.shape)
    if dim is None:
        expected_shape = (
            torch.Size([1] * len(input_shape)) if keepdim else torch.Size([])
        )
    elif keepdim:
        if input_shape:
            input_shape[dim] = 1
        expected_shape = torch.Size(input_shape)
    else:
        if input_shape:
            input_shape.pop(dim)
        expected_shape = torch.Size(input_shape)

    return expected_shape


def _collect_shard_mesh_dims(
    op_call_repr: str,
    local_tensor: torch.Tensor,
    placements: tuple[Placement, ...],
    dim: Optional[int],
) -> list[int]:
    """Collect mesh dimensions that are sharded along the reduction dimension."""
    shard_mesh_dims: list[int] = []
    for mesh_dim, p in enumerate(placements):
        if isinstance(p, Shard):
            if dim is None or p.dim == (dim if dim >= 0 else local_tensor.ndim + dim):
                shard_mesh_dims.append(mesh_dim)
        elif isinstance(p, _StridedShard):
            raise NotImplementedError(f"{op_call_repr} does not support _StridedShard!")
    return shard_mesh_dims


def _convert_to_global_idxs(
    local_idx: torch.Tensor,
    global_shape: torch.Size,
    device_mesh: "torch.distributed.device_mesh.DeviceMesh",
    placements: tuple[Placement, ...],
    dim: Optional[int],
) -> tuple[int, torch.Tensor]:
    """Convert local indices to global indices."""
    local_shape, global_offset = compute_local_shape_and_global_offset(
        global_shape, device_mesh, placements
    )

    if dim is None:
        # Convert flat local index → flat global index using arithmetic ops
        # instead of torch.unravel_index, which doesn't support SymInt shapes.
        gathered_idxs = torch.zeros_like(local_idx)
        remaining = local_idx
        for i in range(len(local_shape)):
            local_stride = reduce(operator.mul, local_shape[i + 1 :], 1)
            global_stride = reduce(operator.mul, global_shape[i + 1 :], 1)
            coord = remaining // local_stride
            remaining = remaining % local_stride
            gathered_idxs = gathered_idxs + (coord + global_offset[i]) * global_stride
        gather_dim = 0
    else:
        gather_dim = dim
        gathered_idxs = local_idx + global_offset[dim]
    return gather_dim, gathered_idxs


def _gather_tensors(
    gather_dim: int,
    gathered_idxs: torch.Tensor,
    local_redux: torch.Tensor,
    device_mesh: "torch.distributed.device_mesh.DeviceMesh",
    shard_mesh_dims: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather the min or max of the tensors and their corresponding indices.

    Args:
        gather_dim: The dim to stack the collected min/max tensors.
        gathered_idxs: The local tensor holding the corresponding indices.
        local_redux: The local tensor holding the operator's value i.e. min/max.
        device_mesh: Device mesh of the DTensor.
        shard_mesh_dims: List of mesh dimensions that are sharded.

    Returns:
        All gathered tensors (gathered_redux, gathered_idxs) of the reducing operator.
    """
    gathered_redux = local_redux
    for mesh_dim in shard_mesh_dims:
        gathered_redux = funcol.all_gather_tensor(
            gathered_redux,
            gather_dim=gather_dim,
            group=(device_mesh, mesh_dim),
        )
        gathered_idxs = funcol.all_gather_tensor(
            gathered_idxs,
            gather_dim=gather_dim,
            group=(device_mesh, mesh_dim),
        )
    return gathered_redux, gathered_idxs


def argminmax_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """
    Handler for aten.argmin.default and aten.argmax.default ops.

    This is a pure function handler that doesn't require instantiation.
    """
    if op_call not in _ARGMINMAX_REDUCTION_OPS:
        raise NotImplementedError(f"Unsupported reduction op: {op_call}")

    local_tensor, global_shape, device_mesh, placements, dim, keepdim = _prep_arguments(
        str(op_call), args, kwargs
    )
    output_sharding = _get_output_sharding(op_call, args, kwargs)

    expected_shape = _get_expected_shape(local_tensor, dim, keepdim)
    shard_mesh_dims = _collect_shard_mesh_dims(
        str(op_call), local_tensor, placements, dim
    )

    # Compute local reduction
    if dim is None:
        val_op = _ARGMINMAX_REDUCTION_OPS[op_call]
        # unsqueeze scalars to 1-d so they can be allgathered
        local_redux = val_op(local_tensor).unsqueeze(0)
        local_idx = op_call(local_tensor).unsqueeze(0)
    else:
        val_op = _ARGMINMAX_REDUCTION_OPS[op_call]
        local_redux, local_idx = val_op(local_tensor, dim=dim, keepdim=True)

    if not shard_mesh_dims:
        return dtensor.DTensor._op_dispatcher.wrap(
            local_idx.reshape(expected_shape), output_sharding.output_spec
        )

    gather_dim, gathered_idxs = _convert_to_global_idxs(
        local_idx, global_shape, device_mesh, placements, dim
    )
    gathered_redux, gather_idxs = _gather_tensors(
        gather_dim, gathered_idxs, local_redux, device_mesh, shard_mesh_dims
    )
    # Select the rank with the best value; use dim=0 when dim was None since
    # the scalars were unsqueezed to 1-d for gathering
    select_dim = 0 if dim is None else dim
    rank_winner = op_call(gathered_redux, select_dim, True)
    final_idx = torch.gather(gather_idxs, dim=gather_dim, index=rank_winner)

    return dtensor.DTensor._op_dispatcher.wrap(
        final_idx.reshape(expected_shape), output_sharding.output_spec
    )


def minmax_dim_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """
    Handler for aten.min.dim and aten.max.dim ops.

    This is a pure function handler that doesn't require instantiation.
    """
    local_tensor, global_shape, device_mesh, placements, dim, keepdim = _prep_arguments(
        str(op_call), args, kwargs
    )
    output_sharding = _get_output_sharding(op_call, args, kwargs)

    expected_shape = _get_expected_shape(local_tensor, dim, keepdim)
    shard_mesh_dims = _collect_shard_mesh_dims(
        str(op_call), local_tensor, placements, dim
    )

    # Compute local reduction - min/max with dim always requires dim
    assert dim is not None
    local_redux, local_idx = op_call(local_tensor, dim=dim, keepdim=True)

    if not shard_mesh_dims:
        return dtensor.DTensor._op_dispatcher.wrap(
            (
                local_redux.reshape(expected_shape),
                local_idx.reshape(expected_shape),
            ),
            output_sharding.output_spec,
        )

    gather_dim, gathered_idxs = _convert_to_global_idxs(
        local_idx, global_shape, device_mesh, placements, dim
    )

    gathered_redux, gather_idxs = _gather_tensors(
        gather_dim, gathered_idxs, local_redux, device_mesh, shard_mesh_dims
    )
    # The op_call here is min/max with dim which returns (values, indices)
    final_redux, rank_winner = op_call(gathered_redux, dim, True)
    final_idx = torch.gather(gather_idxs, dim=gather_dim, index=rank_winner)

    return dtensor.DTensor._op_dispatcher.wrap(
        (
            final_redux.reshape(expected_shape),
            final_idx.reshape(expected_shape),
        ),
        output_sharding.output_spec,
    )
