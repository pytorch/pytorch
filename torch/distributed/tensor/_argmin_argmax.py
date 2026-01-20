import operator
from functools import reduce

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard


_REDUCTION_OPS = {
    torch.ops.aten.argmax.default: torch.max,
    torch.ops.aten.argmin.default: torch.min,
}


def argmin_argmax_handler(
    op_call: torch._ops.OpOverload,
    args: tuple["dtensor.DTensor", int] | tuple["dtensor.DTensor", int, bool],
    kwargs: dict[str, object],
):
    """
    Handles reduces on sharded dimensions locally to limit calls to replicate.
    """
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    if op_call not in _REDUCTION_OPS:
        raise NotImplementedError(f"Unsupported reduction op: {op_call}")
    val_op = _REDUCTION_OPS[op_call]

    input_dtensor = args[0]
    if not isinstance(input_dtensor, dtensor.DTensor):
        raise NotImplementedError

    dim: int | None = args[1] if len(args) > 1 else None  # type: ignore[assignment]
    keepdim = args[2] if len(args) > 2 else False

    placements = input_dtensor.placements

    # check for partial placements and handle it as replicate.
    if any(isinstance(p, Partial) for p in placements):
        target_placements = [
            Replicate() if isinstance(p, Partial) else p for p in placements
        ]
        input_dtensor = input_dtensor.redistribute(
            device_mesh=input_dtensor.device_mesh, placements=target_placements
        )
        placements = input_dtensor.placements
    local_tensor = input_dtensor.to_local()

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

    shard_mesh_dims = []
    for mesh_dim, p in enumerate(placements):
        if isinstance(p, Shard):
            if dim is None or p.dim == (dim if dim >= 0 else local_tensor.ndim + dim):
                shard_mesh_dims.append(mesh_dim)

    device_mesh = input_dtensor.device_mesh

    if dim is None:
        local_idx = op_call(local_tensor)
        local_max = local_tensor.flatten()[local_idx]
    else:
        local_max, local_idx = val_op(local_tensor, dim=dim, keepdim=True)

    if not shard_mesh_dims:
        return dtensor.DTensor._op_dispatcher.wrap(
            local_idx.reshape(expected_shape), output_sharding.output_spec
        )

    # find the correct offset for sharded dim
    global_shape = input_dtensor.shape
    _, global_offset = compute_local_shape_and_global_offset(
        global_shape, device_mesh, placements
    )
    gathered_maxes = local_max
    if dim is None:
        local_coord = torch.unravel_index(local_idx, local_tensor.shape)
        global_coord = torch.stack(local_coord)
        gather_dim = 0
        for i, offset in enumerate(global_offset):
            global_coord[i] += offset
        # compute with proper striding
        gathered_idxs = torch.tensor(0, device=local_tensor.device, dtype=torch.long)
        for i, coord in enumerate(global_coord):
            gathered_idxs += coord * reduce(operator.mul, global_shape[i + 1 :], 1)
    else:
        gather_dim = dim
        gathered_idxs = local_idx + global_offset[dim]

    for mesh_dim in shard_mesh_dims:
        gathered_maxes = funcol.all_gather_tensor(
            gathered_maxes, gather_dim=gather_dim, group=(device_mesh, mesh_dim)
        )
        gathered_idxs = funcol.all_gather_tensor(
            gathered_idxs, gather_dim=gather_dim, group=(device_mesh, mesh_dim)
        )

    rank_winner = op_call(gathered_maxes, dim, True)

    final_idx = torch.gather(gathered_idxs, dim=gather_dim, index=rank_winner)

    return dtensor.DTensor._op_dispatcher.wrap(
        final_idx.reshape(expected_shape), output_sharding.output_spec
    )
