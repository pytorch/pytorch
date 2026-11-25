import operator
from functools import reduce
from typing import cast

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch._logging import LazyString
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor._utils import (
    _format_implicit_redistribution_msg,
    compute_local_shape_and_global_offset,
    ExplicitRedistributionContext,
)
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

# Mapping from any/all ops to the reduction that combines their local results
_BOOL_REDUCTION_OPS = {
    torch.ops.aten.any.default: "max",
    torch.ops.aten.any.dim: "max",
    torch.ops.aten.any.dims: "max",
    torch.ops.aten.all.default: "min",
    torch.ops.aten.all.dim: "min",
    torch.ops.aten.all.dims: "min",
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
    if output_sharding is None:
        raise AssertionError("output sharding should not be None")
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
    int | None,
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
    dim: int | None = None
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
    local_tensor: torch.Tensor, dim: int | None, keepdim: bool
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
    dim: int | None,
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
    dim: int | None,
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
        gathered_redux = funcol.all_gather_single(
            gathered_redux,
            gather_dim=gather_dim,
            group=(device_mesh, mesh_dim),
        )
        gathered_idxs = funcol.all_gather_single(
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
    if dim is None:
        raise AssertionError
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


def _bool_reduction_plan(
    op_call: torch._ops.OpOverload, schema: OpSchema, output_sharding: OutputSharding
) -> tuple[OutputSharding, DTensorSpec | None]:
    """
    Returns the sharding to run any/all with and the Partial spec of its local
    result, or None if no mesh dim is reduced locally.
    """
    from torch.distributed.tensor._ops._math_ops import _infer_reduction_dims

    input_spec = cast(DTensorSpec, schema.args_schema[0])
    output_spec = cast(DTensorSpec, output_sharding.output_spec)
    redistribute_schema = output_sharding.redistribute_schema
    if not output_sharding.needs_redistribute or redistribute_schema is None:
        return output_sharding, None

    target_spec = cast(DTensorSpec, redistribute_schema.args_schema[0])
    partial = Partial(_BOOL_REDUCTION_OPS[op_call])
    dim = schema.args_schema[1] if len(schema.args_schema) > 1 else None
    reduced_dims = _infer_reduction_dims(dim, input_spec.ndim)
    is_bool = cast(TensorMeta, input_spec.tensor_meta).dtype == torch.bool
    input_placements = list(target_spec.placements)
    partial_placements = list(output_spec.placements)
    pairs = zip(input_spec.placements, target_spec.placements, strict=True)
    for i, (src, dst) in enumerate(pairs):
        if src == dst or not (dst.is_replicate() or dst.is_shard()):
            continue
        if isinstance(src, (Shard, _StridedShard)):
            reduce_locally = reduced_dims is None or src.dim in reduced_dims
        else:
            reduce_locally = is_bool and src == partial
        if reduce_locally:
            input_placements[i] = src
            partial_placements[i] = partial
    if partial_placements == list(output_spec.placements):
        return output_sharding, None

    if tuple(input_placements) == input_spec.placements:
        kept_spec = input_spec
    else:
        kept_spec = DTensorSpec(
            input_spec.mesh,
            tuple(input_placements),
            tensor_meta=input_spec.tensor_meta,
            use_strided_shard_as_shard_order=input_spec.use_strided_shard_as_shard_order,
        )
    sharding = OutputSharding(
        output_spec=output_spec,
        redistribute_schema=OpSchema(
            redistribute_schema.op,
            (kept_spec, *redistribute_schema.args_schema[1:]),
            redistribute_schema.kwargs_schema,
            schema_info=redistribute_schema.schema_info,
        ),
        needs_redistribute=kept_spec is not input_spec,
        use_val_from_redistribute_schema=output_sharding.use_val_from_redistribute_schema,
    )
    partial_spec = DTensorSpec(
        output_spec.mesh,
        tuple(partial_placements),
        tensor_meta=output_spec.tensor_meta,
        use_strided_shard_as_shard_order=output_spec.use_strided_shard_as_shard_order,
    )
    return sharding, partial_spec


def bool_reduction_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """
    Handler for the functional aten.any and aten.all overloads.

    The any/all strategy only shards non-reduction dims, so sharding
    propagation redistributes the input on mesh dims that shard a reduction
    dim (an all-gather). This handler keeps those mesh dims, and bool Partial
    inputs matching the op's max/min combine, as they are: the op reduces them
    locally and only the reduced result is redistributed, from that Partial to
    the output placements the strategy chose.
    """
    op_dispatcher = dtensor.DTensor._op_dispatcher
    op_info = op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    output_sharding = cast(
        OutputSharding,
        op_dispatcher._propagate_op_sharding_dispatch_slow_path(
            op_call, args, kwargs, op_info, try_cache=True
        ),
    )
    schema = cast(OpSchema, op_info.schema)
    sharding, partial_spec = _bool_reduction_plan(op_call, schema, output_sharding)
    op_info.output_sharding = sharding
    result = op_dispatcher._dispatch_get_local_results_slow_path(op_call, args, op_info)
    output_spec = cast(DTensorSpec, sharding.output_spec)
    mesh = op_info.compute_mesh
    if partial_spec is not None and mesh._is_current_rank_part_of_mesh():
        msg = LazyString(_format_implicit_redistribution_msg, schema)
        ExplicitRedistributionContext.observe_redistribution(
            partial_spec, output_spec, msg
        )
        local = cast(torch.Tensor, result)
        result = redistribute_local_tensor(local, partial_spec, output_spec)
    return op_dispatcher.wrap(result, output_spec)
