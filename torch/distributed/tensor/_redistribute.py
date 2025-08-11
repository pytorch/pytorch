# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from functools import cache
from typing import cast, NamedTuple, Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


logger = logging.getLogger(__name__)


class _TransformInfo(NamedTuple):
    mesh_dim: int
    src_dst_placements: tuple[Placement, Placement]
    # logical_shape on this mesh dimension
    logical_shape: list[int]


def _gen_transform_infos_non_cached(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    src_device_order: tuple[int, ...],
    dst_device_order: tuple[int, ...],
) -> list[_TransformInfo]:
    """
    Generate the transform infos from the source placements to the target placements.

    To transform from source to target placement it might have multiple steps, i.e. it
    might decompose Si -> Sj into Si -> R -> Sj.
    This would detect if there're mis-aligned/nested shardings between src/dst placements.
    E.g. Suppose the redistribution to perform is (Shard(0), Shard(0)) -> (Replicate(), Shard(0)),
    in this case Shard(0) -> Shard(0) for mesh dimension 1 actually needs resharding, because in
    the former is a nested-sharding of a tensor already already sharded dimension 0, whereras
    the latter is the first sharding on tensor dimension 0.
    """
    transform_infos: list[_TransformInfo] = []

    device_mesh = src_spec.device_mesh
    my_coordinate = device_mesh.get_coordinate()
    assert my_coordinate is not None

    # logical shape records the logic tensor shape on the mesh dimension
    # this is useful to ensure uneven sharding gets correct output shape
    initial_logical_shape = list(src_spec.shape)

    if device_mesh.ndim == 1:
        # if device_mesh is 1D, redistribute is a simple direct transformation
        transform_infos.append(
            _TransformInfo(
                mesh_dim=0,
                src_dst_placements=(src_spec.placements[0], dst_spec.placements[0]),
                logical_shape=initial_logical_shape,
            )
        )
        return transform_infos

    # sort the src_spec based on src_device_order
    def _reorder_placement(placements, device_order):
        return [placement for _, placement in sorted(zip(device_order, placements))]

    sorted_src_placement = _reorder_placement(src_spec.placements, src_device_order)
    sorted_dst_placement = _reorder_placement(dst_spec.placements, dst_device_order)

    # map sharded tensor dim to device mesh dim with device ordering
    def _map_tensor_dim_to_mesh_dim(placements, device_order):
        tensor_dim_to_mesh_dims: dict[int, list[int]] = {}
        for placement, mesh_dim in zip(placements, device_order):
            if placement.is_shard():
                assert isinstance(placement, Shard)
                if placement.dim not in tensor_dim_to_mesh_dims:
                    tensor_dim_to_mesh_dims[placement.dim] = []
                tensor_dim_to_mesh_dims[placement.dim].append(mesh_dim)
        return tensor_dim_to_mesh_dims

    src_tensor_dim_to_mesh_dims = _map_tensor_dim_to_mesh_dim(
        src_spec.placements, src_device_order
    )
    dst_tensor_dim_to_mesh_dims = _map_tensor_dim_to_mesh_dim(
        dst_spec.placements, dst_device_order
    )

    # derive the logical shape of src_spec on each mesh dim
    final_logical_shape = list(src_spec.shape)
    # for mesh_dim, placement in zip(src_device_order, src_spec.placements):
    for mesh_dim, placement in enumerate(sorted_src_placement):
        if placement.is_shard():
            assert isinstance(placement, Shard)
            mesh_dim_size = device_mesh.size(mesh_dim=mesh_dim)
            local_shard_size, _ = placement._local_shard_size_and_offset(
                final_logical_shape[placement.dim],
                mesh_dim_size,
                my_coordinate[mesh_dim],
            )
            final_logical_shape[placement.dim] = local_shard_size
    # now final_logical_shape is the final shape under src_spec.placements.
    current_logical_shape = list(final_logical_shape)
    for tensor_dim in range(src_spec.ndim):
        if (
            tensor_dim in src_tensor_dim_to_mesh_dims
            and tensor_dim in dst_tensor_dim_to_mesh_dims
        ):
            # The rule is to only allow push/pop the rightmost number to turn
            # src_tensor_dim_to_mesh_dims[tensor_dim] into
            # dst_tensor_dim_to_mesh_dims[tensor_dim]. For example, if
            # src_tensor_dim_to_mesh_dims[tensor_dim] = [3, 0, 1, 2] and
            # dst_tensor_dim_to_mesh_dims[tensor_dim] = [3, 1, 0], we need to
            # [3, 0, 1, 2] -> (allgather) [3, 0, 1] -> (allgather) [3, 0] ->
            # (allgather) [3] -> (chunk) [3, 1] -> (chunk) [3, 1, 0]
            src_mesh_dims = (
                src_tensor_dim_to_mesh_dims[tensor_dim].copy()
                if tensor_dim in src_tensor_dim_to_mesh_dims
                else []
            )
            dst_mesh_dims = (
                dst_tensor_dim_to_mesh_dims[tensor_dim].copy()
                if tensor_dim in dst_tensor_dim_to_mesh_dims
                else []
            )

            # find i s.t. src_tensor_dim_to_mesh_dims[:i]==dst_device_order_to_mesh_dims[:i]
            i = 0
            for i in range(len(src_mesh_dims)):
                if i < len(dst_mesh_dims):
                    if src_mesh_dims[i] != dst_mesh_dims[i]:
                        break

            # Build the transform_infos for those gathering operation.
            while len(src_mesh_dims) > i:
                mesh_dim = src_mesh_dims.pop()
                # allgather on the popped mesh_dim
                mesh_dim_size = device_mesh.size(mesh_dim=mesh_dim)
                transform_infos.append(
                    _TransformInfo(
                        mesh_dim=mesh_dim,
                        src_dst_placements=(Shard(tensor_dim), Replicate()),
                        logical_shape=current_logical_shape,
                    )
                )

                current_logical_shape[tensor_dim] = min(
                    current_logical_shape[tensor_dim] * mesh_dim_size,
                    src_spec.shape[tensor_dim],
                )

            assert len(src_mesh_dims) == i
            # Build the transform_infos for those chunk operation.
            for mesh_dim in dst_mesh_dims[i:]:
                # chunk on mesh_dim
                mesh_dim_size = device_mesh.size(mesh_dim=mesh_dim)
                current_placement = sorted_dst_placement[mesh_dim]
                assert isinstance(current_placement, Shard)
                local_shard_size, _ = current_placement._local_shard_size_and_offset(
                    current_logical_shape[tensor_dim],
                    mesh_dim_size,
                    my_coordinate[mesh_dim],
                )
                transform_infos.append(
                    _TransformInfo(
                        mesh_dim=mesh_dim,
                        src_dst_placements=(Replicate(), Shard(tensor_dim)),
                        logical_shape=current_logical_shape,
                    )
                )
                current_logical_shape[tensor_dim] = local_shard_size
            # update all Shard(tensor_dim) to Replicate()
            for mesh_dim, placement in enumerate(sorted_src_placement):
                if isinstance(placement, Shard) and placement.dim == tensor_dim:
                    sorted_src_placement[mesh_dim] = Replicate()
            for mesh_dim, placement in enumerate(sorted_dst_placement):
                if isinstance(placement, Shard) and placement.dim == tensor_dim:
                    sorted_dst_placement[mesh_dim] = Replicate()

        elif tensor_dim in src_tensor_dim_to_mesh_dims:
            # Check if exist Shard() to Shard() pattern I_x -> J_x. In this case
            # we can apply alltoall. x is mesh dim
            if len(src_tensor_dim_to_mesh_dims[tensor_dim]) == 1:
                mesh_dim = src_tensor_dim_to_mesh_dims[tensor_dim][0]
                # check if exist j s.t. dst_tensor_dim_to_mesh_dims[j]==[mesh_dim]
                if not sorted_dst_placement[mesh_dim].is_shard():
                    # sorted_dst_placement[mesh_dim] may have been handled and
                    # replaced with Replicate(), skip
                    continue
                for j in range(src_spec.ndim):
                    if j in dst_tensor_dim_to_mesh_dims and dst_tensor_dim_to_mesh_dims[
                        j
                    ] == [mesh_dim]:
                        mesh_dim_size = device_mesh.size(mesh_dim=mesh_dim)
                        current_placement = sorted_dst_placement[mesh_dim]
                        # alltoall from Shard(tensor_dim) to Shard(j)
                        transform_infos.append(
                            _TransformInfo(
                                mesh_dim=mesh_dim,
                                src_dst_placements=(Shard(tensor_dim), Shard(j)),
                                logical_shape=current_logical_shape,
                            )
                        )
                        current_logical_shape[tensor_dim] = min(
                            current_logical_shape[tensor_dim] * mesh_dim_size,
                            src_spec.shape[tensor_dim],
                        )
                        local_shard_size, _ = (
                            current_placement._local_shard_size_and_offset(
                                current_logical_shape[j],
                                mesh_dim_size,
                                my_coordinate[mesh_dim],
                            )
                        )
                        current_logical_shape[j] = local_shard_size
                        sorted_src_placement[mesh_dim] = Shard(j)
                        # just use the first matching one, delete key j from dst_device_order_to_mesh_dims to prevent reuse
                        del dst_tensor_dim_to_mesh_dims[
                            j
                        ]  # may not be necessary, just to be safe
                        break
    # We have done processing Shard()->Shard() case, now process the remaining mesh dim
    for mesh_dim, (src_placement, dst_placement) in enumerate(
        zip(sorted_src_placement, sorted_dst_placement)
    ):
        if src_placement == dst_placement:
            continue
        mesh_dim_size = device_mesh.size(mesh_dim=mesh_dim)
        if isinstance(src_placement, Shard) and isinstance(dst_placement, Shard):
            # alltoall
            transform_infos.append(
                _TransformInfo(
                    mesh_dim=mesh_dim,
                    src_dst_placements=(src_placement, dst_placement),
                    logical_shape=current_logical_shape,
                )
            )
            current_logical_shape[src_placement.dim] = min(
                current_logical_shape[src_placement.dim] * mesh_dim_size,
                src_spec.shape[src_placement.dim],
            )
            local_shard_size, _ = dst_placement._local_shard_size_and_offset(
                current_logical_shape[dst_placement.dim],
                mesh_dim_size,
                my_coordinate[mesh_dim],
            )
            current_logical_shape[dst_placement.dim] = local_shard_size
        elif isinstance(src_placement, Shard):
            # shard -> replicate/partial
            transform_infos.append(
                _TransformInfo(
                    mesh_dim=mesh_dim,
                    src_dst_placements=(src_placement, dst_placement),
                    logical_shape=current_logical_shape,
                )
            )
            current_logical_shape[src_placement.dim] = min(
                current_logical_shape[src_placement.dim] * mesh_dim_size,
                src_spec.shape[src_placement.dim],
            )
        elif isinstance(dst_placement, Shard):
            # replicate/partial -> shard
            transform_infos.append(
                _TransformInfo(
                    mesh_dim=mesh_dim,
                    src_dst_placements=(src_placement, dst_placement),
                    logical_shape=current_logical_shape,
                )
            )

            local_shard_size, _ = dst_placement._local_shard_size_and_offset(
                current_logical_shape[dst_placement.dim],
                mesh_dim_size,
                my_coordinate[mesh_dim],
            )
            current_logical_shape[dst_placement.dim] = local_shard_size
        else:
            # replicate/partial -> replicate/partial
            transform_infos.append(
                _TransformInfo(
                    mesh_dim=mesh_dim,
                    src_dst_placements=(src_placement, dst_placement),
                    logical_shape=current_logical_shape,
                )
            )
    return transform_infos


@cache
def _gen_transform_infos(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    src_device_order: tuple[int, ...],
    dst_device_order: tuple[int, ...],
) -> list[_TransformInfo]:
    return _gen_transform_infos_non_cached(
        src_spec, dst_spec, src_device_order, dst_device_order
    )


def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    src_device_order: Optional[tuple[int, ...]] = None,
    dst_device_order: Optional[tuple[int, ...]] = None,
    *,
    async_op: bool = False,
    is_backward: bool = False,
) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current DTensorSpec to
    the target DTensorSpec, which involves the necessary collective calls to transform
    the local shard of the DTensor from its current spec to the target spec.
    """

    if current_spec.mesh != target_spec.mesh:
        # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    if not src_device_order:
        src_device_order = tuple(range(current_spec.device_mesh.ndim))
    if not dst_device_order:
        dst_device_order = tuple(range(target_spec.device_mesh.ndim))

    if not isinstance(src_device_order, tuple):
        src_device_order = tuple(src_device_order)
    if not isinstance(dst_device_order, tuple):
        dst_device_order = tuple(dst_device_order)

    new_local_tensor = local_tensor
    device_mesh = current_spec.mesh

    my_coordinate = device_mesh.get_coordinate()

    if my_coordinate is None:
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    has_symints = any(isinstance(s, torch.SymInt) for s in current_spec.shape) or any(
        isinstance(s, torch.SymInt) for s in target_spec.shape
    )
    if has_symints:
        transform_infos = _gen_transform_infos_non_cached(
            current_spec, target_spec, src_device_order, dst_device_order
        )
    else:
        transform_infos = _gen_transform_infos(
            current_spec, target_spec, src_device_order, dst_device_order
        )

    for transform_info in transform_infos:
        i = transform_info.mesh_dim
        current, target = transform_info.src_dst_placements
        device_mesh.size(mesh_dim=i)

        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            continue

        logger.debug("redistribute from %s to %s on mesh dim %s", current, target, i)

        if target.is_replicate():
            # Case 1: target is Replicate
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_value(
                    local_tensor, device_mesh, i
                )
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )
            else:
                raise RuntimeError(
                    f"redistribute from {current} to {target} not supported yet"
                )
        elif target.is_shard():
            # Case 2: target is Shard
            target_placement = cast(Shard, target)
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_shard_value(
                    local_tensor, device_mesh, i, target_placement
                )
            elif current.is_replicate():
                # split the tensor and return the corresponding cloned local shard
                new_local_tensor = target_placement._replicate_to_shard(
                    local_tensor, device_mesh, i, my_coordinate[i]
                )
            else:
                assert current.is_shard(), (
                    f"Current placement should be shard but found {current}"
                )
                shard_spec = cast(Shard, current)
                if shard_spec.dim != target_placement.dim:
                    new_local_tensor = shard_spec._to_new_shard_dim(
                        local_tensor,
                        device_mesh,
                        i,
                        transform_info.logical_shape,
                        target_placement.dim,
                    )
        elif target.is_partial():
            if current.is_replicate():
                partial_spec = cast(Partial, target)
                # skip the replicate to partial transformation when we are in backward pass
                # In this case we keep the grad as replicate, this is because we don't
                # want to convert the replicated gradients back to partial, although
                # that's logically conform with the same layout, converting the gradients
                # back to partial is actually useless as you would have to do reduce later
                # which would be more expensive than keeping it replicate! For this reason,
                # we keep the replicate grad here.
                new_local_tensor = (
                    partial_spec._partition_value(local_tensor, device_mesh, i)
                    if not is_backward
                    else local_tensor
                )
            elif current.is_shard():
                if not is_backward:
                    raise RuntimeError(
                        f"redistribute from {current} to {target} not supported yet"
                    )
                # for backward shard -> partial, we just need to convert the shard to replicate
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )
            else:
                # partial -> partial no op, should never hit
                new_local_tensor = local_tensor

        local_tensor = new_local_tensor

    if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
        new_local_tensor = new_local_tensor.wait()

    return new_local_tensor


class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        input: "dtensor.DTensor",
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        device_order: Optional[tuple[int, ...]] = None,
        async_op: bool = False,
        forward_dtype: Optional[torch.dtype] = None,
        backward_dtype: Optional[torch.dtype] = None,
    ):
        ctx.async_op = async_op
        ctx.backward_dtype = backward_dtype
        ctx.original_dtype = input._local_tensor.dtype
        ctx.original_device_order = input._spec.device_order

        if forward_dtype is not None and forward_dtype != input._local_tensor.dtype:
            local_tensor = input._local_tensor.to(dtype=forward_dtype)
            current_spec = DTensorSpec(
                mesh=device_mesh,
                placements=input._spec.placements,
                device_order=input._spec.device_order,
                tensor_meta=TensorMeta(
                    shape=input.shape,
                    stride=input.stride(),
                    dtype=forward_dtype,
                ),
            )
        else:
            local_tensor = input._local_tensor
            current_spec = input._spec

        ctx.current_spec = current_spec

        if current_spec.placements != placements:
            target_spec = DTensorSpec(
                device_mesh,
                placements,
                device_order=device_order,
                tensor_meta=current_spec.tensor_meta,
            )

            output = redistribute_local_tensor(
                local_tensor,
                current_spec,
                target_spec,
                src_device_order=input._spec.device_order,
                dst_device_order=device_order,
                async_op=async_op,
            )
        else:
            # use the same local tensor if placements are the same.
            output = local_tensor
            target_spec = current_spec

        return dtensor.DTensor(
            output,
            target_spec,
            requires_grad=input.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_output: "dtensor.DTensor"):  # type: ignore[override]
        previous_spec = ctx.current_spec
        async_op = ctx.async_op
        backward_dtype = ctx.backward_dtype or ctx.original_dtype

        if backward_dtype != grad_output._local_tensor.dtype:
            local_tensor = grad_output._local_tensor.to(dtype=backward_dtype)
            current_spec = DTensorSpec(
                mesh=grad_output._spec.device_mesh,
                placements=grad_output._spec.placements,
                tensor_meta=TensorMeta(
                    shape=grad_output.shape,
                    stride=grad_output.stride(),
                    dtype=backward_dtype,
                ),
            )
            previous_spec = DTensorSpec(
                mesh=previous_spec.device_mesh,
                placements=previous_spec.placements,
                tensor_meta=current_spec.tensor_meta,
            )
        else:
            local_tensor = grad_output._local_tensor
            current_spec = grad_output._spec

        output = redistribute_local_tensor(
            local_tensor,
            current_spec,
            previous_spec,
            src_device_order=current_spec.device_order,
            dst_device_order=previous_spec.device_order,
            async_op=async_op,
            is_backward=True,
        )

        if output.dtype != ctx.original_dtype:
            output = output.to(ctx.original_dtype)

        # normalize the target placement to replicate if it is partial
        normalized_placements: list[Placement] = []
        for previous_placement in previous_spec.placements:
            if previous_placement.is_partial():
                # keep target placement to replicate instead of partial in this case
                normalized_placements.append(Replicate())
            else:
                normalized_placements.append(previous_placement)

        spec = DTensorSpec(
            previous_spec.device_mesh,
            tuple(normalized_placements),
            tensor_meta=TensorMeta(
                shape=grad_output.shape,
                stride=grad_output.stride(),
                dtype=output.dtype,
            ),
        )
        output_dtensor = dtensor.DTensor(
            output,
            spec,
            requires_grad=grad_output.requires_grad,
        )

        return (
            output_dtensor,
            None,
            None,
            None,
            None,
            None,
            None,
        )
