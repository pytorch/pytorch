from collections import defaultdict
from collections.abc import Sequence
from typing import cast, Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch._prims_common import ShapeType
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)


def _explicit_order_placements(
    mesh_shape: ShapeType, placements: Sequence[Placement]
) -> Sequence[tuple[int, Placement]]:
    """
    Replace Strided Shards with regular shards in an adjusted order.

    Returns a list of (mesh_dim, placement) tuples where the list order is the sharding order.

    ex.
    [Shard(0), _StridedShard(0, split_factor=2), Shard(0)] ->
    [(0, Shard(0)), (2, Shard(0)), (1, Shard(0))]

    """
    if not len(placements) == len(mesh_shape):
        raise RuntimeError(
            "Expected one placement per mesh dim, "
            f"but found {len(placements)} placements and {len(mesh_shape)} mesh dims."
        )
    ordered = []
    deferred_strided_placements = defaultdict(list)
    strided_part_ended_for_dim = set()
    for mesh_dim, p in enumerate(placements):
        if isinstance(p, _StridedShard):
            # validate the stride is the correct multiple of the meshdim and the earlier shard
            deferred_strided_placements[p.dim].append((mesh_dim, p))

        else:
            ordered.append((mesh_dim, p))
            if isinstance(p, Shard):
                if p.dim in strided_part_ended_for_dim:
                    raise NotImplementedError(
                        f"Strided sharding does not allow Shard() to appear after "
                        f"the strided part has ended. {p} at mesh dim {mesh_dim} in "
                        f"{placements} violates this assumption."
                    )

                if p.dim in deferred_strided_placements:
                    strided_part_ended_for_dim.add(p.dim)
                    strided_placements = deferred_strided_placements.pop(p.dim)
                    aggregate_size = mesh_shape[mesh_dim]
                    while len(strided_placements) > 0:
                        strided_mesh_dim, strided = strided_placements.pop()
                        if not strided.split_factor == aggregate_size:
                            raise RuntimeError(
                                f"Can only convert _StridedShard to ordered Shard if split_factor({strided.split_factor})"
                                f" == aggregate mesh size ({aggregate_size})"
                            )
                        aggregate_size *= mesh_shape[strided_mesh_dim]
                        ordered.append((strided_mesh_dim, Shard(p.dim)))

    return ordered


def compute_local_shape_and_global_offset(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Compute the local tensor shape and the global offsets into the original tensor
    of a DTensor on its current global rank. This is useful for checkpointing purpose.

    Example:
    global_tensor = [[0,  1,  2,  3,  4], sharded on mesh (DP=2, TP=2) with (Shard(1), Shard(1))
                     [10, 11, 12, 13, 14]]

    This table shows the return value of local_shape and global_offset for each rank.
    (`local_tensor` is for illustration only).

    Note how the first coordinate of global_offset is always 0, corresponding to tensor dim 0 being replicated.

    Rank        local_tensor        local_shape     global_offset
    -------------------------------------------------------------
    0           [[0, 1],            (2, 2)          (0, 0)
                 [10, 11]]

    1           [[2],               (2, 1)          (0, 2)
                 [12]]

    2           [[3],               (2, 1)          (0, 3)
                 [13]]

    3           [[4],               (2, 1)          (0, 4)
                 [14]]

    Args:
        global_shape (ShapeType): The global shape of the DTensor.
        mesh (:class:`DeviceMesh`): The device mesh this DTensor is distributed on.
        placements (Sequence[:class:`Placement`]]): The placements of the DTensor.

    Return:
        local_shape: the shape of the DTensor's _local_tensor on the current rank.
        global_offset: a tuple of offsets for each dimension of the global tensor shape,
        identifying how this shard fits into the global tensor in each dimension.

    """
    return _compute_local_shape_and_global_offset(
        global_shape, mesh.shape, mesh.get_coordinate(), placements
    )


# accept 'plain data types' to enable simpler unit testing without creating device mesh
def _compute_local_shape_and_global_offset(
    global_shape: ShapeType,
    mesh_shape: ShapeType,
    my_coordinate: Optional[list[int]],
    placements: Sequence[Placement],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered_placements = _explicit_order_placements(mesh_shape, placements)

    if my_coordinate is None:
        # if rank not in the mesh, return empty offset
        return ((0,), ())
    else:
        local_shape = list(global_shape)
        global_offset = [0] * len(global_shape)
        for mesh_dim, placement in ordered_placements:
            mesh_dim_size = mesh_shape[mesh_dim]
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                local_offset = [0] * len(global_shape)
                assert shard_dim < len(local_shape), (
                    f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
                )
                shard_size, shard_offset = placement._local_shard_size_and_offset(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coordinate[mesh_dim],
                )

                local_shape[shard_dim] = shard_size
                local_offset[shard_dim] = shard_offset
                if shard_size == 0:
                    # Special case to fill in a standardized non-garbage value for the global_offset
                    # of zero-sized shards.  This value is out of bounds of the tensor, so it won't conflict
                    # with any real offsets.  DCP may rely on this value to de-duplicate shards.
                    global_offset[shard_dim] = global_shape[shard_dim]
                else:
                    # On a given dimension, if the local_offset[shard_dim] is smaller than global_offset[shard_dim],
                    # it means that this dimension has been already sharded in previous placement.
                    # Therefore, we cannot simply replace the global_offset[shard_dim] with local_offset[shard_dim].
                    # Instead, for the given shard_dim, we need to add local_offset[shard_dim] to existing global_offset[shard_dim].
                    if global_offset[shard_dim] <= local_offset[shard_dim]:
                        global_offset[shard_dim] = local_offset[shard_dim]
                    else:
                        global_offset[shard_dim] += local_offset[shard_dim]

        # NOTE: the offset compute relies on the local shard index and it has no
        # problem when strided sharding is not present. To correctly compute, we assume
        # that the ``_StridedShard.split_factor`` field encodes how many partitions
        # each local tensor will be further split into when sharding on higher mesh
        # dimensions. However, this number is only correct if the DTensor is not
        # sharded after the strided sharding completes. For example,
        # [Shard(0), _StridedShard(0, split_factor=2), Shard(0)] is the placements
        # where the DTensor's dim-0 is first sharded on device mesh dim-0, then on
        # device mesh dim-2, and last on mesh dim-1. We define the
        # "_StridedShard(0, split_factor=2), Shard(0)" part as the strided sharding
        # part because strided sharding happens on mesh dim-1 and it was caused by
        # the fact that sharding on dim-2 occurred ahead. In this case, there's no
        # further sharding after this strided sharding part and ``split_factor``
        # correctly encodes the number. Another example is
        # [_StridedShard(0, split_factor=2), Shard(0), Shard(0)] where the DTensor's
        # dim-0 is first sharded on mesh dim-1, then on mesh dim-0, and last on mesh
        # dim-2. This violates our assumption that no further sharding shall occur
        # after the strided sharding part and ``split_factor`` won't correctly
        # encode the number of further split. So far, the only case where _StridedShard
        # placement would appear is FSDP2 + TP on 2D mesh and the above case could only
        # happen on mesh of 3 or more dimensions.
        # TODO: change this function to correctly address this.
        # TODO: this logic can be applied to contiguous sharding as well
        return tuple(local_shape), tuple(global_offset)


def compute_global_tensor_info(
    tensor: torch.Tensor, mesh: DeviceMesh, placements: Sequence[Placement]
) -> tuple[list[int], list[int]]:
    """
    Compute the global size and stride of a DTensor from the given local tensor.
    The local size is multiplited by `world_size` per Sharding dim.
    The local stride is multiplited by `world_size` per Sharding dim, as long as the
    dimension is outside sharding dim.

    For example, if we have a local tensor with size (4, 8, 2) and stride (16, 1, 8).
    If the DTensor placements are [Shard(2)] and world_size is 2;
    then the global size is (4, 8, 4) and stride is (16 * 2, 1, 8).

    Args:
        tensor (:class:`torch.Tensor`):
            Local tensor which DTensor will be constructed from.
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        placements (Sequence[:class:`Placement`]]):
            The attribute of the DTensor that describes its layout
            on the mesh topology.

    Return:
        tensor_shape: A List of int which specifies the size of DTensor which build
            on top of the local tensor.
        tensor_stride: A List of int which specifies the stride of DTensor.
    """
    tensor_shape = list(tensor.size())
    tensor_stride = list(tensor.stride())
    for idx, placement in enumerate(placements):
        mesh_dim_size = mesh.size(idx)
        if placement.is_shard():
            shard_placement = cast(Shard, placement)
            if shard_placement.dim < 0:
                raise AssertionError(
                    "Shard placements should have negative dims normalized in "
                    f"the user-facing APIs: {shard_placement}"
                )
            shard_dim = shard_placement.dim

            assert shard_dim < tensor.ndim, (
                f"Sharding dim {shard_dim} greater than tensor ndim {tensor.ndim} for placement number {idx}."
            )

            local_dim_size = tensor_shape[shard_dim]
            tensor_shape[shard_dim] = local_dim_size * mesh_dim_size

            # recover tensor stride by modifying the stride that larger than
            # the current stride on the shard_dim
            for i in range(len(tensor_stride)):
                if i != shard_dim and tensor_stride[i] >= tensor_stride[shard_dim]:
                    # rescale the stride by the shard size
                    tensor_stride[i] = tensor_stride[i] * mesh_dim_size
        elif not isinstance(placement, (Replicate, Partial)):
            raise RuntimeError(f"placement type {type(placement)} not supported!")
    return tensor_shape, tensor_stride


def compute_global_tensor_shape(
    shape: torch.Size, mesh: DeviceMesh, placements: Sequence[Placement]
) -> torch.Size:
    """
    Compute the global size of a DTensor from the given local tensor shape,
    the mesh and placements. Different from `compute_global_tensor_info`,
    which assumes sharding is even, this util allgathers local shards' shapes
    from all ranks and thus can support uneven sharding.
    NOTE: Currently this function only supports 1D mesh.

    Args:
        shape (:class:`torch.Size`):
            Shape of the local tensor
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        placements (Sequence[:class:`Placement`]]):
            The attribute of the DTensor that describes its layout
            on the mesh topology.

    Return:
        tensor_shape: Shape of the global DTensor.
    """
    if len(placements) != 1:
        raise NotImplementedError(
            "compute_global_tensor_shape only supports 1 placement for now."
        )

    if len(placements) != mesh.ndim:
        raise RuntimeError(
            "Expected one placement per mesh dim, "
            f"but found {len(placements)} placements and {mesh.ndim} mesh dims."
        )

    if isinstance(placements[0], Replicate):
        return shape
    elif isinstance(placements[0], Shard):
        local_shape = torch.tensor(list(shape))
        gathered_shaped_tensors = [
            torch.empty_like(local_shape, device=local_shape.device)
            for _ in range(mesh.size())
        ]
        funcol.all_gather_inplace(gathered_shaped_tensors, local_shape)
        sharded_dim_sum = 0
        shard_dim = placements[0].dim
        other_dims = [d for d in range(mesh.ndim) if d != shard_dim]
        for shape_tensor in gathered_shaped_tensors:
            if not torch.equal(local_shape[other_dims], shape_tensor[other_dims]):
                raise RuntimeError(
                    "Non-sharded dimentions should have identical size across ranks."
                )
            shape_tensor_list = shape_tensor.tolist()
            sharded_dim_sum += shape_tensor_list[shard_dim]
        global_shape = list(shape)
        global_shape[placements[0].dim] = sharded_dim_sum
        return torch.Size(global_shape)
    else:
        raise NotImplementedError(
            f"Placement type {type(placements[0])} not supported."
        )


def try_find_mesh_from_args(
    op_call: torch._ops.OpOverload, args: Sequence[object]
) -> DeviceMesh:
    """
    Find the device mesh object from args.
    It returns None if no mesh is found.
    NOTE: we can optimize this search if needed
    """
    for arg in args:
        if isinstance(arg, (dtensor.DTensor, DTensorSpec)):
            return arg.device_mesh
        elif (
            isinstance(arg, (list, tuple))
            and len(arg) > 0
            and isinstance(arg[0], (dtensor.DTensor, DTensorSpec))
        ):
            return arg[0].device_mesh

    raise ValueError(f"Cannot find device mesh from args for op : {op_call}.")


def compute_local_stride(
    global_stride: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> tuple[int, ...]:
    """
    Compute the stride of a local tensor shard, given the global stride of the DTensor.
    NOTE: Currently this function is assuming the DTensor is evenly shardable.
    """
    stride_divisors = [1] * len(global_stride)
    for mesh_idx, p in enumerate(placements):
        if p.is_shard():
            i = cast(Shard, p).dim
            # tensor dimension i is sharded on mesh dimension mesh_idx,
            # so we need to divide all the strides larger than stride[i]
            # (by the submesh size)
            for j in range(len(global_stride)):
                if global_stride[j] > global_stride[i]:
                    stride_divisors[j] *= mesh.size(mesh_idx)
    return tuple(
        global_stride[i] // stride_divisors[i] for i in range(len(global_stride))
    )


def normalize_to_torch_size(size) -> torch.Size:  # type: ignore[no-untyped-def]
    """
    Unify variable types of size argument to torch.Size
    Acceptable types include:
        int, Sequence[int], Tuple[int], Tuple[Sequence[int]],
        or torch.Size
    """
    if isinstance(size, torch.Size):
        return size

    if isinstance(size, int):
        torch_size = [size]
    elif len(size) == 1 and isinstance(size[0], Sequence):
        torch_size = list(size[0])
    else:
        torch_size = list(size)
    return torch.Size(torch_size)
