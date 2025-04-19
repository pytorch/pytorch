from collections.abc import Sequence
from typing import cast

import torch
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


def compute_local_shape_and_global_offset(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Compute the local tensor shape and the global offsets into the original tensor
    of a DTensor on its current global rank. This is useful for checkpointing purpose.

    Example (2 host with 4GPUs each):
    # Below is a DeviceMesh with mesh_shape of (2, 4)
    mesh = DeviceMesh(device_type="cuda",
                        mesh=[
                        [0, 1, 2, 3],
                        [4, 5, 6, 7]
                        ],
    )

    Let's say we distribute a global_tensor of shape (8,4) over the above DeviceMesh
    with a placements of [Shard(0), Shard(0)].
    The local shape and global offset will be as follows:
    rank0 -- local_shape:[1, 4], global_offset:[0, 0]
    rank1 -- local_shape:[1, 4], global_offset:[1, 0]
    rank2 -- local_shape:[1, 4], global_offset:[2, 0]
    rank5 -- local_shape:[1, 4], global_offset:[5, 0]
    rank3 -- local_shape:[1, 4], global_offset:[3, 0]
    rank4 -- local_shape:[1, 4], global_offset:[4, 0]
    rank6 -- local_shape:[1, 4], global_offset:[6, 0]
    rank7 -- local_shape:[1, 4], global_offset:[7, 0]

    Let's say we distribute a global_tensor of shape (2) over the above DeviceMesh with
    a placements of [Shard(0)]. We will not have non-empty local tensor for all the ranks.
    The local shape and global offset will be as follows:
    rank0 -- local_shape:[1,], global_offset:[0,]
    rank1 -- local_shape:[1,], global_offset:[1,]
    rank2 -- local_shape:[0,], global_offset:[2,]
    rank5 -- local_shape:[0,], global_offset:[2,]
    rank3 -- local_shape:[0,], global_offset:[2,]
    rank4 -- local_shape:[0,], global_offset:[2,]
    rank6 -- local_shape:[0,], global_offset:[2,]
    rank7 -- local_shape:[0,], global_offset:[2,]
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # if rank not in the mesh, return empty offset
        return ((0,), ())
    else:
        local_shape = list(global_shape)
        global_offset = [0] * len(global_shape)
        shard_idx_stride_by_mesh_dim = [
            [0] * mesh.ndim for _ in range(len(global_shape))
        ]  # index by (shard_dim, mesh_dim)
        num_shards_by_tensor_dim = [1] * len(global_shape)

        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                local_offset = [0] * len(global_shape)
                assert shard_dim < len(local_shape), (
                    f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
                )
                shard_size, shard_offset = placement._local_shard_size_on_dim(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coordinate[idx],
                    return_offset=True,
                )

                local_shape[shard_dim] = shard_size
                local_offset[shard_dim] = shard_offset

                # On a given dimension, if the local_offset[shard_dim] is smaller than global_offset[shard_dim],
                # it means that this dimension has been already sharded in previous placement.
                # Therefore, we cannot simply replace the global_offset[shard_dim] with local_offset[shard_dim].
                # Instead, for the given shard_dim, we need to add local_offset[shard_dim] to existing global_offset[shard_dim].
                if global_offset[shard_dim] <= local_offset[shard_dim]:
                    global_offset[shard_dim] = local_offset[shard_dim]
                else:
                    global_offset[shard_dim] += local_offset[shard_dim]

                num_shards_by_tensor_dim[shard_dim] *= mesh_dim_size

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
        strided_sharding = any(isinstance(p, _StridedShard) for p in placements)
        if strided_sharding:
            strided_part_seen = [False] * len(global_shape)
            strided_part_end = [False] * len(global_shape)
            for idx, placement in enumerate(placements):
                mesh_dim_size = mesh.size(idx)
                if isinstance(placement, Shard):
                    shard_dim = placement.dim

                    if strided_part_end[shard_dim]:
                        raise NotImplementedError(
                            f"Strided sharding does not allow Shard() to appear after "
                            f"the strided part has ended. {placement} at idx {idx} in "
                            f"{placements} violates this assumption."
                        )

                    if strided_part_seen[shard_dim]:
                        strided_part_end[shard_dim] = True

                    if isinstance(placement, _StridedShard):
                        strided_part_seen[shard_dim] = True
                        shard_idx_stride_by_mesh_dim[shard_dim][idx] = (
                            num_shards_by_tensor_dim[shard_dim]
                            // (placement.split_factor * mesh_dim_size)
                        )
                    else:
                        num_shards_by_tensor_dim[shard_dim] //= mesh_dim_size
                        shard_idx_stride_by_mesh_dim[shard_dim][idx] = (
                            num_shards_by_tensor_dim[shard_dim]
                        )

            shard_idx = [
                sum([x * y for x, y in zip(shard_idx_stride, my_coordinate)])
                for shard_dim, shard_idx_stride in enumerate(
                    shard_idx_stride_by_mesh_dim
                )
            ]

            global_offset = [x * y for x, y in zip(local_shape, shard_idx)]

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
