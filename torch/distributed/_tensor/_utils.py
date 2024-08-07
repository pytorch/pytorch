from functools import reduce
from typing import cast, List, Sequence, Tuple

import torch
import torch.distributed._tensor.api as dtensor
from torch._prims_common import ShapeType
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh


# TODO: audit existing code base to see if we can safely remove this API.
def compute_local_shape(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[int, ...]:
    """
    Compute the shape of a local shard of the given DTensor on its current
    coordinate of the mesh.
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # if rank not in the mesh, return empty shape
        return (0,)
    else:
        local_shape = list(global_shape)  # start with global shape
        ndim = len(global_shape)
        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                assert (
                    shard_dim < ndim
                ), f"Sharding dim {shard_dim} greater than tensor ndim {ndim}"
                local_shard_size, _ = placement._local_shard_size_on_dim(
                    local_shape[shard_dim], mesh_dim_size, my_coordinate[idx]
                )
                assert isinstance(local_shard_size, int)
                local_shape[shard_dim] = local_shard_size

        return tuple(local_shape)


def compute_local_shape_and_global_offset(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
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
        return ((), ())
    else:
        local_shape = list(global_shape)
        global_offset = [0] * len(global_shape)

        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                local_offset = [0] * len(global_shape)
                assert shard_dim < len(
                    local_shape
                ), f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
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

        return tuple(local_shape), tuple(global_offset)


def compute_global_tensor_info(
    tensor: torch.Tensor, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[List[int], List[int]]:
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

            assert (
                shard_dim < tensor.ndim
            ), f"Sharding dim {shard_dim} greater than tensor ndim {tensor.ndim} for placement number {idx}."

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
) -> Tuple[int, ...]:
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


def get_shard_idx_on_dim(
    dim_map: List[List[int]], mesh: DeviceMesh, tensor_dim: int
) -> int:
    """
    Returns the index of the current shard on a given tensor dimension.
    """
    shard_mesh_dims = dim_map[tensor_dim]
    coordinate = mesh.get_coordinate()

    shard_idx_on_tensor_dim = 0
    for i, mesh_dim in enumerate(shard_mesh_dims):
        if i != len(shard_mesh_dims) - 1:
            # For a given coordinate, the stride equals to the product of the mesh dim
            # of all the mesh dimensions sharded on the same tensor dimension
            # For a given coordinate i, the stride equals to the product of the mesh dim size
            # of all the mesh dimensions sharded on the same tensor dimension
            # when the same tensor dimenstion is further sharded on additional mesh dimensions.
            # For example, with 3D 2*2*2 mesh with all placements on Shard(0), the stride of
            # coordinate 0 is mesh.size(1) * mesh.size(2) = 4, and the stride of coordinate 1 is
            # mesh.size(2) = 2.
            stride = reduce(
                lambda a, b: a * b, [mesh.size(j) for j in shard_mesh_dims[i + 1 :]]
            )
            shard_idx_on_tensor_dim += coordinate[mesh_dim] * stride  # type: ignore[index]
        else:
            # For the last mesh dimension, the stride is simply 1.
            shard_idx_on_tensor_dim += coordinate[mesh_dim]  # type: ignore[index]

    return shard_idx_on_tensor_dim


def get_dim_map(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> List[List[int]]:
    """
    Returns a dim_map of list of list such that dim_map[i] includes a list of mesh dimensions
    that tensor dimension i is sharded on. For example, if we have a dist tensor that have the
    shape of [18, 20, 30, 40] with a 2*2*2 device_mesh and placements [shard(0), shard(0), shard(1)],
    we would have a dim_map of [[0, 1], [2], [-1], [-1]].

    TODO: replace the current dim_map with this updated dim_map, as the current dim_map cannot
    express a tensor dim being sharded on multiple mesh dims.
    """
    dim_map = [[-1]] * len(global_shape)
    for mesh_dim, placement in enumerate(placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            if dim_map[shard_dim] == [-1]:
                dim_map[shard_dim] = [mesh_dim]
            else:
                dim_map[shard_dim].append(mesh_dim)
    return dim_map


def compute_padded_and_unpadded_local_shape(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    This util computes the padded and unpadded local shape of a DTensor. The padded shape is computed by considering
    applying padding to the global tensor such that each padded shard would have the exact same shape across all ranks.
    This means sharding happens after padding.

    # TODO: Remove compute_local_shape once we switch to static padding.
    This differs from `compute_local_shape`. The local shape from `compute_local_shape` considers padding after sharding,
    meaning padding is applied on each placements instead of globally. Therefore, the local shape on each shard
    after padding could be different. The local shape returned from `compute_local_shape` is different from the
    unpadded local shape.

    Padded and unpadded local shape could be the same depending on whether padding is needed on the current shard.
    """

    # Calculate globally how many chunks a given tensor dim will have globally.
    num_chunks_on_dim = [1 for _ in enumerate(global_shape)]
    for mesh_idx, placement in enumerate(placements):
        if placement.is_shard():
            tensor_dim = placement.dim  # type: ignore[attr-defined]
            mesh_dim_size = mesh.size(mesh_idx)
            num_chunks_on_dim[tensor_dim] *= mesh_dim_size

    dim_map = get_dim_map(global_shape, mesh, placements)
    full_shard_size, cur_unpadded_shard_size = [], []
    for tensor_dim, _ in enumerate(zip(global_shape, num_chunks_on_dim)):
        size_on_dim, num_chunks = _
        if num_chunks == 1:
            # This means no sharding is happening on the ith dimension of the global tensor.
            # Therefore, the padded and unpadded size of the ith dimension is the same as global_shape[i].
            full_shard_size.append(size_on_dim)
            cur_unpadded_shard_size.append(size_on_dim)
        else:
            # Calculate the full chunk size and the number of full chunks on a given tensor dim
            full_chunk_size = (size_on_dim + num_chunks - 1) // num_chunks
            num_full_chunks = size_on_dim // full_chunk_size
            tail_chunk_size = size_on_dim % full_chunk_size
            full_shard_size.append(full_chunk_size)

            cur_chunk = get_shard_idx_on_dim(dim_map, mesh, tensor_dim)

            # If the index of cur chunk is smaller than num_full_chunks,
            # this means cur_chunk would be a full chunk on the given tensor dimension.
            if cur_chunk < num_full_chunks:
                cur_unpadded_shard_size.append(full_chunk_size)
            # If the index of cur_chunk is num_full_chunks and the tail_chunk_size is not 0,
            # this means the cur_chunk is the non-empty tail chunk.
            # There should be only 1 non-empty tail chunk.
            # For example, shard [1, 1, 1, 1, 1] to 4 chunks, we would have [1, 1], [1, 1], [1].
            # The third shard is a non-empty tail chunk and the last shard is an empty chunk.
            elif cur_chunk == num_full_chunks and tail_chunk_size != 0:
                cur_unpadded_shard_size.append(tail_chunk_size)
            # Otherwise, the cur_chunk is an empty chunk on the tensor_dim. There could be more than 1 empty chunks.
            # For example, chunk a tensor([1, 1]) into 4 chunks, the last two chunks would be empty.
            else:
                cur_unpadded_shard_size.append(0)

    return tuple(full_shard_size), tuple(cur_unpadded_shard_size)


def compute_padding_size(
    padded_size: Sequence[int], unpadded_size: Sequence[int]
) -> Tuple[int, ...]:
    """
    Given the padded and unpadded shape of a tensor, returns a tuple of padding.
    padding_size[i] is the size of padding needed on the i-th tensor dimension.
    """
    assert len(padded_size) == len(unpadded_size)
    padding_size = [
        padded_size_on_dim - unpadded_size_on_dim
        for padded_size_on_dim, unpadded_size_on_dim in zip(padded_size, unpadded_size)
    ]
    return tuple(padding_size)
    
def compute_global_padding(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> List[int]:
    """
    Compute the padding needed to make the tensor evenly shardable. It returns
    a list of ints where pad_sizes[i] means tensor dim i needs to be padded by
    pad_sizes[i] in order to make the tensor evenly shardable on the given mesh
    and placements.

    For example, we have a dist tensor with shape (5, 1) on
    device_mesh([[0, 1],[2, 3]]) with placements [Shard(0), Shard(0)].
    The pad_sizes would be [3, 0]. This means, to make the tensor evenly shardable,
    we need to pad the tensor on dim 0 by 5 elements and no padding is needed on dim 1.
    """
    num_shard_by_dim = [1 for _ in range(len(global_shape))]
    for mesh_idx, placement in enumerate(placements):
        if placement.is_shard():
            tensor_shard_dim = placement.dim  # type: ignore[attr-defined]
            mesh_dim_size = mesh.size(mesh_idx)
            num_shard_by_dim[tensor_shard_dim] *= mesh_dim_size

    pad_sizes = []
    for tensor_dim, num_shard in enumerate(num_shard_by_dim):
        tensor_dim_size = global_shape[tensor_dim]
        if num_shard == 1 or num_shard == tensor_dim_size:
            pad_sizes.append(0)
        elif tensor_dim_size > num_shard:
            padded_tensor_dim_size = (
                tensor_dim_size + num_shard - (tensor_dim_size % num_shard)
            )
            pad_sizes.append(padded_tensor_dim_size - tensor_dim_size)
        elif tensor_dim_size < num_shard:
            pad_sizes.append(num_shard - tensor_dim_size)

    return pad_sizes
