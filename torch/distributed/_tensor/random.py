# Copyright (c) Meta Platforms, Inc. and affiliates
import math

import torch
import torch.distributed as dist

from torch import Tensor
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.placement_types import DTensorSpec


def set_rng_state(new_state: Tensor, device_mesh: DeviceMesh) -> None:
    """Sets the random number generator state of the specified device mesh.

    Args:
        new_state (:class:`torch.ByteTensor`): The desired state.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the RNG state.

    Returns:
        None

    .. warning::
        Current implementation only supports a GPU device mesh.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        `set_rng_state` will not set its GPU device's generator state.
    """
    assert isinstance(
        device_mesh, DeviceMesh
    ), f"expect a DeviceMesh but {device_mesh} was passed in."

    if device_mesh.get_coordinate() is not None:
        # the current rank is in mesh
        if device_mesh.device_type == "cuda":
            torch.cuda.set_rng_state(new_state)
        else:
            raise NotImplementedError(
                f"DTensor randomness only supports cuda device type, but got {device_mesh.device_type}"
            )


def get_rng_state(device_mesh: DeviceMesh) -> Tensor:
    """Returns the random number generator state of the calling rank as a
    :class:`torch.ByteTensor` object.

    Args:
        device_mesh (:class:`DeviceMesh`): The device mesh to return the RNG state of.

    Returns:
        A :class:`torch.ByteTensor` object that contains the random number generator state.

    .. warning::
        Current implementation only supports a GPU device mesh.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        `get_rng_state` still returns its GPU device's generator state.
    """
    assert isinstance(
        device_mesh, DeviceMesh
    ), f"expect a DeviceMesh but {device_mesh} was passed in."

    if device_mesh.device_type == "cuda":
        return torch.cuda.get_rng_state()
    else:
        raise NotImplementedError(
            f"DTensor randomness only supports cuda device type, but got {device_mesh.device_type}"
        )


def manual_seed(seed: int, device_mesh: DeviceMesh) -> None:
    """Sets the seed for generating random numbers for the calling rank.

    Args:
        seed (int): The desired seed.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the seed.

    Returns:
        None

    .. warning::
        When calling this function, :func:`manual_seed` must be called from all ranks of the
        default `ProcessGroup` even if some ranks may not be a part of the `device_mesh`,
        with the same `seed` value.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        `manual_seed` will not set its GPU device's generator seed.
        Current implementation only supports a GPU device mesh.
    """
    assert isinstance(
        device_mesh, DeviceMesh
    ), f"expect a DeviceMesh but {device_mesh} was passed in."

    # allgather the seed from rank 0 over the default PG
    object_list = [seed] * dist.get_world_size()
    dist.all_gather_object(object_list, seed)
    for rank, object in enumerate(object_list):
        if seed != int(object):
            raise RuntimeError(
                f"calling manual_seed function over {device_mesh} but received different seed values on ranks:",
                f"seed on rank {dist.get_rank()} is {seed}, and seed on rank {rank} is {object}!",
            )

    # the current rank is in mesh
    if device_mesh.get_coordinate() is not None:
        if device_mesh.device_type == "cuda":
            torch.cuda.manual_seed(seed)
        else:
            raise NotImplementedError(
                f"DTensor randomness only supports cuda device type, but got {device_mesh.device_type}"
            )


def _set_offset(new_offset: int, device_mesh: DeviceMesh) -> None:
    """Sets the random number generator state offset for the calling rank.

    Args:
        new_offset (int): The desired random number generator state offset.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the offset.

    Returns:
        None

    .. warning::
        Current implementation only supports a GPU device mesh.
        Different offset values can be passed in on different ranks so that each rank
        can generate different random numbers in following rand calls.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        `_set_offset` will not set its GPU device's generator offset.
    """
    assert isinstance(
        device_mesh, DeviceMesh
    ), f"expect a DeviceMesh but {device_mesh} was passed in."

    if device_mesh.get_coordinate() is not None:
        # the current rank is in mesh
        if device_mesh.device_type == "cuda":
            # source: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
            # the RNG state tensor returned from torch.cuda.get_rng_state() is a ByteTensor
            # first 200 * sizeof(4120) bytes in tensor are 0xFF
            # next sizeof(uint64_t) bytes are the random seed
            # last sizeof(int64_t) bytes are the offset
            state = get_rng_state(device_mesh)
            offset = state[-8:].view(torch.int64)
            offset[0] = new_offset
            set_rng_state(state, device_mesh)
        else:
            raise NotImplementedError(
                f"DTensor randomness only supports cuda device type, "
                f"but got {device_mesh.device_type}"
            )


def _increase_offset(increment: int, device_mesh: DeviceMesh) -> None:



def _calc_start_offset(spec: DTensorSpec) -> int:
    """Find out the starting RNG offset for current device's local shard.

    Args:
        spec (:class:`DTensorSpec`): the spec of the DTensor object on which
            we compute the offset.

    Returns:
        An int value denoting the starting offset for current device's local shard.

    .. warning::
        ``spec.mesh`` must be a total mesh (cannot be a sub-mesh).
        Note that, current implementation does not consider DTensor's continguity and
        requires the DTensor can be evenly sharded on all dimensions.

    Example:
        take a DTensor of shape [8, 16] as an example. Assume that the DTensor
        is placed on a device mesh with placements ([Shard(1), Replicate(), Shard(0)]),
        and the mesh is:
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        ``spec.mesh.get_coordinate()`` provides the coordinate of the current rank
        in the mesh. For example, the coordinate of rank 5 is (1, 0, 1).

        Another concept to introduce besides rank coordinate is shard coordinate.
        Each rank holds a local shard of the DTensor. In the example, the DTensor
        is partitioned into 4 [4, 8] shards. The first shard has 2 replicas and
        rank 0 (coord (0, 0, 0)) and rank 2 (coord (0, 1, 0)) have 1 replica each.
        That being said, the local shard on rank 0 and rank 2 correspond to the same
        shard of the DTensor. To denote each DTensor shard, we use a shard coordinate
        (in the example, it will be a tuple (i, j) where shard (i, j) has the slice
        DTensor[4 * i : 4 * (i + 1), 8 * j : 8 * (j + 1)], 0 <= i < 2, 0 <= j < 2).

        Once we have rank coordinate and shard coordinate, we can calculate on each rank
        what shard of the DTensor the rank holds, with the help of dim_map. The dim_map
        of the above DTensor is [2, 0] so the shard coordinate of a rank with rank coord
        (x, y, z) is simply (z, x) by taking(rank_coord[dim_map[0]],rank_coord[dim_map[1]]).
        Following this calculation,
        rank 0 and rank 2 holds the shard of coord (0, 0);
        rank 1 and rank 3 holds the shard of coord (0, 1);
        rank 4 and rank 6 holds the shard of coord (1, 0);
        rank 5 and rank 7 holds the shard of coord (1, 1);

        The last value to calculate before obtaining the starting offset is the shard linear index.
        The starting offset for each rank will be its shard_linear_index * local_tensor_numel.
    """
    dtensor_shape = spec.shape
    mesh = spec.mesh
    dim_map = spec.dim_map

    # check if DTensor can be evenly sharded. If so, compute local tensor size
    local_tensor_size = []
    for tensor_dim, mesh_dim in enumerate(dim_map):
        if mesh_dim >= 0:  # tensor_dim is sharded over mesh_dim
            local_tensor_size.append(dtensor_shape[tensor_dim] // mesh.size(mesh_dim))

            if dtensor_shape[tensor_dim] % mesh.size(mesh_dim) != 0:
                raise RuntimeError(
                    "DTensor is expected to be evenly sharded but cannot evenly shard",
                    f"tensor dim {tensor_dim} on mesh dim {mesh_dim}\n",
                    f"DTensor shape = {dtensor_shape}\nMesh shape = {mesh.mesh.size()}",
                )
        else:
            local_tensor_size.append(dtensor_shape[tensor_dim])

    # get rank coordinate
    rank_coord = mesh.get_coordinate()
    if rank_coord is None:
        raise RuntimeError(
            "Do not support calculating starting offset for DTensor over sub-mesh"
        )

    # Compute shard coordinate:
    # The coordinate on each tensor dim is a tuple (idx, range)
    # If a DTensor is partitioned on its dim i into n shards, and the current rank
    # holds the j-th, then its shard coordinate will be (idx=j, range=n) on dim i
    shard_coord = [
        (rank_coord[mesh_dim], mesh.size(mesh_dim)) if mesh_dim >= 0 else (0, 1)
        for mesh_dim in dim_map
    ]

    # compute shard linear index
    shard_linear_idx = 0
    shard_coord_stride = 1
    for idx, size in shard_coord:
        shard_linear_idx += idx * shard_coord_stride
        shard_coord_stride *= size

    # compute starting offset
    local_size = math.prod(local_tensor_size)
    return local_size
