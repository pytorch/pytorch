import threading
from collections.abc import Sequence
from typing import cast, Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch._prims_common import ShapeType
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)


class ExplicitRedistributionContext:
    """
    Within this context manager, DTensor will refuse to perform implicit redistribution,
    instead raising an error.  Manual calls to ``redistribute()`` are required wherever a redistribution
    must occur to avoid erroring.  This can be used to ensure that the user is aware of all redistribution.

    Note: it is easier to use this mode on just the forward pass of a typical DTensor program, as the backwards pass
    may contain implicit redistribution calls that are not visible to the user and difficult to replace with manual
    calls.  Redistribution during backward can be made explicit by writing `autograd.Function`s that are no-op
    during forward and perform a manual redistribution during backwards.
    """

    _local = threading.local()

    def __init__(self, enable: bool = True, strict: bool = False):
        self._enable = enable
        self._strict = strict

    @classmethod
    def is_redistribute_allowed(cls, src_spec: DTensorSpec, dst_spec: DTensorSpec):
        if instance := getattr(cls._local, "_active", None):
            if instance._enable:
                if instance._strict:
                    return False
                return redistribute_cost(src_spec, dst_spec) <= 0
        return True

    def __enter__(self):
        self._prev = getattr(ExplicitRedistributionContext._local, "_active", None)
        ExplicitRedistributionContext._local._active = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ExplicitRedistributionContext._local._active = self._prev


def compute_local_shape_and_global_offset(
    global_shape: ShapeType,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
    skip_offset: bool = False,
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
        global_shape, mesh.shape, mesh.get_coordinate(), placements, skip_offset
    )


# accept 'plain data types' to enable simpler unit testing without creating device mesh
def _compute_local_shape_and_global_offset(
    global_shape: ShapeType,
    mesh_shape: ShapeType,
    my_coordinate: Optional[list[int]],
    placements: Sequence[Placement],
    skip_offset: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Suppose you have a full tensor with size global_shape, and you have sharded
    it according to placements for mesh_shape.  This function returns, for a
    specific coordinate my_coordinate in the device mesh:

        - The size of your local shard WITHOUT padding (i.e., if you have
          an uneven split, your size might be smaller than the other entries
          in your dim), and

        - Where the data for your shard begins, in the full tensor.

    This function is fairly simple if your tensor is evenly sharded; the complication
    is around uneven splits.  There is also some complication for handling StridedShard,
    which changes the order you should apply sharding.
    """

    empty_offset = ()
    if my_coordinate is None:
        # if rank not in the mesh, return empty offset
        return ((0,), empty_offset)

    local_shape = list(global_shape)
    # Perform shard from left to right. For example,
    #   global tensor: [0, 1, 2, 3, 4, 5, 6, 7]
    #   placements: S(0), SS(0, split_factor=2)
    #   mesh_shape: (2, 2)
    # After S(0), shard_dim_to_global_offsets are
    #   {0: [0, 1, 2, 3]} on my_coordinate [0, 0] [0, 1]
    #   {0: [4, 5, 6, 7]} on my_coordinate [1, 0] [1, 1]
    # After SS(0, split_factor=2), shard_dim_to_global_offsets are
    #   {0: [0, 2]} on my_coordinate [0, 0]
    #   {0: [1, 3]} on my_coordinate [0, 1]
    #   {0: [4, 6]} on my_coordinate [1, 0]
    #   {0: [5, 7]} on my_coordinate [1, 1]
    shard_dim_to_global_offsets = {}
    for mesh_dim, placement in enumerate(placements):
        mesh_dim_size = mesh_shape[mesh_dim]
        if not isinstance(placement, (Shard, _StridedShard)):
            continue
        shard_dim = placement.dim
        zero_global_offset = global_shape[shard_dim]
        assert shard_dim < len(local_shape), (
            f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
        )
        shard_size, shard_offsets = placement._local_shard_size_and_offset(
            local_shape[shard_dim],
            mesh_dim_size,
            my_coordinate[mesh_dim],
        )
        local_shape[shard_dim] = shard_size
        if skip_offset:
            continue
        if shard_size == 0:
            shard_dim_to_global_offsets[shard_dim] = [zero_global_offset]
            continue
        if not isinstance(placement, _StridedShard):
            assert shard_offsets is not None and isinstance(shard_offsets, int)
            shard_offsets = list(range(shard_offsets, shard_offsets + shard_size))
        assert isinstance(shard_offsets, list)
        if shard_dim not in shard_dim_to_global_offsets:
            shard_dim_to_global_offsets[shard_dim] = shard_offsets
        else:
            shard_dim_to_global_offsets[shard_dim] = [
                shard_dim_to_global_offsets[shard_dim][i] for i in shard_offsets
            ]
    if skip_offset:
        return tuple(local_shape), empty_offset
    global_offset = [0] * len(global_shape)
    for shard_dim, global_offsets in shard_dim_to_global_offsets.items():
        global_offset[shard_dim] = global_offsets[0]
    return tuple(local_shape), tuple(global_offset)


compute_global_tensor_info = torch._C._DTensor_compute_global_tensor_info


def compute_local_tensor_info(
    global_tensor: torch.Tensor,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> tuple[list[int], list[int]]:
    """
    Compute the local size and stride of a DTensor from the given global tensor info.

    For example, if we have a global tensor with size (4, 8, 4) and stride (32, 1, 8).
    If the DTensor placements are [Shard(2)] and world_size is 2;
    then the local size is (4, 8, 2) and stride is (16, 1, 8).

    Args:
        tensor (:class:`torch.Tensor`):
            Global tensor which DTensor will distribute
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        placements (Sequence[:class:`Placement`]):
            The attribute of the DTensor that describes its layout
            on the mesh topology.

    Returns:
        local_shape: A List of int which specifies the size of the local tensor.
        local_stride: A List of int which specifies the stride of the local tensor.
    """
    local_shape = list(global_tensor.size())
    local_stride = list(global_tensor.stride())

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
            assert shard_dim < len(local_shape), (
                f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)} "
                f"for placement number {idx}."
            )

            global_dim_size = local_shape[shard_dim]
            assert global_dim_size % mesh_dim_size == 0, (
                f"Global dim {global_dim_size} not divisible by mesh size {mesh_dim_size}"
            )
            local_shape[shard_dim] = global_dim_size // mesh_dim_size

            # shrink strides that were scaled up globally
            for i in range(len(local_stride)):
                if (
                    i != shard_dim
                    and local_stride[i] >= local_stride[shard_dim] * mesh_dim_size
                ):
                    local_stride[i] = local_stride[i] // mesh_dim_size

        elif not isinstance(placement, (Replicate, Partial)):
            raise RuntimeError(f"placement type {type(placement)} not supported!")

    return local_shape, local_stride


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
        local_shape = torch.tensor(list(shape), device=mesh.device_type)
        gathered_shaped_tensors = [
            torch.empty_like(local_shape, device=local_shape.device)
            for _ in range(mesh.size())
        ]
        funcol.all_gather_inplace(gathered_shaped_tensors, local_shape, mesh)
        sharded_dim_sum = 0
        shard_dim = placements[0].dim
        other_dims = [d for d in range(mesh.ndim) if d != shard_dim]
        for shape_tensor in gathered_shaped_tensors:
            if not torch.equal(local_shape[other_dims], shape_tensor[other_dims]):
                raise RuntimeError(
                    "Non-sharded dimensions should have identical size across ranks."
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
