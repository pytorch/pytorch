import itertools
import logging
import math
import threading
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch._logging import LazyString
from torch._prims_common import ShapeType
from torch.distributed import RankType
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema
from torch.distributed.tensor.placement_types import (
    _is_shard_like,
    _StridedShard,
    _StridedShardOffsetMode,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.types import IntLikeType


logger = logging.getLogger(__name__)


def _format_implicit_redistribution_msg(schema: OpSchema) -> str:
    return f"Implicit redistribution occurred for {schema} while ExplicitRedistributionContext was active"


class ExplicitRedistributionContext:
    """
    Within this context manager, DTensor will refuse to perform implicit redistribution,
    instead raising an error.  Manual calls to ``redistribute()`` are required wherever a redistribution
    must occur to avoid erroring.  This can be used to ensure that the user is aware of all redistribution.

    Note: it is easier to use this mode on just the forward pass of a typical DTensor program, as the backwards pass
    may contain implicit redistribution calls that are not visible to the user and difficult to replace with manual
    calls.  Redistribution during backward can be made explicit by writing `autograd.Function`s that are no-op
    during forward and perform a manual redistribution during backwards.

    enable (bool) if False, disables the context manager. Can be used nested inside an enabled region.

    strict (bool) if True, triggers on any redistribution.  If False, only triggers on redistributions that perform
    communication.

    mode (str) Determines what happens when ExplicitRedistributionContext triggers:
    "raise": raises an exception, "warn" issues a warning
    """

    _local = threading.local()

    def __init__(self, enable: bool = True, strict: bool = False, mode="raise"):
        self._enable = enable
        self._strict = strict
        if mode not in ("raise", "warn"):
            raise RuntimeError(f"Invalid mode {mode}")
        self._raise_on_redistribution = mode == "raise"

    @classmethod
    def observe_redistribution(
        cls,
        src_spec: DTensorSpec,
        dst_spec: DTensorSpec,
        redistribution_msg: LazyString,
    ):
        if instance := getattr(cls._local, "_active", None):
            allowed = True
            if instance._enable:
                if instance._strict:
                    allowed = False
                else:
                    allowed = redistribute_cost(src_spec, dst_spec) <= 0
            if not allowed:
                if instance._raise_on_redistribution:
                    raise RuntimeError(redistribution_msg)
                else:
                    logger.warning(redistribution_msg)

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
        skip_offset (bool): If True, skip computing the global offsets and return an empty
            tuple for global_offset. This can improve performance when only the local shape
            is needed. Defaults to False.

    Return:
        local_shape: the shape of the DTensor's _local_tensor on the current rank.
        global_offset: a tuple of offsets for each dimension of the global tensor shape,
        identifying how this shard fits into the global tensor in each dimension. If
        skip_offset is True, this will be an empty tuple.

    .. note::
        Along each dim, ``global_offset`` is the global index of the first row or
        column the local tensor holds, or ``global_shape[dim]`` if it holds none.
        With ``local_shape``, it describes the local tensor only if the local tensor
        is one chunk: a rectangle of consecutive rows and columns of the global
        tensor. A ``_StridedShard`` can leave a rank holding several separate chunks.
        For such placements, use :func:`compute_local_shape_and_chunks`.

    """
    empty_offset = ()
    if not mesh._is_current_rank_part_of_mesh():
        # if rank not in the mesh, return empty offset
        return ((0,), empty_offset)

    return _compute_local_shape_and_global_offset(
        global_shape, mesh.shape, mesh._sym_get_coordinate, placements, skip_offset
    )


@maybe_run_for_local_tensor
def _get_shard_size_and_offsets(
    curr_local_size: int,
    mesh_dim_size: int,
    rank: RankType,
    placement: Shard | _StridedShard,
    previous_offsets,
    zero_global_offset: int,
    skip_offset: bool,
) -> tuple[int, torch.Tensor | None]:
    kwargs: dict[str, Any] = {
        "curr_local_size": curr_local_size,
        "num_chunks": mesh_dim_size,
        "rank": rank,
    }
    if isinstance(placement, _StridedShard):
        kwargs["offset_mode"] = (
            _StridedShardOffsetMode.NONE if skip_offset else _StridedShardOffsetMode.ALL
        )
        # _StridedShard.local_shard_size_and_offset materializes the offsets list
        # via .tolist() on a (potentially fake) index tensor; under FakeTensorMode
        # that allocates one unbacked SymInt per element. Skip when the caller
        # discards the offsets anyway.
    shard_size, shard_offsets = placement._local_shard_size_and_offset(**kwargs)
    if skip_offset:
        return shard_size, None
    if shard_size == 0:
        return shard_size, torch.arange(zero_global_offset, zero_global_offset + 1)
    if isinstance(placement, Shard) and not isinstance(placement, _StridedShard):
        if not isinstance(shard_offsets, int):
            raise AssertionError
        index = torch.arange(shard_offsets, shard_offsets + shard_size)
    else:
        if not isinstance(shard_offsets, list):
            raise AssertionError
        index = torch.tensor(shard_offsets)
    if previous_offsets is None:
        return shard_size, index
    else:
        return shard_size, previous_offsets[index]


@maybe_run_for_local_tensor
def _get_first_offset(offsets: torch.Tensor) -> int:
    return int(offsets[0])


# accept 'plain data types' to enable simpler unit testing without creating device mesh
def _compute_local_shape_and_global_offset(
    global_shape: ShapeType,
    mesh_shape: ShapeType,
    my_coordinate: list[int] | Callable[[int], RankType] | None,
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

    Args:
        global_shape (ShapeType): The global shape of the tensor.
        mesh_shape (ShapeType): The shape of the device mesh.
        my_coordinate (Optional[list[int]]): The coordinate of the current rank in the device mesh.
        placements (Sequence[Placement]): The placements of the DTensor.
        skip_offset (bool): If True, skip computing the global offsets and return an empty
            tuple for global_offset. This can improve performance when only the local shape
            is needed. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - local_shape (tuple[int, ...]): The shape of the local shard on the current rank.
            - global_offset (tuple[int, ...]): The offsets for each dimension identifying where
              this shard begins in the global tensor. If skip_offset is True, this will be an
              empty tuple.
    """

    if isinstance(my_coordinate, (list, tuple)):
        _coord: list | tuple = my_coordinate

        def coordinate_lookup(dim: int) -> RankType:
            return _coord[dim]
    else:
        if my_coordinate is None:
            raise AssertionError
        coordinate_lookup = my_coordinate

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
        if not isinstance(placement, (Shard, _StridedShard)):
            continue
        shard_dim = placement.dim
        zero_global_offset = global_shape[shard_dim]
        if shard_dim >= len(local_shape):
            raise AssertionError(
                f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
            )
        previous_offsets = shard_dim_to_global_offsets.get(shard_dim)
        shard_size, shard_offsets = _get_shard_size_and_offsets(
            local_shape[shard_dim],
            mesh_shape[mesh_dim],
            coordinate_lookup(mesh_dim),
            placement,
            previous_offsets,
            zero_global_offset,
            skip_offset,
        )
        local_shape[shard_dim] = shard_size
        shard_dim_to_global_offsets[shard_dim] = shard_offsets
    if skip_offset:
        return tuple(local_shape), ()
    global_offset = [0] * len(global_shape)
    for shard_dim, global_offsets in shard_dim_to_global_offsets.items():
        global_offset[shard_dim] = _get_first_offset(global_offsets)
    return tuple(local_shape), tuple(global_offset)


def compute_local_shape_and_chunks(
    global_shape: ShapeType,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> tuple[
    tuple[IntLikeType, ...],
    tuple[tuple[IntLikeType, ...], ...],
    tuple[tuple[IntLikeType, ...], ...],
    tuple[tuple[IntLikeType, ...], ...],
]:
    """
    Compute the local tensor shape of a DTensor on its current global rank, and the
    chunks of the global tensor that the local tensor holds.

    :func:`compute_local_shape_and_global_offset` returns where the local tensor
    starts in the global tensor, which describes it only if it is one chunk of
    consecutive rows and columns. A ``_StridedShard`` can leave a rank holding
    several separate chunks. The chunks describe the local tensor either way. Each is
    a rectangle of consecutive indices along every dim, of the global tensor and of
    the local tensor: chunk ``i`` starts at ``global_offsets[i]`` in the global
    tensor and at ``local_offsets[i]`` in the local tensor, and has shape
    ``local_sizes[i]``. This is the layout of the fields of
    :class:`torch.distributed.checkpoint.protocol.CheckpointableTensor`::

        size = local_sizes[i]
        local_index = tuple(slice(o, o + n) for o, n in zip(local_offsets[i], size))
        global_index = tuple(slice(o, o + n) for o, n in zip(global_offsets[i], size))
        assert torch.equal(local_tensor[local_index], global_tensor[global_index])

    The number of chunks depends only on ``placements``: along each dim it is the
    product of the ``split_factor`` of the ``_StridedShard`` placements on that dim,
    and some chunks can be empty. Because it depends on neither the rank nor the
    sizes, every rank in the mesh returns the same structure, and this can be traced
    with a symbolic mesh coordinate, as under
    ``torch.compiler.config.compile_on_one_rank``. An empty chunk can have the same
    global offset as another chunk, and DCP looks up the chunks of a
    ``CheckpointableTensor`` by global offset, so drop the chunks with a zero in
    ``local_sizes`` before building one. Under ``LocalTensorMode``, do that in a
    function decorated with ``maybe_run_for_local_tensor`` that also uses the
    chunks, so that it sees each rank's sizes as ints. Outside one, a per-rank size
    equals 0 only if it is 0 on every rank, and ``<`` and ``>`` raise on it. The
    function can't return the filtered chunks, because their number can differ
    between ranks. With a symbolic mesh coordinate, the comparison can need a
    data-dependent guard.

    Example: a tensor of shape ``(8, 2)``, with its elements labelled ``a`` to ``p``,
    and placements ``[_StridedShard(0, split_factor=2), Replicate()]`` on a 2 x 2
    mesh::

        row 0  [ a, b ]
        row 1  [ c, d ]
        row 2  [ e, f ]
        row 3  [ g, h ]
        row 4  [ i, j ]
        row 5  [ k, l ]
        row 6  [ m, n ]
        row 7  [ o, p ]

                         mesh dim 1
                         0                          1
        mesh dim 0   0   rank 0: rows 0, 1, 4, 5    rank 1: rows 0, 1, 4, 5
                     1   rank 2: rows 2, 3, 6, 7    rank 3: rows 2, 3, 6, 7

    Rank 0 has local tensor ``[[a, b], [c, d], [i, j], [k, l]]``: global rows 0, 1
    at local rows 0, 1, and global rows 4, 5 at local rows 2, 3::

        local_shape = (4, 2)
        global_offsets = ((0, 0), (4, 0))
        local_offsets = ((0, 0), (2, 0))
        local_sizes = ((2, 2), (2, 2))

    With ``[_StridedShard(0, split_factor=2), Shard(0)]`` instead, rank 0 holds only
    rows 0, 1, and its second chunk is empty::

        local_shape = (2, 2)
        global_offsets = ((0, 0), (4, 0))
        local_offsets = ((0, 0), (2, 0))
        local_sizes = ((2, 2), (0, 2))

    Args:
        global_shape (ShapeType): The global shape of the DTensor.
        mesh (:class:`DeviceMesh`): The device mesh this DTensor is distributed on.
        placements (Sequence[:class:`Placement`]): The placements of the DTensor.

    Return:
        local_shape: the shape of the DTensor's _local_tensor on the current rank.
        global_offsets: for each chunk, where it starts in the global tensor.
        local_offsets: for each chunk, where it starts in the local tensor.
        local_sizes: for each chunk, its shape.
        Under ``LocalTensorMode``, values that differ between ranks are ``SymInt`` s
        holding each rank's value. If the current rank is not part of the mesh,
        ``local_shape`` is ``(0,)`` and there are no chunks.
    """
    if not mesh._is_current_rank_part_of_mesh():
        return (0,), (), (), ()
    coordinate = [
        mesh._sym_get_coordinate(mesh_dim) if _is_shard_like(placement) else None
        for mesh_dim, placement in enumerate(placements)
    ]
    return _compute_local_shape_and_chunks(
        global_shape, mesh.shape, coordinate, placements
    )


# accept 'plain data types' to enable simpler unit testing without creating device mesh
@maybe_run_for_local_tensor
def _compute_local_shape_and_chunks(
    global_shape: ShapeType,
    mesh_shape: ShapeType,
    my_coordinate: Sequence[IntLikeType | None],
    placements: Sequence[Placement],
) -> tuple[
    tuple[IntLikeType, ...],
    tuple[tuple[IntLikeType, ...], ...],
    tuple[tuple[IntLikeType, ...], ...],
    tuple[tuple[IntLikeType, ...], ...],
]:
    """
    :func:`compute_local_shape_and_chunks` for the mesh coordinate ``my_coordinate``.
    Under ``LocalTensorMode`` this runs once per rank, and the ranks' results are
    combined value by value, which needs the same structure on every rank.
    """
    # For each tensor dim, the global indices the local tensor holds, in local
    # order, run-length encoded as (global_start, length). Sharding is applied
    # left to right.
    runs_per_dim: list[list[tuple[IntLikeType, IntLikeType]]] = [
        [(0, size)] for size in global_shape
    ]
    for mesh_dim, placement in enumerate(placements):
        if not _is_shard_like(placement):
            continue
        shard_dim = placement.dim
        if shard_dim >= len(global_shape):
            raise ValueError(
                f"Sharding dim {shard_dim} greater than tensor ndim {len(global_shape)}"
            )
        rank = my_coordinate[mesh_dim]
        if rank is None:
            raise AssertionError(f"No mesh coordinate for shard mesh dim {mesh_dim}")
        runs_per_dim[shard_dim] = _shard_runs(
            runs_per_dim[shard_dim], mesh_shape[mesh_dim], rank, placement
        )

    # A chunk is one run along each dim, as ((global_start, length), local_start).
    local_starts = [
        list(itertools.accumulate((length for _, length in r), initial=0))[:-1]
        for r in runs_per_dim
    ]
    chunks = list(
        itertools.product(*(zip(r, s) for r, s in zip(runs_per_dim, local_starts)))
    )
    num_chunks = math.prod(
        p._split_factor_int() for p in placements if isinstance(p, _StridedShard)
    )
    if len(chunks) != num_chunks:
        raise AssertionError(
            f"Expected {num_chunks} chunks for {placements}, got {len(chunks)}"
        )
    return (
        tuple(sum(length for _, length in r) for r in runs_per_dim),
        tuple(tuple(start for (start, _), _ in chunk) for chunk in chunks),
        tuple(tuple(local_start for _, local_start in chunk) for chunk in chunks),
        tuple(tuple(length for (_, length), _ in chunk) for chunk in chunks),
    )


def _shard_runs(
    shard_dim_runs: list[tuple[IntLikeType, IntLikeType]],
    mesh_dim_size: int,
    rank: IntLikeType,
    placement: Shard | _StridedShard,
) -> list[tuple[IntLikeType, IntLikeType]]:
    """
    The global indices left on ``rank`` along the tensor dim that ``placement``
    shards. ``shard_dim_runs`` is the global indices the local tensor holds
    along that dim so far, in local order, run-length encoded as
    ``(global_start, length)``: ``[(0, 2), (3, 2)]`` stands for 0, 1, 3, 4.
    Returns them encoded the same way.
    """
    size = sum(length for _, length in shard_dim_runs)
    split_factor = (
        placement._split_factor_int() if isinstance(placement, _StridedShard) else 1
    )
    selected: list[tuple[IntLikeType, IntLikeType]] = []
    # TODO: share this two-level split with _StridedShard.local_shard_size_and_offset.
    for group in range(split_factor):
        group_size, group_start = Shard.local_shard_size_and_offset(
            size, split_factor, group
        )
        piece_size, piece_start = Shard.local_shard_size_and_offset(
            group_size, mesh_dim_size, rank
        )
        start = group_start + piece_start
        selected.extend(_select_from_runs(shard_dim_runs, start, start + piece_size))
    return selected


def _select_from_runs(
    runs: list[tuple[IntLikeType, IntLikeType]],
    start: IntLikeType,
    stop: IntLikeType,
) -> list[tuple[IntLikeType, IntLikeType]]:
    """
    The part of ``runs`` at positions ``start`` to ``stop - 1`` of the local
    tensor, as runs of global indices: one for each of ``runs``, cut down to what
    falls in that range and possibly empty, so the structure doesn't depend on
    ``start`` or ``stop``, and nothing here branches on them.
    """
    selected: list[tuple[IntLikeType, IntLikeType]] = []
    position: IntLikeType = 0  # position of the run's first index in the local tensor
    for global_start, length in runs:
        first = _clamp(start - position, 0, length)
        last = _clamp(stop - position, 0, length)
        selected.append((global_start + first, last - first))
        position = position + length
    return selected


def _clamp(x: IntLikeType, lo: IntLikeType, hi: IntLikeType) -> IntLikeType:
    return torch.sym_min(torch.sym_max(x, lo), hi)


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
        if _is_shard_like(placement):
            if placement.dim < 0:
                raise AssertionError(
                    "Shard placements should have negative dims normalized in "
                    f"the user-facing APIs: {placement}"
                )
            shard_dim = placement.dim
            if shard_dim >= len(local_shape):
                raise AssertionError(
                    f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)} "
                    f"for placement number {idx}."
                )

            global_dim_size = local_shape[shard_dim]
            if global_dim_size % mesh_dim_size != 0:
                raise AssertionError(
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
    # NOTE: isinstance(_, Shard) does not match _StridedShard; see _is_shard_like().
    elif isinstance(placements[0], Shard):

        @maybe_run_for_local_tensor
        def _create_local_shape_tensor(shape):
            return torch.tensor(list(shape), device=mesh.device_type)

        local_shape = _create_local_shape_tensor(shape)
        gathered_shaped_tensors = [
            torch.empty_like(local_shape, device=local_shape.device)
            for _ in range(mesh.size())
        ]
        funcol.all_gather_inplace(gathered_shaped_tensors, local_shape, mesh)

        @maybe_run_for_local_tensor
        def _validate_and_compute_global_shape(local_shape, gathered_shaped_tensors):
            sharded_dim_sum = 0
            shard_dim = placements[0].dim  # type: ignore[union-attr]
            other_dims = [d for d in range(len(shape)) if d != shard_dim]
            for shape_tensor in gathered_shaped_tensors:
                if not torch.equal(local_shape[other_dims], shape_tensor[other_dims]):
                    raise RuntimeError(
                        "Non-sharded dimensions should have identical size across ranks."
                    )
                shape_tensor_list = shape_tensor.tolist()
                sharded_dim_sum += shape_tensor_list[shard_dim]
            return sharded_dim_sum

        sharded_dim_sum = _validate_and_compute_global_shape(
            local_shape, gathered_shaped_tensors
        )
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
    global_stride: ShapeType, local_shape: ShapeType
) -> tuple[int, ...]:
    """
    Compute the stride of a local tensor shard, given the global stride and local shape.

    Derives strides by preserving the memory layout (dimension ordering) implied
    by the global strides, then computing contiguous strides for the local shape
    in that order.  Assumes the global tensor is non-overlapping and dense.
    """
    ndim = len(global_stride)
    # Sort dims by global stride descending to recover memory layout order.
    # Stable sort preserves original dim order for ties, which only occur
    # on size-1 dims where the stride value is semantically irrelevant.
    perm = sorted(range(ndim), key=lambda d: global_stride[d], reverse=True)
    local_strides = [0] * ndim
    s = 1
    for d in reversed(perm):
        local_strides[d] = s
        s *= local_shape[d]
    return tuple(local_strides)


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


def assert_no_mixed_partial_types(placements: Sequence[Placement]) -> None:
    """
    Assert that a placement list doesn't contain mixed Partial reduce types.

    Mixed Partial types (e.g., ``Partial("sum")`` and ``Partial("max")`` together in the
    same placement list) are not supported and will raise a ``ValueError``. This restriction
    exists because nonlinear reductions (e.g., max) don't commute with linear reductions
    (e.g., sum), which means the relative ordering of different partial types would be
    semantically critical during redistribution. Rather than introducing complex ordering
    constraints, we prohibit mixing different Partial reduce types.

    Note: Partial("sum") and Partial("avg") DO commute with each other, so they can be ordered
    arbitrarily, and we allow this.

    This function is called internally by public APIs like :meth:`DTensor.from_local` and
    :func:`distribute_tensor` to validate placements early, before DTensor construction.

    Args:
        placements (Sequence[:class:`Placement`]): A sequence of placement specifications
            to validate.

    Raises:
        ValueError: If the placements contain more than one distinct Partial reduce type.
    """
    partial_reduce_ops: set[str] = set()
    for p in placements:
        if isinstance(p, Partial):
            partial_reduce_ops.add(p.reduce_op)

    if len(partial_reduce_ops) > 1 and partial_reduce_ops != {"sum", "avg"}:
        raise ValueError(
            f"Mixed Partial reduce types are not supported in the same placement list. "
            f"Found reduce ops: {partial_reduce_ops}. "
            f"Please ensure all Partial placements use the same reduce operation."
        )
