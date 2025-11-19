import itertools
import math
from collections import defaultdict
from typing import cast, NamedTuple, Optional

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    MaskPartial,
    Partial,
    Placement,
    Replicate,
    Shard,
)


__all__ = [
    "ShardOrderEntry",
    "convert_shard_order_to_StridedShard",
    "maybe_convert_StridedShard_to_shard_order",
    "format_shard_order_str",
]


class ShardOrderEntry(NamedTuple):
    """
    Represents how a single tensor dimension is sharded across mesh dimensions.

    Attributes:
        tensor_dim: The tensor dimension being sharded (e.g., 0, 1, 2 for a 3D tensor).
        mesh_dims: Tuple of mesh dimensions across which this tensor dimension is sharded,
                   in execution order. The first mesh dim is applied first, second is applied
                   second, etc. This tuple is guaranteed to be non-empty.

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_DISTRIBUTED)
        >>> # Tensor dim 1 sharded across mesh dim 2, then mesh dim 0
        >>> ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 0))

        >>> # Tensor dim 0 sharded only on mesh dim 1
        >>> ShardOrderEntry(tensor_dim=0, mesh_dims=(1,))
    """

    tensor_dim: int
    mesh_dims: tuple[int, ...]  # guaranteed to be non-empty


# Type alias for the complete shard order specification
# A tuple of ShardOrderEntry, one per sharded tensor dimension
#
# Example:
#   shard_order = (
#       ShardOrderEntry(tensor_dim=0, mesh_dims=(1,)),
#       ShardOrderEntry(tensor_dim=2, mesh_dims=(0, 3)),
#   )
#   This means:
#     - Tensor dimension 0 is sharded on mesh dimension 1
#     - Tensor dimension 2 is sharded on mesh dimension 0 first, then mesh dimension 3
ShardOrder = tuple[ShardOrderEntry, ...]


def _verify_shard_order(
    placements: tuple[Placement, ...], shard_order: ShardOrder
) -> None:
    """Verify that the shard_order is valid and matches the placements."""
    total_shard = 0
    if any(isinstance(p, _StridedShard) for p in placements):
        return
    prev_tensor_dim = -1
    for entry in shard_order:
        tensor_dim = entry.tensor_dim
        mesh_dims = entry.mesh_dims
        assert len(mesh_dims) > 0, f"shard_order {shard_order} has empty mesh dim"
        assert tensor_dim >= 0, (
            f"shard_order {shard_order} has invalid tensor dim {tensor_dim}"
        )
        assert tensor_dim > prev_tensor_dim, (
            "tensor dim should be sorted in shard_order"
        )
        prev_tensor_dim = tensor_dim
        total_shard += len(mesh_dims)
        for mesh_dim in mesh_dims:
            assert 0 <= mesh_dim < len(placements), (
                f"shard_order {shard_order} has invalid mesh dim {mesh_dims}"
            )
            assert placements[mesh_dim] == Shard(tensor_dim), (
                f"placement[{mesh_dim}] doesn't have a matching shard in shard_order"
            )
    assert total_shard == sum(1 for p in placements if isinstance(p, Shard))


def _normalize_placements_into_shard_order(
    placements: tuple[Placement, ...], mesh: DeviceMesh
) -> tuple[tuple[Placement, ...], Optional[ShardOrder]]:
    # if no _StridedShard in placements, we create default order
    if not any(isinstance(p, _StridedShard) for p in placements):
        return placements, _compute_default_shard_order(placements)
    # _StridedShard in placements, try check if it can be decoded as shard order
    shard_order = maybe_convert_StridedShard_to_shard_order(placements, mesh)
    if shard_order is not None:
        normalized_placements = tuple(
            [
                p if not isinstance(p, _StridedShard) else Shard(p.dim)
                for p in placements
            ]
        )
        return normalized_placements, shard_order
    # unable to decode placements to shard order(e.g., the _StridedShard is
    # also used by `view` op shard propagation).
    return placements, None


def _compute_default_shard_order(
    placements: tuple[Placement, ...],
) -> ShardOrder:
    """
    Compute the default shard order from placements.

    Returns a ShardOrder where each ShardOrderEntry maps a tensor dimension
    to the mesh dimensions it's sharded on, in left-to-right order.
    """
    # follow default left-to-right device order if shard_order is not specified
    tensor_dim_to_mesh_dims: defaultdict[int, list[int]] = defaultdict(list)
    mesh_ndim = len(placements)
    for mesh_dim in range(mesh_ndim):
        # _StridedShard may not be a subclass of Shard in the future
        if isinstance(placements[mesh_dim], Shard | _StridedShard):
            placement = cast(Shard, placements[mesh_dim])
            shard_dim = placement.dim
            assert shard_dim >= 0, (
                f"Shard dim {shard_dim} in placements {placements} must be normalized"
            )
            tensor_dim_to_mesh_dims[shard_dim].append(mesh_dim)

    # Convert dict into ShardOrderEntry tuples
    default_shard_order = tuple(
        ShardOrderEntry(tensor_dim=key, mesh_dims=tuple(value))
        for key, value in sorted(tensor_dim_to_mesh_dims.items())
        if value
    )
    return default_shard_order


def _is_default_shard_order(shard_order: ShardOrder) -> bool:
    """
    Check if the device order is the default left-to-right order.
    """
    for entry in shard_order:
        mesh_dims = entry.mesh_dims
        is_increasing = all(prev < nxt for prev, nxt in itertools.pairwise(mesh_dims))
        if not is_increasing:
            return False
    return True


def convert_shard_order_to_StridedShard(
    shard_order: ShardOrder, placements: tuple[Placement, ...], mesh: DeviceMesh
) -> tuple[Placement, ...]:
    """
    Convert ShardOrder to placements with _StridedShard.

    This function converts a ShardOrder specification into a tuple of Placement objects,
    using _StridedShard when a tensor dimension is sharded across multiple mesh dimensions
    in a non-default order. The split_factor of each _StridedShard is determined by the
    product of mesh dimension sizes that appear earlier in the shard order but later in
    the placement tuple.

    Args:
        shard_order: ShardOrder specification indicating which tensor dimensions are
            sharded on which mesh dimensions and in what execution order.
        placements: Tuple of Placement objects that does not contain _StridedShard.
        mesh: DeviceMesh containing the size information for each mesh dimension.

    Returns:
        Updated tuple of Placement objects with Shard or _StridedShard placements.

    Algorithm:
        For each ShardOrderEntry in shard_order:
          - For each mesh dimension in the entry's mesh_dims (in order):
            - Calculate split_factor as the product of mesh sizes for all mesh dimensions
              that appear:
              1. Earlier in the shard order (lower index in mesh_dims), and
              2. Later in the placement tuple (higher mesh dimension index)
            - If split_factor == 1: use normal Shard
            - Otherwise: use _StridedShard with the calculated split_factor

    Example:
        >>> # xdoctest: +SKIP("Requires DeviceMesh")
        >>> # Tensor dimension 0 sharded on mesh dims [2, 0, 1] in that order
        >>> # mesh = DeviceMesh([4, 3, 2])  # sizes: mesh[0]=4, mesh[1]=3, mesh[2]=2
        >>> shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 0, 1)),)
        >>> placements = (Shard(0), Shard(0), Shard(0))
        >>> # For mesh_dim=2 (index 0 in mesh_dims): no earlier dims, split_factor=1
        >>> #   -> placements[2] = Shard(0)
        >>> # For mesh_dim=0 (index 1 in mesh_dims): mesh_dim=2 is earlier and has index 2>0
        >>> #   -> split_factor = mesh.size(2) = 2
        >>> #   -> placements[0] = _StridedShard(0, split_factor=2)
        >>> # For mesh_dim=1 (index 2 in mesh_dims): mesh_dim=2 is earlier and has index 2>1
        >>> #   -> split_factor = mesh.size(2) = 2
        >>> #   -> placements[1] = _StridedShard(0, split_factor=2)
        >>> # Result: (_StridedShard(0, sf=2), _StridedShard(0, sf=2), Shard(0))
    """
    placements_list = list(placements)
    for entry in shard_order:
        tensor_dim = entry.tensor_dim
        mesh_dims = entry.mesh_dims
        for idx in range(len(mesh_dims)):
            # TODO(zpcore): split_factor from `view` and `shard order`
            # should be able to be multiplied into one. Need to loosen the
            # condition here.
            mesh_dim = mesh_dims[idx]
            if type(placements[mesh_dim]) is not Shard:
                raise ValueError(
                    f"Only Shard placement can be converted to _StridedShard, "
                    f"found {placements[mesh_dim]} in {placements=}."
                )
            split_factor = math.prod(
                mesh.size(i) for i in mesh_dims[:idx] if i > mesh_dim
            )
            if split_factor == 1:
                # use normal Shard
                placements_list[mesh_dim] = Shard(tensor_dim)
            else:
                placements_list[mesh_dim] = _StridedShard(
                    tensor_dim, split_factor=split_factor
                )
    return tuple(placements_list)


def maybe_convert_StridedShard_to_shard_order(
    placements: tuple[Placement, ...], mesh: DeviceMesh
) -> Optional[ShardOrder]:
    """
    Try to convert _StridedShard placements to ShardOrder.

    This is the inverse of `_convert_shard_order_to_StridedShard`. It reconstructs the shard
    order by examining the split_factor of each _StridedShard and determining its position
    in the execution order. If the _StridedShard configuration cannot be represented as a
    valid ShardOrder (i.e., there's no shard order that produces the observed split_factors),
    this function returns None.

    Args:
        placements: Tuple of Placement objects that may contain _StridedShard.
        mesh: DeviceMesh containing the size information for each mesh dimension.

    Returns:
        ShardOrder if conversion is possible, None otherwise. For placements without
        _StridedShard, returns the default shard order.

        Algorithm:
            1. If no _StridedShard in placements, return default shard order
            2. Create an empty list for each tensor dimension to represent mesh dim ordering
            3. Iterate through placements in reverse order (right to left):
                - For each Shard/_StridedShard on a tensor dimension:
                - Extract its split_factor (1 for Shard, split_factor for _StridedShard)
                - Find the position in mesh_dims_order where accumulated_sf equals split_factor
                - accumulated_sf is the product of mesh sizes of mesh dimensions that appear
                    earlier in mesh_dims_order (lower indices)
                - Insert mesh_dim at the found position
            4. If no valid position found for any split_factor, return None (unable to convert)
            5. Construct ShardOrderEntry for each tensor dimension from mesh_dims_order

    Example:
        >>> # xdoctest: +SKIP("Requires DeviceMesh")
        >>> # mesh = DeviceMesh([4, 3, 2])  # sizes: mesh[0]=4, mesh[1]=3, mesh[2]=2
        >>> # placements = (_StridedShard(0, sf=2), _StridedShard(0, sf=2), Shard(0))
        >>> # Process tensor_dim=0 from right to left:
        >>> #   - mesh_dim=2: Shard(0) with sf=1
        >>> #     Try position 0: accumulated_sf=1, matches! Insert at position 0
        >>> #     Current mesh_dims_order order: [2]
        >>> #   - mesh_dim=1: _StridedShard(0, sf=2) with sf=2
        >>> #     Try position 0: accumulated_sf=1, no match
        >>> #     Try position 1: accumulated_sf=1*mesh.size(2)=2, matches! Insert at position 1
        >>> #     Current mesh_dims_order order: [2, 1]
        >>> #   - mesh_dim=0: _StridedShard(0, sf=2) with sf=2
        >>> #     Try position 0: accumulated_sf=1, no match
        >>> #     Try position 1: accumulated_sf=1*mesh.size(2)=2, matches! Insert at position 1
        >>> #     Final mesh_dims_order order: [2, 0, 1]
        >>> # Result: ShardOrder((ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 0, 1)),))
        >>> # This means: first shard on mesh_dim=2, then mesh_dim=0, then mesh_dim=1

    Note:
        This function validates that _StridedShard can be represented as a ShardOrder.
        Not all _StridedShard configurations are valid - the split_factor must match
        the product of mesh sizes in some execution order.
    """
    if not any(isinstance(p, _StridedShard) for p in placements):
        return _compute_default_shard_order(placements)
    max_tensor_dim = (
        max([i.dim for i in placements if isinstance(i, Shard | _StridedShard)]) + 1
    )
    shard_order = []

    tensor_dim_to_mesh_dims_order: list[list[int]] = [[] for i in range(max_tensor_dim)]
    for mesh_dim in reversed(range(len(placements))):
        cur_placement = placements[mesh_dim]
        # _StridedShard may not be a subclass of Shard in the future, so write in this way:
        if isinstance(cur_placement, Shard | _StridedShard):
            tensor_dim = cur_placement.dim
            mesh_dims_order = tensor_dim_to_mesh_dims_order[tensor_dim]
            cur_sf = 1
            if isinstance(cur_placement, _StridedShard):
                cur_sf = cur_placement.split_factor
            accumulated_sf = 1
            find_order = False
            for i in range(len(mesh_dims_order) + 1):
                if accumulated_sf == cur_sf:
                    mesh_dims_order.insert(i, mesh_dim)
                    find_order = True
                    break
                if i < len(mesh_dims_order):
                    accumulated_sf *= mesh.size(mesh_dims_order[i])
            if not find_order:
                # _StridedShard is not convertible to ShardOrder
                return None
        else:
            if not isinstance(cur_placement, Replicate | Partial | MaskPartial):
                raise ValueError(
                    f"Unsupported placement type {type(cur_placement)} encountered in "
                    f"{placements}; expected Replicate, Partial, or MaskPartial."
                )
    for tensor_dim in range(max_tensor_dim):
        if len(tensor_dim_to_mesh_dims_order[tensor_dim]) > 0:
            shard_order.append(
                ShardOrderEntry(
                    tensor_dim=tensor_dim,
                    mesh_dims=tuple(tensor_dim_to_mesh_dims_order[tensor_dim]),
                )
            )
    return tuple(shard_order)


def format_shard_order_str(
    placements: tuple[Placement, ...],
    shard_order: Optional[ShardOrder] = None,
) -> str:
    """
    Format DTensor sharding information as a human-readable string.

    This method formats the sharding pattern in mesh-centric order, showing the placement
    for each mesh dimension sequentially. When a tensor dimension is sharded across multiple
    mesh dimensions, the order index indicates the execution sequence of the sharding operations.

    Args:
        placements: Tuple of placement objects for each mesh dimension.
        shard_order: Optional ShardOrder specifying the sharding order.

    Returns:
        String representation of the sharding pattern in mesh-centric format.

    Example:
        For a 3D tensor on a 2x2x2x2 mesh (16 devices) with::

            placements = [Partial(), Shard(1), Shard(1), Replicate()]
            shard_order = (ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 1)),)

        Mesh configuration:
            - mesh_dim_0: Partial reduction (sum)
            - mesh_dim_1: Shard tensor dimension 1 (executed second, order index 1)
            - mesh_dim_2: Shard tensor dimension 1 (executed first, order index 0)
            - mesh_dim_3: Replicate

        Output: ``"PS(1)[1]S(1)[0]R"``

        Explanation:
            - ``P``: mesh dimension 0 has partial reduction
            - ``S(1)[1]``: mesh dimension 1 shards tensor dimension 1 (order index 1 means second)
            - ``S(1)[0]``: mesh dimension 2 shards tensor dimension 1 (order index 0 means first)
            - ``R``: mesh dimension 3 replicates

        The format follows mesh dimension order (0, 1, 2, 3), and when a tensor dimension
        is sharded across multiple mesh dimensions, the bracketed index shows the execution
        order: ``[0]`` is executed first, ``[1]`` is executed second, etc.
    """
    out_str = ""
    if any(isinstance(p, _StridedShard) for p in placements):
        # _StridedShard should not co-exist with shard_order here. Return _StridedShard representation.
        return placements.__str__()
    # native dtensor-style sharding representation: map from mesh
    # dim to tensor dim
    for mesh_dim, placement in enumerate(placements):
        # _StridedShard may not be a subclass of Shard in the future
        if isinstance(placement, Shard | _StridedShard):
            if shard_order is not None:
                for entry in shard_order:
                    tensor_dim = entry.tensor_dim
                    mesh_dims = entry.mesh_dims
                    if placement.dim == tensor_dim:
                        assert mesh_dim in mesh_dims
                        if len(mesh_dims) > 1:
                            out_str += f"{placement}[{mesh_dims.index(mesh_dim)}]"
                        else:
                            # no need to show device order if the tensor dim is
                            # only sharded in one mesh dim
                            out_str += str(placement)
                        break
            else:
                out_str += str(placement)
        else:
            out_str += str(placement)
    return out_str
