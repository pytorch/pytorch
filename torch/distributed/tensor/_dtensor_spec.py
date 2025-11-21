from dataclasses import dataclass
from typing import Any, cast, NamedTuple, Optional

import torch
import torch.distributed.tensor.placement_utils as putils
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor.placement_utils import ShardOrder
from torch.utils._debug_mode import _stringify_shape
from torch.utils._dtype_abbrs import dtype_abbrs


class TensorMeta(NamedTuple):
    # simple named tuple to represent tensor metadata
    # intentionally to stay simple only for sharding
    # propagation purposes.
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype


# Note [Sharding representation in DTensorSpec]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# DTensorSpec uses two complementary representations for expressing how tensor dimensions
# are sharded across device mesh dimensions:
#
# 1. **Placements with _StridedShard** (internal representation stored in DTensorSpec.placements):
#    A tuple of Placement objects where each position corresponds to a mesh dimension.
#    When a tensor dimension is sharded across multiple mesh dimensions in a specific order,
#    we use _StridedShard to encode that ordering information via split_factor.
#
# 2. **ShardOrder** (logical representation):
#    A tuple of ShardOrderEntry objects that explicitly describe the sharding execution order.
#    Each ShardOrderEntry specifies which tensor dimension is sharded and the ordered sequence
#    of mesh dimensions used for sharding.
#
# Why two representations?
# - Placements align with the device mesh structure (one placement per mesh dimension)
# - ShardOrder provides an intuitive, execution-order view of how sharding is applied
# - _StridedShard.split_factor encodes the ordering information needed to recover ShardOrder
#
# Example: Sharding a tensor dimension across mesh dims [2, 0, 1] in that order
# Given:
#   - mesh = DeviceMesh([4, 3, 6])  # mesh.size(0)=4, mesh.size(1)=3, mesh.size(2)=6
#   - tensor dim 1 should be sharded first on mesh dim 2, then mesh dim 0, then mesh dim 1
#
# ShardOrder representation:
#   shard_order = (ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 0, 1)),)
#   This clearly states: "shard tensor dim 1 on mesh dim 2, then 0, then 1"
#
# Placements representation (using _StridedShard):
#   placements = (_StridedShard(1, split_factor=6), _StridedShard(1, split_factor=6), Shard(1))
#   Position 0 (mesh dim 0): _StridedShard(1, sf=6) - shard tensor dim 1, accounting for prior shards on mesh dim 2
#   Position 1 (mesh dim 1): _StridedShard(1, sf=6) - shard tensor dim 1, accounting for prior shards on mesh dim 2
#   Position 2 (mesh dim 2): Shard(1) - first shard of tensor dim 1, no split_factor needed
#
# The split_factor calculation:
# For each mesh dimension in ShardOrder, split_factor is the product of mesh.size() for all
# mesh dimensions that:
#   1. Appear earlier in the shard order (lower index in mesh_dims), AND
#   2. Have a higher mesh dimension index (later in the placements tuple)
#
# For mesh dim 0 (index 1 in mesh_dims=(2,0,1)):
#   - Mesh dim 2 appears earlier (index 0) and has higher dim index (2 > 0)
#   - split_factor = mesh.size(2) = 6
# For mesh dim 1 (index 2 in mesh_dims=(2,0,1)):
#   - Mesh dim 2 appears earlier (index 0) and has higher dim index (2 > 1)
#   - split_factor = mesh.size(2) = 6
# For mesh dim 2 (index 0 in mesh_dims=(2,0,1)):
#   - No earlier mesh dims, split_factor = 1, so use regular Shard(1)
#
# Conversion between representations:
# - putils.convert_shard_order_to_StridedShard(): ShardOrder -> Placements with _StridedShard
# - putils.maybe_convert_StridedShard_to_shard_order(): Placements -> ShardOrder and replace _StridedShard
#       with Shard in Placements (if possible)
# - DTensorSpec.shard_order property: Returns ShardOrder derived from placements
#
# See https://github.com/pytorch/pytorch/pull/166740 for the detailed algorithm illustration


@dataclass
class DTensorSpec:
    mesh: DeviceMesh
    placements: tuple[Placement, ...]

    # tensor meta will only be set during sharding propagation
    tensor_meta: Optional[TensorMeta] = None

    def __post_init__(self) -> None:
        if not isinstance(self.placements, tuple):
            self.placements = tuple(self.placements)
        self._hash: int | None = None

    def __setattr__(self, attr: str, value: Any) -> None:
        super().__setattr__(attr, value)
        # Make sure to recompute the hash in case any of the hashed attributes
        # change (though we do not expect `mesh`, `placements` or `shard_order`
        # to change)
        if hasattr(self, "_hash") and attr in (
            "mesh",
            "placements",
            "tensor_meta",
        ):
            self._hash = None
        # This assert was triggered by buggy handling for dict outputs in some
        # FX passes, where you accidentally iterate over a dict and try to put
        # keys into TensorMeta.  See https://github.com/pytorch/pytorch/issues/157919
        if attr == "tensor_meta" and value is not None:
            from torch.fx.passes.shape_prop import TensorMetadata

            # TODO: the TensorMetadata arises from
            # test/distributed/tensor/experimental/test_tp_transform.py::TensorParallelTest::test_tp_transform_e2e
            # but I actually can't reproduce it, maybe it is also a bug!
            assert isinstance(value, TensorMeta | TensorMetadata), value

    def _maybe_update_placements_given_shard_order(
        self, shard_order: ShardOrder
    ) -> bool:
        """Check if original placements can be normalized and updated given a shard order. Update if can."""
        # check to see if original placements can be normalized into normal
        # placements w/o StridedShard plus shard order
        normalized_placements, original_shard_order = (
            putils._normalize_placements_into_shard_order(self.placements, self.mesh)
        )
        if original_shard_order is None:
            return False
        try:
            putils._verify_shard_order(normalized_placements, shard_order)
        except Exception:
            # invalid shard_order argument
            return False
        strided_placements = putils.convert_shard_order_to_StridedShard(
            shard_order, self.placements, self.mesh
        )
        self.__setattr__("placements", strided_placements)
        return True

    def _hash_impl(self) -> int:
        # hashing and equality check for DTensorSpec are used to cache the sharding
        # propagation results. We only need to consider the mesh, placements, shape
        # dtype and stride.
        # Caveat: we need to keep this in mind and sync hash and eq if we add more
        # fields to them.
        if self.tensor_meta is not None:
            return hash(
                (
                    self.mesh,
                    self.placements,
                    self.tensor_meta.shape,
                    self.tensor_meta.stride,
                    self.tensor_meta.dtype,
                )
            )
        return hash((self.mesh, self.placements))

    def __hash__(self) -> int:
        # We lazily cache the spec to avoid recomputing the hash upon each
        # use, where we make sure to update the hash when the `tensor_meta`
        # changes by overriding `__setattr__`. This must be lazy so that Dynamo
        # does not try to hash non-singleton `SymInt`s for the stride.
        if self._hash is None:
            self._hash = self._hash_impl()
        return self._hash

    def _check_equals(self, other: object, skip_shapes: bool = False) -> bool:
        if not (
            isinstance(other, DTensorSpec)
            and self.mesh == other.mesh
            and self.placements == other.placements
        ):
            return False
        if self.tensor_meta is None or other.tensor_meta is None:
            return self.tensor_meta == other.tensor_meta

        if skip_shapes:
            return self.tensor_meta.dtype == other.tensor_meta.dtype
        return (
            self.tensor_meta.shape == other.tensor_meta.shape  # type: ignore[union-attr]
            and self.tensor_meta.stride == other.tensor_meta.stride  # type: ignore[union-attr]
            and self.tensor_meta.dtype == other.tensor_meta.dtype  # type: ignore[union-attr]
        )

    def __eq__(self, other: object, /) -> bool:
        return self._check_equals(other)

    def __str__(self) -> str:
        """
        human readable representation of the DTensorSpec
        """
        placement_str = putils.format_shard_order_str(self.placements, self.shard_order)
        if self.tensor_meta is not None:
            tensor_shape = _stringify_shape(self.tensor_meta.shape)
            tensor_dtype = dtype_abbrs[self.tensor_meta.dtype]
        else:
            tensor_shape = "unknown shape"
            tensor_dtype = "unknown dtype"

        return f"Spec({tensor_dtype}{tensor_shape}({placement_str}))"

    @property
    def shape(self) -> torch.Size:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.shape

    @property
    def stride(self) -> tuple[int, ...]:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.stride

    @property
    def ndim(self) -> int:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return len(self.tensor_meta.shape)

    @property
    def num_shards(self) -> int:
        num_shards = 1
        for i, placement in enumerate(self.placements):
            if placement.is_shard():
                num_shards *= self.mesh.size(i)
        return num_shards

    @property
    def device_mesh(self) -> DeviceMesh:
        # simple aliasing for the mesh field, make some
        # checks that mixes DTensor/DTensorSpec easier
        return self.mesh

    @property
    def shard_order(self) -> ShardOrder:
        """
        Returns the shard order specifying the ordered mesh dims over all
        sharded tensor dimensions.

        This property attempts to convert _StridedShard placements to a shard
        order. If conversion is unsuccessful (i.e., no _StridedShard placements
        exist or they cannot be decoded), falls back to computing the default
        left-to-right shard order from the current placements.

        Returns:
            ShardOrder: The derived shard order from _StridedShard placements,
            or the default left-to-right shard order if derivation fails.
        """
        derived_order = putils.maybe_convert_StridedShard_to_shard_order(
            self.placements, self.mesh
        )
        if derived_order is None:
            # use the default left-to-right order if unable to decode
            # _StridedShard to shard_order
            return putils._compute_default_shard_order(self.placements)
        return derived_order

    @property
    def dim_map(self) -> list[int]:
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. It simply return a list of ints
        where dim_map[i] denotes the sharding mapping to the mesh
        dimension, and len(dim_map) == dist_tensor.ndim
        dim_map[i] = -1: means tensor dim i replicate on mesh
        dim_map[i] = j: means tensor dim i shard on mesh dim j

        For example, we have a dist tensor that have the shape of
        [18, 20, 30], and device_mesh([0, 1, 2, 3]), placements:
        [Shard(1)], the dim_map of this placement would be:
        [-1, 0, -1]. This representation is pretty helpful during
        sharding propagation where we could know exactly each
        tensor dimension is sharded or not.

        Note that if placements contains `_Partial`, we have to
        explicitly deal with it, so that when we create a DTensorSpec
        with dim_map, we could properly record the pending sums.
        """
        # dims mapping of dist tensor sharding
        # return size of tensor ndim, -1 represent replicate
        # and int >=0 represent shard on that device mesh dim
        r = [-1] * self.ndim
        for i, placement in enumerate(self.placements):
            if placement.is_shard():
                shard_dim = cast(Shard, placement).dim
                if r[shard_dim] > -1:
                    raise ValueError(
                        f"Tensor dim {shard_dim} is already sharded on mesh dim {r[shard_dim]},"
                        " DTensor operator implementation does not support things like hybrid"
                        " sharding strategies yet (i.e. [Shard(0), Shard(0)])"
                    )
                r[shard_dim] = i
        return r

    @property
    def num_shards_map(self) -> list[int]:
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. Unlike `dim_map`, `num_shards_map`
        denotes how many shards each tensor dim has. Like `dim_map`:
            len(num_shards_map) == dist_tensor.ndim
            num_shards_map[i] = 1: means tensor dim i is not sharded
            num_shards_map[i] = j: means tensor dim i has j shards in total

        For example, we have a dist tensor of shape [18, 20, 30],
        a device_mesh ([[0, 1, 2, 3], [4, 5, 6, 7]]), and placements
        ([Shard(1), Shard(0)]), the num_shards_map of this distributed tensor
        would be: [4, 2, 1].
        """
        r = [1] * self.ndim
        for i, placement in enumerate(self.placements):
            if placement.is_shard():
                shard_dim = cast(Shard, placement).dim
                r[shard_dim] *= self.mesh.size(i)

        return r

    @property
    def sums(self) -> list[int]:
        """
        sums is a property we derive from `placements` of the
        distributed tensor. It simply return a list of ints where
        sums[i] denotes the pending sum (partial) on mesh dim i
        """
        return [
            idx
            for idx, placement in enumerate(self.placements)
            if placement.is_partial()
        ]

    @classmethod
    def from_dim_map(
        cls,
        mesh: DeviceMesh,
        dim_map: list[int],
        sums: list[int],
        tensor_meta: Optional[TensorMeta] = None,
    ) -> "DTensorSpec":
        """
        Construct a DTensorSpec from dim_map list and pending sum.

        Args:
            mesh (class:`DeviceMesh`): device mesh to be used in the DTensorSpec
            dim_map (List[int]): a list of integer that represents sharding on each
                tensor dimension, see `dim_map` property doc for details
            sums (List[int]): a list of integer that represents the dist tensor have
                pending sum on which device mesh dimension.
            tensor meta (TensorMeta): DTensor metadata

        Return:
            a class:`DTensorSpec` object
        """
        # by default replicate on device mesh dims
        placements: list[Placement] = [Replicate() for _ in range(mesh.ndim)]

        # find all mesh dims that need pending reductions
        for s in sums:
            placements[s] = Partial()

        for i, m in enumerate(dim_map):
            if m >= 0:
                placement = placements[m]
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    raise RuntimeError(
                        f"DeviceMesh dimension can't be mapped to two dimension of the same tensor: {i} and {placement.dim}"
                    )
                elif placement.is_partial():
                    raise RuntimeError(
                        f"DeviceMesh dimension {m} cannot be both shard and partial!"
                    )
                placements[m] = Shard(i)

        return cls(mesh, tuple(placements), tensor_meta=tensor_meta)

    def is_replicated(self) -> bool:
        """
        return True if the current DTensorSpec replicates on all mesh dims (devices)
        """
        return all(placement.is_replicate() for placement in self.placements)

    def is_sharded(self) -> bool:
        """
        return True if the current DTensorSpec is sharded on any mesh dims (devices)
        """
        return any(placement.is_shard() for placement in self.placements)

    def shallow_copy_with_tensor_meta(
        self, tensor_meta: Optional[TensorMeta]
    ) -> "DTensorSpec":
        """
        Shallow copy the DTensorSpec with a new tensor_meta.
        """
        assert tensor_meta is not None, "shallow copy with no tensor_meta!"
        return DTensorSpec(
            self.mesh,
            self.placements,
            tensor_meta=tensor_meta,
        )
