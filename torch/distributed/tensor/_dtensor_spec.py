from dataclasses import dataclass
from typing import Any, cast, NamedTuple, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


class TensorMeta(NamedTuple):
    # simple named tuple to represent tensor metadata
    # intentionally to stay simple only for sharding
    # propagation purposes.
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype


# used internally to propagate the placements
@dataclass
class DTensorSpec:
    mesh: DeviceMesh
    placements: tuple[Placement, ...]

    # tensor meta will only be set during sharding propagation
    tensor_meta: Optional[TensorMeta] = None

    def __post_init__(self) -> None:
        if not isinstance(self.placements, tuple):
            self.placements = tuple(self.placements)
        if not len(self.placements) == self.mesh.ndim:
            raise ValueError(
                f"DTensorSpec requires one placement per mesh dim (mesh.ndim={self.mesh.ndim}), got {self.placements=}"
            )
        self._hash: Optional[int] = None

    def __setattr__(self, attr: str, value: Any) -> None:
        super().__setattr__(attr, value)
        # Make sure to recompute the hash in case any of the hashed attributes
        # change (though we do not expect `mesh` or `placements` to change)
        if hasattr(self, "_hash") and attr in ("mesh", "placements", "tensor_meta"):
            self._hash = None

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

    def __eq__(self, other: object, /) -> bool:
        if not (
            isinstance(other, DTensorSpec)
            and self.mesh == other.mesh
            and self.placements == other.placements
        ):
            return False
        if self.tensor_meta is None or other.tensor_meta is None:
            return self.tensor_meta == other.tensor_meta

        return (
            self.tensor_meta.shape == other.tensor_meta.shape  # type: ignore[union-attr]
            and self.tensor_meta.stride == other.tensor_meta.stride  # type: ignore[union-attr]
            and self.tensor_meta.dtype == other.tensor_meta.dtype  # type: ignore[union-attr]
        )

    def __str__(self) -> str:
        """
        human readable representation of the DTensorSpec
        """
        if len(self.placements) == 1:
            placement_str = str(self.placements[0])
        else:
            placement_str = str(self.placements)

        if self.tensor_meta is not None:
            tensor_shape = str(tuple(self.tensor_meta.shape))
        else:
            tensor_shape = "unknown shape"

        return f"Spec({placement_str} on {tensor_shape})"

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
                        f"DeviceMesh dimension cann't be mapped to two dimension of the same tensor: {i} and {placement.dim}"
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
