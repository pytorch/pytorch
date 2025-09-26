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
    # Tensor meta will only be set during sharding propagation
    tensor_meta: Optional[TensorMeta] = None
    # When a tensor dimension is sharded across multiple mesh axes,
    # `shard_order` specifies the sequence in which these shardings are applied,
    # which in turn determines the placement of tensor shards on devices.
    # `len(shard_order)` is equal to tensor dimensions and `shard_order[i]` is a
    # tuple of mesh axis indices indicating the order in which sharding is
    # applied to the tensor dimensions `i`. Note that since `tensor_meta` can be
    # None, we are unable to tell the rank of the tensor. Therefore, the size of
    # `shard_order` is only extended to the largest mesh axis index appeared in
    # `placements` if we let __post_init__ to help fill the `shard_order`.
    shard_order: Optional[tuple[tuple[int, ...], ...]] = None

    @staticmethod
    def compute_default_shard_order(
        placements: tuple[Placement, ...],
        mesh: DeviceMesh,
        tensor_rank: Optional[int] = None,
    ) -> tuple[tuple[int, ...], ...]:
        # follow default left-to-right device order if shard_order is not specified
        tensor_dim_to_mesh_dims: list[list[int]] = [[]]
        for mesh_dim in range(0, mesh.ndim):
            if isinstance(placements[mesh_dim], Shard):
                placement = cast(Shard, placements[mesh_dim])
                shard_dim = placement.dim
                assert shard_dim >= 0, (
                    f"Shard dim {shard_dim} in placements {placements} must be normalized"
                )
                # Extend tensor_dim_to_mesh_dims to have at least (shard_dim + 1) elements
                while len(tensor_dim_to_mesh_dims) <= shard_dim:
                    tensor_dim_to_mesh_dims.append([])
                tensor_dim_to_mesh_dims[shard_dim].append(mesh_dim)
        if tensor_rank:
            while len(tensor_dim_to_mesh_dims) < tensor_rank:
                tensor_dim_to_mesh_dims.append([])
        default_shard_order = tuple(
            tuple(mesh_dims) for mesh_dims in tensor_dim_to_mesh_dims
        )
        return default_shard_order

    def __post_init__(self) -> None:
        if not isinstance(self.placements, tuple):
            self.placements = tuple(self.placements)

        if self.shard_order is None:
            tensor_rank = len(self.tensor_meta.shape) if self.tensor_meta else None
            self.shard_order = DTensorSpec.compute_default_shard_order(
                self.placements, self.mesh, tensor_rank
            )

        self._hash: Optional[int] = None

    def __setattr__(self, attr: str, value: Any) -> None:
        super().__setattr__(attr, value)
        # Make sure to recompute the hash in case any of the hashed attributes
        # change (though we do not expect `mesh` or `placements` to change)
        if hasattr(self, "_hash") and attr in (
            "mesh",
            "placements",
            "tensor_meta",
            "shard_order",
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
            assert isinstance(value, (TensorMeta, TensorMetadata)), value

    def _hash_impl(self) -> int:
        # hashing and equality check for DTensorSpec are used to cache the sharding
        # propagation results. We only need to consider the mesh, placements, shape
        # dtype and stride.
        # Caveat: we need to keep this in mind and sync hash and eq if we add more
        # fields to them.
        hash_items = [self.mesh, self.placements, self.shard_order]
        if self.tensor_meta is not None:
            hash_items.extend(
                [
                    self.tensor_meta.shape,
                    self.tensor_meta.stride,
                    self.tensor_meta.dtype,
                ]
            )
        return hash(tuple(hash_items))

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
            and self.shard_order == other.shard_order
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
        placement_str = self.format_shard_order_str(self.placements, self.shard_order)
        if self.tensor_meta is not None:
            tensor_shape = str(tuple(self.tensor_meta.shape))
        else:
            tensor_shape = "unknown shape"

        return f"Spec({placement_str} on {tensor_shape})"

    @staticmethod
    def format_shard_order_str(
        placements: tuple[Placement, ...],
        shard_order: Optional[tuple[tuple[int, ...], ...]] = None,
        use_jax_style_print: bool = False,
    ) -> str:
        out_str = ""
        # jax-style sharding representation: map from tensor dim to mesh dim
        if shard_order and use_jax_style_print:
            for tensor_dim, mesh_dims in enumerate(shard_order):
                if len(mesh_dims) > 0:
                    out_str += f"S({tensor_dim})"
                    out_str += f"[{', '.join([str(m) for m in mesh_dims])}]"
            # in addition, add the partial placement
            partial_to_mesh_dim: dict[Partial, list[int]] = {}
            for mesh_dim, p in enumerate(placements):
                if isinstance(p, Partial):
                    if p not in partial_to_mesh_dim:
                        partial_to_mesh_dim[p] = []
                    partial_to_mesh_dim[p].append(mesh_dim)
            for p, mesh_dims in partial_to_mesh_dim.items():
                out_str += f"P({p.reduce_op})"
                out_str += f"[{', '.join([str(m) for m in mesh_dims])}]"
        else:
            # native dtensor-style sharding representation: map from mesh
            # dim to tensor dim
            for mesh_dim, placement in enumerate(placements):
                if isinstance(placement, Replicate):
                    out_str += "R"
                elif isinstance(placement, Shard):
                    if shard_order is not None:
                        assert mesh_dim in shard_order[placement.dim]
                        out_str += f"S({placement.dim})[{shard_order[placement.dim].index(mesh_dim)}]"
                    else:
                        out_str += f"S({placement.dim})"
                else:
                    assert isinstance(placement, Partial)
                    out_str += f"P({placement.reduce_op})"
        return out_str

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
