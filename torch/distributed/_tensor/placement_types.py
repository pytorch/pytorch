# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor._collective_utils import (
    fill_empty_tensor_to_shards,
    mesh_broadcast,
    mesh_scatter,
    pad_tensor,
    shard_dim_alltoall,
    unpad_tensor,
)
from torch.distributed.device_mesh import DeviceMesh


class Placement:
    # base class Placement type

    # convenient utils to check for placement types
    def is_shard(self, dim: Optional[int] = None) -> bool:
        is_shard_instance = isinstance(self, Shard)
        if dim is not None and is_shard_instance:
            return cast(Shard, self).dim == dim
        else:
            return is_shard_instance

    def is_replicate(self) -> bool:
        return isinstance(self, Replicate)

    def is_partial(self) -> bool:
        return isinstance(self, Partial)


@dataclass(frozen=True)
class Shard(Placement):
    """
    The ``Shard(dim)`` placement describes the DTensor sharding on tensor dimension
    ``dim`` over a corresponding ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension only holds a shard/piece of the global Tensor. The
    ``Shard(dim)`` placement follows the ``torch.chunk(dim)`` semantic, where the
    last few shards on the DeviceMesh dimension might be empty. The ``Shard``
    placement can be used by all DTensor APIs (i.e. distribute_tensor, from_local, etc.)

    Args:
        dim (int): The tensor dimension that describes the DTensor is sharded over its
            corresponding DeviceMesh dimension.

    ::note:: sharding on a tensor dimension that is not evenly
        divisible on a DeviceMesh dimension is currently experimental.
    """

    dim: int

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        This function uses torch.chunk to split a tensor into num_chunks shards along
        the Shard placement dimension, and return a list of shards with their pad sizes.

        Keyword args:
            with_padding (bool, optional): when True, we pad the tensor on the last
            few ranks before calling the collectives (i.e. scatter/all_gather, etc.).
            This is because collectives usually require equal size tensor inputs
        """
        assert (
            self.dim <= tensor.ndim
        ), f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"

        # chunk tensor over dimension `dim` into n slices
        tensor_list = list(torch.chunk(tensor, num_chunks, dim=self.dim))
        num_empty_tensors = num_chunks - len(tensor_list)

        # if no need to have padding or tensor dim size is evenly sharded already
        # we can return early.
        if not with_padding or tensor.size(self.dim) % num_chunks == 0:
            if contiguous:
                tensor_list = [t.contiguous() for t in tensor_list]
            return (
                fill_empty_tensor_to_shards(tensor_list, self.dim, num_empty_tensors),
                [],
            )

        # compute the chunk size inline with ``torch.chunk`` to calculate padding
        full_chunk_size = (tensor.size(self.dim) + num_chunks - 1) // num_chunks

        # Compute chunk size for each chunk for ``self.dim``
        chunk_sizes = [
            tensor_list[idx].size(self.dim) if idx < len(tensor_list) else 0
            for idx in range(num_chunks)
        ]
        # Compute pad size on each chunk
        pad_sizes = [full_chunk_size - chunk_size for chunk_size in chunk_sizes]

        # Reuse tensor to fill empty chunk with empty tensor
        tensor_list = fill_empty_tensor_to_shards(
            tensor_list, self.dim, num_empty_tensors
        )
        shard_list = []
        for shard, pad_size in zip(tensor_list, pad_sizes):
            # Fill the empty tensor with zeroes with padding.
            if with_padding and pad_size > 0:
                shard = pad_tensor(shard, self.dim, pad_size)
            shard = shard.contiguous() if contiguous else shard
            shard_list.append(shard)
        return shard_list, pad_sizes

    @staticmethod
    def _local_shard_size_on_dim(
        size_on_dim: int,
        num_chunks: int,
        rank: int,
        return_offset: bool = False,
    ) -> Tuple[int, int]:
        """
        returns the local shard size and offset on a given tensor dim
        """
        # Compute the chunk size inline with ``torch.chunk``
        if size_on_dim % num_chunks == 0:
            full_chunk_size = size_on_dim // num_chunks
            return full_chunk_size, full_chunk_size * rank if return_offset else -1

        # uneven sharding case
        full_chunk_size = (size_on_dim + num_chunks - 1) // num_chunks
        shard_starting_idx = full_chunk_size * rank

        if size_on_dim < shard_starting_idx:
            return 0, size_on_dim if return_offset else -1
        else:
            local_shard_size = (
                min(size_on_dim, shard_starting_idx + full_chunk_size)
                - shard_starting_idx
            )
            return local_shard_size, shard_starting_idx if return_offset else -1

    def _shard_tensor(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        """
        shard and scatter a tensor on a mesh dimension (use coordinate
        0 on the mesh dimension as source of truth)
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        scatter_list, pad_sizes = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )

        mesh_dim_local_rank = my_coordinate[mesh_dim]
        output = torch.empty_like(scatter_list[mesh_dim_local_rank])
        mesh_scatter(output, scatter_list, mesh, mesh_dim=mesh_dim)

        # Only unpad if the local_tensor was padded on the dimension.
        if pad_sizes and pad_sizes[mesh_dim_local_rank] > 0:
            output = unpad_tensor(output, self.dim, pad_sizes[mesh_dim_local_rank])
        return output

    def _reduce_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        reduce_op: str,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        reduce and scatter a tensor on a mesh dimension
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return tensor

        is_padded = tensor.size(self.dim) % num_chunks != 0
        if is_padded:
            scattered_list, pad_sizes = self._split_tensor(
                tensor, num_chunks, with_padding=True, contiguous=True
            )
            tensor = torch.cat(scattered_list, dim=self.dim)
        elif not tensor.is_contiguous():
            tensor = tensor.contiguous()

        output = funcol.reduce_scatter_tensor(
            tensor, reduce_op, scatter_dim=self.dim, group=(mesh, mesh_dim)
        )

        if is_padded:
            output = unpad_tensor(output, self.dim, pad_sizes[my_coordinate[mesh_dim]])  # type: ignore[possibly-undefined]
        return output

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: List[int],
    ) -> torch.Tensor:
        """
        This function all_gather all shards and return a tensor that
        is replicated on the previously sharded mesh dimension
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        # check if it's uneven, so we need to pad input tensor before all_gather
        local_shape = list(local_tensor.size())

        logical_dim_size = current_logical_shape[self.dim]
        is_padded = logical_dim_size % num_chunks != 0

        if is_padded:
            full_chunk_size = (logical_dim_size + num_chunks - 1) // num_chunks
            pad_size = full_chunk_size - local_shape[self.dim]
            local_tensor = pad_tensor(local_tensor, self.dim, pad_size)

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        result = funcol.all_gather_tensor(
            local_tensor,
            gather_dim=self.dim,
            group=(mesh, mesh_dim),
        )
        if is_padded:
            unpad_size = full_chunk_size * num_chunks - logical_dim_size  # type: ignore[possibly-undefined]
            result = unpad_tensor(result, self.dim, unpad_size)
        return result

    def _replicate_to_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: int,
    ) -> torch.Tensor:
        """
        transform from replicated tensor to a sharded tensor on
        the current rank, which would perform a local chunk
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        shards, _ = self._split_tensor(
            local_tensor,
            num_chunks,
            with_padding=False,
            contiguous=False,
        )
        return shards[shard_index].clone()

    def _to_new_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: List[int],
        new_shard_dim: int,
    ) -> torch.Tensor:
        """
        transform from existing sharded tensor to a new sharded tensor on
        that shard on a new dimension, which performs an alltoall
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return local_tensor

        num_chunks = mesh.size(mesh_dim=mesh_dim)

        old_dim_logical_size = current_logical_shape[self.dim]
        new_dim_logical_size = current_logical_shape[new_shard_dim]
        old_dim_padding = old_dim_logical_size % num_chunks != 0
        new_dim_padding = new_dim_logical_size % num_chunks != 0
        if old_dim_padding:
            old_dim_full_chunk_size = (
                old_dim_logical_size + num_chunks - 1
            ) // num_chunks
            old_dim_pad_size = old_dim_full_chunk_size - local_tensor.size(self.dim)
            local_tensor = pad_tensor(local_tensor, self.dim, old_dim_pad_size)
        if new_dim_padding:
            new_dim_full_chunk_size = (
                new_dim_logical_size + num_chunks - 1
            ) // num_chunks
            new_dim_pad_size = new_dim_full_chunk_size * num_chunks - local_tensor.size(
                new_shard_dim
            )
            local_tensor = pad_tensor(local_tensor, new_shard_dim, new_dim_pad_size)

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        new_tensor = shard_dim_alltoall(
            local_tensor, self.dim, new_shard_dim, mesh, mesh_dim
        )

        if old_dim_padding:
            old_dim_unpad_size = (
                old_dim_full_chunk_size * num_chunks - current_logical_shape[self.dim]  # type: ignore[possibly-undefined]
            )
            new_tensor = unpad_tensor(new_tensor, self.dim, old_dim_unpad_size)  # type: ignore[possibly-undefined]

        if new_dim_padding:
            local_shard_size_on_new_dim = self._local_shard_size_on_dim(
                new_dim_logical_size, num_chunks, my_coordinate[mesh_dim]
            )[0]
            new_dim_unpad_size = new_dim_full_chunk_size - local_shard_size_on_new_dim  # type: ignore[possibly-undefined]
            new_tensor = unpad_tensor(new_tensor, new_shard_dim, new_dim_unpad_size)  # type: ignore[possibly-undefined]

        return new_tensor

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Shard):
            return False
        return self.dim == other.dim

    def __hash__(self) -> int:
        return hash(self.dim)

    def __repr__(self) -> str:
        """
        machine readable representation of the Shard placement
        """
        return f"Shard(dim={self.dim})"

    def __str__(self) -> str:
        """human readable representation of the Shard placement"""
        return f"S({self.dim})"


@dataclass(frozen=True)
class Replicate(Placement):
    """
    The ``Replicate()`` placement describes the DTensor replicating on a corresponding
    ``DeviceMesh`` dimension, where each rank on the DeviceMesh dimension holds a
    replica of the global Tensor. The ``Replicate`` placement can be used by all
    DTensor APIs (i.e. distribute_tensor, from_local, etc.)
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Replicate):
            return False
        return True

    def __hash__(self) -> int:
        # every replicate placement is the same
        return -1

    def __repr__(self) -> str:
        """
        machine readable representation of the Replicate placement
        """
        return "Replicate()"

    def __str__(self) -> str:
        """
        human readable representation of the Replicate placement
        """
        return "R"

    def _replicate_tensor(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        """
        Replicate (broadcast) a torch.Tensor on a mesh dimension (use
        the first coordinate on the mesh dimension as source of truth)
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        tensor = tensor.contiguous()
        mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim)
        return tensor


@dataclass(frozen=True)
class Partial(Placement):
    """
    The ``Partial(reduce_op)`` placement describes the DTensor that is pending
    reduction on a specified ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension holds the partial value of the global Tensor. User can
    redistribute the ``Partial`` DTensor to a ``Replicate`` or ``Shard(dim)``
    placement on the specified ``DeviceMesh`` dimension using ``redistribute``,
    which would trigger necessary communication operations under the hood (i.e.
    ``allreduce``, ``reduce_scatter``).

    Args:
        reduce_op (str, optional): The reduction op to be used for the partial DTensor
        to produce Replicated/Sharded DTensor. Only element-wise reduction operations
        are supportd, including: "sum", "avg", "prod", "max", "min", default: "sum".

    ::note:: The ``Partial`` placement can be generated as a result of the DTensor operators,
        and can only be used by the ``DTensor.from_local`` API.
    """

    reduce_op: str = "sum"

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # Partial placement contract #1:
        # _reduce_value: reduce the value of the tensor on the mesh dimension
        return funcol.all_reduce(
            tensor, reduceOp=self.reduce_op, group=(mesh, mesh_dim)
        )

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # Partial placement contract #2:
        # _reduce_shard_value: reduce_scatter the value of the tensor over the mesh dimension
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # Partial placement contract #3:
        # _partition_value: partition the value of a replicated tensor on the mesh dimension

        # _partition_value is the conjugate operation of _reduce_value
        # - i.e. _partition_value on a sum reduce op is just a divison operation
        # - the _reduce_value on a sum reduce op would just be a sum(allreduce) operation
        # TODO: if the reduce_op is min/max, etc. the _partition_value should be a
        # different operation
        assert self.reduce_op == "sum", "only support replicate to PartialSUM for now!"
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        return tensor / num_chunks

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partial):
            return False
        return self.reduce_op == other.reduce_op

    def __hash__(self) -> int:
        return 1 + hash(self.reduce_op)

    def __repr__(self) -> str:
        """
        machine readable representation of the Partial placement
        """
        return f"Partial({self.reduce_op})"

    def __str__(self) -> str:
        """
        human readable representation of the Partial placement
        """
        return "P"


# We keep the old _Partial name for a while for BC reason
_Partial = Partial


class TensorMeta(NamedTuple):
    # simple named tuple to represent tensor metadata
    # intentionally to stay simple only for sharding
    # propagation purposes.
    shape: torch.Size
    stride: Tuple[int, ...]
    dtype: torch.dtype


# used internally to propagate the placements
@dataclass
class DTensorSpec:
    mesh: DeviceMesh
    placements: Tuple[Placement, ...]

    # tensor meta will only be set during sharding propagation
    tensor_meta: Optional[TensorMeta] = None

    def __post_init__(self):
        if not isinstance(self.placements, tuple):
            self.placements = tuple(self.placements)
        self._hash: Optional[int] = None

    def __setattr__(self, attr: str, value: Any):
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

    def __eq__(self, __o: object) -> bool:
        if not (
            isinstance(__o, DTensorSpec)
            and self.mesh == __o.mesh
            and self.placements == __o.placements
        ):
            return False
        if self.tensor_meta is None or __o.tensor_meta is None:
            return self.tensor_meta == __o.tensor_meta

        return (
            self.tensor_meta.shape == __o.tensor_meta.shape  # type: ignore[union-attr]
            and self.tensor_meta.stride == __o.tensor_meta.stride  # type: ignore[union-attr]
            and self.tensor_meta.dtype == __o.tensor_meta.dtype  # type: ignore[union-attr]
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
    def stride(self) -> Tuple[int, ...]:
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
    def dim_map(self) -> List[int]:
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
    def sums(self) -> List[int]:
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
        dim_map: List[int],
        sums: List[int],
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
        placements: List[Placement] = [Replicate() for _ in range(mesh.ndim)]

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

    def is_replicated(self):
        """
        return True if the current DTensorSpec replicates on all mesh dims (devices)
        """
        return all(placement.is_replicate() for placement in self.placements)

    def is_sharded(self):
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
