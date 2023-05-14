# Copyright (c) Meta Platforms, Inc. and affiliates

from dataclasses import dataclass
from typing import cast, List, Optional, Sequence, Tuple

import torch
import torch.distributed.distributed_c10d as c10d

from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.fx.passes.shape_prop import TensorMetadata


class Placement:
    # base class Placement type

    # convenient utils to check for placement types
    def is_shard(self, dim: Optional[int] = None) -> bool:
        if dim is not None and isinstance(self, Shard):
            return self.dim == dim
        else:
            return isinstance(self, Shard)

    def is_replicate(self) -> bool:
        return isinstance(self, Replicate)

    def is_partial(self) -> bool:
        return isinstance(self, _Partial)


class Shard(Placement):
    # shard placement, shard on a dim
    def __init__(self, dim):
        self.dim = dim

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        # NOTE: For with_padding option, we pad the tensor on each rank before calling
        # the collectives (i.e. scatter/all_gather, etc.). This is because for gloo
        # backend, it does not support uneven collectives, nccl supports some, but
        # it might be slow compared to even size collective, we need to pad tensor
        # before really calling the collective, and unpad/narrow it afterwards
        # TODO: consider if we should remove this logic once ProcessGroupGloo
        # support uneven list, and collective performance on par
        assert (
            self.dim <= tensor.ndim
        ), f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"
        assert (
            tensor.size(self.dim) > 0
        ), f"Tensor size along dim{self.dim} is 0. There is nothing to be sharded."

        # chunk tensor over dimension `dim` into n slices with padding if necessary
        tensor_list = list(torch.chunk(tensor, num_chunks, dim=self.dim))
        # compute the chunk size inline with ``torch.chunk``
        full_chunk_size = (tensor.size(self.dim) + num_chunks - 1) // num_chunks

        # Compute chunk size for each chunk for ``self.dim``
        chunk_sizes = [
            tensor_list[idx].size(self.dim) if idx < len(tensor_list) else 0
            for idx in range(num_chunks)
        ]
        # Compute pad size on each chunk
        pad_sizes = [full_chunk_size - chunk_size for chunk_size in chunk_sizes]

        # Reuse tensor to fill empty chunk with empty tensor
        num_empty_tensors = num_chunks - len(tensor_list)
        tensor_size = list(tensor_list[0].size())
        tensor_size = [
            size if idx != self.dim else 0 for idx, size in enumerate(tensor_size)
        ]
        tensor = tensor.new_zeros(tensor_size)
        for _ in range(num_empty_tensors):
            tensor_list.append(tensor)

        if with_padding or contiguous:
            shard_list = []
            for shard, pad_size in zip(tensor_list, pad_sizes):
                # Fill the empty tensor with zeroes with padding.
                if with_padding and pad_size > 0:
                    shard = self._pad_tensor(shard, pad_size)
                shard = shard.contiguous() if contiguous else shard
                shard_list.append(shard)
            return shard_list, pad_sizes
        else:
            return tensor_list, pad_sizes

    def _pad_tensor(
        self,
        tensor: torch.Tensor,
        pad_size: int,
    ) -> torch.Tensor:
        pad = [0, 0] * (tensor.ndim - self.dim)
        pad[-1] = pad_size
        return torch.nn.functional.pad(tensor, pad)

    def _unpad_tensor(
        self,
        tensor: torch.Tensor,
        pad_size: int,
    ) -> torch.Tensor:
        return tensor.narrow(
            self.dim,
            start=0,
            length=tensor.size(self.dim) - pad_size,
        )

    def _local_shard_size_on_dim(
        self,
        size_on_dim: int,
        num_chunks: int,
        rank: int,
        return_offset: bool = False,
    ) -> Tuple[int, int]:
        """
        returns the local shard size and offset on a given tensor dim
        """
        assert (
            size_on_dim >= num_chunks
        ), f"Size to be sharded on dim {self.dim} must be at least as large as the number of devices in that dimension {num_chunks}"

        # Compute the chunk size inline with ``torch.chunk``
        full_chunk_size = (size_on_dim + num_chunks - 1) // num_chunks

        # Compute chunk size for each chunk on the dimension.
        chunk_sizes = [
            max(
                min(size_on_dim, full_chunk_size * (idx + 1)) - full_chunk_size * idx,
                0,
            )
            for idx in range(num_chunks)
        ]
        local_shard_size = chunk_sizes[rank]

        local_offset_on_dim = -1
        if return_offset:
            # Return global tensor dim size of current dimension if for empty shard
            # to represent the end of the corresponding tensor dim.
            local_offset_on_dim = sum(chunk_sizes[:rank])

        return (local_shard_size, local_offset_on_dim)

    def _shard_tensor(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        """
        shard and scatter a tensor on a mesh dimension (use coordinate
        0 on the mesh dimension as source of truth)
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        scatter_list, pad_sizes = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )

        output = torch.empty_like(scatter_list[my_coordinate[mesh_dim]])
        mesh.scatter(output, scatter_list, mesh_dim=mesh_dim)

        # Only unpad if the local_tensor was padded on the dimension.
        pad_size = pad_sizes[my_coordinate[mesh_dim]]
        if pad_size > 0:
            output = self._unpad_tensor(output, pad_size)
        return output

    def _reduce_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        reduce_op: c10d.ReduceOp,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        reduce and scatter a tensor on a mesh dimension
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(dim=mesh_dim)

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

        output = mesh.reduce_scatter(
            tensor, op=reduce_op, mesh_dim=mesh_dim, scatter_dim=self.dim
        )

        if is_padded:
            output = self._unpad_tensor(output, pad_sizes[my_coordinate[mesh_dim]])
        return output

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        size: torch.Size,
        mesh: DeviceMesh,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        This function all_gather all shards and return a tensor that
        is replicated on the previously sharded mesh dimension
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return local_tensor

        # check if it needs to pad input tensor before all_gather
        full_chunk_size = (size[self.dim] + num_chunks - 1) // num_chunks
        chunk_sizes = [
            max(
                min(size[self.dim], full_chunk_size * (idx + 1))
                - full_chunk_size * idx,
                0,
            )
            for idx in range(num_chunks)
        ]
        pad_sizes = [full_chunk_size - chunk_size for chunk_size in chunk_sizes]
        is_padded = size[self.dim] % num_chunks != 0

        pad_size = pad_sizes[my_coordinate[mesh_dim]]
        if pad_size > 0:
            local_tensor = self._pad_tensor(local_tensor, pad_size)
        local_tensor = local_tensor.contiguous()

        result = mesh.all_gather(
            tensor=local_tensor,
            mesh_dim=mesh_dim,
            gather_dim=self.dim,
        )

        # Unpad the tensor if the input tensor was padded
        if is_padded:
            gathered_list = torch.chunk(result, num_chunks, dim=self.dim)
            gathered_list = [
                self._unpad_tensor(gathered_tensor, pad_size)  # type: ignore[misc]
                if pad_size > 0
                else gathered_tensor
                for gathered_tensor, pad_size in zip(gathered_list, pad_sizes)
            ]

            result = torch.cat(gathered_list, dim=self.dim)
        return result

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


class Replicate(Placement):
    # replicate placement
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
        mesh.broadcast(tensor, mesh_dim=mesh_dim)
        return tensor


class _Partial(Placement):
    # This is a default partial placement with element-wise reduce op
    # when doing reduction it follows the contract of `_to_replicate`
    # and `_to_shard` to do the reduction and convert the local tensor
    # to the corresponding state (replicate or shard)
    #
    # We can implement custom reductions as needed by subclassing this
    # class and override those contracts.

    def __init__(self, reduce_op: c10d.ReduceOp = c10d.ReduceOp.SUM):  # type: ignore[assignment]
        self.reduce_op: c10d.ReduceOp = reduce_op

    def _to_replicate(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        return mesh.all_reduce(
            tensor, self.reduce_op, mesh_dim=mesh_dim  # type: ignore[call-arg]
        )

    def _to_shard(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # by default call reduce_shard_tensor of the shard_spec.
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(
            tensor, mesh, self.reduce_op, mesh_dim  # type: ignore[call-arg]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Partial):
            return False
        return self.reduce_op == other.reduce_op

    def __hash__(self) -> int:
        return hash(self.reduce_op)

    def __repr__(self) -> str:
        """
        machine readable representation of the Partial placement
        """
        return f"_Partial(reduce_op={self.reduce_op})"

    def __str__(self) -> str:
        """
        human readable representation of the Partial placement
        """
        return "P"


# used internally to propagate the placements
@dataclass
class DTensorSpec:
    mesh: DeviceMesh
    placements: Sequence[Placement]

    # tensor meta will only be set during sharding propagation
    tensor_meta: Optional[TensorMetadata] = None

    def __hash__(self) -> int:
        # hashing and equality check for DTensorSpec are used to cache the sharding
        # propagation results. We only need to consider the mesh, placements and shape
        # Caveat: we need to keep this in mind and sync hash and eq if we add more
        # fields to them,
        if self.tensor_meta is not None:
            return hash((self.mesh, tuple(self.placements), self.tensor_meta.shape))
        else:
            return hash((self.mesh, tuple(self.placements)))

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, DTensorSpec)
            and self.mesh == __o.mesh
            and self.placements == __o.placements
            and self.tensor_meta == __o.tensor_meta
        )

    @property
    def shape(self) -> torch.Size:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.shape

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
        tensor_meta: Optional[TensorMetadata] = None,
    ) -> "DTensorSpec":
        """
        Construct a DTensorSpec from dim_map list and pending sum.

        Args:
            mesh (class:`DeviceMesh`): device mesh to be used in the DTensorSpec
            dim_map (List[int]): a list of integer that represents sharding on each
                tensor dimension, see `dim_map` property doc for details
            sums (List[int]): a list of integer that represents the dist tensor have
                pending sum on which device mesh dimension.
            tensor meta (TensorMetadata): DTensor metadata

        Return:
            a class:`DTensorSpec` object
        """
        # by default replicate on device mesh dims
        placements: List[Placement] = [Replicate() for _ in range(mesh.ndim)]

        # find all mesh dims that need pending reductions
        for s in sums:
            placements[s] = _Partial()

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

        return cls(mesh, placements, tensor_meta=tensor_meta)
