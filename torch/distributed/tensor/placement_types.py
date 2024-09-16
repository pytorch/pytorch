# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from dataclasses import dataclass
from typing import cast, List, Optional, Tuple

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._collective_utils import (
    fill_empty_tensor_to_shards,
    mesh_broadcast,
    mesh_scatter,
    pad_tensor,
    shard_dim_alltoall,
    unpad_tensor,
)


__all__ = ["Placement", "Shard", "Replicate", "Partial"]


class Placement:
    """
    The base class for the Placement type, where it describes how a DTensor is placed onto the
    ``DeviceMesh``. ``Placement`` and ``DeviceMesh`` together could describe the DTensor Layout.
    It is the base class of the three main DTensor Placement types: ``Shard``, ``Replicate``,
    and ``Partial``.

    This class is not meant to be used directly, mainly served as a typing stub.
    """

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
    last few shards on the DeviceMesh dimension might be empty when the tensor dimension
    is not evenly divisble on the DeviceMesh dimension. The ``Shard`` placement can be
    used by all DTensor APIs (i.e. distribute_tensor, from_local, etc.)

    Args:
        dim (int): The tensor dimension that describes the DTensor is sharded over its
            corresponding DeviceMesh dimension.

    .. warning:: sharding on a tensor dimension where the tensor dimension size is not
        evenly divisible on a DeviceMesh dimension is currently experimental and subject to change.
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


# kw_only is only available in python >= 3.10
kw_only_dataclass = dict(kw_only=True) if "kw_only" in dataclass.__kwdefaults__ else {}


@dataclass(frozen=True, **kw_only_dataclass)
class _StridedShard(Shard):
    """
    _StridedShard is only introduced to support 2D FSDP2 + TP sharding where the tensor
    is sharded on the TP mesh dimension first, then sharded on the FSDP mesh dimension.
    We call this right-to-left sharding which is the opposite of the default
    left-to-right sharding. See the example below:
        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [Shard(0), Shard(0)]

    The default sharding behavior shards the tensor on "dp" mesh dimension first then
    "tp" dimension. The sharding result will be:
        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   1 (row 2-3)
        2       |   (1, 0)          |   2 (row 4-5)
        3       |   (1, 1)          |   3 (row 6-7)

    While the FSDP2 + TP sharding behavior does the opposite: it shards the tensor on
    "tp" mesh dim first then "dp" dim. This right-to-left sharding will produce the
    result:
        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   2 (row 4-5)
        2       |   (1, 0)          |   1 (row 2-3)
        3       |   (1, 1)          |   3 (row 6-7)

    The consequence is, any attempt to redistribute this DTensor to a full replica will
    produce a wrong result because the shard-to-replicate redistribution always happens
    right-to-left, regardless it's left-to-right sharding or right-to-left. To address
    this, we use _StridedShard placement to make this right-to-left sharding compatible
    with our left-to-right convention on both tensor distribution and redistribution.

    Now with _StridedShard, the right-to-left sharding above can be represented as:
        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [_StridedShard(0, split_factor=2), Shard(0)]

    And a left-to-right processing of `placements` will produce the same result, which is
    different from using the `Shard` placement:
        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   2 (row 4-5)
        2       |   (1, 0)          |   1 (row 2-3)
        3       |   (1, 1)          |   3 (row 6-7)

    The argument `split_factor` is the number of existing shards over the tensor sharding
    dimension before processing the _StridedShard placement, as if the sharding happened
    right-to-left. In the example above, the tensor should first be sharded on the "tp"
    dimension into 2 shards before being sharded on the "dp" dimension. Therefore, the
    `split_factor` of the _StridedShard placement on "dp" dim is 2.

    TODO: strided sharding needs to work fine with uneven sharding. Now it forbids
    resharding if the tensor is unevenly sharded.
    TODO: we should remove _StridedShard placement once we can unify it with Shard
    """

    split_factor: int

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _StridedShard):
            return self.dim == other.dim and self.split_factor == other.split_factor
        elif isinstance(other, Shard):
            # TODO: this is to avoid extra all-gather in dtensor op dispatch
            # note that sharding prop would not produce _StridedShard and an
            # placement inequality would introduce an all-gather for resharding
            return self.dim == other.dim
        return False

    def __hash__(self) -> int:
        return hash((self.dim, self.split_factor))

    def __repr__(self) -> str:
        """
        machine readable representation of the _StridedShard placement
        """
        return f"_StridedShard(dim={self.dim}, sf={self.split_factor})"

    def __str__(self) -> str:
        """human readable representation of the _StridedShard placement"""
        return f"_S({self.dim}, {self.split_factor})"

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        TODO: currently _StridedShard does not support padding
        """
        assert (
            self.dim <= tensor.ndim
        ), f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"

        total_split = num_chunks * self.split_factor
        assert tensor.size(self.dim) % total_split == 0, (
            "_StridedShard currently only allows even sharding but got tensor size"
            f" {tensor.size(self.dim)} on dim {self.dim} and total split"
            f" {total_split}={num_chunks} * {self.split_factor}"
        )

        group_size = self.split_factor
        total_split_tensor_list = list(torch.chunk(tensor, total_split, dim=self.dim))
        tensor_list = [
            torch.cat(
                [
                    total_split_tensor_list[i + j * num_chunks]  # stride is num_chunks
                    for j in range(group_size)
                ],
                dim=self.dim,
            )
            for i in range(num_chunks)
        ]

        if contiguous:
            tensor_list = [t.contiguous() for t in tensor_list]

        return tensor_list, []

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: List[int],
    ) -> torch.Tensor:
        """
        Note: currently _StridedShard does not support padding
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        total_split = num_chunks * self.split_factor
        # NOTE: we require Strided Sharding to be even for now
        assert current_logical_shape[self.dim] % total_split == 0, (
            "_StridedShard requires even sharding but got tensor size "
            f"{current_logical_shape[self.dim]} on dim {self.dim} and "
            f"total split {total_split}=num_chunks {num_chunks} "
            f"* split_factor {self.split_factor}"
        )

        result = funcol.all_gather_tensor(
            local_tensor,
            gather_dim=self.dim,
            group=(mesh, mesh_dim),
        )
        if isinstance(result, funcol.AsyncCollectiveTensor):
            result = result.wait()

        tensor_shard_list = torch.chunk(result, total_split, dim=self.dim)
        # rearrange the order
        new_tensor_shard_list = []
        for idx in range(len(tensor_shard_list)):
            # the shard split of index `idx` is assigned a new index within
            # _StridedShard._split_tensor:
            # the original tensor was split into `total_split` chunks,
            # all chunks with the same `idx % num_chunks` are merged into one
            # new shard and placed on mesh's local rank `idx % num_chunks`
            idx_after_split = idx % num_chunks * self.split_factor + idx // num_chunks
            new_tensor_shard_list.append(tensor_shard_list[idx_after_split])

        return torch.cat(new_tensor_shard_list, dim=self.dim).contiguous()


@dataclass(frozen=True)
class Replicate(Placement):
    """
    The ``Replicate()`` placement describes the DTensor replicating on a corresponding
    ``DeviceMesh`` dimension, where each rank on the DeviceMesh dimension holds a
    replica of the global Tensor. The ``Replicate`` placement can be used by all
    DTensor APIs (i.e. ``distribute_tensor``, ``DTensor.from_local``, etc.)
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Replicate)

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
            are supported, including: "sum", "avg", "product", "max", "min", default: "sum".

    .. note:: The ``Partial`` placement can be generated as a result of the DTensor operators,
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
