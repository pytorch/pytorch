# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import typing
from dataclasses import dataclass, field
from typing import cast, TypeVar

import torch
import torch._C
import torch.distributed._functional_collectives as funcol
from torch import sym_min
from torch._C._distributed import Placement
from torch.distributed import RankType
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._collective_utils import (
    fill_empty_tensor_to_shards,
    mesh_broadcast,
    mesh_scatter,
    pad_tensor,
    shard_dim_alltoall,
    unpad_tensor,
)
from torch.distributed.tensor._ops._mask_buffer import MaskBuffer
from torch.distributed.tensor._placement_utils import CollectivePaddingContext


__all__ = ["Placement", "Shard", "Replicate", "Partial"]

_RankTypeT = TypeVar("_RankTypeT", bound=RankType)


# Appease TestPublicBindings.test_correct_module_names
Placement.__module__ = "torch.distributed.tensor.placement_types"


class Shard(torch._C._distributed.Shard):
    """
    The ``Shard(dim)`` placement describes the DTensor sharding on tensor dimension
    ``dim`` over a corresponding ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension only holds a shard/piece of the global Tensor. The
    ``Shard(dim)`` placement follows the ``torch.chunk(dim)`` semantic, where the
    last few shards on the DeviceMesh dimension might be empty when the tensor dimension
    is not evenly divisible on the DeviceMesh dimension. The ``Shard`` placement can be
    used by all DTensor APIs (i.e. distribute_tensor, from_local, etc.)

    Args:
        dim (int): The tensor dimension that describes the DTensor is sharded over its
            corresponding DeviceMesh dimension.

    .. warning:: sharding on a tensor dimension where the tensor dimension size is not
        evenly divisible on a DeviceMesh dimension is currently experimental and subject to change.
    """

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> tuple[list[torch.Tensor], list[int]]:
        """
        This function uses torch.chunk to split a tensor into num_chunks shards along
        the Shard placement dimension, and return a list of shards with their pad sizes.

        Keyword args:
            with_padding (bool, optional): when True, we pad the tensor on the last
            few ranks before calling the collectives (i.e. scatter/all_gather, etc.).
            This is because collectives usually require equal size tensor inputs
        """
        return self._split_tensor_helper(
            tensor, num_chunks, with_padding, contiguous, self.dim
        )

    @staticmethod
    def _split_tensor_helper(
        tensor: torch.Tensor,
        num_chunks: int,
        with_padding: bool,
        contiguous: bool,
        dim: int,
    ) -> tuple[list[torch.Tensor], list[int]]:
        assert dim <= tensor.ndim, (
            f"Sharding dim {dim} greater than tensor ndim {tensor.ndim}"
        )

        # chunk tensor over dimension `dim` into n slices
        tensor_list = list(torch.chunk(tensor, num_chunks, dim=dim))
        tensor_list = fill_empty_tensor_to_shards(
            tensor_list, dim, num_chunks - len(tensor_list)
        )

        # compute the chunk size inline with ``torch.chunk`` to calculate padding
        full_chunk_size = (tensor.size(dim) + num_chunks - 1) // num_chunks

        shard_list: list[torch.Tensor] = []
        pad_sizes: list[int] = []
        for shard in tensor_list:
            if with_padding:
                pad_size = Shard._get_shard_pad_size(full_chunk_size, shard, dim)
                shard = pad_tensor(shard, dim, pad_size)
                pad_sizes.append(pad_size)
            if contiguous:
                shard = shard.contiguous()
            shard_list.append(shard)
        return shard_list, pad_sizes

    @maybe_run_for_local_tensor
    def _select_split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        index: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
        clone: bool = True,
    ) -> torch.Tensor:
        """
        Like _split_tensor() but only returns a single Tensor
        """
        shards, _ = self._split_tensor(
            tensor, num_chunks, with_padding=with_padding, contiguous=False
        )
        result = shards[index]
        if clone:
            result = result.clone()
        elif contiguous:
            result = result.contiguous()
        return result

    @staticmethod
    @maybe_run_for_local_tensor
    def _select_shard(shards: list[torch.Tensor], shard_index) -> torch.Tensor:
        return shards[shard_index].clone()

    @staticmethod
    @maybe_run_for_local_tensor
    def _get_shard_pad_size(
        full_size: int, local_tensor: torch.Tensor, dim: int
    ) -> int:
        """
        Get the padding size of the local tensor on the shard dimension.
        """
        return full_size - local_tensor.size(dim)

    @staticmethod
    @maybe_run_for_local_tensor
    def _maybe_unpad_tensor_with_sizes(
        dim, local_tensor, pad_sizes, mesh_dim_local_rank, make_contiguous
    ) -> torch.Tensor:
        # Only unpad if the local_tensor was padded on the dimension.
        if pad_sizes[mesh_dim_local_rank] > 0:
            local_tensor = unpad_tensor(
                local_tensor, dim, pad_sizes[mesh_dim_local_rank]
            )
            if make_contiguous:
                local_tensor = local_tensor.contiguous()
        return local_tensor

    @staticmethod
    @maybe_run_for_local_tensor
    def local_shard_size_and_offset(
        curr_local_size: int,
        num_chunks: int,
        rank: _RankTypeT,
    ) -> tuple[int, _RankTypeT]:
        """
        Given the size of the current local tensor (which may already be sharded on some dimensions),
        computes the new local shard size and offset given the desired number of chunks
        (num_chunks is generally equal to the size of the current sharding dim).

        Note: new local shard offset is relative to the current sharded tensor, not the global tensor.
        See `_utils.compute_local_shape_and_global_offset` for computing global offset.

        Returns (new local shard size, offset)

        """
        # Compute the chunk size inline with ``torch.chunk``
        if curr_local_size % num_chunks == 0:
            full_chunk_size = curr_local_size // num_chunks
            # pyrefly: ignore[bad-assignment] # pyrefly bug?
            shard_starting_idx: _RankTypeT = full_chunk_size * rank
            return full_chunk_size, shard_starting_idx

        # uneven sharding case
        full_chunk_size = (curr_local_size + num_chunks - 1) // num_chunks
        # pyrefly: ignore[bad-assignment] # pyrefly bug?
        shard_starting_idx: _RankTypeT = full_chunk_size * rank

        if curr_local_size < shard_starting_idx:
            return 0, typing.cast(_RankTypeT, curr_local_size)
        else:
            local_shard_size = (
                sym_min(curr_local_size, shard_starting_idx + full_chunk_size)
                - shard_starting_idx
            )
            return local_shard_size, shard_starting_idx

    def _local_shard_size_and_offset(
        self,
        curr_local_size: int,
        num_chunks: int,
        rank: RankType,
    ) -> tuple[int, RankType]:
        return Shard.local_shard_size_and_offset(curr_local_size, num_chunks, rank)

    def _shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        """
        Shard and scatter a tensor on a mesh dimension (use coordinate 0 on the
        mesh dimension as source of truth).

        Create the local tensor for this rank following the given Shard
        placement. If src_data_rank is None, perform only local splitting.
        Otherwise, additionally scatter data from src_data_rank. Unlike
        ``_split_tensor``, which supports uneven sharding via padding, this
        method requires the tensor dimension to be evenly divisible by the
        number of chunks (mesh dimension size).
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        mesh_dim_local_rank = my_coordinate[mesh_dim]

        if src_data_rank is None:
            # src_data_rank specified as None explicitly means to skip the
            # communications, simply split
            return self._select_split_tensor(
                tensor,
                num_chunks,
                mesh_dim_local_rank,
                with_padding=False,
                contiguous=True,
            )

        scatter_list, pad_sizes = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )

        it = iter(scatter_list)
        first = next(it)
        # Tensors in the scatter list are expected to have the same shape because
        # split is requested with padding.
        assert all(first.shape == v.shape for v in it)

        output = torch.empty_like(first)

        # perform scatter from the src_data_rank as data source when it is not None
        mesh_scatter(
            output, scatter_list, mesh, mesh_dim=mesh_dim, group_src=src_data_rank
        )

        return Shard._maybe_unpad_tensor_with_sizes(
            self.dim, output, pad_sizes, mesh_dim_local_rank, True
        )

    @classmethod
    def _make_shard_tensor(
        cls,
        dim: int,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        shard_placement = cls(dim)
        return shard_placement._shard_tensor(tensor, mesh, mesh_dim, src_data_rank)

    def _reduce_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        reduce_op: str,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        Reduce and scatter a tensor on a mesh dimension.

        This performs a reduce_scatter operation where the input tensor (replicated
        across ranks) is reduced and then scattered along self.dim. The result is
        a sharded tensor where each rank holds a shard of the reduced result.

        Args:
            tensor: The replicated input tensor to reduce and scatter.
            mesh: The device mesh over which the operation is performed.
            reduce_op: The reduction operation (e.g., "sum", "avg").
            mesh_dim: The mesh dimension along which to reduce_scatter.

        Returns:
            The local shard of the reduced tensor.
        """
        if not mesh._is_current_rank_part_of_mesh():
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return tensor

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # For reduce_scatter, the input is replicated and output is sharded.
        # We use pad_new_shard since we're creating a new shard dimension.
        # tensor.size(self.dim) is the logical size since input is replicated.
        return (
            CollectivePaddingContext(mesh, mesh_dim)
            .pad_new_shard(self.dim, tensor.size(self.dim))
            .run(
                tensor,
                lambda t: funcol.reduce_scatter_tensor(
                    t, reduce_op, scatter_dim=self.dim, group=(mesh, mesh_dim)
                ),
            )
        )

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
    ) -> torch.Tensor:
        """
        Transform from Shard to Replicate via all_gather.

        Gathers all shards across ranks on the specified mesh dimension and
        returns a fully replicated tensor. Handles uneven sharding by padding
        before the collective and unpadding after.

        Args:
            local_tensor: The local shard on this rank.
            mesh: The device mesh over which the tensor is distributed.
            mesh_dim: The mesh dimension along which to gather.
            current_logical_shape: The global/logical shape of the full tensor.

        Returns:
            The fully replicated tensor with shape matching current_logical_shape.
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            return local_tensor

        return (
            CollectivePaddingContext(mesh, mesh_dim)
            .pad_old_shard(self.dim, current_logical_shape[self.dim])
            .run(
                local_tensor,
                lambda t: funcol.all_gather_tensor(
                    t, gather_dim=self.dim, group=(mesh, mesh_dim)
                ),
            )
        )

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
        return self._select_split_tensor(
            local_tensor,
            num_chunks,
            shard_index,
            with_padding=False,
            clone=True,
        )

    def _to_new_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
        new_shard_dim: int,
    ) -> torch.Tensor:
        """
        Transform from Shard(self.dim) to Shard(new_shard_dim) via alltoall.

        Redistributes the tensor from being sharded on self.dim to being sharded
        on new_shard_dim. Each rank exchanges data with other ranks so that after
        the operation, each rank holds the appropriate shard of the new dimension.

        Args:
            local_tensor: The local shard on this rank.
            mesh: The device mesh over which the tensor is distributed.
            mesh_dim: The mesh dimension along which to perform alltoall.
            current_logical_shape: The global/logical shape of the full tensor.
            new_shard_dim: The target tensor dimension for sharding.

        Returns:
            The local shard after redistribution to Shard(new_shard_dim).
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return local_tensor

        return (
            CollectivePaddingContext(mesh, mesh_dim)
            .pad_old_shard(self.dim, current_logical_shape[self.dim])
            .pad_new_shard(new_shard_dim, current_logical_shape[new_shard_dim])
            .run(
                local_tensor,
                lambda t: shard_dim_alltoall(
                    t, self.dim, new_shard_dim, mesh, mesh_dim
                ),
            )
        )

    def _to_new_strided_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
        new_shard_dim: int,
        split_factor: int,
    ) -> torch.Tensor:
        """
        Transform from Shard(old_dim) to _StridedShard(new_dim, split_factor) via alltoall.

        This method redistributes a tensor that is currently sharded on self.dim (using
        regular Shard placement) to a new _StridedShard placement on new_shard_dim. The
        _StridedShard placement represents right-to-left sharding order used by FSDP2 + TP,
        where the tensor is first sharded by TP then by FSDP.

        The transformation requires:
        1. Rearranging local data into the strided pattern expected by the target placement
        2. Padding to ensure uniform chunk sizes across ranks for alltoall
        3. Performing alltoall to exchange data between ranks
        4. Unpadding to restore correct local shard sizes

        Args:
            local_tensor: The local shard of the tensor on this rank.
            mesh: The device mesh over which the tensor is distributed.
            mesh_dim: The mesh dimension along which the redistribution occurs.
            current_logical_shape: The global/logical shape of the full tensor.
            new_shard_dim: The tensor dimension for the target _StridedShard placement.
            split_factor: The split factor for _StridedShard, indicating how many
                virtual shards exist from a prior (right-side) sharding operation.

        Returns:
            The local shard after redistribution to _StridedShard(new_shard_dim, split_factor).

        Example:
            For a tensor with shape [7, 9] on a mesh of size 4:
            - Source: Shard(0) - tensor is sharded on dim 0
            - Target: _StridedShard(1, split_factor=2) - strided shard on dim 1

            The alltoall exchanges data so each rank receives interleaved pieces
            along dim 1, matching the strided sharding pattern.
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            return local_tensor

        return (
            CollectivePaddingContext(mesh, mesh_dim)
            .pad_old_shard(self.dim, current_logical_shape[self.dim])
            .pad_new_strided(
                new_shard_dim, current_logical_shape[new_shard_dim], split_factor
            )
            .run(
                local_tensor,
                lambda t: shard_dim_alltoall(
                    t, self.dim, new_shard_dim, mesh, mesh_dim
                ),
            )
        )

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


class _StridedShard(torch._C._distributed.StridedShard):
    """
    _StridedShard was originally introduced to support 2D FSDP2 + TP sharding where the tensor
    is sharded on the TP mesh dimension first, then sharded on the FSDP mesh dimension.
    We call this right-to-left sharding which is the opposite of the default
    left-to-right sharding. See the example below::

        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [Shard(0), Shard(0)]

    The default sharding behavior shards the tensor on "dp" mesh dimension first then
    "tp" dimension. The sharding result will be::

        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   1 (row 2-3)
        2       |   (1, 0)          |   2 (row 4-5)
        3       |   (1, 1)          |   3 (row 6-7)

    While the FSDP2 + TP sharding behavior does the opposite: it shards the tensor on
    "tp" mesh dim first then "dp" dim. This right-to-left sharding will produce the
    result::

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

    Now with _StridedShard, the right-to-left sharding above can be represented as::

        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [_StridedShard(0, split_factor=2), Shard(0)]

    And a left-to-right processing of `placements` will produce the same result, which is
    different from using the `Shard` placement::

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
    """

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> tuple[list[torch.Tensor], list[int]]:
        assert self.dim <= tensor.ndim, (
            f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"
        )

        # Essentially _StridedShard express the right-to-left sharding in the
        # reversed order. Here we perform first_split as the virtual "right" sharding,
        # and then second_split as the virtual "left" sharding, and finally assemble
        # results in the transposed left-first order.

        # First split: chunk into split_factor pieces
        first_split = list(torch.chunk(tensor, self.split_factor, dim=self.dim))
        first_split = fill_empty_tensor_to_shards(
            first_split, self.dim, self.split_factor - len(first_split)
        )

        # Second split: chunk each piece into num_chunks pieces
        second_split = []
        for s in first_split:
            chunks = list(torch.chunk(s, num_chunks, dim=self.dim))
            chunks = fill_empty_tensor_to_shards(
                chunks, self.dim, num_chunks - len(chunks)
            )
            second_split.append(chunks)

        shard_list: list[torch.Tensor] = []
        for i in range(num_chunks):
            shard = torch.cat(
                [second_split[j][i] for j in range(self.split_factor)],
                dim=self.dim,
            )
            if contiguous:
                shard = shard.contiguous()
            shard_list.append(shard)

        # The amount of padding is determined by the local chunk with the largest size.
        pad_sizes: list[int] = []
        max_chunk_size = max([shard.size(self.dim) for shard in shard_list])
        if with_padding:
            pad_sizes = [max_chunk_size - shard.size(self.dim) for shard in shard_list]

        return shard_list, pad_sizes

    @staticmethod
    @maybe_run_for_local_tensor
    def _select_shard(shards: list[torch.Tensor], shard_index) -> torch.Tensor:
        return shards[shard_index].clone()

    @staticmethod
    @maybe_run_for_local_tensor
    def local_shard_size_and_offset(
        tensor_dim: int,
        split_factor: int,
        curr_local_size: int,
        num_chunks: int,
        rank: RankType,
        return_first_offset: bool = True,
    ) -> tuple[int, list[int] | int]:
        """
        Compute the local shard size and offset(s) for a _StridedShard placement.

        Unlike the regular Shard placement which produces contiguous offsets, _StridedShard
        produces non-contiguous (strided) offsets due to the right-to-left sharding semantics.
        This method computes the actual indices that belong to the local shard.

        Args:
            tensor_dim: The tensor dimension being sharded.
            split_factor: The split factor for the _StridedShard placement.
            curr_local_size: The current size of the tensor dimension to be sharded.
            num_chunks: Number of chunks to split the dimension into (typically the mesh dimension size).
            rank: The rank index to compute the shard for.
            return_first_offset: If True, return only the first offset as an int. If False,
                return all offsets as a list. Defaults to True.

        Returns:
            A tuple containing:
                - local_shard_size: The number of elements in the local shard for this rank.
                - offset: If return_first_offset is True, returns the first offset
                  as an int (-1 for empty shards). If False, returns a list of all offsets.
        """
        # indices_tensor is 1D torch.arange(logical_dim_size) unsqueezed
        # so that we can reuse self._split_tensor which splits on self.dim
        shape = [1] * tensor_dim + [curr_local_size]
        indices_tensor = torch.arange(
            curr_local_size,
        ).view(shape)

        sharded_indices, _ = _StridedShard(
            tensor_dim, split_factor=split_factor
        )._split_tensor(
            indices_tensor,
            num_chunks,
            with_padding=False,
            contiguous=False,
        )
        # squeeze back to 1D indices tensor
        sharded_indices = [shard.view(-1) for shard in sharded_indices]

        local_shard_size = _StridedShard._local_shard_size(sharded_indices, rank)
        if local_shard_size > 0:
            offsets = sharded_indices[rank].tolist()
        else:
            offsets = []

        if return_first_offset:
            # Always return an int for consistency across ranks.
            # For empty shards, return -1 as an invalid offset indicator.
            offsets = offsets[0] if len(offsets) > 0 else -1

        return local_shard_size, offsets

    def _local_shard_size_and_offset(
        self,
        curr_local_size: int,
        num_chunks: int,
        rank: RankType,
        return_first_offset: bool = True,
    ) -> tuple[int, int | list[int]]:
        return self.local_shard_size_and_offset(
            self.dim,
            self.split_factor,
            curr_local_size,
            num_chunks,
            rank,
            return_first_offset,
        )

    @staticmethod
    @maybe_run_for_local_tensor
    def _local_shard_size(sharded_indices: list[torch.Tensor], rank: RankType) -> int:
        return len(sharded_indices[rank])

    def _shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        """
        Shard and scatter a tensor on a mesh dimension (use coordinate 0 on the
        mesh dimension as source of truth).

        Create the local tensor for this rank following the given StridedShard
        placement. If src_data_rank is None, perform only local splitting.
        Otherwise, additionally scatter data from src_data_rank. Unlike
        ``_split_tensor``, which supports uneven sharding via padding, this
        method requires the tensor dimension to be evenly divisible by the
        number of chunks (mesh dimension size).
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        mesh_dim_local_rank = my_coordinate[mesh_dim]

        if src_data_rank is None:
            # src_data_rank specified as None explicitly means to skip the
            # communications, simply split
            scatter_list, _ = self._split_tensor(
                tensor, num_chunks, with_padding=False, contiguous=True
            )

            return self._select_shard(scatter_list, mesh_dim_local_rank)

        scatter_list, pad_sizes = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )

        it = iter(scatter_list)
        first = next(it)
        # Tensors in the scatter list are expected to have the same shape because
        # split is requested with padding.
        assert all(first.shape == v.shape for v in it)

        output = torch.empty_like(first)

        # perform scatter from the src_data_rank as data source when it is not None
        mesh_scatter(
            output, scatter_list, mesh, mesh_dim=mesh_dim, group_src=src_data_rank
        )

        return Shard._maybe_unpad_tensor_with_sizes(
            self.dim, output, pad_sizes, mesh_dim_local_rank, True
        )

    @classmethod
    def _make_shard_tensor(
        cls,
        dim: int,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
        split_factor: int = 1,
    ) -> torch.Tensor:
        strided_shard_placement = cls(dim=dim, split_factor=split_factor)
        return strided_shard_placement._shard_tensor(
            tensor, mesh, mesh_dim, src_data_rank
        )

    @maybe_run_for_local_tensor
    def _select_split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        index: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
        clone: bool = True,
    ) -> torch.Tensor:
        """
        Like _split_tensor() but only returns a single Tensor
        """
        shards, _ = self._split_tensor(
            tensor, num_chunks, with_padding=with_padding, contiguous=False
        )
        result = shards[index]
        if clone:
            result = result.clone()
        elif contiguous:
            result = result.contiguous()
        return result

    def _reduce_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        reduce_op: str,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        Reduce and scatter a tensor on a mesh dimension to a _StridedShard placement.

        This performs a reduce_scatter operation where the input tensor (replicated
        across ranks) is reduced and then scattered along self.dim with strided
        sharding semantics. The result is a strided-sharded tensor where each rank
        holds interleaved pieces of the reduced result according to the split_factor.

        Args:
            tensor: The replicated input tensor to reduce and scatter.
            mesh: The device mesh over which the operation is performed.
            reduce_op: The reduction operation (e.g., "sum", "avg").
            mesh_dim: The mesh dimension along which to reduce_scatter.

        Returns:
            The local strided shard of the reduced tensor.
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            return tensor

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # `tensor.size(self.dim)` is the logical size since input is the same
        # size as replicated on that mesh dim.
        return (
            CollectivePaddingContext(mesh, mesh_dim)
            .pad_new_strided(self.dim, tensor.size(self.dim), self.split_factor)
            .run(
                tensor,
                lambda t: funcol.reduce_scatter_tensor(
                    t, reduce_op, scatter_dim=self.dim, group=(mesh, mesh_dim)
                ),
            )
        )

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
    ) -> torch.Tensor:
        """
        Transform from _StridedShard to Replicate via all_gather.

        Gathers all strided shards across ranks and reorders elements to produce
        a fully replicated tensor. Unlike regular Shard, _StridedShard has an
        interleaved pattern that requires special handling during unpadding to
        restore the correct logical ordering.

        Args:
            local_tensor: The local strided shard on this rank.
            mesh: The device mesh over which the tensor is distributed.
            mesh_dim: The mesh dimension along which to gather.
            current_logical_shape: The global/logical shape of the full tensor.

        Returns:
            The fully replicated tensor with shape matching current_logical_shape.
        """
        return (
            CollectivePaddingContext(mesh, mesh_dim)
            .pad_old_strided(
                self.dim, current_logical_shape[self.dim], self.split_factor
            )
            .run(
                local_tensor,
                lambda t: funcol.all_gather_tensor(
                    t, gather_dim=self.dim, group=(mesh, mesh_dim)
                ),
            )
        ).contiguous()

    def _replicate_to_strided_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: int,
    ) -> torch.Tensor:
        """
        Transform from replicated tensor to a strided-sharded tensor on the current rank.

        This performs a local chunking operation using the _StridedShard pattern,
        where the tensor is split according to the strided sharding semantics
        (interleaved pieces based on split_factor).

        Args:
            local_tensor: The replicated tensor on this rank.
            mesh: The device mesh over which the tensor is distributed.
            mesh_dim: The mesh dimension for the sharding.
            shard_index: The index of the shard to select (typically the rank's
                coordinate on the mesh dimension).

        Returns:
            The local strided shard for this rank.
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        shards, _ = self._split_tensor(
            local_tensor,
            num_chunks,
            with_padding=False,
            contiguous=False,
        )

        return _StridedShard._select_shard(shards, shard_index)

    def _to_new_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
        new_shard_dim: int,
    ) -> torch.Tensor:
        """
        Transform from _StridedShard(self.dim) to Shard(new_shard_dim) via alltoall.

        Redistributes the tensor from _StridedShard on self.dim to regular Shard on
        new_shard_dim. This handles the interleaved pattern of _StridedShard during
        padding and reordering, producing a contiguously sharded output.

        Args:
            local_tensor: The local strided shard on this rank.
            mesh: The device mesh over which the tensor is distributed.
            mesh_dim: The mesh dimension along which to perform alltoall.
            current_logical_shape: The global/logical shape of the full tensor.
            new_shard_dim: The target tensor dimension for sharding.

        Returns:
            The local shard after redistribution to Shard(new_shard_dim).
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return local_tensor

        return (
            CollectivePaddingContext(mesh, mesh_dim)
            .pad_old_strided(
                self.dim, current_logical_shape[self.dim], self.split_factor
            )
            .pad_new_shard(new_shard_dim, current_logical_shape[new_shard_dim])
            .run(
                local_tensor,
                lambda t: shard_dim_alltoall(
                    t, self.dim, new_shard_dim, mesh, mesh_dim
                ),
            )
        )

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


class Replicate(torch._C._distributed.Replicate):
    """
    The ``Replicate()`` placement describes the DTensor replicating on a corresponding
    ``DeviceMesh`` dimension, where each rank on the DeviceMesh dimension holds a
    replica of the global Tensor. The ``Replicate`` placement can be used by all
    DTensor APIs (i.e. ``distribute_tensor``, ``DTensor.from_local``, etc.)
    """

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

    @classmethod
    def _make_replicate_tensor(
        cls,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
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

        if src_data_rank is not None:
            # perform broadcast from the src_data_rank as data source when it is not None
            mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim, group_src=src_data_rank)
        return tensor

    def _replicate_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        return Replicate._make_replicate_tensor(tensor, mesh, mesh_dim, src_data_rank)


class Partial(torch._C._distributed.Partial):
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
            to produce Replicated/Sharded DTensor. Corresponds to the reduce operations
            supported by ``torch.distributed.ReduceOp``. Default: "sum".

            Supported values:

            * ``"sum"``: Element-wise sum across all ranks.
            * ``"avg"``: Element-wise average across all ranks.
            * ``"min"``: Element-wise minimum across all ranks.
            * ``"max"``: Element-wise maximum across all ranks.
            * ``"product"``: Element-wise product across all ranks.
            * ``"band"``: Bitwise AND across all ranks (integer tensors only).
            * ``"bor"``: Bitwise OR across all ranks (integer tensors only).
            * ``"bxor"``: Bitwise XOR across all ranks (integer tensors only).

    .. note:: The ``Partial`` placement can be generated as a result of the DTensor operators,
        and can only be used by the ``DTensor.from_local`` API.
    """

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
        if not isinstance(shard_spec, Shard | _StridedShard):
            raise ValueError(
                f"Partial can only reduce scatter into Shard or _StridedShard, but got {shard_spec}"
            )
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        """
        Partition a replicated tensor to create partial values for Replicate → Partial.

        This is the conjugate operation of _reduce_value. The partition operation
        must satisfy the invariant that applying _reduce_value to the partitioned
        values recovers the original replicated value (modulo floating-point error).

        Mathematical analysis by reduce_op:

        * "sum": partition(v) = v / n, then sum([v/n] * n) = v
          Introduces floating-point error from the division and summation.
          Error grows with n (the number of ranks).

        * "avg": partition(v) = v, then avg([v] * n) = v
          Numerically exact (averaging identical values).

        * "min": partition(v) = v, then min([v] * n) = v
          Numerically exact.

        * "max": partition(v) = v, then max([v] * n) = v
          Numerically exact.

        * "product": Would need partition(v) = v^(1/n), but n-th root is not exact
          for general values and undefined for negative values with even n.
          NOT SUPPORTED.

        * "band"/"bor"/"bxor": Bitwise operations have no well-defined inverse
          that partitions a value such that reducing recovers the original.
          NOT SUPPORTED.
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        if self.reduce_op == "sum":
            return tensor / num_chunks
        elif self.reduce_op in ("avg", "min", "max"):
            return tensor
        else:
            raise ValueError(
                f"Replicate to Partial({self.reduce_op}) conversion is not supported."
            )

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
        return f"P({self.reduce_op})"


# We keep the old _Partial name for a while for BC reason
_Partial = Partial


@dataclass(frozen=True)
class _MaskPartial(Partial):
    """
    A partial mask placement devised for rowwise sharded embedding op, where we need
    to mask and adjust the indices to the local embedding shard, embedding masking
    is a special type of the Partial placement

    NOTE: the lifecycle of this _MaskPartial placement follows the corresponding DTensor
    lifecycle, i.e. the indices_mask would only be alive during the lifetime of the DTensor.
    """

    mask_buffer: MaskBuffer = field(default_factory=MaskBuffer)

    # required fields for computing the local offset and deriving the mask
    offset_shape: torch.Size | None = None
    offset_dim: int = 0

    def __init__(
        self,
        reduce_op=None,
        mask_buffer=None,
        offset_shape=None,
        offset_dim=0,
        *args,
        **kwargs,
    ):
        super().__init__(reduce_op)
        if mask_buffer is None:
            mask_buffer = MaskBuffer()
        object.__setattr__(self, "mask_buffer", mask_buffer)
        object.__setattr__(self, "offset_shape", offset_shape)
        object.__setattr__(self, "offset_dim", offset_dim)

    @staticmethod
    @maybe_run_for_local_tensor
    def _mask_tensor(
        tensor: torch.Tensor, local_offset_on_dim: int, local_shard_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Build the input mask and save it for the current partial placement
        # this is so that the output of embedding op can reuse the same partial
        # placement saved mask to perform mask + reduction
        mask = (tensor < local_offset_on_dim) | (
            tensor >= local_offset_on_dim + local_shard_size
        )
        # mask the input tensor
        masked_tensor = tensor.clone() - local_offset_on_dim
        masked_tensor[mask] = 0
        return mask, masked_tensor

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        assert mesh._is_current_rank_part_of_mesh(), "rank is not part of mesh"
        # override parent logic to perform partial mask for embedding
        num_chunks = mesh.size(mesh_dim)
        # get local shard size and offset on the embedding_dim
        assert self.offset_shape is not None, (
            "offset_shape needs to be set for _MaskPartial"
        )
        local_shard_size, local_offset_on_dim = Shard.local_shard_size_and_offset(
            self.offset_shape[self.offset_dim],
            num_chunks,
            mesh._sym_get_coordinate(mesh_dim),
        )
        mask, masked_tensor = _MaskPartial._mask_tensor(
            tensor, local_offset_on_dim, local_shard_size
        )
        # materialize the mask buffer to be used for reduction
        self.mask_buffer.materialize_mask(mask)
        return masked_tensor

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # by the time we need reduction, we should have already saved the mask
        assert self.mask_buffer.data is not None

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # perform sum reduction
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
        # by the time we need reduction, we should have already saved the mask
        assert self.mask_buffer.data is not None

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # call reduce_shard_tensor of the shard_spec.
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MaskPartial):
            return False

        return (
            self.reduce_op == other.reduce_op
            and self.offset_shape == other.offset_shape
            and self.offset_dim == other.offset_dim
            and self.mask_buffer is other.mask_buffer
        )

    def __hash__(self) -> int:
        return 1 + hash(
            (
                self.reduce_op,
                self.offset_shape,
                self.offset_dim,
                id(self.mask_buffer),
            )
        )

    def __repr__(self) -> str:
        """
        machine readable representation of the _MaskPartial placement
        """
        return f"_MaskPartial(reduce_op={self.reduce_op}, offset_shape={self.offset_shape}, offset_dim={self.offset_dim})"

    def __str__(self) -> str:
        """
        human readable representation of the _MaskPartial placement
        """
        return f"MaskP({self.reduce_op}, {self.offset_shape}, {self.offset_dim})"
