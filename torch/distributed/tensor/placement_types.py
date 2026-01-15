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
        reduce and scatter a tensor on a mesh dimension
        """
        if not mesh._is_current_rank_part_of_mesh():
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return tensor

        num_chunks = mesh.size(mesh_dim=mesh_dim)
        is_padded = tensor.size(self.dim) % num_chunks != 0
        pad_sizes = None
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
            assert pad_sizes is not None
            output = Shard._maybe_unpad_tensor_with_sizes(
                self.dim, output, pad_sizes, mesh._sym_get_coordinate(mesh_dim), False
            )
        return output

    @maybe_run_for_local_tensor
    def _maybe_pad_tensor(
        self,
        local_tensor: torch.Tensor,
        logical_dim_size: int,
        num_chunks: int,
    ) -> torch.Tensor:
        is_padded = logical_dim_size % num_chunks != 0

        if is_padded:
            full_chunk_size = (logical_dim_size + num_chunks - 1) // num_chunks
            pad_size = full_chunk_size - local_tensor.size(self.dim)
            local_tensor = pad_tensor(local_tensor, self.dim, pad_size)

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        return local_tensor

    @maybe_run_for_local_tensor
    def _maybe_unpad_tensor(
        self,
        local_tensor: torch.Tensor,
        logical_dim_size: int,
        num_chunks: int,
    ) -> torch.Tensor:
        is_padded = logical_dim_size % num_chunks != 0

        if is_padded:
            full_chunk_size = (logical_dim_size + num_chunks - 1) // num_chunks
            unpad_size = full_chunk_size * num_chunks - logical_dim_size  # type: ignore[possibly-undefined]
            local_tensor = unpad_tensor(local_tensor, self.dim, unpad_size)

        return local_tensor

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
    ) -> torch.Tensor:
        """
        This function all_gather all shards and return a tensor that
        is replicated on the previously sharded mesh dimension
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        logical_dim_size = current_logical_shape[self.dim]

        local_tensor = self._maybe_pad_tensor(
            local_tensor, logical_dim_size, num_chunks
        )

        result = funcol.all_gather_tensor(
            local_tensor,
            gather_dim=self.dim,
            group=(mesh, mesh_dim),
        )

        result = self._maybe_unpad_tensor(result, logical_dim_size, num_chunks)

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
        return self._select_split_tensor(
            local_tensor,
            num_chunks,
            shard_index,
            with_padding=False,
            clone=True,
        )

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
    def _compute_padding_info(
        current_logical_shape: list[int],
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
    ) -> tuple[bool, int, int, bool, int, int]:
        results = []
        for shard_dim in [old_shard_dim, new_shard_dim]:
            dim_logical_size = current_logical_shape[shard_dim]
            dim_padding = dim_logical_size % num_chunks != 0
            dim_full_chunk_size = (dim_logical_size + num_chunks - 1) // num_chunks
            results.append((dim_padding, dim_logical_size, dim_full_chunk_size))

        return results[0] + results[1]

    @staticmethod
    @maybe_run_for_local_tensor
    def _pad_for_new_shard_dim(
        current_logical_shape: list[int],
        local_tensor: torch.Tensor,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
    ) -> torch.Tensor:
        (
            old_dim_padding,
            _,
            old_dim_full_chunk_size,
            new_dim_padding,
            _,
            new_dim_full_chunk_size,
        ) = Shard._compute_padding_info(
            current_logical_shape, num_chunks, old_shard_dim, new_shard_dim
        )

        if old_dim_padding:
            old_dim_pad_size = Shard._get_shard_pad_size(
                old_dim_full_chunk_size, local_tensor, old_shard_dim
            )
            local_tensor = pad_tensor(local_tensor, old_shard_dim, old_dim_pad_size)
        if new_dim_padding:
            new_dim_pad_size = Shard._get_shard_pad_size(
                new_dim_full_chunk_size * num_chunks, local_tensor, new_shard_dim
            )
            local_tensor = pad_tensor(local_tensor, new_shard_dim, new_dim_pad_size)

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()
        return local_tensor

    @staticmethod
    @maybe_run_for_local_tensor
    def _unpad_for_new_shard_dim(
        current_logical_shape: list[int],
        local_tensor: torch.Tensor,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
        local_rank: int,
    ) -> torch.Tensor:
        (
            old_dim_padding,
            _,
            old_dim_full_chunk_size,
            new_dim_padding,
            new_dim_logical_size,
            new_dim_full_chunk_size,
        ) = Shard._compute_padding_info(
            current_logical_shape, num_chunks, old_shard_dim, new_shard_dim
        )

        if old_dim_padding:
            old_dim_unpad_size = (
                old_dim_full_chunk_size * num_chunks
                - current_logical_shape[old_shard_dim]  # type: ignore[possibly-undefined]
            )
            local_tensor = unpad_tensor(local_tensor, old_shard_dim, old_dim_unpad_size)  # type: ignore[possibly-undefined]

        if new_dim_padding:
            local_shard_size_on_new_dim = Shard.local_shard_size_and_offset(
                new_dim_logical_size, num_chunks, local_rank
            )[0]
            new_dim_unpad_size = new_dim_full_chunk_size - local_shard_size_on_new_dim  # type: ignore[possibly-undefined]
            local_tensor = unpad_tensor(local_tensor, new_shard_dim, new_dim_unpad_size)  # type: ignore[possibly-undefined]

        return local_tensor

    def _to_new_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
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

        local_tensor = Shard._pad_for_new_shard_dim(
            current_logical_shape, local_tensor, num_chunks, self.dim, new_shard_dim
        )

        new_tensor = shard_dim_alltoall(
            local_tensor, self.dim, new_shard_dim, mesh, mesh_dim
        )

        new_tensor = Shard._unpad_for_new_shard_dim(
            current_logical_shape,
            new_tensor,
            num_chunks,
            self.dim,
            new_shard_dim,
            my_coordinate[mesh_dim],
        )

        return new_tensor

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
    _StridedShard is only introduced to support 2D FSDP2 + TP sharding where the tensor
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

    TODO: we should remove _StridedShard placement once we can unify it with Shard
    """

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

    @staticmethod
    @maybe_run_for_local_tensor
    def _select_shard(shards: list[torch.Tensor], shard_index) -> torch.Tensor:
        return shards[shard_index].clone()

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

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
    ) -> torch.Tensor:
        """
        Replay the replicate-to-shard process to understand how to stitch shards back.

        This method performs all_gather to collect all shards and then reconstructs
        the original replicated tensor by handling padding, unpadding, and reordering.

        Example:
            Consider a 1D input tensor [0, 1, 2, 3, 4, 5, 6, 7, 8] with 9 elements.
            Using _StridedShard(dim=0, split_factor=2) and num_chunks=4:

            Preparation (via _split_tensor, before _to_replicate_tensor is called):
                _split_tensor produces 4 shards with strided indices:
                - First split (split_factor=2): [0,1,2,3,4] and [5,6,7,8]
                - Second split (num_chunks=4) on each piece:
                    [0,1,2,3,4] (5 elements) -> [[0,1], [2,3], [4], []]
                    [5,6,7,8]   (4 elements) -> [[5], [6], [7], [8]]
                - Transpose and concatenate for each chunk:
                    Chunk 0: [0,1] + [5] = [0,1,5]  (indices)
                    Chunk 1: [2,3] + [6] = [2,3,6]  (indices)
                    Chunk 2: [4] + [7]   = [4,7]    (indices)
                    Chunk 3: [] + [8]    = [8]      (indices)

                So we get shards with values:
                    Rank 0: [0, 1, 5]  (size=3)
                    Rank 1: [2, 3, 6]  (size=3)
                    Rank 2: [4, 7]     (size=2)
                    Rank 3: [8]        (size=1)

                These shards are the `local_tensor` input to _to_replicate_tensor
                on each rank. Each rank only has its own shard when this function
                is called.

            Step 1: Pad all shards to max_chunk_size=3:
                    Rank 0: [0, 1, 5]     (no padding needed)
                    Rank 1: [2, 3, 6]     (no padding needed)
                    Rank 2: [4, 7, P]     (padded with 1 element)
                    Rank 3: [8, P, P]     (padded with 2 elements)

            Step 2: all_gather produces concatenated padded tensor:
                [0, 1, 5, | 2, 3, 6, | 4, 7, P, | 8, P, P]
                 chunk 0    chunk 1    chunk 2    chunk 3
                (pos 0-2)  (pos 3-5)  (pos 6-8)  (pos 9-11)

            Step 3: Compute select_indices to extract valid elements and reorder:
                sharded_indices = [[0,1,5], [2,3,6], [4,7], [8]]
                padded_positions:
                    chunk 0: base=0 -> [0, 1, 2]  (positions of [0,1,5] in gathered)
                    chunk 1: base=3 -> [3, 4, 5]  (positions of [2,3,6] in gathered)
                    chunk 2: base=6 -> [6, 7]     (positions of [4,7] in gathered)
                    chunk 3: base=9 -> [9]        (position of [8] in gathered)

                permutation = cat(sharded_indices) = [0, 1, 5, 2, 3, 6, 4, 7, 8]
                select_positions = cat(padded_positions) = [0, 1, 2, 3, 4, 5, 6, 7, 9]

                inv_permutation = argsort(permutation)
                    permutation[0]=0 -> inv_permutation[0]=0
                    permutation[1]=1 -> inv_permutation[1]=1
                    permutation[2]=5 -> inv_permutation[5]=2
                    permutation[3]=2 -> inv_permutation[2]=3
                    permutation[4]=3 -> inv_permutation[3]=4
                    permutation[5]=6 -> inv_permutation[6]=5
                    permutation[6]=4 -> inv_permutation[4]=6
                    permutation[7]=7 -> inv_permutation[7]=7
                    permutation[8]=8 -> inv_permutation[8]=8
                    => inv_permutation = [0, 1, 3, 4, 6, 2, 5, 7, 8]

                select_indices = select_positions[inv_permutation]
                               = [0, 1, 2, 3, 4, 5, 6, 7, 9][inv_permutation]
                    For original position 0: select_indices[0] = select_positions[0] = 0
                    For original position 1: select_indices[1] = select_positions[1] = 1
                    For original position 2: select_indices[2] = select_positions[3] = 3
                    For original position 3: select_indices[3] = select_positions[4] = 4
                    For original position 4: select_indices[4] = select_positions[6] = 6
                    For original position 5: select_indices[5] = select_positions[2] = 2
                    For original position 6: select_indices[6] = select_positions[5] = 5
                    For original position 7: select_indices[7] = select_positions[7] = 7
                    For original position 8: select_indices[8] = select_positions[8] = 9
                    => select_indices = [0, 1, 3, 4, 6, 2, 5, 7, 9]

            Step 4: index_select from gathered tensor using select_indices:
                gathered = [0, 1, 5, 2, 3, 6, 4, 7, P, 8, P, P]
                result = gathered[select_indices]
                       = gathered[[0, 1, 3, 4, 6, 2, 5, 7, 9]]
                       = [0, 1, 2, 3, 4, 5, 6, 7, 8]

            The result is the original replicated tensor [0, 1, 2, 3, 4, 5, 6, 7, 8].
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        logical_dim_size = current_logical_shape[self.dim]

        # indices_tensor is 1D torch.arange(logical_dim_size) unsqueezed
        # so that we can reuse self._split_tensor which splits on self.dim
        shape = [1] * self.dim + [logical_dim_size]
        indices_tensor = torch.arange(
            logical_dim_size, device=local_tensor.device
        ).view(shape)

        sharded_indices, _ = self._split_tensor(
            indices_tensor,
            num_chunks,
            with_padding=False,
            contiguous=False,
        )
        # squeeze back to 1D indices tensor
        sharded_indices = [shard.view(-1) for shard in sharded_indices]

        # First chunk should be one of those biggest chunks.
        max_chunk_size = len(sharded_indices[0])
        local_pad_size = max_chunk_size - local_tensor.size(self.dim)
        local_tensor_padded = pad_tensor(local_tensor, self.dim, local_pad_size)

        if not local_tensor_padded.is_contiguous():
            local_tensor_padded = local_tensor_padded.contiguous()

        replicate_tensor_permuted_padded = funcol.all_gather_tensor(
            local_tensor_padded,
            gather_dim=self.dim,
            group=(mesh, mesh_dim),
        )
        if isinstance(replicate_tensor_permuted_padded, funcol.AsyncCollectiveTensor):
            replicate_tensor_permuted_padded = replicate_tensor_permuted_padded.wait()

        # After all_gather, the tensor is [chunk0, chunk1 (may padded), ...].
        # Each chunk may have padding at the end. Use a single index_select to
        # both extract non-padding data and reorder to the original positions.
        #
        # Build select_indices where select_indices[original_pos] = position in
        # the padded tensor that holds the element for original_pos.
        padded_positions = []
        for i, shard in enumerate(sharded_indices):
            base_offset = i * max_chunk_size
            positions = base_offset + torch.arange(
                len(shard), device=local_tensor.device
            )
            padded_positions.append(positions)

        # Permutation ends up containing strided indices because we create it by
        # chunking over a particular dimension of an N-D shaped arange tensor.
        permutation = torch.cat(sharded_indices)
        # Choose the position by skipping padding indices from
        # replicate_tensor_permuted_padded.
        select_positions = torch.cat(padded_positions)

        inv_permutation = torch.argsort(permutation)
        select_indices = select_positions.index_select(0, inv_permutation)

        replicate_tensor = torch.index_select(
            replicate_tensor_permuted_padded, self.dim, select_indices
        )

        return replicate_tensor.contiguous()

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
        return self._select_split_tensor(
            local_tensor,
            num_chunks,
            shard_index,
            with_padding=False,
            clone=True,
        )

    @staticmethod
    @maybe_run_for_local_tensor
    def _local_shard_size(sharded_indices: list[torch.Tensor], rank: RankType) -> int:
        return len(sharded_indices[rank])

    def _local_shard_size_and_offset(
        self,
        curr_local_size: int,
        num_chunks: int,
        rank: RankType,
        return_first_offset: bool = True,
    ) -> tuple[int, int | list[int]]:
        return self.local_shard_size_and_offset(
            curr_local_size, num_chunks, rank, return_first_offset
        )

    @maybe_run_for_local_tensor
    def local_shard_size_and_offset(
        self,
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
            self (_StridedShard): The _StridedShard placement instance.
            curr_local_size (int): The current size of the tensor dimension to be sharded.
            num_chunks (int): Number of chunks to split the dimension into (typically the mesh dimension size).
            rank (int): The rank index to compute the shard for.
            return_first_offset (bool): If True, return only the first offset as an int. If False,
                return all offsets as a list. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - local_shard_size (int): The number of elements in the local shard for this rank.
                - offset (int | list[int]): If return_first_offset is True, returns the first offset
                  as an int. If False or if the shard size is 0, returns a list of all offsets
                  (which may be empty for empty shards).
        """
        # indices_tensor is 1D torch.arange(logical_dim_size) unsqueezed
        # so that we can reuse self._split_tensor which splits on self.dim
        shape = [1] * self.dim + [curr_local_size]
        indices_tensor = torch.arange(
            curr_local_size,
        ).view(shape)

        sharded_indices, _ = self._split_tensor(
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
        shard_spec = cast(Shard, shard_spec)
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
