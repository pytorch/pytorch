from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union
from typing import TYPE_CHECKING

import torch

from ._internals import (
    check_tensor,
    get_chunked_dim_size,
    get_split_size,
    validate_non_overlapping_shards_metadata
)
from torch.distributed._shard.metadata import ShardMetadata

from torch.distributed._shard.sharded_tensor.utils import (
    _parse_and_validate_remote_device
)

import torch.distributed as dist
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.sharded_tensor.shard import Shard

if TYPE_CHECKING:
    # Only include ShardedTensor when do type checking, exclude it
    # from run-time to resolve circular dependency.
    from torch.distributed._shard.sharded_tensor import ShardedTensor

class PlacementSpec(ABC):
    """
    Base class representing the placement of an entity. Subclasses of this
    class can be used to specify customized placements which might not be
    covered by existing APIs.
    """
    pass


@dataclass
class DevicePlacementSpec(PlacementSpec):
    """
    Associates placement of an entity with a single device.

    Args:
        device(:class:`torch.distributed._remote_device`): The device to place the entity on.
    """

    device: torch.distributed._remote_device

    def __post_init__(self):
        if not isinstance(self.device, torch.distributed._remote_device):
            self.device = torch.distributed._remote_device(self.device)

class ShardingSpec(object):
    """
    Base class representing sharding specifications.
    """
    @abstractmethod
    def build_metadata(self,
                       tensor_sizes: torch.Size,
                       tensor_properties: sharded_tensor_meta.TensorProperties,
                       ) -> sharded_tensor_meta.ShardedTensorMetadata:
        """
        Given a global tensor size, define how to shard a tensor like this shape
        across ranks, return ShardedTensorMetadata
        Args:
            tensor_sizes (:class:`torch.Size`):
                The tensor shape to shard on, a `torch.Size` object that represents the
                tensor shape to be sharded according to the ShardingSpec.
            tensor_properties(:class:`torch.distributed._shard.sharded_tensor.TensorProperties):
                Tensor properties used to create a ShardedTensor.
        Returns:
            A :class:`ShardedTensorMetadata` object that encodes the information about
            the layout of the ShardedTensor and its properties.
        """

    @abstractmethod
    def shard(self, tensor: torch.Tensor, src_rank: int = 0, process_group=None) -> "ShardedTensor":
        """
        Given a global tensor on src_rank, shard this tensor
        across ranks within the process group, return a ShardedTensor.
        Args:
            tensor (:class:`torch.Tensor`): Tensor needs to be sharded.
        Keyword args:
            src_rank (int, optional): The source rank which is used as the ground truth of
                the data for the parameter that would be sharded and scattered
                across the rest of the ranks.
                Default: 0.
            process_group (ProcessGroup, optional): The process group to work on. If None,
                the default process group will be used.
        Returns:
            A :class:`ShardedTensor` sharded from the given tensor.
        """

@dataclass
class ChunkShardingSpec(ShardingSpec):
    """
    This is a type of PlacementSpec that defines the placement as being sharded
    across multiple devices. In particular, it represents sharding a Tensor
    along a single dimension into equal chunks (similar to :meth:`torch.chunk`).

    The semantics of how a tensor is partitioned is inline with
    :meth:`torch.chunk`, where ``dim`` in torch.chunk corresponds to the
    specified ``dim`` and ``chunks`` in torch.chunk is the number of elements
    in the placement specified.

    Args:
        dim (int or str):
            The dimension to shard on, could be an integer representing the
            dimension or a string in case of named tensors where dimensions are
            named. Note that named tensor support is not added yet.
        placement(List[Union[_remote_device, str]]):
            Specifies the placement of each shard of the Tensor. The size of
            the list represents the number of shards to be created. This could
            be a list of
            :class:`torch.distributed._remote_device`'s. This list
            could also contain a string which represents remote
            device as accepted by
            :class:`torch.distributed._remote_device`
    """

    ShardingDim = Union[int, str]

    dim: ShardingDim
    placements: List[Union[torch.distributed._remote_device, str]]

    def __post_init__(self):
        self._verify_dim(self.dim)
        for i, remote_device in enumerate(self.placements):
            if not isinstance(remote_device, torch.distributed._remote_device):
                self.placements[i] = torch.distributed._remote_device(remote_device)

    @staticmethod
    def _verify_dim(dim):
        # Validate the sharding spec.
        # TODO: support named dimension
        if isinstance(dim, str):
            raise NotImplementedError(
                "ChunkShardingSpec does not support named dimension yet!"
            )

        if not isinstance(dim, int):
            raise ValueError(
                f"Sharding dim needs to be an integer, found: {dim}"
            )

    def build_metadata(self,
                       tensor_sizes: torch.Size,
                       tensor_properties: sharded_tensor_meta.TensorProperties,
                       ) -> sharded_tensor_meta.ShardedTensorMetadata:
        tensor_num_dim = len(tensor_sizes)

        self._verify_dim(self.dim)
        if self.dim >= tensor_num_dim or self.dim < -tensor_num_dim:  # type: ignore[operator]
            raise ValueError(f"Invalid sharding dim: {self.dim}")

        shards_metadata = []
        sharding_dim_size = tensor_sizes[self.dim]  # type: ignore[index]
        chunks = len(self.placements)
        split_size = get_split_size(sharding_dim_size, chunks)
        for idx, placement in enumerate(self.placements):
            # generate ShardMetadata for each placement device
            chunked_dim_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
            if chunked_dim_size > 0:
                shard_size = list(tensor_sizes)
                current_offsets = [0] * tensor_num_dim
                current_offsets[self.dim] = split_size * idx  # type: ignore[index]
                shard_size[self.dim] = chunked_dim_size  # type: ignore[index]

                shard_metadata = ShardMetadata(
                    shard_offsets=current_offsets,
                    shard_sizes=shard_size,
                    placement=placement,
                )
                shards_metadata.append(shard_metadata)

                # current_offsets[self.dim] += chunked_dim_size  # type: ignore[index]

        return sharded_tensor_meta.ShardedTensorMetadata(
            shards_metadata,
            tensor_sizes,
            tensor_properties
        )


    def shard(self, tensor: torch.Tensor, src_rank: int = 0, process_group=None) -> "ShardedTensor":
        # relative imports to avoid circular dependency
        from torch.distributed._shard.sharded_tensor import (
            ShardedTensor
        )
        tensor_properties = sharded_tensor_meta.TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned()
        )
        current_rank = dist.get_rank(process_group)
        tensor_meta = self.build_metadata(tensor.size(), tensor_properties)
        local_shards = []
        local_tensor = None
        local_metadata = None
        tensors_to_scatter = []

        for shard_meta in tensor_meta.shards_metadata:
            rank, device = _parse_and_validate_remote_device(process_group, shard_meta.placement)
            shard_offsets = shard_meta.shard_offsets
            shard_sizes = shard_meta.shard_sizes
            if current_rank == src_rank:
                narrowed_tensor = tensor
                for idx, (offset, size) in enumerate(zip(shard_offsets, shard_sizes)):
                    if size < tensor.size(idx):
                        # Reshape to get shard for this rank and we don't want autograd
                        # recording here for the narrow op and 'local_shard' should be a
                        # leaf variable in the autograd graph.
                        narrowed_tensor = narrowed_tensor.narrow(
                            idx,
                            shard_offsets[idx],
                            shard_sizes[idx]
                        ).clone().detach().contiguous()
                tensors_to_scatter.append(narrowed_tensor)

            if current_rank == rank:
                local_tensor = torch.empty(
                    shard_sizes, dtype=tensor.dtype, layout=tensor.layout, device=device)
                local_metadata = shard_meta

        # Scatter the shards to all ranks in the pg
        dist.scatter(
            local_tensor,
            scatter_list=tensors_to_scatter if current_rank == src_rank else None,
            src=src_rank,
            group=process_group
        )

        assert local_tensor is not None
        assert local_metadata is not None
        # Sync requires_grad to local_shard.
        local_tensor.requires_grad = tensor.requires_grad

        local_shards.append(Shard(tensor=local_tensor, metadata=local_metadata))

        st = ShardedTensor._init_from_local_shards(local_shards, tensor.size(), process_group=process_group)
        # Manually set sharding_spec
        st._sharding_spec = self

        return st


@dataclass
class EnumerableShardingSpec(ShardingSpec):
    """
    This is a type of PlacementSpec that allows users to specify a generic
    sharding scheme by enumerating exactly how each shard is laid out.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard. Note that none of the shards should overlap.
    """

    shards: List[ShardMetadata]

    def __post_init__(self):
        if len(self.shards) == 0:
            raise ValueError(f'Empty shard list provided: {self.shards}')

        # Validate each shard has same rank.
        rank = -1
        for shard in self.shards:
            if rank != -1 and rank != len(shard.shard_offsets):
                raise ValueError(f'Found inconsistent ranks for shards: {rank} and {len(shard.shard_offsets)}')
            rank = len(shard.shard_offsets)

        validate_non_overlapping_shards_metadata(self.shards)

    def build_metadata(self,
                       tensor_sizes: torch.Size,
                       tensor_properties: sharded_tensor_meta.TensorProperties,
                       ) -> sharded_tensor_meta.ShardedTensorMetadata:
        # check if shards form a valid tensor
        check_tensor(self.shards, tensor_sizes)
        return sharded_tensor_meta.ShardedTensorMetadata(
            self.shards,
            tensor_sizes,
            tensor_properties
        )

    def shard(self, tensor: torch.Tensor, src_rank: int = 0, process_group=None) -> "ShardedTensor":
        # TODO: figure out a generic and efficient way to scatter the shards for EnumerableShardingSpec
        raise NotImplementedError("EnumerableShardingSpec.shard not implemented yet!")


def _infer_sharding_spec_from_shards_metadata(shards_metadata):
    """
    Infer the sharding spec from the metadata of each shard of a ShardedTensor.
    If the tensor is sharded only on one dimension, we then assume it's a ChunkShardingSpec.
    Otherwise, we assume it's enum sharded.

    Args:
        shards_metadata (List[ShardMetadata]): List of Metadata of local shards.

    Returns:
        A :class:`torch.distributed._shard.sharding_spec.ShardingSpec` object of sharding
            spec for one sharded tensor.
    """
    placements = []
    chunk_sharding_dim = None
    chunk_offset_list = []
    shard_size_list = []
    # collect local shard metadatas from the global sharded_tensor_metadata
    for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
        placements.append(shard_metadata.placement)
        local_offsets = shard_metadata.shard_offsets
        chunk_offset_list.append(sum(local_offsets))
        shard_size_list.append(shard_metadata.shard_sizes)
        shard_dims = [idx for idx, e in enumerate(local_offsets) if e != 0]
        # If the offset is [0, 0, ..., 0] (all zeros),
        # we cannot decide whether how the tensor is sharded.
        if len(shard_dims) == 0:
            continue
        # If the offset is [0, N, .,0, M, 0, .., 0],
        # we are sure it's sharded by more than one dimension.
        if len(shard_dims) != 1:
            chunk_sharding_dim = None
            break
        # If the offset is [0, 0, .,0, M, 0, .., 0], aka, it's sharded by just
        # one dimension, we need to make sure all ranks share the same dimension.
        if not chunk_sharding_dim:
            chunk_sharding_dim = shard_dims[0]
        elif chunk_sharding_dim != shard_dims[0]:
            chunk_sharding_dim = None
            break

    if chunk_sharding_dim is not None:
        # Ensure we infer the correct placement order from offsets
        placements = [
            x for _, x in sorted(zip(chunk_offset_list, placements), key=lambda e: e[0])
        ]
        chunk_spec = ChunkShardingSpec(
            dim=chunk_sharding_dim,
            placements=placements,
        )
        shard_sizes = [
            x[chunk_sharding_dim]
            for _, x in sorted(zip(chunk_offset_list, shard_size_list))
        ]
        if len(shard_sizes) == 1 or (
            len(set(shard_sizes[:-1])) == 1 and shard_sizes[0] >= shard_sizes[-1]
        ):
            return chunk_spec
        # Corner case when length = 5 and chunks = 4, local size is [2, 2, 1, 0]
        if (
            len(set(shard_sizes[:-2])) == 1
            and shard_sizes[0] >= shard_sizes[-2]
            and shard_sizes[-2] >= shard_sizes[-1]
        ):
            return chunk_spec
    return EnumerableShardingSpec(shards_metadata)
