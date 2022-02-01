from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
from typing import List, Union
import torch

from ._internals import (
    check_tensor,
    get_chunked_dim_size,
    get_split_size,
    ShardMetadata,
    validate_non_overlapping_shards_metadata
)

# from torch.distributed._sharded_tensor.utils import (
#     _parse_and_validate_remote_device
# )

import torch.distributed as dist

class PlacementSpec(ABC):
    """
    Base class representing the placement of an entity. Subclasses of this
    class can be used to specify customized placements which might not be
    covered by existing APIs.
    """
    @abstractmethod
    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Retrieves a tensor and places it on the appropriate device.
        """


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

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor.to(self.device)


class ShardingSpec(object):
    """
    Base class representing sharding specifications.
    """
    @abstractmethod
    def shard(self, tensor_sizes: List[int], process_group=None) -> List[ShardMetadata]:
        """
        Given a global tensor size list, define how to shard a tensor like this shape
        across ranks, return a list of ShardMetadata.
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
            named.
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
        if not (isinstance(dim, int) or isinstance(dim, str)):
            raise ValueError(f'{dim} needs to either be an int or str')

    def shard(self, tensor_sizes: List[int], process_group=None) -> List[ShardMetadata]:
        """
        Shard a tensor base on ChunkShardingSpec, and return a list of shards on the current rank.
        """
        pg = process_group if process_group is not None else dist._get_default_group()
        world_size = dist.get_world_size(pg)
        tensor_num_dim = len(tensor_sizes)

        if self.dim >= tensor_num_dim or self.dim < -tensor_num_dim:
            raise ValueError(f"Invalid sharding dim: {self.dim}")

        shards_metadata = []
        current_offsets = [0] * tensor_num_dim
        sharding_dim_size = tensor_sizes[self.dim]  # type: ignore[arg-type]
        split_size = get_split_size(sharding_dim_size, world_size)
        for idx, placement in enumerate(self.placements):
            # check if the placement is valid or not
            # _parse_and_validate_remote_device(process_group, placement)
            # generate ShardMetadata for each placement device
            chunked_dim_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
            shard_size = copy.deepcopy(tensor_sizes)
            shard_size[self.dim] = chunked_dim_size  # type: ignore[index]

            shard_metadata = ShardMetadata(
                shard_offsets=copy.deepcopy(current_offsets),
                shard_sizes=shard_size,
                placement=placement,
            )
            shards_metadata.append(shard_metadata)

            current_offsets[self.dim] += chunked_dim_size  # type: ignore[index]

        return shards_metadata


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

    def shard(self, tensor_sizes: List[int], process_group=None) -> List[ShardMetadata]:
        # check if shards form a valid tensor
        check_tensor(self.shards, tensor_sizes)
        return self.shards
