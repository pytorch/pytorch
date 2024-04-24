from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from typing import Callable, Dict, List, TYPE_CHECKING

import torch

from ._internals import (
    check_tensor,
    get_chunked_dim_size,
    get_split_size,
    validate_non_overlapping_shards_metadata
)
from torch.distributed._shard.metadata import ShardMetadata

import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.op_registry_utils import _decorator_func

if TYPE_CHECKING:
    # Only include ShardedTensor when do type checking, exclude it
    # from run-time to resolve circular dependency.
    from torch.distributed._shard.sharded_tensor import ShardedTensor

class PlacementSpec(ABC):  # noqa: B024
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

class ShardingSpec(ABC):
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

# Ops customized for a particular ShardingSpec.
_CUSTOM_SHARDING_SPEC_OPS: Dict[str, Dict[Callable, Callable]] = {}

def _has_custom_op(sharding_spec, op):
    """
    Returns whether or not the ShardingSpec has a custom op implementation.
    """
    class_name = type(sharding_spec).__qualname__
    return class_name in _CUSTOM_SHARDING_SPEC_OPS and op in _CUSTOM_SHARDING_SPEC_OPS[class_name]

def _dispatch_custom_op(sharding_spec, op: Callable, types, args, kwargs, process_group):
    """
    Calls the custom op for this ShardingSpec if it exists.
    """
    class_name = type(sharding_spec).__qualname__
    if not _has_custom_op(sharding_spec, op):
        raise RuntimeError(f'Custom op: {op} not registered for {class_name}')
    func = _CUSTOM_SHARDING_SPEC_OPS[class_name][op]
    return func(types, args, kwargs, process_group)

def custom_sharding_spec_op(sharding_spec_class, func):
    """
    Decorator to allow custom registration of ops.
    Args:
        sharding_spec_class(type): The ShardingSpec for which we need to add this custom op.
        func(Callable): The op to override (ex: torch.bmm)
    """
    class_name = sharding_spec_class.__qualname__
    if class_name not in _CUSTOM_SHARDING_SPEC_OPS:
        _CUSTOM_SHARDING_SPEC_OPS[class_name] = {}
    return functools.partial(
        _decorator_func,
        op=func,
        op_table=_CUSTOM_SHARDING_SPEC_OPS[class_name]
    )


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
    If the tensor is sharded only on one dimension, we can then verify whether it's
    a ChunkShardingSpec or not. The way to verify it is to first get the total length
    and perform a chunk sharding with the given placements to see if we can have the
    same chunk size as the given shards_metadata. If not, we assume it's enum sharded.

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
    shard_offset_list = []
    # collect local shard metadatas from the global sharded_tensor_metadata
    for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
        placements.append(shard_metadata.placement)
        local_offsets = shard_metadata.shard_offsets
        chunk_offset_list.append(sum(local_offsets))
        shard_size_list.append(shard_metadata.shard_sizes)
        shard_offset_list.append(shard_metadata.shard_offsets)
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

        from .chunk_sharding_spec import ChunkShardingSpec
        chunk_spec = ChunkShardingSpec(
            dim=chunk_sharding_dim,
            placements=placements,
        )

        shard_sizes = sorted([x[chunk_sharding_dim] for x in shard_size_list])
        shard_total_length = sum(shard_sizes)
        shard_offsets = sorted([x[chunk_sharding_dim] for x in shard_offset_list])

        chunks = len(placements)
        split_size = get_split_size(shard_total_length, chunks)
        chunk_shard_sizes = sorted(
            [
                get_chunked_dim_size(shard_total_length, split_size, idx)
                for idx in range(chunks)
            ]
        )
        # Should match ChunkShardingSpec offsets calculation
        chunk_shard_offsets = [split_size * idx for idx in range(chunks)]
        if shard_sizes == chunk_shard_sizes and shard_offsets == chunk_shard_offsets:
            return chunk_spec
    return EnumerableShardingSpec(shards_metadata)
