from dataclasses import dataclass
from typing import List

import torch
from torch.distributed._sharding_spec import ShardMetadata
from torch.distributed.remote_device import _remote_device


@dataclass
class Shard(object):
    """
    Container which holds the data for a shard as a Tensor and also
    the associated metadata for that shard.

    Args:
        tensor(torch.Tensor): local tensor for the shard.
        metadata(:class `torch.distributed._sharded_tensor.ShardMetadata`):
            metadata for this shard, including offsets, lengths and device placement.
    """
    __slots__ = ['tensor', 'metadata']
    tensor: torch.Tensor
    metadata: ShardMetadata

    def __post_init__(self):
        # verification between local tensor and metadata
        if list(self.tensor.size()) != self.metadata.shard_lengths:
            raise ValueError(
                "Shard tensor size does not match with metadata.shard_lengths! "
                f"Found shard tensor size: {list(self.tensor.size())}, "
                f"metadata.shard_lengths: {self.metadata.shard_lengths}, "
            )
        if self.metadata.placement.device() != self.tensor.device:
            raise ValueError(
                f"Local shard tensor device does not match with local Shard's placement! "
                f"Found local shard tensor device: {self.tensor.device}, "
                f"local shard metadata placement device: {self.metadata.placement.device()}"
            )

    @classmethod
    def from_tensor_and_offsets(cls, tensor: torch.Tensor, shard_offsets: List[int], rank: int):
        """
        Class method to create Shard from local tensor, shard_offsets, and rank

        Args:
            tensor(torch.Tensor): local tensor for the shard.
            shard_offsets(List[int]): list of integers specify the offset
                of this shard on each dimension.
            rank(int): specify the rank for this shard.
        """
        shard_lengths = list(tensor.size())
        placement = _remote_device(f"rank:{rank}/{str(tensor.device)}")
        shard_meta = ShardMetadata(
            shard_offsets=shard_offsets,
            shard_lengths=shard_lengths,
            placement=placement
        )
        return Shard(tensor, shard_meta)
