from dataclasses import dataclass

import torch
from torch.distributed._sharding_spec import ShardMetadata

@dataclass
class Shard(object):
    """
    Container which holds the data for a shard as a Tensor and also
    the associated metadata for that shard.
    """
    __slots__ = ['tensor', 'metadata']

    tensor: torch.Tensor
    metadata: ShardMetadata
