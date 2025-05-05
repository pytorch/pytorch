from dataclasses import dataclass
from typing import Dict, List, Optional, TypeAlias
import torch


@dataclass
class Shard:
    shard_offsets: List[int]
    shard_lengths: List[int]


@dataclass
class TensorMetadata:
    """
    Dataclass which holds information about a tensor.
    """

    dtype: torch.dtype
    device: torch.types.Device
    sizes: list[int]
    shards: list[Shard]


@dataclass
class Param:
    fqn: str
    type_name: str  # type(obj).__name__

    tensor_metadata: Optional[TensorMetadata] = (
        None  # this is populated only for tensors
    )


MANIFEST: TypeAlias = Optional[Dict[str, List[Param]]]


@dataclass
class Metadata:
    """
    Manifest for a checkpoint with FQNs.
    """

    manifest: MANIFEST
    dcp_checkpointer_version: int = 1
