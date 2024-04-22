from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, Sequence, Any
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed.checkpoint.stateful import StatefulT

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)

__all__ = [
    "ChunkStorageMetadata",
    "TensorStorageMetadata",
    "BytesStorageMetadata",
    "Metadata",
    "MetadataIndex",
]


@dataclass
class ChunkStorageMetadata:
    """Each chunk is expected to have the same properties of the TensorStorageMetadata that includes it."""

    offsets: torch.Size
    sizes: torch.Size


@dataclass
class TensorStorageMetadata:
    properties: TensorProperties
    size: torch.Size
    chunks: List[ChunkStorageMetadata]


@dataclass
class BytesStorageMetadata:
    pass


TENSOR_TYPE = Union[torch.Tensor, ShardedTensor]
STORAGE_TYPES = Union[TensorStorageMetadata, BytesStorageMetadata]
STATE_DICT_TYPE = Dict[str, Union[StatefulT, Any]]


@dataclass
class Metadata:
    # Keys are the same from the `state_dict` used.
    state_dict_metadata: Dict[str, STORAGE_TYPES]
    planner_data: Any = None
    storage_data: Any = None


@dataclass(frozen=True)
class MetadataIndex:
    """This class represents a lookup key for items in a state dict or Metadata."""

    fqn: str
    """Fully Qualified Name of the object"""

    offset: Optional[torch.Size] = None
    """If the object is a tensor, offset into the tensor we're looking for"""

    index: Optional[int] = field(hash=False, compare=False, default=None)
    """
    Index hint when searching for tensor chunk to speedup lookups (optional)

    A common representation of a sharded tensor is as a list of chunks so to
    find the index in such a list you need to linear search it.

    When constructing an instance of MetadataIndex that points to that list,
    one can provide the index as a hint and it will be probed first before
    the linear search and thus making it significantly faster.
    """

    def __init__(
        self,
        fqn: str,
        offset: Optional[Sequence[int]] = None,
        index: Optional[int] = None,
    ):
        # We must use object.__setattr__ due to frozen=True
        object.__setattr__(self, "fqn", fqn)
        object.__setattr__(self, "index", index)
        if offset is not None:
            object.__setattr__(self, "offset", torch.Size(offset))
