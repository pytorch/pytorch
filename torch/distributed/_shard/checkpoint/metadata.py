import io
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional, Sequence, Any

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)

TENSOR_TYPE = Union[torch.Tensor, ShardedTensor]
STATE_DICT_TYPE = Dict[str, Any]

@dataclass
class ShardStorageMetadata:
    shard_metadata: ShardMetadata
    # storage key used for this particular Shard
    storage_key: str


# Metadata for each param.
@dataclass
class ShardedTensorStorageMetadata:
    # Metadata for the sharded tensor itself
    tensor_metadata: ShardedTensorMetadata

    # Storage info for each Shard. There's no ordering requirement for this list.
    storage_metadata: List[ShardStorageMetadata]


@dataclass
class TensorStorageMetadata:
    # Storage key used for this tensor
    storage_key: str

    # Tensor sizes
    size: torch.Size

@dataclass
class BytesStorageMetadata:
    # Storage key used for this tensor
    storage_key: str

    # serialized payload size
    length: int

STORAGE_TYPES = Union[ShardedTensorStorageMetadata, TensorStorageMetadata, BytesStorageMetadata]

@dataclass
class Metadata:
    # Keys are the same from the `state_dict` used.
    state_dict_metadata: Dict[str, STORAGE_TYPES]

@dataclass
class BytesWriteRequest:
    bytes: io.BytesIO
    storage_key: str


@dataclass
class BytesReadRequest:
    bytes: io.BytesIO
    storage_key: str
    fqn: str


@dataclass
class TensorWriteRequest:
    tensor: torch.Tensor
    storage_key: str


@dataclass
class TensorReadRequest:
    tensor: torch.Tensor
    storage_key: str
    # offset and length w.r.t. to the storage identified by ``storage_key``
    offsets: Tuple[int, ...]
    lengths: Tuple[int, ...]


@dataclass(frozen=True)
class MetadataIndex:
    """
    This class represents a lookup key for items in a state dict or Metadata.
    """
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

    def __init__(self, fqn: str, offset: Optional[Sequence[int]] = None, index: Optional[int] = None):
        # We must use object.__setattr__ due to frozen=True
        object.__setattr__(self, "fqn", fqn)
        object.__setattr__(self, "index", index)
        if offset is not None:
            object.__setattr__(self, "offset", torch.Size(offset))
