import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)

TENSOR_TYPE = Union[torch.Tensor, ShardedTensor]

@dataclass
class ShardStorageMetadata:
    shard_metadata: ShardMetadata
    # storage key used for this particular Shard
    storage_key: str
    # Length in bytes for this shard
    length: int


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
