from typing import Any, Dict, List, Union, Optional
import torch
from dataclasses import dataclass
import pickle
from torch.distributed._sharded_tensor import ShardedTensor, ShardedTensorMetadata, ShardMetadata

TENSOR_TYPE = Union[torch.Tensor, ShardedTensor]

@dataclass
class StorageMetadata:
    shard_metadata: Optional[ShardMetadata]
    # Unique identifier for this particular entity (Tensor or Shard of ShardedTensor)
    storage_key: str
    length: int
    offset: int

# Metadata for each param.
@dataclass
class ExtendedTensorMetadata:
    # Details of Tensor/ShardedTensor (dtype, shape, sharding config etc.)
    tensor_metadata: ShardedTensorMetadata
    storage_metadata: List[StorageMetadata]

@dataclass
class Metadata:
    # Metadata for the state dict.
    state_dict_metadata: Dict[str, ExtendedTensorMetadata]

    def __getstate__(self) -> Dict[str, Any]:
        serialized = pickle.dumps(self.state_dict_metadata)
        return serialized

    def __setstate__(self, state: Dict[str, Any]):
        self.state_dict_metadata = pickle.loads(state)

@dataclass
class ReadWriteRequest:
    # Tensor to read and write
    # TODO: replace this with a buffer
    target_tensor: torch.Tensor
    # The storage key for read write.
    storage_key: str
    # offset and length to read/write w.r.t. to the storage identified by ``storage_key``
    offset: int
    length: int
