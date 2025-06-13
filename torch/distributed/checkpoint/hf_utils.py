from dataclasses import dataclass

import torch

_metadata_fn: str = "model.safetensors.index.json"

FILE_NAME = "model-{cpt_idx}-of-{num_files}"
SHARDED_FILE_NAME = "shard-{shard_idx}-model-{cpt_idx}-of-{num_files}"
SUFFIX = ".safetensors"

# metadata keys
CUSTOM_METADATA_KEY = "DCP_SHARDING_INFO"
DEFAULT_EXTRA_METADATA_KEY = "__metadata__"
SAVED_OFFSETS_KEY = "saved_offsets"
SHAPE_KEY = "shape"
DATA_KEY = "data"
DTYPE_KEY = "dtype"
DATA_OFFSETS_KEY = "data_offsets"

DTYPE_MAP = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BF16": torch.bfloat16,
}

HF_DCP_VERSION: float = 1.0
DCP_VERSION_KEY = "DCP_VERSION"
DCP_SHARDING_INFO_KEY = "DCP_SHARDING_INFO"


@dataclass
class _StorageInfo:
    """This is the per entry storage info."""

    relative_path: str
    offset: int
    length: int
    shape: torch.Size
    dtype: torch.dtype

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}
