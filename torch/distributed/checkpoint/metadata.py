# mypy: allow-untyped-defs
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

import torch
from torch.distributed.checkpoint.stateful import StatefulT


__all__ = [
    "ChunkStorageMetadata",
    "TensorStorageMetadata",
    "BytesStorageMetadata",
    "Metadata",
    "MetadataIndex",
    "TensorProperties",
    "StorageMeta",
]


@dataclass
class ChunkStorageMetadata:
    """
    Each chunk is expected to have the same properties of the TensorStorageMetadata
    that includes it.
    """

    offsets: torch.Size
    sizes: torch.Size


class _MEM_FORMAT_ENCODING(Enum):
    """Describe the memory format of a tensor."""

    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2


@dataclass
class TensorProperties:
    """Properties used to create :class:`Tensor`"""

    # Regular tensor fields
    dtype: torch.dtype = field(default_factory=torch.get_default_dtype)
    # This field is deprecated.
    layout: torch.layout = field(default=torch.strided)
    # This field is deprecated.
    requires_grad: bool = False
    # This field is deprecated.
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    # This field is deprecated.
    pin_memory: bool = False

    def __getstate__(self):
        # Since torch.memory_format cannot be pickled!
        memory_format = self.memory_format
        if memory_format == torch.contiguous_format:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        elif memory_format == torch.channels_last:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        elif memory_format == torch.preserve_format:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
        else:
            raise RuntimeError(f"Invalid torch.memory_format: {memory_format}")

        return (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        )

    def __setstate__(
        self,
        state,
    ):
        (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        ) = state

        if mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
            memory_format = torch.contiguous_format
        elif mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
            memory_format = torch.channels_last
        elif mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
            memory_format = torch.preserve_format
        else:
            raise RuntimeError(
                f"Invalid torch.memory_format encoding: {mem_format_encoding}"
            )

        self.memory_format = memory_format

    @staticmethod
    def create_from_tensor(tensor: torch.Tensor) -> "TensorProperties":
        return TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        )


@dataclass
class TensorStorageMetadata:
    properties: TensorProperties
    size: torch.Size
    chunks: list[ChunkStorageMetadata]


@dataclass
class BytesStorageMetadata:
    pass


STORAGE_TYPES = Union[TensorStorageMetadata, BytesStorageMetadata]
STATE_DICT_TYPE = dict[str, StatefulT | Any]


@dataclass
class StorageMeta:
    checkpoint_id: str | os.PathLike | None = None
    save_id: str | None = None
    load_id: str | None = None
    modules: list[str] = field(default_factory=list)


@dataclass
class Metadata:
    """This class represents the metadata of the checkpoint."""

    # Keys are the same from the `state_dict` used.
    state_dict_metadata: dict[str, STORAGE_TYPES]
    # It is the responsibility of the planner and storage plugins to ensure
    # backward compatibility of the planner_data and storage_data. DCP will
    # also ensure the backward compatibility of the metadata in this file and
    # the metadata of the built-in planner and storage plugins.
    planner_data: Any = None
    storage_data: Any = None
    storage_meta: StorageMeta | None = None
    version: str | None = None


@dataclass(frozen=True)
class MetadataIndex:
    """This class represents a lookup key for items in a state dict or Metadata."""

    fqn: str
    """Fully Qualified Name of the object"""

    offset: torch.Size | None = None
    """If the object is a tensor, offset into the tensor we're looking for"""

    index: int | None = field(hash=False, compare=False, default=None)
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
        offset: Sequence[int] | None = None,
        index: int | None = None,
    ):
        # We must use object.__setattr__ due to frozen=True
        object.__setattr__(self, "fqn", fqn)
        object.__setattr__(self, "index", index)
        if offset is not None:
            object.__setattr__(self, "offset", torch.Size(offset))
