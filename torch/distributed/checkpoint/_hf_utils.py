import io
import json
import math
import os
import struct
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.distributed.checkpoint.filesystem import FileSystem, FileSystemBase
    from torch.distributed.checkpoint.fsspec_filesystem import _FileSystem

_Index = int | slice | EllipsisType


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

FORMAT_KEY = "format"
FORMAT_VALUE = "pt"

NUM_BYTES_FOR_HEADER_LEN = 8

SHARDED_DIR_NAME = "sharded"


@dataclass
class _HFStorageInfo:
    """This is the per entry storage info."""

    relative_path: str
    shape: torch.Size
    dtype: torch.dtype


def _gen_file_name(
    index: int, largest_index: int, shard_index: int | None = None
) -> str:
    if shard_index is not None:
        return (
            SHARDED_FILE_NAME.format(
                shard_idx=f"{shard_index}".zfill(5),
                cpt_idx=f"{index}".zfill(5),
                num_files=f"{largest_index}".zfill(5),
            )
            + SUFFIX
        )
    else:
        return (
            FILE_NAME.format(
                cpt_idx=f"{index}".zfill(5), num_files=f"{largest_index}".zfill(5)
            )
            + SUFFIX
        )


def _get_safetensors_file_metadata(file_bytes: io.IOBase) -> tuple[Any, int]:
    # this uses the same logic that's done in HF code base
    # https://github.com/2404589803/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L5308
    # and follows their documentation on how their files are serialized
    # https://huggingface.co/docs/safetensors/index#format

    header_len_bytes = file_bytes.read(NUM_BYTES_FOR_HEADER_LEN)
    header_len = struct.unpack("<Q", header_len_bytes)[0]
    header_json = file_bytes.read(header_len)
    metadata = json.loads(header_json)
    return (metadata, header_len + NUM_BYTES_FOR_HEADER_LEN)


def _get_fs_and_path(
    path: str | os.PathLike,
) -> tuple["FileSystem | _FileSystem", str | os.PathLike]:
    """Return the DCP filesystem for ``path`` (fsspec for URIs) and the initialized path."""
    from torch.distributed.checkpoint.filesystem import FileSystem
    from torch.distributed.checkpoint.fsspec_filesystem import _FileSystem

    if "://" not in str(path):
        fs = FileSystem()
        return fs, fs.init_path(path)
    remote_fs = _FileSystem()
    # Other ranks and processes write the files listed and opened here, and
    # fsspec's listing cache only sees writes made through the same instance.
    return remote_fs, remote_fs.init_path(path, use_listings_cache=False)


class _FsspecSafetensorsSlice:
    def __init__(self, file: "_FsspecSafetensorsFile", name: str) -> None:
        self._file = file
        self._info = file._header[name]

    def get_shape(self) -> list[int]:
        return list(self._info[SHAPE_KEY])

    def get_dtype(self) -> str:
        return self._info[DTYPE_KEY]

    def __getitem__(self, index: _Index | tuple[_Index, ...]) -> torch.Tensor:
        from safetensors.torch import _getdtype  # type: ignore[import]

        shape = self.get_shape()
        dtype = _getdtype(self.get_dtype())
        begin, end = self._info[DATA_OFFSETS_KEY]
        if not shape:
            return self._file._read_tensor(begin, end - begin, dtype, [])[index]
        index = index if isinstance(index, tuple) else (index,)
        if any(i is Ellipsis for i in index):
            e = next(n for n, i in enumerate(index) if i is Ellipsis)
            fill = (slice(None),) * (len(shape) - len(index) + 1)
            index = index[:e] + fill + index[e + 1 :]
        first, rest = (index[0], index[1:]) if index else (slice(None), ())
        if isinstance(first, int):
            # range() normalizes negative indices and raises IndexError like torch.
            start = range(shape[0])[first]
            stop, first = start + 1, 0
        elif isinstance(first, slice):
            start, stop, step = first.indices(shape[0])
            stop, first = max(stop, start), slice(None, None, step)
        else:
            raise TypeError(f"Unsupported index: {index!r}")
        # Read only the dim-0 rows covering the request, then index them in memory.
        row_nbytes = math.prod(shape[1:]) * dtype.itemsize
        rows = self._file._read_tensor(
            begin + start * row_nbytes,
            (stop - start) * row_nbytes,
            dtype,
            [stop - start, *shape[1:]],
        )
        return rows[(first, *rest)]


class _FsspecSafetensorsFile:
    """The ``safetensors.safe_open`` API that DCP uses, over an fsspec stream."""

    def __init__(self, stream: io.IOBase) -> None:
        self._stream = stream
        header, self._data_start = _get_safetensors_file_metadata(stream)
        self._metadata = header.pop(DEFAULT_EXTRA_METADATA_KEY, None)
        self._header = header

    def keys(self) -> list[str]:
        return sorted(self._header)

    def metadata(self) -> dict[str, str] | None:
        return self._metadata

    def get_slice(self, name: str) -> _FsspecSafetensorsSlice:
        return _FsspecSafetensorsSlice(self, name)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.get_slice(name)[()]

    def _read_tensor(
        self, offset: int, nbytes: int, dtype: torch.dtype, shape: list[int]
    ) -> torch.Tensor:
        if nbytes == 0:
            return torch.empty(shape, dtype=dtype)
        self._stream.seek(self._data_start + offset)
        data = bytearray(self._stream.read(nbytes))
        return torch.frombuffer(data, dtype=dtype).view(shape)


@contextmanager
def _open_safetensors(
    fs: "FileSystemBase", path: str | os.PathLike
) -> Generator[Any, None, None]:
    from torch.distributed.checkpoint.filesystem import FileSystem

    if isinstance(fs, FileSystem):
        from safetensors import safe_open  # type: ignore[import]

        with safe_open(path, framework="pt") as f:
            yield f
    else:
        with fs.create_stream(path, "rb") as stream:
            yield _FsspecSafetensorsFile(stream)


def _get_dtype(dtype_str: str) -> torch.dtype:
    try:
        dtype = DTYPE_MAP[dtype_str]
    except KeyError:
        dtype = torch.get_default_dtype()

    return dtype


def _get_dcp_custom_metadata(metadata: Any) -> Any | None:
    if DEFAULT_EXTRA_METADATA_KEY in metadata:
        custom_metadata = metadata[DEFAULT_EXTRA_METADATA_KEY]
        if CUSTOM_METADATA_KEY in custom_metadata:
            return json.loads(custom_metadata[CUSTOM_METADATA_KEY])
    return None
