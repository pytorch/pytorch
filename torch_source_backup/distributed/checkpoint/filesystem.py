# mypy: allow-untyped-defs
import collections
import dataclasses
import io
import json
import operator
import os
import pickle
import queue
import threading
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from io import UnsupportedOperation
from pathlib import Path
from typing import Any, Callable, cast, IO, Optional, Union

# introduced as collections.abc.Buffer in Python 3.12
from typing_extensions import Buffer

import torch
from torch import Tensor
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint._extension import (
    ExtensionRegistry,
    StreamTransformExtension,
)
from torch.distributed.checkpoint._hf_utils import (
    CUSTOM_METADATA_KEY,
    DCP_VERSION_KEY,
    FORMAT_KEY,
    FORMAT_VALUE,
    HF_DCP_VERSION,
)
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE, StorageMeta
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.staging import BlockingAsyncStager
from torch.distributed.checkpoint.storage import (
    StorageReader,
    StorageWriter,
    WriteResult,
)
from torch.distributed.checkpoint.utils import _create_file_view
from torch.futures import Future


__all__ = [
    "FileSystemWriter",
    "FileSystemReader",
    "FileSystem",
    "FileSystemBase",
    "SerializationFormat",
]

_metadata_fn: str = ".metadata"


@dataclass
class _StorageInfo:
    """This is the per entry storage info."""

    relative_path: str
    offset: int
    length: int
    transform_descriptors: Optional[Sequence[str]] = None

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class _StoragePrefix:
    prefix: str


class SerializationFormat(Enum):
    TORCH_SAVE = "torch_save"
    SAFETENSORS = "safetensors"


DEFAULT_SUFFIX = ".distcp"


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class _TensorLoader(ABC):
    @abstractmethod
    def add(self, size: int, obj: object) -> None:
        pass

    @abstractmethod
    def start_loading(self) -> None:
        pass

    @abstractmethod
    def values(self) -> Iterator[tuple[torch.Tensor, object]]:
        pass


class _SerialCpuLoader(_TensorLoader):
    def __init__(self, resolve_fun: Callable) -> None:
        self.resolve_fun = resolve_fun
        self.items: list[tuple[int, object]] = []

    def add(self, size: int, obj: object) -> None:
        self.items.append((size, obj))

    def start_loading(self) -> None:
        pass

    def values(self) -> Iterator[tuple[torch.Tensor, object]]:
        for _, obj in self.items:
            tensor = self.resolve_fun(obj).detach()
            tensor = tensor.cpu()
            if tensor.storage().size() != tensor.numel():
                tensor = tensor.clone()
            yield (
                tensor,
                obj,
            )


class _OverlappingCpuLoader(_TensorLoader):
    def __init__(
        self,
        resolve_fun: Callable,
        stream: Optional[torch.Stream] = None,
        inflight_threshhold: int = 1_000_000,
    ) -> None:
        self.resolve_fun = resolve_fun
        self.items: list[tuple[int, object]] = []
        self.inflight_threshhold = inflight_threshhold
        self.in_flight_data = 0
        self.current_items: collections.deque = collections.deque()
        self.idx = 0
        self.started = False
        self.device_type = (
            stream.device_type if stream else _get_available_device_type()
        )
        self.device_module = _get_device_module(self.device_type)
        self.stream = cast(
            torch.cuda.Stream, stream or self.device_module.current_stream()
        )
        if self.stream != self.device_module.current_stream():
            self.stream.wait_stream(self.device_module.current_stream())

    @property
    def _done(self) -> bool:
        return self.idx >= len(self.items)

    def _drain(self) -> list[tuple[torch.Tensor, object]]:
        drained = []
        if self.in_flight_data >= self.inflight_threshhold:
            self.stream.synchronize()
        while self.in_flight_data >= self.inflight_threshhold:
            val = self.current_items.popleft()
            self.in_flight_data -= val[0].numel() * val[0].element_size()
            drained.append(val)
        return drained

    def _refill(self) -> None:
        with self.device_module.stream(self.stream):
            while not self._done and self.in_flight_data < self.inflight_threshhold:
                _, obj = self.items[self.idx]
                self.idx += 1
                tensor = self.resolve_fun(obj).detach()
                if tensor.device.type == self.device_type:
                    tensor = tensor.to(device="cpu", non_blocking=True)
                elif tensor.device == torch.device("cpu"):
                    if (
                        tensor.untyped_storage().size()
                        != tensor.numel() * tensor.itemsize
                    ):
                        # this forces the tensor to be both contiguous and with minimal storage
                        tensor = tensor.clone()

                self.current_items.append(
                    (
                        tensor,
                        obj,
                    )
                )
                self.in_flight_data += tensor.numel() * tensor.element_size()

    def _finish(self) -> Iterable[tuple[torch.Tensor, object]]:
        assert self._done
        if len(self.current_items) > 0:
            self.stream.synchronize()
        return self.current_items

    def add(self, size: int, obj: object) -> None:
        if self.started:
            raise RuntimeError("cannot add items after loading started")
        self.items.append((size, obj))

    def start_loading(self) -> None:
        if self.started:
            return
        self.started = True
        self.items.sort(key=operator.itemgetter(0))
        self._refill()

    def values(self) -> Iterator[tuple[torch.Tensor, object]]:
        self.start_loading()
        while not self._done:
            drained = self._drain()
            self._refill()
            yield from drained

        yield from self._finish()


class _StorageWriterTransforms:
    """
    This is experimental, and will likely move elsewhere in the
    future.  It lives here to minimize changes while we are still
    learning and gathering feedback.
    """

    def __init__(
        self, extensions: Optional[Sequence[StreamTransformExtension]] = None
    ) -> None:
        """
        If the extensions arg is None, this means the implementation
        should provide whatever defaults it chooses.  An empty
        sequence indicates no extensions should be used.  At this
        time, the default extensions sequence is empty.
        """
        self.extensions = () if extensions is None else extensions

    def transform_save_stream(
        self, write_item: WriteItem, raw_stream: io.IOBase
    ) -> tuple[IO[bytes], list[str]]:
        # In order to avoid leaking fds, transformers' close must
        # cascade to wrapped streams, but since this function can
        # append to the raw stream, we can't close the actual stream.
        # So, we use this to put a wrapper around the raw stream's
        # close() to make it a noop, and it gets closed once all files
        # are appended.

        class NoCloseWriter(io.IOBase):
            def __init__(self, raw: io.IOBase):
                self.raw = raw

            def writeable(self) -> bool:
                return True

            def write(self, b: Buffer) -> int:
                return self.raw.write(b)

            def close(self):
                self.flush()
                self.raw.flush()
                # but not close.

        transform_to = cast(IO[bytes], NoCloseWriter(raw_stream))

        for ex in self.extensions:
            transform_to = ex.transform_to(transform_to)

        return (transform_to, [ex.get_descriptor() for ex in reversed(self.extensions)])


def _item_size(item: WriteItem) -> int:
    size = 1
    assert item.tensor_data is not None
    # can't use math.prod as PT needs to support older python
    for s in item.tensor_data.size:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _split_by_size_and_type(bins: int, items: list[WriteItem]) -> list[list[WriteItem]]:
    if bins == 1:
        return [items]

    bytes_w = [wi for wi in items if wi.type == WriteItemType.BYTE_IO]
    tensor_w = [wi for wi in items if wi.type != WriteItemType.BYTE_IO]

    buckets: list[list[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    tensor_w.sort(key=_item_size, reverse=True)

    for i, wi in enumerate(bytes_w):
        buckets[i % bins].append(wi)

    for wi in tensor_w:
        # TODO replace with headq
        idx = min(enumerate(bucket_sizes), key=operator.itemgetter(1))[0]
        buckets[idx].append(wi)
        bucket_sizes[idx] += _item_size(wi)

    return buckets


def _write_item(
    transforms: _StorageWriterTransforms,
    stream: io.IOBase,
    data: Union[io.BytesIO, torch.Tensor],
    write_item: WriteItem,
    storage_key: str,
    serialization_format: SerializationFormat,
) -> WriteResult:
    offset = stream.tell()

    (transform_to, transform_descriptors) = transforms.transform_save_stream(
        write_item, stream
    )

    if write_item.type == WriteItemType.BYTE_IO:
        assert isinstance(data, io.BytesIO)
        transform_to.write(data.getbuffer())
    else:
        assert isinstance(data, torch.Tensor)
        assert data.device == torch.device("cpu")
        if serialization_format == SerializationFormat.TORCH_SAVE:
            torch.save(data, transform_to)

    transform_to.close()

    if serialization_format == SerializationFormat.TORCH_SAVE or isinstance(
        data, io.BytesIO
    ):
        length = stream.tell() - offset
    else:
        length = data.numel() * data.element_size()

    # For consistency with earlier versions, leave this field out of the
    # metadata if there are no extensions.
    info_transform_descriptors = (
        None if len(transform_descriptors) == 0 else transform_descriptors
    )

    return WriteResult(
        index=write_item.index,
        size_in_bytes=length,
        storage_data=_StorageInfo(
            storage_key,
            offset,
            length,
            transform_descriptors=info_transform_descriptors,
        ),
    )


def _write_files_from_queue(
    create_stream: Callable,
    file_queue: queue.Queue,
    result_queue: queue.Queue,
    planner: SavePlanner,
    transforms: _StorageWriterTransforms,
    inflight_threshhold: int,
    use_fsync: bool,
    thread_count: int,
    serialization_format: SerializationFormat,
) -> None:
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: _TensorLoader

            custom_backend_name = torch._C._get_privateuse1_backend_name()
            custom_device_mod = getattr(torch, custom_backend_name, None)

            # TODO: Using the OverlappingCpuLoader with multiple threads creates significant
            # performance degradation, observed as being related to cuda stream syncs. We
            # should try to fix this and use _OverlappingCpuLoader for all threaded cases
            if (
                thread_count == 1
                and (
                    torch.cuda.is_available()
                    or (custom_device_mod and custom_device_mod.is_available())
                )
                and inflight_threshhold > 0
            ):
                loader = _OverlappingCpuLoader(
                    planner.resolve_data,
                    inflight_threshhold=inflight_threshhold,
                )
            else:
                loader = _SerialCpuLoader(
                    planner.resolve_data,
                )

            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            loader.start_loading()

            bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            write_results = []

            with create_stream(file_name, "wb") as stream:
                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(
                        _write_item(
                            transforms,
                            stream,
                            data,
                            write_item,
                            storage_key,
                            serialization_format,
                        )
                    )

                tensor_dict = {}
                metadata_dict = {}
                for tensor, write_item in loader.values():
                    assert tensor.is_cpu
                    write_results.append(
                        _write_item(
                            transforms,
                            stream,
                            tensor,
                            write_item,  # type: ignore[arg-type]
                            storage_key,
                            serialization_format,
                        )
                    )
                    tensor_dict[write_item.index.fqn] = tensor  # type: ignore[attr-defined]
                    metadata_dict[write_item.index.fqn] = {  # type: ignore[attr-defined]
                        "saved_offsets": write_item.tensor_data.chunk.offsets  # type: ignore[attr-defined]
                    }

                if serialization_format == SerializationFormat.SAFETENSORS:
                    from safetensors.torch import save  # type: ignore[import-not-found]

                    stream.write(
                        save(
                            tensor_dict,
                            metadata={
                                CUSTOM_METADATA_KEY: json.dumps(metadata_dict),
                                DCP_VERSION_KEY: str(HF_DCP_VERSION),
                                FORMAT_KEY: FORMAT_VALUE,
                            },
                        )
                    )

                if use_fsync:
                    try:
                        os.fsync(stream.fileno())
                    except (AttributeError, UnsupportedOperation):
                        os.sync()
                stream.close()
            result_queue.put(write_results)
    except queue.Empty:
        pass


class FileSystemBase(ABC):
    @contextmanager
    @abstractmethod
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]: ...

    @abstractmethod
    def concat_path(
        self, path: Union[str, os.PathLike], suffix: str
    ) -> Union[str, os.PathLike]: ...

    @abstractmethod
    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None: ...

    @abstractmethod
    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]: ...

    @abstractmethod
    def mkdir(self, path: Union[str, os.PathLike]) -> None: ...

    @classmethod
    @abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool: ...

    @abstractmethod
    def exists(self, path: Union[str, os.PathLike]) -> bool: ...

    @abstractmethod
    def rm_file(self, path: Union[str, os.PathLike]) -> None: ...


class FileSystem(FileSystemBase):
    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        if not isinstance(path, Path):
            path = Path(path)
        with path.open(mode) as stream:
            yield cast(io.IOBase, stream)

    def concat_path(
        self, path: Union[str, os.PathLike], suffix: str
    ) -> Union[str, os.PathLike]:
        if not isinstance(path, Path):
            path = Path(path)
        return path / suffix

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        if not isinstance(path, Path):
            path = Path(path)
        return path

    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        if not isinstance(path, Path):
            path = Path(path)

        path.rename(cast(Path, new_path))

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        if isinstance(checkpoint_id, Path):
            return True

        if "://" in str(checkpoint_id):
            return False

        for p in Path(checkpoint_id).parents:
            if p.exists() and os.access(str(p), os.W_OK):
                return True

        return False

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        if not isinstance(path, Path):
            path = Path(path)
        return path.exists()

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        path.unlink()

    def ls(self, path: Union[str, os.PathLike]) -> list[str]:
        if not isinstance(path, Path):
            path = Path(path)
        return [str(p) for p in path.iterdir()]


class _FileSystemWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        overwrite: bool = True,
        _extensions: Optional[Sequence[StreamTransformExtension]] = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__()
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files
        self.thread_count = thread_count
        self.per_thread_copy_ahead = per_thread_copy_ahead
        self.save_id = _generate_uuid()
        self.overwrite = overwrite
        self.transforms = _StorageWriterTransforms(_extensions)
        self.serialization_format = serialization_format

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.save_id = _generate_uuid()

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        pass

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self.fs.mkdir(self.path)
        if self.fs.exists(self.metadata_path):
            if self.overwrite:
                warnings.warn(
                    f"Detected an existing checkpoint in {self.metadata_path}, overwriting since {self.overwrite=}."
                    " Past version 2.5 of PyTorch, `overwrite` will default to False. Set this variable to True to"
                    " maintain this functionality or False to raise when an existing checkpoint is found."
                )
            else:
                raise RuntimeError(f"Checkpoint already exists and {self.overwrite=}.")

        return plan

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_"))
            for i, plan in enumerate(plans)
        ]
        return new_plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        def gen_file():
            nonlocal file_count
            file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        file_queue: queue.Queue = queue.Queue()
        if self.single_file_per_rank:
            for bucket in _split_by_size_and_type(self.thread_count, plan.items):
                file_name = gen_file()
                path = self.fs.concat_path(self.path, file_name)
                file_queue.put((path, file_name, bucket))
        else:
            for item in plan.items:
                file_name = gen_file()
                path = self.fs.concat_path(self.path, file_name)
                file_queue.put((path, file_name, [item]))

        return self._write_data(planner, file_queue)

    def _write_data(
        self,
        planner: SavePlanner,
        file_queue: queue.Queue,
    ) -> Future[list[WriteResult]]:
        result_queue: queue.Queue = queue.Queue()

        threads = []
        for _ in range(1, self.thread_count):
            t = threading.Thread(
                target=_write_files_from_queue,
                args=(
                    self.fs.create_stream,
                    file_queue,
                    result_queue,
                    planner,
                    self.transforms,
                    self.per_thread_copy_ahead,
                    self.sync_files,
                    self.thread_count,
                    self.serialization_format,
                ),
            )
            t.start()
            threads.append(t)

        _write_files_from_queue(
            create_stream=self.fs.create_stream,
            file_queue=file_queue,
            result_queue=result_queue,
            planner=planner,
            transforms=self.transforms,
            inflight_threshhold=self.per_thread_copy_ahead,
            use_fsync=self.sync_files,
            thread_count=self.thread_count,
            serialization_format=self.serialization_format,
        )

        for t in threads:
            t.join()

        res = []
        try:
            while True:
                res += result_queue.get_nowait()
        except queue.Empty:
            fut: Future[list[WriteResult]] = Future()
            fut.set_result(res)
            return fut

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        storage_md = {}
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md

        metadata.storage_meta = self.storage_meta()

        tmp_path = cast(Path, self.fs.concat_path(self.path, f"{_metadata_fn}.tmp"))
        with self.fs.create_stream(tmp_path, "wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            if self.sync_files:
                try:
                    os.fsync(metadata_file.fileno())
                except (AttributeError, UnsupportedOperation):
                    os.sync()

        # delete in-case other checkpoints were present.
        if self.fs.exists(self.metadata_path):
            self.fs.rm_file(self.metadata_path)

        self.fs.rename(tmp_path, self.metadata_path)

    def storage_meta(self) -> Optional[StorageMeta]:
        return StorageMeta(checkpoint_id=self.checkpoint_id, save_id=self.save_id)

    @property
    def metadata_path(self) -> Union[str, os.PathLike]:
        return cast(Path, self.fs.concat_path(self.path, _metadata_fn))

    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        """
        return the checkpoint_id that will be used to save the checkpoint.
        """
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class _StorageReaderTransforms:
    """
    This is experimental, and will likely move elsewhere in the
    future.  It lives here to minimize changes while we are still
    learning and gathering feedback.
    """

    def __init__(self, extension_registry: Optional[ExtensionRegistry] = None) -> None:
        self.extension_registry = (
            ExtensionRegistry() if extension_registry is None else extension_registry
        )

    def transform_load_stream(
        self,
        read_item: ReadItem,
        transform_descriptors: Sequence[str],
        raw_stream: IO[bytes],
    ) -> IO[bytes]:
        extensions = self.extension_registry.from_descriptor_list(transform_descriptors)
        transform_from = raw_stream
        for ex in extensions:
            if isinstance(ex, StreamTransformExtension):
                transform_from = ex.transform_from(transform_from)
        return transform_from


class FileSystemReader(StorageReader):
    def __init__(
        self,
        path: Union[str, os.PathLike],
        _extension_registry: Optional[ExtensionRegistry] = None,  # EXPERIMENTAL
    ) -> None:
        super().__init__()
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)
        self.storage_data: dict[Any, Any] = {}
        self.load_id = _generate_uuid()
        self.transforms = _StorageReaderTransforms(_extension_registry)

    def _slice_file(self, file, sinfo: _StorageInfo) -> IO[bytes]:
        return cast(IO[bytes], _create_file_view(file, sinfo.offset, sinfo.length))

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        self.storage_data = {}
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.load_id = _generate_uuid()

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # group requests by file
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md: _StorageInfo = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    transform_from = self.transforms.transform_load_stream(
                        req,
                        # This field wasn't present in older
                        # implementations so provide a fallback.
                        item_md.transform_descriptors or (),
                        file_slice,
                    )

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                    else:
                        if transform_from.seekable():
                            seekable = transform_from
                        else:
                            # torch.load requires a seekable input, so read the transform
                            # stream now and store the output if needed
                            seekable = io.BytesIO(transform_from.read(-1))
                            seekable.seek(0)

                        tensor = cast(
                            Tensor,
                            torch.load(
                                seekable,
                                map_location="cpu",
                                weights_only=True,
                            ),
                        )
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
                        target_tensor = planner.resolve_tensor(req).detach()

                        assert target_tensor.size() == tensor.size(), (
                            f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                        )
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # Implementing the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = pickle.load(metadata_file)

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id

        return metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        return plans

    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        """
        return the checkpoint_id that will be used to load the checkpoint.
        """
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class FileSystemWriter(_FileSystemWriter, BlockingAsyncStager):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        cache_staged_state_dict: bool = False,
        overwrite: bool = True,
        _extensions: Optional[Sequence[StreamTransformExtension]] = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.
            cache_staged_state_dict: Whether to cache the staged state_dict. This option decreases staging latency
                at the cost of increases memory usage. Additionally, if this parameter is set to True, it's the expectation
                that the stager is maintained and reused for multiple dcp.async_save calls. Default to False.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        _FileSystemWriter.__init__(
            self,
            path=path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            overwrite=overwrite,
            _extensions=_extensions,
            serialization_format=serialization_format,
        )
        BlockingAsyncStager.__init__(
            self,
            cache_staged_state_dict=cache_staged_state_dict,
        )

    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        """Override of AsyncStager.stage"""
        # in the async case, the state dict is already on CPU, so maintaining this
        # buffer makes no sense
        self.per_thread_copy_ahead = 0
        return super().stage(state_dict)
