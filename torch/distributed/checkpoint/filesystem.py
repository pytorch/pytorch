# mypy: allow-untyped-defs
import collections
import dataclasses
import io
import operator
import os
import pickle
import queue
import threading
import uuid
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    IO,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from torch import Tensor
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed._shard._utils import narrow_tensor_by_index

from torch.distributed.checkpoint.metadata import (
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    StorageMeta,
)
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

__all__ = ["FileSystemWriter", "FileSystemReader", "FileSystem", "FileSystemBase"]

_metadata_fn: str = ".metadata"


@dataclass
class _StorageInfo:
    """This is the per entry storage info."""

    relative_path: str
    offset: int
    length: int


@dataclass
class _StoragePrefix:
    prefix: str


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
    def values(self) -> Iterator[Tuple[torch.Tensor, object]]:
        pass


class _SerialCpuLoader(_TensorLoader):
    def __init__(self, resolve_fun: Callable) -> None:
        self.resolve_fun = resolve_fun
        self.items: List[Tuple[int, object]] = []

    def add(self, size: int, obj: object) -> None:
        self.items.append((size, obj))

    def start_loading(self) -> None:
        pass

    def values(self) -> Iterator[Tuple[torch.Tensor, object]]:
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
        self.items: List[Tuple[int, object]] = []
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

    def _drain(self) -> List[Tuple[torch.Tensor, object]]:
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

    def _finish(self) -> Iterable[Tuple[torch.Tensor, object]]:
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

    def values(self) -> Iterator[Tuple[torch.Tensor, object]]:
        self.start_loading()
        while not self._done:
            drained = self._drain()
            self._refill()
            yield from drained

        yield from self._finish()


def _item_size(item: WriteItem) -> int:
    size = 1
    assert item.tensor_data is not None
    # can't use math.prod as PT needs to support older python
    for s in item.tensor_data.size:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _split_by_size_and_type(bins: int, items: List[WriteItem]) -> List[List[WriteItem]]:
    if bins == 1:
        return [items]

    bytes_w = [wi for wi in items if wi.type == WriteItemType.BYTE_IO]
    tensor_w = [wi for wi in items if wi.type != WriteItemType.BYTE_IO]

    buckets: List[List[WriteItem]] = [[] for _ in range(bins)]
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
    stream: io.IOBase,
    data: Union[io.BytesIO, torch.Tensor],
    write_item: WriteItem,
    storage_key: str,
) -> WriteResult:
    offset = stream.tell()

    if write_item.type == WriteItemType.BYTE_IO:
        assert isinstance(data, io.BytesIO)
        stream.write(data.getbuffer())
    else:
        assert isinstance(data, torch.Tensor)
        assert data.device == torch.device("cpu")
        torch.save(data, cast(IO[bytes], stream))
    length = stream.tell() - offset

    return WriteResult(
        index=write_item.index,
        size_in_bytes=length,
        storage_data=_StorageInfo(storage_key, offset, length),
    )


def _write_files_from_queue(
    create_stream: Callable,
    file_queue: queue.Queue,
    result_queue: queue.Queue,
    planner: SavePlanner,
    inflight_threshhold: int,
    use_fsync: bool,
    thread_count: int,
) -> None:
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: _TensorLoader

            custom_backend_name = torch._C._get_privateuse1_backend_name()
            custom_device_mod = getattr(torch, custom_backend_name, None)

            # TODO: Using the OverlappingCpuLoader with multiple threads creates significant
            # performance degredation, observed as being related to cuda stream syncs. We
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
                        _write_item(stream, data, write_item, storage_key)
                    )

                for tensor, write_item in loader.values():
                    assert tensor.is_cpu
                    write_results.append(
                        _write_item(stream, tensor, write_item, storage_key)
                    )

                if use_fsync:
                    try:
                        os.fsync(stream.fileno())
                    except AttributeError:
                        os.sync()
            result_queue.put(write_results)
    except queue.Empty:
        pass


class FileSystemBase(ABC):
    @contextmanager
    @abstractmethod
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        ...

    @abstractmethod
    def concat_path(
        self, path: Union[str, os.PathLike], suffix: str
    ) -> Union[str, os.PathLike]:
        ...

    @abstractmethod
    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        ...

    @abstractmethod
    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        ...

    @abstractmethod
    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        ...

    @classmethod
    @abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        ...

    @abstractmethod
    def exists(self, path: Union[str, os.PathLike]) -> bool:
        ...

    @abstractmethod
    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        ...


class FileSystem(FileSystemBase):
    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        with cast(Path, path).open(mode) as stream:
            yield cast(io.IOBase, stream)

    def concat_path(
        self, path: Union[str, os.PathLike], suffix: str
    ) -> Union[str, os.PathLike]:
        return cast(Path, path) / suffix

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        if not isinstance(path, Path):
            path = Path(path)
        return path

    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        cast(Path, path).rename(cast(Path, new_path))

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        cast(Path, path).mkdir(parents=True, exist_ok=True)

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
        return cast(Path, path).exists()

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        cast(Path, path).unlink()


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

    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_"))
            for i, plan in enumerate(plans)
        ]
        return new_plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[List[WriteResult]]:
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
                    self.per_thread_copy_ahead,
                    self.sync_files,
                    self.thread_count,
                ),
            )
            t.start()
            threads.append(t)

        _write_files_from_queue(
            create_stream=self.fs.create_stream,
            file_queue=file_queue,
            result_queue=result_queue,
            planner=planner,
            inflight_threshhold=self.per_thread_copy_ahead,
            use_fsync=self.sync_files,
            thread_count=self.thread_count,
        )

        for t in threads:
            t.join()

        res = []
        try:
            while True:
                res += result_queue.get_nowait()
        except queue.Empty:
            pass

            fut: Future[List[WriteResult]] = Future()
            fut.set_result(res)
            return fut

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        storage_md = dict()
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
                except AttributeError:
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


class FileSystemReader(StorageReader):
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__()
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()
        self.load_id = _generate_uuid()

    def _slice_file(self, file, sinfo: _StorageInfo) -> io.IOBase:
        return _create_file_view(file, sinfo.offset, sinfo.length)

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        self.storage_data = dict()
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.load_id = _generate_uuid()

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # group requests by file
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(file_slice.read(item_md.length))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                    else:
                        tensor = cast(
                            Tensor,
                            torch.load(
                                cast(IO[bytes], file_slice),
                                map_location="cpu",
                                weights_only=True,
                            ),
                        )
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
                        target_tensor = planner.resolve_tensor(req).detach()

                        assert (
                            target_tensor.size() == tensor.size()
                        ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
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

    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        return plans

    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        """
        return the checkpoint_id that will be used to save the checkpoint.
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
                that the stager is maintained and re-used for multiple dcp.async_save calls. Default to False.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__(
            path=path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            cache_staged_state_dict=cache_staged_state_dict,
            overwrite=overwrite,
        )

    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        """Override of AsyncStager.stage"""
        # in the async case, the state dict is already on CPU, so maintaining this
        # buffer makes no sense
        self.per_thread_copy_ahead = 0
        return super().stage(state_dict)
