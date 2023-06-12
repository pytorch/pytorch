# pyre-ignore-all-errors
# mypy: ignore-errors

import collections
import dataclasses
import io
import os
import pickle
import queue
import threading
from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import Callable, cast, Dict, List, Optional, Union

import fsspec

import torch
from fsspec.core import url_to_fs
from torch import Tensor

from torch.distributed._shard._utils import narrow_tensor_by_index

from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex

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
from torch.distributed.checkpoint.storage import (
    StorageReader,
    StorageWriter,
    WriteResult,
)
from torch.futures import Future

__all__ = [
    "FsspecWriter",
    "FsspecReader",
]


@dataclass
class _StorageInfo:
    """
    This is the per entry storage info
    """

    relative_path: str
    offset: int
    length: int


@dataclass
class _StoragePrefix:
    prefix: str


DEFAULT_SUFFIX = ".distcp"


def _result_from_write_item(
    item: WriteItem, size_in_bytes, storage_data
) -> WriteResult:
    return WriteResult(
        index=item.index, size_in_bytes=size_in_bytes, storage_data=storage_data
    )


class _TensorLoader(ABC):
    @abstractmethod
    def add(self, size: int, obj: object):
        pass

    @abstractmethod
    def start_loading(self):
        pass

    @abstractmethod
    def values(self):
        pass


class _SerialCpuLoader(_TensorLoader):
    def __init__(self, resolve_fun: Callable):
        self.resolve_fun = resolve_fun
        self.items = []

    def add(self, size: int, obj: object):
        self.items.append((size, obj))

    def start_loading(self):
        pass

    def values(self):
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
        stream: Union[None, io.RawIOBase, torch._C._CudaStreamBase] = None,
        inflight_threshhold: int = 1_000_000,
    ):
        self.resolve_fun = resolve_fun
        self.items = []
        self.inflight_threshhold = inflight_threshhold
        self.in_flight_data = 0
        self.current_items: collections.deque = collections.deque()
        self.idx = 0
        self.started = False
        self.stream = stream or torch.cuda.current_stream()
        if self.stream != torch.cuda.current_stream():
            self.stream.wait_stream(torch.cuda.current_stream())

    @property
    def _done(self):
        return self.idx >= len(self.items)

    def _drain(self):
        drained = []
        if self.in_flight_data >= self.inflight_threshhold:
            self.stream.synchronize()
        while self.in_flight_data >= self.inflight_threshhold:
            val = self.current_items.popleft()
            self.in_flight_data -= val[0].numel() * val[0].element_size()
            drained.append(val)
        return drained

    def _refill(self):
        with torch.cuda.stream(self.stream):
            while (
                not self._done
                and self.in_flight_data < self.inflight_threshhold
            ):
                _, obj = self.items[self.idx]
                self.idx += 1
                tensor = self.resolve_fun(obj).detach()
                if tensor.is_cuda:
                    tensor = tensor.to(device="cpu", non_blocking=True)
                elif tensor.device == torch.device("cpu"):
                    if tensor.storage().size() != tensor.numel():
                        # this forces the tensor to be both contiguous and with minimal storage
                        tensor = tensor.clone()

                self.current_items.append(
                    (
                        tensor,
                        obj,
                    )
                )
                self.in_flight_data += tensor.numel() * tensor.element_size()

    def _finish(self):
        assert self._done
        if len(self.current_items) > 0:
            self.stream.synchronize()
        return self.current_items

    def add(self, size: int, obj: object):
        if self.started:
            raise RuntimeError("cannot add items after loading started")
        self.items.append((size, obj))

    def start_loading(self):
        if self.started:
            return
        self.started = True
        self.items.sort(key=lambda x: x[0])
        self._refill()

    def values(self):
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


def _split_by_size_and_type(
    bins: int, items: List[WriteItem]
) -> List[List[WriteItem]]:
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
        idx = min(enumerate(bucket_sizes), key=lambda x: x[1])[0]
        buckets[idx].append(wi)
        bucket_sizes[idx] += _item_size(wi)

    return buckets


def _write_item(
    stream: Optional[Union[io.RawIOBase, torch._C._CudaStreamBase]],
    data: Union[io.BytesIO, torch.Tensor],
    write_item: WriteItem,
    storage_key: str,
):
    offset = stream.tell()

    if write_item.type == WriteItemType.BYTE_IO:
        assert isinstance(data, io.BytesIO)
        stream.write(data.getbuffer())
    else:
        assert isinstance(data, torch.Tensor)
        assert data.device == torch.device("cpu")
        torch.save(data, stream)
    length = stream.tell() - offset

    return _result_from_write_item(
        write_item, length, _StorageInfo(storage_key, offset, length)
    )


def _write_files_from_queue(
    file_queue: queue.Queue,
    result_queue: queue.Queue,
    planner: SavePlanner,
    inflight_threshhold: int,
):
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: _TensorLoader

            if torch.cuda.is_available() and inflight_threshhold > 0:
                loader = _OverlappingCpuLoader(
                    lambda x: planner.resolve_data(x),
                    inflight_threshhold=inflight_threshhold,
                )
            else:
                loader = _SerialCpuLoader(
                    lambda x: planner.resolve_data(x),
                )

            tensor_w = [
                wi for wi in write_items if wi.type != WriteItemType.BYTE_IO
            ]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            loader.start_loading()

            bytes_w = [
                wi for wi in write_items if wi.type == WriteItemType.BYTE_IO
            ]
            write_results = []

            with fsspec.open(file_name, "wb") as stream:
                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(
                        _write_item(stream, data, write_item, storage_key)
                    )

                for tensor, write_item in loader.values():
                    assert not tensor.is_cuda
                    write_results.append(
                        _write_item(stream, tensor, write_item, storage_key)
                    )
            result_queue.put(write_results)
    except queue.Empty:
        pass


class FsspecWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using FFspec.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
    ) -> None:
        """
        Initialize the writer pointing to `path`

        Args:
            path: diretory where the checkpoint will be writen to.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.

        N. B. There's no guarantee that the checkpoint will be consistent in the case of a failure.
        """

        super().__init__()
        self.path = path
        self.fs, _ = url_to_fs(path)
        self.thread_count = thread_count
        self.per_thread_copy_ahead = per_thread_copy_ahead

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        pass

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self.fs.makedirs(self.path, exist_ok=True)
        return plan

    def prepare_global_plan(
        self, global_plan: List[SavePlan]
    ) -> List[SavePlan]:
        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_"))
            for i, plan in enumerate(global_plan)
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
        for item in plan.items:
            file_name = gen_file()
            file_path = os.path.join(self.path, file_name)
            file_queue.put((file_path, file_name, [item]))

        result_queue: queue.Queue = queue.Queue()

        threads = []
        for _ in range(1, self.thread_count):
            t = threading.Thread(
                target=_write_files_from_queue,
                args=(
                    file_queue,
                    result_queue,
                    planner,
                    self.per_thread_copy_ahead,
                ),
            )
            t.start()
            threads.append(t)

        _write_files_from_queue(
            file_queue=file_queue,
            result_queue=result_queue,
            planner=planner,
            inflight_threshhold=self.per_thread_copy_ahead,
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

    def finish(
        self, metadata: Metadata, results: List[List[WriteResult]]
    ) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        metadata_path = os.path.join(self.path, ".metadata")
        with self.fs.transaction:
            with fsspec.open(metadata_path, "wb") as metadata_file:
                pickle.dump(metadata, metadata_file)


class FsspecReader(StorageReader):
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__()
        self.path = path
        self.fs, _ = url_to_fs(path)
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # group requests by file
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            abs_path = os.path.join(self.path, relative_path)
            with fsspec.open(abs_path, "rb") as file:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    if req.type == LoadItemType.BYTE_IO:
                        bytes = io.BytesIO(file.read(item_md.length))
                        bytes.seek(0)
                        planner.load_bytes(req, bytes)
                    else:
                        tensor = cast(
                            Tensor, torch.load(file, map_location="cpu")
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

    # Implementating the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        metadata_path = os.path.join(self.path, ".metadata")
        with fsspec.open(metadata_path, "rb") as metadata_file:
            return pickle.load(metadata_file)

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool
    ) -> None:
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(
        self, global_plan: List[LoadPlan]
    ) -> List[LoadPlan]:
        return global_plan
