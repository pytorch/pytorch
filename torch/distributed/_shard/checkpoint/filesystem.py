from dataclasses import dataclass
import os
import dataclasses
import io
import pickle
from typing import List, Union, Dict, cast

import torch
from torch import Tensor
from torch.futures import Future
from pathlib import Path

from .metadata import (
    Metadata,
    MetadataIndex,
)
from .storage import (
    StorageReader,
    StorageWriter,
    WriteResult,
)

from .planner import (
    LoadItemType,
    LoadPlanner,
    LoadPlan,
    SavePlan,
    SavePlanner,
    ReadItem,
    WriteItem,
    WriteItemType,
)

from torch.distributed._shard._utils import narrow_tensor_by_index


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

DEFAULT_SUFIX = ".distcp"

def _trim(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.storage().size() != tensor.numel():
        tensor = tensor.clone()
    return tensor

def _result_from_write_item(item: WriteItem, size_in_bytes, storage_data) -> WriteResult:
    return WriteResult(
        index=item.index,
        size_in_bytes=size_in_bytes,
        storage_data=storage_data)

def _write_item(stream, data, write_item, storage_key):
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
        write_item,
        length,
        _StorageInfo(storage_key, offset, length)
    )

def _write_files_from_queue(
    file_queue: List,
    planner: SavePlanner,
    use_fsync: bool,
):
    write_results = []

    for file_path, file_name, write_items in file_queue:
        tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
        bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]

        with open(file_path, "wb") as stream:
            for write_item in bytes_w:
                data = planner.resolve_data(write_item)
                write_results.append(_write_item(stream, data, write_item, file_name))

            for write_item in tensor_w:
                tensor = _trim(cast(torch.Tensor, planner.resolve_data(write_item)))
                assert not tensor.is_cuda
                write_results.append(_write_item(stream, tensor, write_item, file_name))

            if use_fsync:
                os.fsync(stream.fileno())

    return write_results

class FileSystemWriter(StorageWriter):
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
        single_file_per_rank: bool = False,
        sync_files: bool = True,
    ) -> None:
        """
        Initialize the writer pointing to `path`

        Args:
            path: diretory where the checkpoint will be writen to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files: force files to be synced to permanent storage. Default to True.

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__()
        self.path = Path(path)
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files

    def init(self, is_coordinator: bool) -> None:
        pass

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        # There's no storage input in the local plan
        return plan

    def prepare_global_plan(self, global_plan: List[SavePlan]) -> List[SavePlan]:
        self.path.mkdir(parents=True, exist_ok=True)

        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_")) for i, plan in enumerate(global_plan)
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
            file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFIX}"
            file_count += 1
            return file_name

        file_queue = []
        if self.single_file_per_rank:
            file_name = gen_file()
            file_queue.append((self.path / file_name, file_name, plan.items))
        else:
            for item in plan.items:
                file_name = gen_file()
                file_queue.append((self.path / file_name, file_name, [item]))

        results = _write_files_from_queue(
            file_queue=file_queue,
            planner=planner,
            use_fsync=self.sync_files,
        )

        fut: Future[List[WriteResult]] = Future()
        fut.set_result(results)
        return fut

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({
                wr.index: wr.storage_data for wr in wr_list
            })
        metadata.storage_data = storage_md
        with (self.path / ".metadata.tmp").open("wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            os.fsync(metadata_file.fileno())

        (self.path / ".metadata.tmp").rename(self.path / ".metadata")


class SlicedBufferedReader(io.BufferedReader):
    # TODO override read to handle (-1) correctly
    def __init__(self, base_stream: io.RawIOBase, offset: int, len: int):
        super().__init__(base_stream)
        self.offset = offset
        self.len = len
        self.seek(0)

    def seek(self, __offset: int, __whence: int = os.SEEK_SET) -> int:
        if __whence == os.SEEK_SET:
            __offset = self.offset + __offset
        elif __whence == os.SEEK_END:
            __whence = os.SEEK_SET
            __offset = (self.offset + self.len) - __offset
        return super().seek(__offset, __whence)

    def tell(self) -> int:
        return super().tell() - self.offset

class FileSystemReader(StorageReader):
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__()
        self.path = Path(path)
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()

    def _slice_file(self, file, sinfo: _StorageInfo):
        return SlicedBufferedReader(
            io.FileIO(file.fileno(), closefd=False),
            sinfo.offset, sinfo.length
        )

    def read_data(
        self,
        plan: LoadPlan,
        planner: LoadPlanner
    ) -> Future[None]:
        # group requests by file
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            with (self.path / relative_path).open("rb") as file:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(file, item_md)
                    if req.type == LoadItemType.BYTE_IO:
                        bytes = io.BytesIO(file_slice.read(item_md.length))
                        bytes.seek(0)
                        planner.load_bytes(req, bytes)
                    else:
                        tensor = cast(Tensor, torch.load(file_slice, map_location="cpu"))
                        tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
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
        with (self.path / ".metadata").open("rb") as metadata_file:
            return pickle.load(metadata_file)

    def init(self, metadata: Metadata, is_coordinator: bool) -> None:
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return global_plan
