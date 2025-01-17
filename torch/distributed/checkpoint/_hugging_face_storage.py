import dataclasses
import io
import pickle
import queue
import threading
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, cast, IO, List, Optional, Union

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.filesystem import (
    _item_size,
    _OverlappingCpuLoader,
    _SerialCpuLoader,
    _split_by_size_and_type,
    _TensorLoader,
)
from torch.distributed.checkpoint.metadata import Metadata, StorageMeta
from torch.distributed.checkpoint.planner import (
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.storage import StorageWriter, WriteResult
from torch.futures import Future


__all__ = ["HuggingFaceWriter", "HuggingFaceReader"]

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


class HuggingFaceHubWriter(StorageWriter):
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
        repo_id: str,
        single_file_per_rank: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        overwrite: bool = False,
        token: Optional[str] = None,
        private: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the huggingface writer for repo_id.

        Args:
            repo_id: huggingface repo where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            token: The token to use to authenticate with the huggingface hub.
            private: Whether to create a private repo. Defaults to False.
        """
        super().__init__()

        import huggingface_hub

        self.api = huggingface_hub.HfApi()
        huggingface_hub.create_repo(repo_id, token=token, exist_ok = overwrite, private=private)
        self.repo_id = repo_id
        self.single_file_per_rank = single_file_per_rank
        self.thread_count = thread_count
        self.per_thread_copy_ahead = per_thread_copy_ahead
        self.save_id = _generate_uuid()
        self.overwrite = overwrite
        self.token = token
        self.private = private

    def reset(self, repo_id: Union[str, None] = None) -> None:
        from huggingface_hub import create_repo
        
        self.save_id = _generate_uuid()
        if repo_id:
            self.repo_id = create_repo(repo_id, token=self.token, exist_ok =self.overwrite, private=self.private).repo_id

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        pass

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        if self.api.file_exists(repo_id=self.repo_id, filename=self.metadata_path):
            if self.overwrite:
                warnings.warn(
                    f"Detected an existing checkpoint in {self.metadata_path}, overwriting since {self.overwrite=}."
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
                file_queue.put((file_name, bucket))
        else:
            for item in plan.items:
                file_name = gen_file()
                file_queue.put((file_name, [item]))

        result_queue: queue.Queue = queue.Queue()

        threads = []
        for _ in range(1, self.thread_count):
            t = threading.Thread(
                target=self._write_files_from_queue,
                args=(
                    file_queue,
                    result_queue,
                    planner,
                    self.per_thread_copy_ahead,
                    self.thread_count,
                ),
            )
            t.start()
            threads.append(t)

        self._write_files_from_queue(
            file_queue=file_queue,
            result_queue=result_queue,
            planner=planner,
            inflight_threshhold=self.per_thread_copy_ahead,
            thread_count=self.thread_count,
        )

        for t in threads:
            t.join()

        res = []
        try:
            while True:
                res += result_queue.get_nowait()
        except queue.Empty:
            fut: Future[List[WriteResult]] = Future()
            fut.set_result(res)
            return fut

    def _write_files_from_queue(
        self,
        file_queue: queue.Queue,
        result_queue: queue.Queue,
        planner: SavePlanner,
        inflight_threshhold: int,
        thread_count: int,
) -> None:
        try:
            while True:
                file_name, write_items = file_queue.get_nowait()
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

                bytes_io = io.BytesIO()

                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(
                        self._write_item(bytes_io, data, write_item, file_name)
                    )

                for tensor, write_item in loader.values():
                    assert tensor.is_cpu
                    write_results.append(
                    self._write_item(bytes_io, tensor, write_item, file_name,)
                    )
                
                self.api.upload_file(
                    path_or_fileobj=bytes_io,
                    path_in_repo=file_name,
                    repo_id=self.repo_id,
                    repo_type="model",
                    token=self.token,
                )
                result_queue.put(write_results)
        except queue.Empty:
            pass

    def _write_item(
        self,
        bytes_io: io.BytesIO,
        data: Union[io.BytesIO, torch.Tensor],
        write_item: WriteItem,
        storage_key: str,
    ) -> WriteResult:
        offset  = bytes_io.tell()
        if write_item.type == WriteItemType.BYTE_IO:
            assert isinstance(data, io.BytesIO)
            bytes_io.write(data.getbuffer())
        else:
            assert isinstance(data, torch.Tensor)
            assert data.device == torch.device("cpu")

            torch.save(data, cast(IO[bytes], bytes_io))

        length = bytes_io.tell() - offset
            
        return WriteResult(
            index=write_item.index,
            size_in_bytes=length,
            storage_data=_StorageInfo(storage_key, offset, length),
        )


    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        storage_md = {}
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md

        metadata.storage_meta = self.storage_meta()

        metadata_bytes_io = io.BytesIO()
        pickle.dump(metadata, metadata_bytes_io)

        self.api.upload_file(
            path_or_fileobj=metadata_bytes_io,
            path_in_repo=self.metadata_path,
            repo_id=self.repo_id,
            repo_type="model",
            token=self.token,
        )
           

    def storage_meta(self) -> Optional[StorageMeta]:
        return StorageMeta(checkpoint_id=self.checkpoint_id, save_id=self.save_id)

    @property
    def checkpoint_id(self) -> str:
        """
        return the checkpoint_id that will be used to save the checkpoint.
        """
        return self.repo_id

    @property
    def metadata_path(self) -> str:
        return _metadata_fn

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str) -> bool:
        import huggingface_hub
        
        return huggingface_hub.HfApi().repo_exists(checkpoint_id)


class HuggingFaceHubReader(FileSystemReader):
   
    def __init__(self, repo_id: str, token : Optional[str] = None) -> None:
        """
        Initialize the huggingface reader for repo_id.

        Args:
            repo_id: huggingface repo where the checkpoint will be read from.
            token: The token to use to authenticate with the huggingface hub.
        """

        from huggingface_hub import snapshot_download
        
        self.repo_id = repo_id
        self.path = snapshot_download(self.repo_id, token=token)
        super().__init__(path=self.path)

    def reset(self, repo_id: Union[str, None] = None) -> None:
        from huggingface_hub import snapshot_download

        self.storage_data = {}
        if repo_id:
            self.repo_id = repo_id
            self.path = snapshot_download(self.repo_id)
        self.load_id = _generate_uuid()

    @property
    def checkpoint_id(self) -> str:
        """
        return the checkpoint_id that will be used to load the checkpoint.
        """
        return self.repo_id

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str) -> bool:
        import huggingface_hub
       
        return huggingface_hub.HfApi().repo_exists(checkpoint_id)
