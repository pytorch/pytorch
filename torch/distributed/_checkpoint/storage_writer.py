from .metadata import Metadata, ReadWriteRequest
from typing import Dict
import os
from torch.futures import Future
import pickle
import torch.distributed as dist
import abc

class StorageWriter(abc.ABC):
    """
    Interface to write to underlying storage system
    """
    @abc.abstractmethod
    def write(self, req: ReadWriteRequest) -> Future[None]:
        """
        Performs a write request and returns a Future to wait on.
        Args:
            req (ReadWriteRequest): see `./metadata.py`
        """
        pass

    @abc.abstractmethod
    def write_metadata(self, metadata: Metadata) -> None:
        """
        Writes the metatdata.

        Args:
            metadata (Metadata): see `./metadata.py`
        """
        pass

    def prepare_storage(self, storage_handles: Dict[str, int]) -> None:
        """
        This blocking call can be overwritten by the subclass.
        It can use `storage_handles` to plan for any write preformace optimization.
        e.g. non sequential and parallel writes.
        By default, it does nothing

        Args:
            storage_handles (Dict[str, int]): key - handle's name. value - size
                of the handle.
        """
        pass

class FileSystemWriter(StorageWriter):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def write(self, req: ReadWriteRequest) -> Future[None]:
        with open(os.path.join(self.path, req.storage_key), "a+b") as storage:
            storage.seek(req.offset)
            # The following couple lines are simple implementation to get things going.
            # They should be updated.
            # Somehow I need to cast the memoryview to "c"ontiguous
            # otherwise I will get a size mismatch at writing
            mv = memoryview(req.target_tensor.detach().cpu().numpy()).cast("c")
            storage.write(mv)

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in Storage Writer
    def write_metadata(self, metadata: Metadata) -> None:
        # Only need to write the metadata once, since each ShardMetadata has the global view
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        with open(os.path.join(self.path, ".metadata"), "wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
