import os
import operator
import pickle
from typing import List, Optional, Union, cast

import torch
from torch import Tensor
from torch.futures import Future
from pathlib import Path

from .metadata import (
    BytesReadRequest,
    BytesWriteRequest,
    Metadata,
    TensorReadRequest,
    TensorWriteRequest,
)
from .storage import StorageReader, StorageWriter
from torch.distributed._shard._utils import narrow_tensor_by_index


class FileSystemWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        """
        Initialize the writer pointing to `path`

        Args:
            path: diretory where the checkpoint will be writen to.
        """
        super().__init__()
        self.path = Path(path)

    def write_bytes(self, requests: List[BytesWriteRequest]) -> Future[None]:
        for req in requests:
            with (self.path / req.storage_key).open("wb") as w:
                w.write(req.bytes.getbuffer())
                os.fsync(w.fileno())

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    def write_tensors(self, requests: List[TensorWriteRequest]) -> Future[None]:
        for req in requests:
            # The following couple lines are simple implementation to get
            # things going.
            #
            # At load time, to enable resharding, we use (sub)view of the tensor.
            # Since the storage of the tensor might not be contiguous. we need to
            # preserve the original view, to calculate the correct sub view at load.
            #
            # `torch.save` saves both the view and storage, it is a good option
            # for unblocking. There are two drawbacks:
            # 1. `torch.save` is pickle based, and pickle is not known for its
            #   compatibility, we should consider replacing it with a more
            #   stable option.
            # 2. pickle is not streamable.
            with (self.path / req.storage_key).open("wb") as w:
                torch.save(req.tensor, w)
                os.fsync(w.fileno())

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    def prepare(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def finish(self, metadata: Metadata) -> None:
        with (self.path / ".metadata.tmp").open("wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            os.fsync(metadata_file.fileno())

        (self.path / ".metadata.tmp").rename(self.path / ".metadata")

class FileSystemReader(StorageReader):
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__()
        self.path = Path(path)

    def read_tensors(self, requests: List[TensorReadRequest]) -> Future[None]:
        """
        Very basic implementation that read from file system.
        """
        # Sort the the requests by storage key and try to reuse the loaded tensors
        requests.sort(key=operator.attrgetter("storage_key"))

        cached_storage_key = None
        view_cached: Optional[Tensor] = None

        for req in requests:
            if cached_storage_key != req.storage_key or \
                    (view_cached is not None and view_cached.device != req.tensor.device):

                with (self.path / req.storage_key).open("rb") as storage:
                    view_cached = cast(Tensor, torch.load(storage, map_location=req.tensor.device))
                    cached_storage_key = req.storage_key

            view_to_copy: Tensor = cast(Tensor, view_cached)
            # FileSystemWrite writes the tensor as is during save.
            # During load time, we will load the Tensor (with it orignal view)
            # narrow it along all dimemsions, and copy_ it to the
            # target tensor, which will be the same size.
            view_to_copy = narrow_tensor_by_index(view_to_copy, req.offsets, req.lengths)

            assert (
                view_to_copy.size() == req.tensor.size()
            ), f"The {req.storage_key} src/dst size does not match."


            assert (
                view_to_copy.device == req.tensor.device
            ), f"cannot load across devices {view_to_copy.device} vs {req.tensor.device}"

            req.tensor.copy_(view_to_copy)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_bytes(self, requests: List[BytesReadRequest]) -> Future[None]:
        for req in requests:
            with (self.path / req.storage_key).open("rb") as storage:
                req.bytes.write(storage.read())

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        with (self.path / ".metadata").open("rb") as metadata_file:
            return pickle.load(metadata_file)
