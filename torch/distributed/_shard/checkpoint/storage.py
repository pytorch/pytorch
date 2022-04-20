import abc
from typing import List, Dict

from torch.futures import Future

from .metadata import (
    BytesReadRequest,
    BytesWriteRequest,
    Metadata,
    TensorReadRequest,
    TensorWriteRequest,
)

class StorageWriter(abc.ABC):
    """
    Interface used by ``save_state_dict`` to write to storage.

    A subclass should expect the following sequence of calls by ``save_state_dict``

    1) (if rank == 0) prepare()
    2) prepare_storage with a dict of storage_key -> size. This can be used to
        pre-allocate storage.
    3) write_bytes
    4) write_tensors.
    5) Wait for (2) and (3) futures. If either fail, abort checkpoint.
    6) (if rank == 0) write_metadata().
    7) (if rank == 0) finish().

    Rank 0 will wait for (3) and (4) before writing metadata.
    The writes from (3) and (4) are initiated before any waiting is done.
    """
    @abc.abstractmethod
    def prepare(self) -> None:
        """
        Initialize storage to receive the checkpoint.

        This method will only be called on rank0.

        Returns:
            Future to signal intialization is complete.
        """
        pass

    @abc.abstractmethod
    def write_bytes(self, requests: List[BytesWriteRequest]) -> Future[None]:
        """
        Initiate writes for all requests in `requests`.

        Writing can happen asynchronously and/or concurrently. A blocking
        implementation is valid.

        Args:
            requests (List[BytesWriteRequest]): A list of requests to write
        Returns:
            A future that completes once all writes have finished.
        """
        pass

    @abc.abstractmethod
    def write_tensors(self, requests: List[TensorWriteRequest]) -> Future[None]:
        """
        Initiate writes for all requests in `requests`.

        Writing can happen asynchronously and/or concurrently. A blocking
        implementation is valid.

        Implementors are responsible for any device to host transfers required
        to copy.

        Args:
            requests (List[TensorWriteRequest]): A list of requests to write

        Returns:
            A future that completes once all writes have finished.
        """
        pass

    @abc.abstractmethod
    def write_metadata(self, metadata: Metadata) -> None:
        """
        Write the global metadata and commit the checkpoint.

        This method is called on rank 0 after all data was writen
        and is used to commit the checkpoint.

        The `metadata` object includes a global view of the checkpoint
        and, while writing it is optional, it must be recoverable by the
        StorageReader implementation.

        The actual format/schema used for serializing `metadata` is
        considered and implementation detail.

        Args:
            metadata (Metadata): metadata for the new checkpoint

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def finish(self) -> None:
        """
        Mark the current checkpoint as sucessfully done.

        This method will only be called from rank0.
        """
        pass


    def prepare_storage(self, storage_keys: Dict[str, int]) -> None:
        """
        Prepare the underlying storage for upcoming writes.

        This is an optional override intended for advanced scenarios where
        a storage layer needs wants to do some work ahead of the writing itself.

        The default implementation does nothing.

        NB: Size is a lowerbound estimate and doesn't account for any
        serialization overhead from the storage layer. For example, for a tensor `t`
        it will be `t.numel() * t.element_size()` but if `torch.save` is used,
        there will be aditional serialization overhead.

        Args:
            storage_keys (Dict[str, int]): key - handle's name. value - size
                of the handle.

        Returns:
            None
        """
        pass


class StorageReader(abc.ABC):
    """
    Interface used by ``load_state_dict`` to read from storage.

    A subclass should expected the following sequence of calls by ``load_state_dict``:

    1) read_metadata() - on all ranks
    2) read_bytes
    3) read_tensors

    The reads from (2) and (3) are initiated before any waiting is done.

    Implementors must ensure host/device synchronization as part of
    completion of both read requests.
    """

    @abc.abstractmethod
    def read_bytes(self, requests: List[BytesReadRequest]) -> Future[None]:
        """
        Initiate read for all requests in `requests`.

        Reading happen asynchronously and/or concurrently. A blocking
        implementation is valid.

        Args:
            requests (List[BytesReadRequest]): A list of requests to read.

        Return:
            A future that completes once all read have finished.
        """
        pass

    @abc.abstractmethod
    def read_tensors(self, requests: List[TensorReadRequest]) -> Future[None]:
        """
        Initiate read for all requests in `requests`.

        Reading happen asynchronously and/or concurrently. A blocking
        implementation is valid.

        Implementors must not assume that the original device
        at write time will be the same at read time.

        If an implementation uses asynchronous copies to device, it must
        ensure proper synchronization W.R.T. the returned future.

        Args:
            requests (List[BytesReadRequest]): A list of requests to read.

        Returns:
            A future that completes once all read have finished.
        """
        pass

    @abc.abstractmethod
    def read_metadata(self) -> Metadata:
        """
        Reads the checkpoint metadata.

        Returnss:
            The metatada object associated with the checkpoint being loaded.

        """
        pass
