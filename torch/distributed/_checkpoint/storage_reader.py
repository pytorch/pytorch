import abc
from .metadata import Metadata, ReadWriteRequest
from torch.futures import Future
import torch
import pickle
import numpy

class StorageReader(abc.ABC):
    @abc.abstractmethod
    def read(self, req: ReadWriteRequest) -> Future:
        """
        Performs a read and returns a Future to enable async reading.
        """
        pass

    @abc.abstractmethod
    def read_metadata(self) -> Metadata:
        pass

class FileSystemReader(StorageReader):
    def __init__(self, base_folder_name: str) -> None:
        super().__init__()
        self.base_folder_name = base_folder_name

    def read(self, req: ReadWriteRequest) -> Future:
        """
        Performs a read and returns a Future to enable async reading.
        """
        with open(f"{self.base_folder_name}/{req.storage_key}", "rb") as storage:
            storage.seek(req.offset)

            # Read everything for the storage and write to the buffer
            buffer = storage.read(req.length)
            # Hack, I am not able to convert torch dtype to numpy dtype, hence the hard coding
            np_array = numpy.frombuffer(buffer, dtype="float32")

            # The tensor conversion triggers this warning, will deal with it later
            # https://www.internalfb.com/code/fbsource/[81a627eaf12b3da7ba04e0f13c9609bd6a8be7eb]/fbcode/caffe2/torch/csrc/utils/tensor_numpy.cpp?lines=172-178
            buffer_tensor = torch.from_numpy(np_array)
            req.target_tensor.copy_(buffer_tensor)

        fut = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in StateDictLoader
    def read_metadata(self) -> Metadata:
        with open(f"{self.base_folder_name}/metadata", "rb") as metadata_file:
            return pickle.load(metadata_file)
