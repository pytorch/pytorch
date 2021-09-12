import os.path
from glob import glob
from typing import Any, List

import torch

_storages: List[Any] = [
    torch.DoubleStorage,
    torch.FloatStorage,
    torch.LongStorage,
    torch.IntStorage,
    torch.ShortStorage,
    torch.CharStorage,
    torch.ByteStorage,
    torch.BoolStorage,
]
_dtype_to_storage = {data_type(0).dtype: data_type for data_type in _storages}

# because get_storage_from_record returns a tensor!?
class _HasStorage(object):
    def __init__(self, storage):
        self._storage = storage

    def storage(self):
        return self._storage


class DirectoryReader(object):
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, directory):
        self.directory = directory

    def get_record(self, name):
        filename = f"{self.directory}/{name}"
        with open(filename, "rb") as f:
            return f.read()

    def get_storage_from_record(self, name, numel, dtype):
        storage = _dtype_to_storage[dtype]
        filename = f"{self.directory}/{name}"
        return _HasStorage(storage.from_file(filename=filename, size=numel))

    def has_record(self, path):
        full_path = os.path.join(self.directory, path)
        return os.path.isfile(full_path)

    def get_all_records(
        self,
    ):
        files = []
        for filename in glob(f"{self.directory}/**", recursive=True):
            if not os.path.isdir(filename):
                files.append(filename[len(self.directory) + 1 :])
        return files
