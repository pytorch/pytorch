import torch
from glob import glob
import os.path
from typing import List, Any

_storages : List[Any] = [
    torch.DoubleStorage,
    torch.FloatStorage,
    torch.LongStorage,
    torch.IntStorage,
    torch.ShortStorage,
    torch.CharStorage,
    torch.ByteStorage,
    torch.BoolStorage,
]
_dtype_to_storage = {
    data_type(0).dtype: data_type for data_type in _storages
}

# because get_storage_from_record returns a tensor!?
class _HasStorage(object):
    def __init__(self, storage):
        self._storage = storage

    def storage(self):
        return self._storage


class MockZipReader(object):
    def __init__(self, directory):
        self.directory = directory

    def get_record(self, name):
        filename = f'{self.directory}/{name}'
        with open(filename, 'rb') as f:
            return f.read()

    def get_storage_from_record(self, name, numel, dtype):
        storage = _dtype_to_storage[dtype]
        filename = f'{self.directory}/{name}'
        return _HasStorage(storage.from_file(filename=filename, size=numel))

    def get_all_records(self, ):
        files = []
        for filename in glob(f'{self.directory}/**', recursive=True):
            if not os.path.isdir(filename):
                files.append(filename[len(self.directory) + 1:])
        return files
