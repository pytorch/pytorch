import os.path
from glob import glob
from typing import cast

import torch
import zipfile
from torch.types import Storage

# because get_storage_from_record returns a tensor!?
class _HasStorage(object):
    def __init__(self, storage):
        self._storage = storage

    def storage(self):
        return self._storage


class PackageZipFile(ZipFile):
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, file_name, mode='r'):
        self.zip_file = zipfile.ZipFile(file_name, mode)
        self.records = set(self.zip_file.namelist())



    def get_record(self, name):
        return self.zip_file.read(name)

    def get_storage_from_record(self, filename, numel, dtype):
        # filename = f"{self.directory}/{name}"
        nbytes = torch._utils._element_size(dtype) * numel
        storage = cast(Storage, torch.UntypedStorage)
        ret = None
        # with open(filename) as
        return _HasStorage(storage.from_file(filename=filename, nbytes=nbytes))

    def has_record(self, path):
        return (path in self.zip_file.namelist())

    def write_record(self, file_name, str_or_bytes, size=None):
        self.writestr(file_name, str_or_bytes)

    def get_all_records(
        self
    ):
        return list(self.records)
