from typing import cast

import torch
from torch.types import Storage

from ._directory_reader import DirectoryReader

# because get_storage_from_record returns a tensor!?
class _HasStorage(object):
    def __init__(self, storage):
        self._storage = storage

    def storage(self):
        return self._storage


class TorchScriptDirectoryReader(DirectoryReader):
    """
    Class to allow PackageImporter to operate on unzipped packages which include
    torchscript modules. Methods copy the behavior of the internal PyTorchFileReader
    class (which is used for accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this TorchScriptDirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, directory):
        super().__init__(directory)

    def get_storage_from_record(self, name, numel, dtype):
        filename = f"{self.directory}/{name}"
        nbytes = torch._utils._element_size(dtype) * numel
        storage = cast(Storage, torch._UntypedStorage)
        return _HasStorage(storage.from_file(filename=filename, nbytes=nbytes))
