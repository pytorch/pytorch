import os.path
from glob import glob
from typing import cast
import torch
from ._zip_file import PackageZipFileReader, PackageZipFileWriter
import zipfile
from torch.types import Storage
from abc import ABC, abstractmethod

class TorchScriptPackageZipFileWriter(torch._C.PyTorchFileWriter, PackageZipFileWriter):
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, file_name):
        super().__init__(file_name)
        print(isinstance(self, PackageZipFileWriter))

    def get_record(self, name):
        return super().get_record(name)

    def has_record(self, path):
        return super().has_record(name)

    def get_all_records(self):
        return super.get_all_records()

    def close(self):
        pass

class TorchScriptPackageZipFileReader(torch._C.PyTorchFileReader):
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, file_name):
        super().__init__(file_name)

    # def close(self):
    #     pass
