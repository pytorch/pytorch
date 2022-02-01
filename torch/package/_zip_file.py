import os.path
from glob import glob
from typing import cast

import torch
import zipfile
from torch.types import Storage
from abc import ABC, abstractmethod

class PackageZipFileReader(ABC):

    @abstractmethod
    def get_record(self, name: str):
        raise NotImplementedError(
            f"get_record(self, name: str) is not implemented in {type(self)}"
        )

    @abstractmethod
    def has_record(self, path: str):
        raise NotImplementedError(
            f"has_record(self, path: str) is not implemented in {type(self)}"
        )

    @abstractmethod
    def get_all_records(self):
        raise NotImplementedError(
            f"get_all_records(self) is not implemented in {type(self)}"
        )

    @abstractmethod
    def close(self):
        raise NotImplementedError(
            f"close(self) is not implemented in {type(self)}"
        )

class PackageZipFileWriter(ABC):


    @abstractmethod
    def write_record(self, file_name, str_or_bytes, size):
        raise NotImplementedError(
            f"write_record(self, file_name, str_or_bytes, size) is not implemented in {type(self)}"
        )

    @abstractmethod
    def close(self):
        raise NotImplementedError(
            f"close(self) is not implemented in {type(self)}"
        )

class DefaultPackageZipFileWriter(zipfile.ZipFile, PackageZipFileWriter):
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, file_name):
        super().__init__(file_name, mode='w')

    def write_record(self, file_name, str_or_bytes, size=None):
        super().writestr(file_name, str_or_bytes)

    def close(self):
        super().close()

class DefaultPackageZipFileReader(zipfile.ZipFile, PackageZipFileReader):
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, file_name):
        super().__init__(file_name, mode='r')
        self.records = set(super().namelist())

    def get_record(self, name):
        return super().read(name)

    def has_record(self, path):
        return path in records

    def write_record(self, file_name, str_or_bytes, size=None):
        super().writestr(file_name, str_or_bytes)

    def get_all_records(self):
        return list(self.records)

    def close(self):
        super().close()
