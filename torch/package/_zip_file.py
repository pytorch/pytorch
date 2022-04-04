import os.path
import zipfile
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import List, Union, BinaryIO, Optional

from ._directory_reader import DirectoryReader


class PackageZipFileReader(ABC):
    """
    Class to allow PackageImporter to operate objects. To create a custom
    zip file reader for PackageImporter simply inherit this class.
    """

    def __init__(self, file_or_buffer: Union[str, Path, BinaryIO]):
        raise NotImplementedError(
            f"init(self, name: str) is not implemented in {type(self)}"
        )

    @abstractmethod
    def get_record(self, name: str) -> bytes:
        raise NotImplementedError(
            f"get_record(self, name: str) is not implemented in {type(self)}"
        )

    @abstractmethod
    def has_record(self, path: str) -> bool:
        raise NotImplementedError(
            f"has_record(self, path: str) is not implemented in {type(self)}"
        )

    @abstractmethod
    def get_all_records(self) -> List[str]:
        raise NotImplementedError(
            f"get_all_records(self) is not implemented in {type(self)}"
        )

    @abstractmethod
    def get_filename(self) -> str:
        raise NotImplementedError(
            f"get_filename(self) is not implemented in {type(self)}"
        )

    def is_directory(self) -> bool:
        raise NotImplementedError(
            f"is_directory(self) is not implemented in {type(self)}"
        )

    @abstractmethod
    def close(self):
        raise NotImplementedError(f"close(self) is not implemented in {type(self)}")


class PackageZipFileWriter(ABC):
    """
    Class to allow PackageExporter to operate objects. To create a custom
    zip file writer for PackageExporter simply inherit this class.
    """

    def __init__(self, f: Union[str, Path, BinaryIO]):
        raise NotImplementedError(
            f"init(self, name: str) is not implemented in {type(self)}"
        )

    @abstractmethod
    def write_record(self, f, str_or_bytes: Union[str, bytes], size: int):
        raise NotImplementedError(
            f"write_record(self, f, str_or_bytes, size) is not implemented in {type(self)}"
        )

    @abstractmethod
    def close(self):
        raise NotImplementedError(f"close(self) is not implemented in {type(self)}")


class DefaultPackageZipFileWriter(zipfile.ZipFile, PackageZipFileWriter):
    """
    Class to allow PackageExporter to operate general objects. This is default
    zipfile reader. This is effectively a wrapper around ZipFile to have a similar
    API to torch._C.PyTorchWriter.
    """

    def __init__(self, f: Union[str, Path, BinaryIO]):

        if isinstance(f, (Path, str)):
            f = str(f)
            self.buffer: Optional[BinaryIO] = None
        else:  # is a byte buffer
            self.buffer = f

        super().__init__(f, mode="w")

        self.prefix: str = "archive"
        if isinstance(f, (Path, str)):
            self.prefix = "/".join(str(f).strip("/").split("/")[1:])
        super().writestr(f"{self.prefix}/.data/version", "6\n")

    def write_record(self, f: str, str_or_bytes: Union[str, bytes], size: int = None):
        super().writestr(f"{self.prefix}/{f}", str_or_bytes)

    def close(self):
        if self.buffer:
            self.buffer.flush()
        super().close()


class DefaultPackageZipFileReader(PackageZipFileReader):
    """
    Class to allow PackageImporter to operate general objects. This is default
    zipfile reader.  This is effectively a wrapper around ZipFile to have a similar
    API to torch._C.PyTorchReader.
    """

    def __init__(self, file_or_buffer: Union[str, Path, BinaryIO]):

        if isinstance(file_or_buffer, (Path, str)):
            self.filename = str(file_or_buffer)
            if not os.path.isdir(self.filename):
                self.zip_reader: Union[
                    zipfile.ZipFile, DirectoryReader
                ] = zipfile.ZipFile(self.filename)
            else:
                self.zip_reader = DirectoryReader(self.filename)
        else:
            self.filename = "<binary>"
            self.zip_reader = zipfile.ZipFile(file_or_buffer)

        if isinstance(self.zip_reader, DirectoryReader):
            self.records = self.zip_reader.get_all_records()

        elif isinstance(self.zip_reader, zipfile.ZipFile):
            prefixed_records = self.zip_reader.namelist()

            self.records = []
            if isinstance(file_or_buffer, BytesIO):
                self.prefix = "archive"
            else:
                self.prefix = "/".join(str(file_or_buffer).strip("/").split("/")[1:])
            for record in prefixed_records:
                self.records.append(record[len(self.prefix) + 1 :])

    def get_record(self, name: str) -> bytes:
        if isinstance(self.zip_reader, DirectoryReader):
            return self.zip_reader.get_record(f"{name}")
        else:
            return self.zip_reader.read(f"{self.prefix}/{name}")

    def has_record(self, path: str) -> bool:
        return path in self.records

    def get_all_records(self) -> List[str]:
        return list(self.records)

    def get_filename(self) -> str:
        return self.filename

    def is_directory(self) -> bool:
        return isinstance(self.zip_reader, DirectoryReader)

    def close(self):
        self.zip_reader.close()
