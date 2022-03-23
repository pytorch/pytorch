import zipfile
from abc import ABC, abstractmethod
from io import BytesIO

class PackageZipFileReader(ABC):
    """
    Class to allow PackageImporter to operate objects. To create a custom
    zip file reader for PackageImporter simply inherit this class.
    """
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
    """
    Class to allow PackageExporter to operate objects. To create a custom
    zip file writer for PackageExporter simply inherit this class.
    """

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
    Class to allow PackageExporter to operate general objects. This is default
    zipfile reader. This is effectively a wrapper around ZipFile to have a similar
    API to torch._C.PyTorchWriter.
    """

    def __init__(self, file_name):
        super().__init__(file_name, mode='w')
        if isinstance(file_name, BytesIO):
            self.prefix = "archive"
        else:
            self.prefix = "/".join(file_name.strip("/").split('/')[1:])
        super().writestr(f"{self.prefix}/.data/version", "6\n")

    def write_record(self, file_name, str_or_bytes, size=None):
        super().writestr(f"{self.prefix}/{file_name}", str_or_bytes)

    def close(self):
        super().close()

class DefaultPackageZipFileReader(zipfile.ZipFile, PackageZipFileReader):
    """
    Class to allow PackageImporter to operate general objects. This is default
    zipfile reader.  This is effectively a wrapper around ZipFile to have a similar
    API to torch._C.PyTorchReader.
    """

    def __init__(self, file_name):
        super().__init__(file_name, mode='r')
        prefixed_records = super().namelist()
        self.records = []
        if isinstance(file_name, BytesIO):
            self.prefix = "archive"
        else:
            self.prefix = "/".join(file_name.strip("/").split('/')[1:])
        for record in prefixed_records:
            self.records.append(record[len(self.prefix) + 1:])


    def get_record(self, name):
        return super().read(f"{self.prefix}/{name}")

    def has_record(self, path):
        return path in self.records

    def get_all_records(self):
        return list(self.records)

    def close(self):
        super().close()
