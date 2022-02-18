import torch
from ._zip_file import PackageZipFileReader, PackageZipFileWriter

class TorchScriptPackageZipFileWriter(PackageZipFileWriter):
    """
    Class to allow PackageExporter to operate torchscript objects. This
    is a wrapper around the PyTorchFileWriter and ScriptModuleSerializer classes.
    """

    def __init__(self, file_name):
        self.zip_file_writer = torch._C.PyTorchFileWriter(file_name)
        self.zip_file_writer.set_min_version(6)


    def write_record(self, file_name, str_or_bytes, size):
        self.zip_file_writer.write_record(file_name, str_or_bytes, size)

    def close(self):
        pass

class TorchScriptPackageZipFileReader(PackageZipFileReader):
    """
    Class to allow PackageImporter to operate torchscript objects.  This
    is a wrapper around the PyTorchReader class.
    """

    def __init__(self, file_name):
        self.zip_file_reader = torch._C.PyTorchFileReader(file_name)

    def get_record(self, name):
        return self.zip_file_reader.get_record(name)

    def has_record(self, path):
        return self.zip_file_reader.has_record(path)

    def get_all_records(self):
        return self.zip_file_reader.get_all_records()

    def get_storage_from_record(self, name, numel, dtype):
        return self.zip_file_reader.get_storage_from_record(name, numel, dtype)

    def close(self):
        pass
