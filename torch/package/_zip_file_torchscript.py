import torch
from ._zip_file import PackageZipFileReader, PackageZipFileWriter
from ._directory_reader_torchscript import TorchScriptDirectoryReader
from pathlib import Path
import os.path

class TorchScriptPackageZipFileWriter(PackageZipFileWriter):
    """
    Class to allow PackageExporter to operate torchscript objects. This
    is a wrapper around the PyTorchFileWriter and ScriptModuleSerializer classes.
    """

    def __init__(self, f):

        if isinstance(f, (Path, str)):
            f = str(f)
            self.buffer: Optional[BinaryIO] = None
        else:  # is a byte buffer
            self.buffer = f

        self.zip_file_writer = torch._C.PyTorchFileWriter(f)
        self.zip_file_writer.set_min_version(6)
        self.script_module_serializer = torch._C.ScriptModuleSerializer(self.zip_file_writer)
        self.storage_context = self.script_module_serializer.storage_context()


    def write_record(self, f, str_or_bytes, size):
        self.zip_file_writer.write_record(f, str_or_bytes, size)

    def close(self):
        self.script_module_serializer.write_files()
        if self.buffer:
            self.buffer.flush()

class TorchScriptPackageZipFileReader(PackageZipFileReader):
    """
    Class to allow PackageImporter to operate torchscript objects.  This
    is a wrapper around the PyTorchReader class.
    """

    def __init__(self, file_or_buffer):
        if isinstance(file_or_buffer, torch._C.PyTorchFileReader):
            self.filename = "<pytorch_file_reader>"
            self.zip_reader = file_or_buffer
        elif isinstance(file_or_buffer, (Path, str)):
            self.filename = str(file_or_buffer)
            if not os.path.isdir(self.filename):
                self.zip_reader = torch._C.PyTorchFileReader(self.filename)
            else:
                self.zip_reader = TorchScriptDirectoryReader(self.filename)
        else:
            self.filename = "<binary>"
            self.zip_reader = torch._C.PyTorchFileReader(file_or_buffer)

    def get_record(self, name):
        return self.zip_reader.get_record(name)

    def has_record(self, path):
        return self.zip_reader.has_record(path)

    def get_all_records(self):
        return self.zip_reader.get_all_records()

    def get_storage_from_record(self, name, numel, dtype):
        return self.zip_reader.get_storage_from_record(name, numel, dtype)

    def get_filename(self):
        return self.filename

    def is_directory(self):
        return isinstance(self.zip_reader, TorchScriptDirectoryReader)

    def close(self):
        pass
