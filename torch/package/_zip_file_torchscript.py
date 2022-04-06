import os.path
from pathlib import Path
from typing import List, BinaryIO, Union, Optional

import torch

from ._directory_reader_torchscript import TorchScriptDirectoryReader, _HasStorage
from ._zip_file import PackageZipFileReader, PackageZipFileWriter


class TorchScriptPackageZipFileWriter(PackageZipFileWriter):
    """
    Class to allow PackageExporter to operate torchscript objects. This
    is a wrapper around the PyTorchFileWriter and ScriptModuleSerializer classes.
    """

    def __init__(self, f: Union[str, Path, BinaryIO]):

        if isinstance(f, (Path, str)):
            f = str(f)
            self.buffer: Optional[BinaryIO] = None
        else:  # is a byte buffer
            self.buffer = f

        self.zip_file_writer = torch._C.PyTorchFileWriter(f)
        self.zip_file_writer.set_min_version(6)
        self.script_module_serializer = torch._C.ScriptModuleSerializer(
            self.zip_file_writer
        )
        self.storage_context = self.script_module_serializer.storage_context()

    def write_record(self, f: str, str_or_bytes: Union[str, bytes], size: int):
        if isinstance(str_or_bytes, str):
            str_or_bytes = str.encode(f)
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

    def __init__(
        self, file_or_buffer: Union[str, torch._C.PyTorchFileReader, Path, BinaryIO]
    ):
        if isinstance(file_or_buffer, torch._C.PyTorchFileReader):
            self.filename = "<pytorch_file_reader>"
            self.zip_reader: Union[
                torch._C.PyTorchFileReader, TorchScriptDirectoryReader
            ] = file_or_buffer
        elif isinstance(file_or_buffer, (Path, str)):
            self.filename = str(file_or_buffer)
            if not os.path.isdir(self.filename):
                self.zip_reader = torch._C.PyTorchFileReader(self.filename)
            else:
                self.zip_reader = TorchScriptDirectoryReader(self.filename)
        else:
            self.filename = "<binary>"
            self.zip_reader = torch._C.PyTorchFileReader(file_or_buffer)

    def get_record(self, name: str) -> bytes:
        return self.zip_reader.get_record(name)

    # NOTE: for has_record, get_all_records, and get_storage_from_record pybind doesn't reaveal
    #       the attributes of PyTorchFileReader, so it'll call an error. Strangely, this error
    #       doesn't have an error code which is why it's ignored
    def has_record(self, path: str) -> bool:
        return self.zip_reader.has_record(path)  # type: ignore[union-attr]

    def get_all_records(self) -> List[str]:
        return self.zip_reader.get_all_records()  # type: ignore[union-attr]

    def get_storage_from_record(
        self, name: str, numel: int, dtype: torch.dtype
    ) -> _HasStorage:
        return self.zip_reader.get_storage_from_record(name, numel, dtype)  # type: ignore[union-attr]

    def get_filename(self) -> str:
        return self.filename

    def is_directory(self) -> bool:
        return isinstance(self.zip_reader, TorchScriptDirectoryReader)

    def close(self):
        pass
