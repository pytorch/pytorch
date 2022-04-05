import torch
from torch.types import Storage
from ._zip_file_torchscript import TorchScriptPackageZipFileWriter
from torch.serialization import location_tag, normalize_storage_type
from .package_exporter_no_torch import PackageExporter as DefaultPackageExporter
from .importer import sys_importer, Importer
from typing import (
    cast,
    BinaryIO,
    Sequence,
    Union,
)
from pathlib import Path

class PackageExporter(DefaultPackageExporter):
    """
    A package exporter for specialized functionality for torch. Specifically it uses the optimizations
    of torch's storage in order to not duplicate tensors.
    """
    def __init__(
        self,
        f: Union[str, Path, BinaryIO],
        importer: Union[Importer, Sequence[Importer]] = sys_importer
    ):

        super(PackageExporter, self).__init__(f, importer, zip_file_writer_type=TorchScriptPackageZipFileWriter)


    def persistent_id(self, obj):
        assert isinstance(self.zip_file, TorchScriptPackageZipFileWriter)
        # needed for 'storage' typename which is a way in which torch models are saved
        if torch.is_storage(obj) or isinstance(obj, torch.storage._TypedStorage):
            if isinstance(obj, torch.storage._TypedStorage):
                # TODO: Once we decide to break serialization FC, we can
                # remove this case
                storage = obj._storage
                storage_type_str = obj.pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                dtype = obj.dtype
                storage_numel = obj.size()

            else:
                storage = obj
                storage_type = normalize_storage_type(type(storage))
                dtype = torch.uint8
                storage_numel = storage.nbytes()

            storage = cast(Storage, storage)
            location = location_tag(storage)

            # serialize storage if not already written
            storage_present = self.zip_file.storage_context.has_storage(storage)
            storage_id = self.zip_file.storage_context.get_or_add_storage(storage)
            if not storage_present:
                if storage.device.type != "cpu":
                    storage = storage.cpu()
                num_bytes = storage.nbytes()
                self.zip_file.write_record(
                    f".data/{storage_id}.storage", storage.data_ptr(), num_bytes
                )
            return ("storage", storage_type, storage_id, location, storage_numel)
        return None
