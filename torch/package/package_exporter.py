import torch
from torch.types import Storage
from ._zip_file_torchscript import TorchScriptPackageZipFileWriter
from torch.serialization import location_tag, normalize_storage_type
from .package_exporter_no_torch import PackageExporter

class PackageExporter(PackageExporter):

    def __init__(
        self,
        f: Union[str, Path, BinaryIO],
        importer: Union[Importer, Sequence[Importer]] = sys_importer
    ):

        super(PackageExporter, self).__init__(f, importer)
        self.setup_zipfile(f, TorchScriptPackageZipFileWriter)
        self.script_module_serializer = torch._C.ScriptModuleSerializer(self.zip_file.zip_file_writer)
        self.storage_context = self.script_module_serializer.storage_context()

    def close(self):
        """Write the package to the filesystem. Any calls after :meth:`close` are now invalid.
        It is preferable to use resource guard syntax instead::

            with PackageExporter("file.zip") as e:
                ...
        """
        self._execute_dependency_graph()
        if self.torch_is_available:
            self.script_module_serializer.write_files()
        self._finalize_zip()

    def closing_function(self):
        self.script_module_serializer.write_files()

    def persistent_id(self, obj):
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
            storage_present = self.storage_context.has_storage(storage)
            storage_id = self.storage_context.get_or_add_storage(storage)
            if not storage_present:
                if storage.device.type != "cpu":
                    storage = storage.cpu()
                num_bytes = storage.nbytes()
                self.zip_file.write_record(
                    f".data/{storage_id}.storage", storage.data_ptr(), num_bytes
                )
            return ("storage", storage_type, storage_id, location, storage_numel)
        return None
