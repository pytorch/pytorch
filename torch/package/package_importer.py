import io
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union, BinaryIO, Callable, Dict

import torch
from torch.serialization import _get_restore_location

from ._directory_reader_torchscript import TorchScriptDirectoryReader
from ._zip_file_torchscript import TorchScriptPackageZipFileReader
from .package_importer_no_torch import _maybe_decode_ascii
from .package_importer_no_torch import PackageImporter as DefaultPackageImporter


class PackageImporter(DefaultPackageImporter):
    def __init__(
        self,
        file_or_buffer: Union[str, Path, BinaryIO],
        module_allowed: Callable[[str], bool] = lambda module_name: True,
    ):
        """Open ``file_or_buffer`` for importing. This checks that the imported package only requires modules
        allowed by ``module_allowed``

        Args:
            file_or_buffer: a file-like object (has to implement :meth:`read`, :meth:`readline`, :meth:`tell`, and :meth:`seek`),
                a string, or an ``os.PathLike`` object containing a filename.
            module_allowed (Callable[[str], bool], optional): A method to determine if a externally provided module
                should be allowed. Can be used to ensure packages loaded do not depend on modules that the server
                does not support. Defaults to allowing anything.

        Raises:
            ImportError: If the package will use a disallowed module.
        """
        super(PackageImporter, self).__init__(
            file_or_buffer,
            module_allowed,
            zip_file_reader_type=TorchScriptPackageZipFileReader,
        )

    def persistent_load(self, typename, data):
        assert isinstance(
            self.zip_reader,
            (TorchScriptDirectoryReader, TorchScriptPackageZipFileReader),
        )

        def load_tensor(dtype, size, key, location, restore_location):
            assert self.loaded_storages is not None
            name = f"{key}.storage"

            if self.storage_context.has_storage(name):
                storage = self.storage_context.get_storage(name, dtype).storage()
            else:
                tensor = self.zip_reader.get_storage_from_record(  # type: ignore[attr-defined]
                    ".data/" + name, size, dtype
                )
                if not self.zip_reader.is_directory():
                    self.storage_context.add_storage(name, tensor)
                storage = tensor.storage()
            self.loaded_storages[key] = restore_location(storage, location)

        if typename == "storage":
            storage_type, key, location, size = data
            dtype = storage_type.dtype
            assert self.loaded_storages is not None
            if key not in self.loaded_storages:
                load_tensor(
                    dtype,
                    size,
                    key,
                    _maybe_decode_ascii(location),
                    self.restore_location,
                )
            storage = self.loaded_storages[key]
            # TODO: Once we decide to break serialization FC, we can
            # stop wrapping with TypedStorage
            return torch.storage._TypedStorage(
                wrap_storage=storage._untyped(), dtype=dtype
            )
        return None

    @contextmanager
    def set_torch_deserialization_context(self, map_location):
        # to let reduce_package access deserializaiton context
        self.storage_context = torch._C.DeserializationStorageContext()
        self.last_map_location = map_location
        self.restore_location = _get_restore_location(map_location)
        self.loaded_storages: Union[Dict[int, Any], None] = {}
        try:
            yield
        finally:
            self.storage_context = None
            self.last_map_location = None
            self.restore_location = None
            self.loaded_storages = None

    # TODO: load_pickle to reduce the repeated code between this and the non-torch version
    def load_pickle(self, package: str, resource: str, map_location=None) -> Any:
        """Unpickles the resource from the package, loading any modules that are needed to construct the objects
        using :meth:`import_module`.

        Args:
            package (str): The name of module package (e.g. ``"my_package.my_subpackage"``).
            resource (str): The unique name for the resource.
            map_location: Passed to `torch.load` to determine how tensors are mapped to devices. Defaults to ``None``.

        Returns:
            Any: The unpickled object.
        """
        pickle_file = self._zipfile_path(package, resource)
        loaded_reduces = {}

        def _persistent_load(saved_id):
            assert isinstance(saved_id, tuple)
            typename = _maybe_decode_ascii(saved_id[0])
            data = saved_id[1:]
            module = self.persistent_load(typename, data)
            if module is not None:
                return module
            if typename == "reduce_package":
                # to fix BC breaking change, objects on this load path
                # will be loaded multiple times erroneously
                if len(data) == 2:
                    func, args = data
                    return func(self, *args)
                reduce_id, func, args = data
                if reduce_id not in loaded_reduces:
                    loaded_reduces[reduce_id] = func(self, *args)
                return loaded_reduces[reduce_id]
            else:
                f"Unknown typename for persistent_load, expected 'storage' or 'reduce_package' but got '{typename}'"

        # Load the data (which may in turn use `persistent_load` to load tensors)
        data_file = io.BytesIO(self.zip_reader.get_record(pickle_file))
        unpickler = self.Unpickler(data_file)
        unpickler.persistent_load = _persistent_load
        with self.set_torch_deserialization_context(map_location):
            result = unpickler.load()
            # TODO from zdevito:
            #   This stateful weird function will need to be removed in our efforts
            #   to unify the format. It has a race condition if multiple python
            #   threads try to read independent files
            torch._utils._validate_loaded_sparse_tensors()

        return result
