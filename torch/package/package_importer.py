import torch
from typing import Dict, Union, BinaryIO, Callable, Any
import types
from pathlib import Path
from ._zip_file import PackageZipFileReader
from ._zip_file_torchscript import TorchScriptPackageZipFileReader
from ._directory_reader_torchscript import TorchScriptDirectoryReader
from torch.serialization import _get_restore_location, _maybe_decode_ascii
import io
from contextlib import contextmanager
from .package_importer_oss import PackageImporter as PI

class PackageImporter(PI):
    """Importers allow you to load code written to packages by :class:`PackageExporter`.
    Code is loaded in a hermetic way, using files from the package
    rather than the normal python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external during export.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.
    """

    """The dictionary of already loaded modules from this package, equivalent to ``sys.modules`` but
    local to this importer.
    """
    modules: Dict[str, types.ModuleType]

    def __init__(
        self,
        file_or_buffer: Union[str, torch._C.PyTorchFileReader, PackageZipFileReader, Path, BinaryIO],
        module_allowed: Callable[[str], bool] = lambda module_name: True
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
        if isinstance(file_or_buffer, torch._C.PyTorchFileReader):
            file_or_buffer = TorchScriptPackageZipFileReader(file_or_buffer)
        super().__init__(file_or_buffer,
                         module_allowed,
                         zip_file_reader_type=TorchScriptPackageZipFileReader,
                         directory_reader_type=TorchScriptDirectoryReader)

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
        restore_location = _get_restore_location(map_location)
        self.restore_location = _get_restore_location(map_location)
        loaded_storages = {}
        loaded_reduces = {}
        storage_context = torch._C.DeserializationStorageContext()

        # TODO move out and add deprecration warning for this behavior
        # TODO move to package shim
        def load_tensor(dtype, size, key, location, restore_location):
            name = f"{key}.storage"

            if storage_context.has_storage(name):
                storage = storage_context.get_storage(name, dtype).storage()
            else:
                tensor = self.zip_reader.get_storage_from_record(
                    ".data/" + name, size, dtype
                )
                if isinstance(self.zip_reader, torch._C.PyTorchFileReader):
                    storage_context.add_storage(name, tensor)
                storage = tensor.storage()
            loaded_storages[key] = restore_location(storage, location)

        def persistent_load(saved_id):
            assert isinstance(saved_id, tuple)
            typename = _maybe_decode_ascii(saved_id[0])
            data = saved_id[1:]

            if typename == "storage":
                storage_type, key, location, size = data
                dtype = storage_type.dtype

                if key not in loaded_storages:
                    load_tensor(
                        dtype,
                        size,
                        key,
                        _maybe_decode_ascii(location),
                        restore_location,
                    )
                storage = loaded_storages[key]
                # TODO: Once we decide to break serialization FC, we can
                # stop wrapping with TypedStorage
                return torch.storage.TypedStorage(
                    wrap_storage=storage._untyped(), dtype=dtype
                )
            elif typename == "reduce_package":
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
        unpickler.persistent_load = persistent_load

        # TODO: it might make sense to have a seperate packager in torch which uses the OSS pacakge but saves state
        @contextmanager
        def set_deserialization_context():
            # to let reduce_package access deserializaiton context
            self.external_registry = {}
            self.storage_context = storage_context
            self.last_map_location = map_location
            try:
                yield
            finally:
                self.external_registry = None
                self.storage_context = None
                self.last_map_location = None

        with set_deserialization_context():
            result = unpickler.load()

        # TODO from zdevito:
        #   This stateful weird function will need to be removed in our efforts
        #   to unify the format. It has a race condition if multiple python
        #   threads try to read independent files
        torch._utils._validate_loaded_sparse_tensors()

        return result
