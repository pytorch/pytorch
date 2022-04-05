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
<<<<<<< HEAD

import torch
from torch.serialization import location_tag, normalize_storage_type
from torch.types import Storage
from torch.utils.hooks import RemovableHandle

from ._digraph import DiGraph
from ._importlib import _normalize_path
from ._mangling import demangle, is_mangled
from ._package_pickler import create_pickler
from ._stdlib import is_stdlib_module
from ._zip_file import PackageZipFileWriter
from ._zip_file_torchscript import TorchScriptPackageZipFileWriter
from .find_file_dependencies import find_files_source_depends_on
from .glob_group import GlobGroup, GlobPattern
from .importer import Importer, OrderedImporter, sys_importer

_gate_torchscript_serialization = True

ActionHook = Callable[["PackageExporter", str], None]


class _ModuleProviderAction(Enum):
    """Represents one of the actions that :class:`PackageExporter` can take on a module.

    See :meth:`PackageExporter.extern` and friends for a description of what the actions do.
    """

    INTERN = 1
    EXTERN = 2
    MOCK = 3
    DENY = 4
    # Special case: when a module is mocked, PackageExporter writes out a
    # `_mock` module that implements our mocking stubs. If we re-package code,
    # we may encounter a `_mock` module from the original package. If we do,
    # just ignore it and write a `_mock` module once.
    REPACKAGED_MOCK_MODULE = 5
    # Special case: PackageImporter adds a fake module
    # (`torch_package_importer`) that allows packaged code to access it. Don't
    # re-export this.
    SKIP = 6


class PackagingErrorReason(Enum):
    """Listing of different reasons a dependency may fail to package.

    This enum is used to provide good error messages when
    :class:`PackagingError` is raised.
    """

    def __repr__(self):
        return "<%s.%s>" % (self.__class__.__name__, self.name)

    IS_EXTENSION_MODULE = (
        "Module is a C extension module. torch.package supports Python modules only."
    )
    NO_DUNDER_FILE = "Module had no __file__ defined."
    SOURCE_FILE_NOT_FOUND = (
        "Module had a __file__, but we could not find it in your filesystem."
    )
    DEPENDENCY_RESOLUTION_FAILED = "Dependency resolution failed."
    NO_ACTION = (
        "Module did not match against any action pattern. Extern, mock, or intern it."
    )
    DENIED = "Module was denied by a pattern."
    MOCKED_BUT_STILL_USED = (
        "Module was mocked out, but is still being used in the package. "
        "Please intern or extern the mocked modules if objects are supposed to be in "
        "the package."
    )


@dataclass
class _PatternInfo:
    """Holds :class:`PackageExporter`-specific info about how to execute matches against"""

    # What action to take on a module that matches this pattern.
    action: _ModuleProviderAction
    # The value of `allow_empty` the user gave when specifying the pattern.
    allow_empty: bool
    # Whether this pattern has been matched during packaging.
    was_matched: bool

    def __init__(self, action, allow_empty):
        self.action = action
        self.allow_empty = allow_empty
        self.was_matched = False


class EmptyMatchError(Exception):
    """This is an exception that is thrown when a mock or extern is marked as
    ``allow_empty=False``, and is not matched with any module during packaging.
    """

    pass


class PackagingError(Exception):
    """This exception is raised when there is an issue with exporting a package.
    ``PackageExporter`` will attempt to gather up all the errors and present
    them to you at once.
    """

    def __init__(self, dependency_graph: DiGraph):
        # Group errors by reason.
        broken: Dict[PackagingErrorReason, List[str]] = defaultdict(list)
        for module_name, attrs in dependency_graph.nodes.items():
            error = attrs.get("error")
            if error is None:
                continue
            if error == PackagingErrorReason.NO_ACTION:
                assert "action" not in attrs
            broken[error].append(module_name)

        message = io.StringIO()
        message.write("\n")

        for reason, module_names in broken.items():
            message.write(f"* {reason.value}\n")
            for module_name in module_names:
                message.write(f"    {module_name}\n")

                # Print additional context if it's provided.
                error_context = dependency_graph.nodes[module_name].get("error_context")
                if error_context is not None:
                    message.write(f"      Context: {error_context}\n")

        # Save the dependency graph so that tooling can get at it.
        self.dependency_graph = dependency_graph
        super().__init__(message.getvalue())


class PackageExporter:
    """Exporters allow you to write packages of code, pickled Python data, and
    arbitrary binary and text resources into a self-contained package.

    Imports can load this code in a hermetic way, such that code is loaded
    from the package rather than the normal Python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The code contained in packages is copied file-by-file from the original
    source when it is created, and the file format is a specially organized
    zip file. Future users of the package can unzip the package, and edit the code
    in order to perform custom modifications to it.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external using :meth:`extern`.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.
=======
from pathlib import Path
>>>>>>> caf0c2a67c (remove need for torch in torch.package)

class PackageExporter(DefaultPackageExporter):
    """
    A package exporter for specialized functionality for torch. Specifically it uses the optimizations
    of torch's storage in order to not duplicate tensors.
    """
    def __init__(
        self,
        f: Union[str, Path, BinaryIO],
<<<<<<< HEAD
        importer: Union[Importer, Sequence[Importer]] = sys_importer,
        zip_file_writer_type: Type[
            PackageZipFileWriter
        ] = TorchScriptPackageZipFileWriter,
    ):
        """
        Create an exporter.

        Args:
            f: The location to export to. Can be a  ``string``/``Path`` object containing a filename
                or a binary I/O object.
            importer: If a single Importer is passed, use that to search for modules.
                If a sequence of importers are passsed, an ``OrderedImporter`` will be constructed out of them.
            zip_file_writer_type: A subclass of PackageZipFileWriter which would be used to instantiate the zip file writer
        """

        self.zip_file = zip_file_writer_type(f)

        self._written_files: Set[str] = set()

        self.serialized_reduces: Dict[int, Any] = {}

        # A graph tracking all the modules and pickle objects added to this
        # package and the dependencies between them.
        # - Each node is a module name (or a pickle name that looks like '<foo.obj.pkl>')
        # - Each directed edge (u, v) means u depends on v.
        # - Nodes may contain metadata that describe how to write the thing to the zipfile.
        self.dependency_graph = DiGraph()

        # These are OrderedDicts for compatibility with RemovableHandle.
        # Generic OrderedDict type annotations are not present until 3.7.
        # The real type signature is OrderedDict[int, Callable[[PackageExporter, str], None]]
        self._extern_hooks: OrderedDict = OrderedDict()
        self._mock_hooks: OrderedDict = OrderedDict()
        self._intern_hooks: OrderedDict = OrderedDict()
        if isinstance(importer, Importer):
            self.importer = importer
        else:
            if not isinstance(importer, collections.abc.Sequence):
                raise TypeError(
                    "importer arg should be an Importer or a sequence of Importers, "
                    f"got {type(importer)} instead."
                )
            self.importer = OrderedImporter(*importer)

        self.patterns: Dict[GlobGroup, _PatternInfo] = {}
        self._unique_id = 0

    def save_source_file(
        self, module_name: str, file_or_directory: str, dependencies=True
    ):
        """Adds the local file system ``file_or_directory`` to the source package to provide the code
        for ``module_name``.

        Args:
            module_name (str): e.g. ``"my_package.my_subpackage"``, code will be saved to provide code for this package.
            file_or_directory (str): the path to a file or directory of code. When a directory, all python files in the directory
                are recursively copied using :meth:`save_source_file`. If a file is named ``"/__init__.py"`` the code is treated
                as a package.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
        path = Path(file_or_directory)
        if path.is_dir():
            to_save = []  # list of tuples with arguments to save_source_string
            module_path = module_name.replace(".", "/")
            for filename in path.glob("**/*.py"):
                relative_path = filename.relative_to(path).as_posix()
                archivename = module_path + "/" + relative_path
                submodule_name = None
                if filename.name == "__init__.py":
                    submodule_name = archivename[: -len("/__init__.py")].replace(
                        "/", "."
                    )
                    is_package = True
                else:
                    submodule_name = archivename[: -len(".py")].replace("/", ".")
                    is_package = False

                # we delay the call to save_source_string so that we record all the source files
                # being provided by this directory structure _before_ attempting to resolve the dependencies
                # on the source. This makes sure we don't try to copy over modules that will just get
                # overwritten by this directory blob
                to_save.append(
                    (
                        submodule_name,
                        _read_file(str(filename)),
                        is_package,
                        dependencies,
                    )
                )

            for item in to_save:
                self.save_source_string(*item)
        else:
            is_package = path.name == "__init__.py"
            self.save_source_string(
                module_name,
                _read_file(file_or_directory),
                is_package,
                dependencies,
            )

    def get_unique_id(self) -> str:
        """Get an id. This id is guaranteed to only be handed out once for this package."""
        ret = str(self._unique_id)
        self._unique_id += 1
        return ret

    def _get_dependencies(
        self, src: str, module_name: str, is_package: bool
    ) -> List[str]:
        """Return all modules that this source code depends on.

        Dependencies are found by scanning the source code for import-like statements.

        Arguments:
            src: The Python source code to analyze for dependencies.
            module_name: The name of the module that ``src`` corresponds to.
            is_package: Whether this module should be treated as a package.
                See :py:meth:`save_source_string` for more info.

        Returns:
            A list containing modules detected as direct dependencies in
            ``src``.  The items in the list are guaranteed to be unique.
        """
        package_name = (
            module_name if is_package else module_name.rsplit(".", maxsplit=1)[0]
        )
        try:
            dep_pairs = find_files_source_depends_on(src, package_name)
        except Exception as e:
            self.dependency_graph.add_node(
                module_name,
                error=PackagingErrorReason.DEPENDENCY_RESOLUTION_FAILED,
                error_context=str(e),
            )
            return []

        # Use a dict to get uniquing but also deterministic order
        dependencies = {}
        for dep_module_name, dep_module_obj in dep_pairs:
            # handle the case where someone did something like `from pack import sub`
            # where `sub` is a submodule. In this case we don't have to save pack, just sub.
            # this ensures we don't pick up additional dependencies on pack.
            # However, in the case where `sub` is not a submodule but an object, then we do have
            # to save pack.
            if dep_module_obj is not None:
                possible_submodule = f"{dep_module_name}.{dep_module_obj}"
                if self._module_exists(possible_submodule):
                    dependencies[possible_submodule] = True
                    # we don't need to save `pack`
                    continue
            if self._module_exists(dep_module_name):
                dependencies[dep_module_name] = True

        return list(dependencies.keys())

    def save_source_string(
        self,
        module_name: str,
        src: str,
        is_package: bool = False,
        dependencies: bool = True,
    ):
        """Adds ``src`` as the source code for ``module_name`` in the exported package.

        Args:
            module_name (str): e.g. ``my_package.my_subpackage``, code will be saved to provide code for this package.
            src (str): The Python source code to save for this package.
            is_package (bool, optional): If ``True``, this module is treated as a package. Packages are allowed to have submodules
                (e.g. ``my_package.my_subpackage.my_subsubpackage``), and resources can be saved inside them. Defaults to ``False``.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
        self.dependency_graph.add_node(
            module_name,
            source=src,
            is_package=is_package,
            provided=True,
            action=_ModuleProviderAction.INTERN,
        )

        if dependencies:
            deps = self._get_dependencies(src, module_name, is_package)

            for dep in deps:
                self.dependency_graph.add_edge(module_name, dep)
                self.add_dependency(dep)

    def _write_source_string(
        self,
        module_name: str,
        src: str,
        is_package: bool = False,
    ):
        """Write ``src`` as the source code for ``module_name`` in the zip archive.

        Arguments are otherwise the same as for :meth:`save_source_string`.
        """
        extension = "/__init__.py" if is_package else ".py"
        filename = module_name.replace(".", "/") + extension

        self._write(filename, src)

    def _import_module(self, module_name: str):
        try:
            return self.importer.import_module(module_name)
        except ModuleNotFoundError as e:
            if not is_mangled(module_name):
                raise
            msg = (
                f"Module not found: '{module_name}'. Make sure the PackageImporter that "
                "created this module is present in `self.importer`"
            )
            raise ModuleNotFoundError(msg) from None

    def _module_exists(self, module_name: str) -> bool:
        try:
            self._import_module(module_name)
            return True
        except Exception:
            return False

    def _get_source_of_module(self, module: types.ModuleType) -> Optional[str]:
        filename = getattr(module, "__file__", None)
        result = (
            None
            if filename is None or not filename.endswith(".py")
            else linecache.getlines(filename, module.__dict__)
        )

        if result is None:
            return None

        return "".join(result)

    def add_dependency(self, module_name: str, dependencies=True):
        """Given a module, add it to the dependency graph according to patterns
        specified by the user.
        """
        if (
            module_name in self.dependency_graph
            and self.dependency_graph.nodes[module_name].get("provided") is True
        ):
            return

        # Special case: PackageImporter provides a special module called
        # `torch_package_importer` that allows packaged modules to reference
        # their PackageImporter. We don't want to re-export this.
        if module_name == "torch_package_importer":
            self.dependency_graph.add_node(
                module_name,
                action=_ModuleProviderAction.SKIP,
                provided=True,
            )
            return

        if module_name == "_mock":
            self.dependency_graph.add_node(
                module_name,
                action=_ModuleProviderAction.REPACKAGED_MOCK_MODULE,
                provided=True,
            )
            return

        if self._can_implicitly_extern(module_name):
            self.dependency_graph.add_node(
                module_name, action=_ModuleProviderAction.EXTERN, provided=True
            )
            return

        for pattern, pattern_info in self.patterns.items():
            if pattern.matches(module_name):
                pattern_info.was_matched = True
                self.dependency_graph.add_node(
                    module_name, action=pattern_info.action, provided=True
                )

                if pattern_info.action == _ModuleProviderAction.DENY:
                    # Requiring a denied module just adds an error to the graph.
                    self.dependency_graph.add_node(
                        module_name, error=PackagingErrorReason.DENIED
                    )

                # If we are interning this module, we need to retrieve its
                # dependencies and package those as well.
                if pattern_info.action == _ModuleProviderAction.INTERN:
                    self._intern_module(module_name, dependencies)
                return

        # No patterns have matched. Explicitly add this as an error.
        self.dependency_graph.add_node(
            module_name, error=PackagingErrorReason.NO_ACTION
        )

    def save_module(self, module_name: str, dependencies=True):
        """Save the code for ``module`` into the package. Code for the module is resolved using the ``importers`` path to find the
        module object, and then using its ``__file__`` attribute to find the source code.

        Args:
            module_name (str): e.g. ``my_package.my_subpackage``, code will be saved to provide code
                for this package.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
        if not isinstance(module_name, str):
            raise TypeError(
                "save_module() expects a string input, did you perhaps mean to pass `__name__`?"
            )

        self._intern_module(module_name, dependencies)

    def _intern_module(
        self,
        module_name: str,
        dependencies: bool,
=======
        importer: Union[Importer, Sequence[Importer]] = sys_importer
>>>>>>> caf0c2a67c (remove need for torch in torch.package)
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
