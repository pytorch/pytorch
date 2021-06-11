import collections
import importlib.machinery
import io
import linecache
import pickletools
import types
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)
from urllib.parse import quote

import torch
from torch.serialization import location_tag, normalize_storage_type
from torch.utils.hooks import RemovableHandle

from ._digraph import DiGraph
from ._importlib import _normalize_path
from ._mangling import is_mangled
from ._package_pickler import create_pickler
from ._stdlib import is_stdlib_module
from .find_file_dependencies import find_files_source_depends_on
from .glob_group import GlobGroup, GlobPattern
from .importer import Importer, OrderedImporter, sys_importer

ActionHook = Callable[["PackageExporter", str], None]


class _ModuleProviderAction(Enum):
    """Represents one of the actions that :class:`PackageExporter` can take on a module.

    See :meth:`PackageExporter.extern` and friends for a description of what the actions do.
    """

    INTERN = 1
    EXTERN = 2
    MOCK = 3
    DENY = 4


class PackagingErrorReason(Enum):
    """Listing of different reasons a dependency may fail to package.

    This enum is used to provide good error messages when
    :class:`PackagingError` is raised.
    """
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)

    IS_EXTENSION_MODULE = "Module is a C extension module. torch.package supports Python modules only."
    NO_DUNDER_FILE = "Module had no __file__ defined."
    SOURCE_FILE_NOT_FOUND = (
        "Module had a __file__, but we could not find it in your filesystem."
    )
    DEPENDENCY_RESOLUTION_FAILED = "Dependency resolution failed."
    NO_ACTION = (
        "Module did not match against any action pattern. Extern, mock, or intern it."
    )
    DENIED = "Module was denied by a pattern."


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

    When source code is added to the package, the exporter can optionally scan it
    for further code dependencies (``dependencies=True``). It looks for import statements,
    resolves relative references to qualified module names, and performs an action specified by the user
    (See: :meth:`extern`, :meth:`mock`, and :meth:`intern`).
    """

    """A importer that will be searched in order to find the modules referenced by other modules or by
    pickled objects. The default module environment just uses sys_importer, which searches the Python environment.
    """
    importer: Importer

    def __init__(
        self,
        f: Union[str, Path, BinaryIO],
        importer: Union[Importer, Sequence[Importer]] = sys_importer,
        verbose: bool = True,
    ):
        """
        Create an exporter.

        Args:
            f: The location to export to. Can be a  ``string``/``Path`` object containing a filename
                or a binary I/O object.
            importer: If a single Importer is passed, use that to search for modules.
                If a sequence of importers are passsed, an ``OrderedImporter`` will be constructed out of them.
            verbose: Print information about dependency resolution to stdout.
                Useful for tracking down why certain files get included.
        """
        if isinstance(f, (Path, str)):
            f = str(f)
            self.buffer: Optional[BinaryIO] = None
        else:  # is a byte buffer
            self.buffer = f

        self.zip_file = torch._C.PyTorchFileWriter(f)
        self.zip_file.set_min_version(6)
        self.serialized_reduces: Dict[int, Any] = {}
        self.serialized_storages: Set[str] = set()

        # A graph tracking all the modules and pickle objects added to this
        # package and the dependencies between them.
        # - Each node is a module name (or a pickle name that looks like '<foo.obj.pkl>')
        # - Each directed edge (u, v) means u depends on v.
        # - Nodes may contain metadata that describe how to write the thing to the zipfile.
        self.dependency_graph = DiGraph()
        self.verbose = verbose
        self.script_module_serializer = torch._C.ScriptModuleSerializer(self.zip_file)

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

        if self.verbose:
            dep_str = "".join(f"  {dep}\n" for dep in dependencies)
            print(f"{module_name} depends on:\n{dep_str}\n")

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
                self.require_module_if_not_provided(dep)

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
                f"Module not found: '{module_name}'. Modules imported "
                "from a torch.package cannot be re-exported directly."
            )
            raise ModuleNotFoundError(msg) from None

    def _module_exists(self, module_name: str) -> bool:
        try:
            self._import_module(module_name)
            return True
        except Exception:
            return False

    def _write_dep_graph(self, failing_module=None):
        edges = "\n".join(f'"{f}" -> "{t}";' for f, t in self.dependency_graph.edges)
        failing = "" if failing_module is None else f'"{failing_module}" [color=red];'
        template = f"""\
digraph G {{
rankdir = LR;
node [shape=box];
{failing}
{edges}
}}
"""
        arg = quote(template, safe="")
        return f"https://dreampuf.github.io/GraphvizOnline/#{arg}"

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

    def require_module_if_not_provided(self, module_name: str, dependencies=True):
        if (
            module_name in self.dependency_graph
            and self.dependency_graph.nodes[module_name].get("provided") is True
        ):
            return

        if self._can_implicitly_extern(module_name):
            if self.verbose:
                print(
                    f"implicitly adding {module_name} to external modules "
                    f"since it is part of the standard library and is a dependency."
                )
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
                    self._add_module_to_dependency_graph(module_name, dependencies)
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

        self.dependency_graph.add_node(
            module_name, provided=True, action=_ModuleProviderAction.INTERN
        )
        self._add_module_to_dependency_graph(module_name, dependencies)

    def _add_module_to_dependency_graph(
        self,
        module_name: str,
        dependencies: bool,
    ):
        module_obj = self._import_module(module_name)

        # Find dependencies of this module and require them as well.
        is_package = hasattr(module_obj, "__path__")
        source = self._get_source_of_module(module_obj)
        if source is None:
            # Couldn't find a source!  Add it to our dependency graph as broken
            # and continue.
            filename = getattr(module_obj, "__file__", None)
            error_context = None
            if filename is None:
                packaging_error = PackagingErrorReason.NO_DUNDER_FILE
            elif filename.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
                packaging_error = PackagingErrorReason.IS_EXTENSION_MODULE
            else:
                packaging_error = PackagingErrorReason.SOURCE_FILE_NOT_FOUND
                error_context = f"filename: {filename}"
            self.dependency_graph.add_node(
                module_name,
                is_package=is_package,
                error=packaging_error,
                error_context=error_context,
            )
            return

        self.dependency_graph.add_node(
            module_name, is_package=is_package, source=source, provided=True
        )

        if dependencies:
            deps = self._get_dependencies(source, module_name, is_package)
            for dep in deps:
                self.dependency_graph.add_edge(module_name, dep)
                self.require_module_if_not_provided(dep)

    def save_pickle(
        self, package: str, resource: str, obj: Any, dependencies: bool = True
    ):
        """Save a python object to the archive using pickle. Equivalent to :func:`torch.save` but saving into
        the archive rather than a stand-alone file. Stanard pickle does not save the code, only the objects.
        If ``dependencies`` is true, this method will also scan the pickled objects for which modules are required
        to reconstruct them and save the relevant code.

        To be able to save an object where ``type(obj).__name__`` is ``my_module.MyObject``,
        ``my_module.MyObject`` must resolve to the class of the object according to the ``importer`` order. When saving objects that
        have previously been packaged, the importer's ``import_module`` method will need to be present in the ``importer`` list
        for this to work.

        Args:
            package (str): The name of module package this resource should go in (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            obj (Any): The object to save, must be picklable.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
        filename = self._filename(package, resource)
        # Write the pickle data for `obj`
        data_buf = io.BytesIO()
        pickler = create_pickler(data_buf, self.importer)
        pickler.persistent_id = self._persistent_id
        pickler.dump(obj)
        data_value = data_buf.getvalue()

        name_in_dependency_graph = f"<{package}.{resource}>"
        self.dependency_graph.add_node(
            name_in_dependency_graph,
            action=_ModuleProviderAction.INTERN,
            provided=True,
            is_pickle=True,
        )

        if dependencies:
            all_dependencies = []
            for opcode, arg, pos in pickletools.genops(data_value):
                if opcode.name == "GLOBAL":  # a global reference
                    assert isinstance(arg, str)
                    module, field = arg.split(" ")
                    if module not in all_dependencies:
                        all_dependencies.append(module)

            if self.verbose:
                dep_string = "".join(f"  {dep}\n" for dep in all_dependencies)
                print(f"{resource} depends on:\n{dep_string}\n")

            for module_name in all_dependencies:
                self.dependency_graph.add_edge(name_in_dependency_graph, module_name)
                self.require_module_if_not_provided(module_name)

        self._write(filename, data_value)

    def save_text(self, package: str, resource: str, text: str):
        """Save text data to the package.

        Args:
            package (str): The name of module package this resource should go it (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            text (str): The contents to save.
        """
        return self.save_binary(package, resource, text.encode("utf-8"))

    def save_binary(self, package, resource, binary: bytes):
        """Save raw bytes to the package.

        Args:
            package (str): The name of module package this resource should go it (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            binary (str): The data to save.
        """
        filename = self._filename(package, resource)
        self._write(filename, binary)

    def register_extern_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers an extern hook on the exporter.

        The hook will be called each time a module matches against an :meth:`extern` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        handle = RemovableHandle(self._extern_hooks)
        self._extern_hooks[handle.id] = hook
        return handle

    def register_mock_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers a mock hook on the exporter.

        The hook will be called each time a module matches against a :meth:`mock` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        handle = RemovableHandle(self._mock_hooks)
        self._mock_hooks[handle.id] = hook
        return handle

    def register_intern_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers an intern hook on the exporter.

        The hook will be called each time a module matches against an :meth:`intern` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        handle = RemovableHandle(self._intern_hooks)
        self._intern_hooks[handle.id] = hook
        return handle

    def intern(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
    ):
        """Specify modules that should be packaged. A module must match some ``intern`` pattern in order to be
        included in the package and have its dependencies processed recursively.

        Args:
            include (Union[List[str], str]): A string e.g. "my_package.my_subpackage", or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as described in :meth:`mock`.

            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the include string.

            allow_empty (bool): An optional flag that specifies whether the intern modules specified by this call
                to the ``intern`` method must be matched to some module during packaging. If an ``intern`` module glob
                pattern is added with ``allow_empty=False``, and :meth:`close` is called (either explicitly or via ``__exit__``)
                before any modules match that pattern, an exception is thrown. If ``allow_empty=True``, no such exception is thrown.

        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.INTERN, allow_empty
        )

    def mock(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
    ):
        """Replace some required modules with a mock implementation.  Mocked modules will return a fake
        object for any attribute accessed from it. Because we copy file-by-file, the dependency resolution will sometimes
        find files that are imported by model files but whose functionality is never used
        (e.g. custom serialization code or training helpers).
        Use this function to mock this functionality out without having to modify the original code.

        Args:
            include (Union[List[str], str]): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be mocked out. Strings can also be a glob-style pattern
                string that may match multiple modules. Any required dependencies that match this pattern
                string will be mocked out automatically.

                Examples :
                    ``'torch.**'`` -- matches ``torch`` and all submodules of torch, e.g. ``'torch.nn'``
                    and ``'torch.nn.functional'``

                    ``'torch.*'`` -- matches ``'torch.nn'`` or ``'torch.functional'``, but not
                    ``'torch.nn.functional'``

            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the include string.
                e.g. ``include='torch.**', exclude='torch.foo'`` will mock all torch packages except ``'torch.foo'``,
                Default: is ``[]``.

            allow_empty (bool): An optional flag that specifies whether the mock implementation(s) specified by this call
                to the :meth:`mock` method must be matched to some module during packaging. If a mock is added with
                ``allow_empty=False``, and :meth:`close` is called (either explicitly or via ``__exit__``) and the mock has
                not been matched to a module used by the package being exported, an exception is thrown.
                If ``allow_empty=True``, no such exception is thrown.

        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.MOCK, allow_empty
        )

    def extern(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
    ):
        """Include ``module`` in the list of external modules the package can import.
        This will prevent dependency discovery from saving
        it in the package. The importer will load an external module directly from the standard import system.
        Code for extern modules must also exist in the process loading the package.

        Args:
            include (Union[List[str], str]): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as
                described in :meth:`mock`.

            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the
                include string.

            allow_empty (bool): An optional flag that specifies whether the extern modules specified by this call
                to the ``extern`` method must be matched to some module during packaging. If an extern module glob
                pattern is added with ``allow_empty=False``, and :meth:`close` is called (either explicitly or via
                ``__exit__``) before any modules match that pattern, an exception is thrown. If ``allow_empty=True``,
                no such exception is thrown.

        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.EXTERN, allow_empty
        )

    def deny(self, include: "GlobPattern", *, exclude: "GlobPattern" = ()):
        """Blocklist modules who names match the given glob patterns from the list of modules the package can import.
        If a dependency on any matching packages is found, a :class:`PackagingError` is raised.

        Args:
            include (Union[List[str], str]): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as described in :meth:`mock`.

            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the include string.
        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.DENY, allow_empty=True
        )

    def _persistent_id(self, obj):
        if torch.is_storage(obj):
            storage_type = normalize_storage_type(type(obj))
            obj_key = str(obj._cdata)
            location = location_tag(obj)
            name = f".data/{obj_key}.storage"

            if name not in self.serialized_storages:
                # check to see if storage was previously serialized
                serialized_files = self.zip_file.get_all_written_records()
                if name not in serialized_files:
                    if obj.device.type != "cpu":
                        obj = obj.cpu()
                    num_bytes = obj.size() * obj.element_size()
                    self.zip_file.write_record(name, obj.data_ptr(), num_bytes)
                self.serialized_storages.add(name)
            return ("storage", storage_type, obj_key, location, obj.size())

        if hasattr(obj, "__reduce_package__"):
            if self.serialized_reduces.get(id(obj)) is None:
                self.serialized_reduces[id(obj)] = ("reduce_package", id(obj), *obj.__reduce_package__(self))

            return self.serialized_reduces[id(obj)]

        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # If __exit__ was called because an exception was raised, we do not attempt to
        # attempt to finalize the package. Instead, control is returned to the
        # caller to continue raising the exception.
        if exc_type is not None:
            # Do the bare minimum to leave the open buffer in a valid state.
            self._finalize_zip()
            return

        self.close()

    def _write(self, filename, str_or_bytes):
        if is_mangled(filename):
            raise RuntimeError(
                f"Tried to save a torch.package'd module as '{filename}'. "
                "Directly saving torch.package'd modules is not allowed."
            )
        if isinstance(str_or_bytes, str):
            str_or_bytes = str_or_bytes.encode("utf-8")
        self.zip_file.write_record(filename, str_or_bytes, len(str_or_bytes))

    def _validate_dependency_graph(self):
        # 1. Check the graph for any errors inserted during dependency analysis.
        for module_name, attrs in self.dependency_graph.nodes.items():
            if "error" in attrs:
                raise PackagingError(self.dependency_graph)

        # 2. Check that all patterns for which allow_empty=False have been matched at least once.
        for pattern, pattern_info in self.patterns.items():
            if not pattern_info.allow_empty and not pattern_info.was_matched:
                raise EmptyMatchError(
                    f"Exporter did not match any modules to {pattern}, which was marked as allow_empty=False"
                )

    def _execute_dependency_graph(self):
        """Takes a finalized dependency graph describing how to package all
        modules and executes it, writing to the ZIP archive.
        """
        self._validate_dependency_graph()

        extern_modules = []
        _mock_written = False
        for module_name, attrs in self.dependency_graph.nodes.items():
            action = attrs["action"]

            if action == _ModuleProviderAction.EXTERN:
                for hook in self._extern_hooks.values():
                    hook(self, module_name)

                extern_modules.append(module_name)

            elif action == _ModuleProviderAction.MOCK:
                for hook in self._mock_hooks.values():
                    hook(self, module_name)

                if not _mock_written:
                    mock_file = str(Path(__file__).parent / "_mock.py")
                    self._write_source_string(
                        "_mock", _read_file(mock_file), is_package=False
                    )
                    _mock_written = True

                is_package = hasattr(self._import_module(module_name), "__path__")
                self._write_source_string(module_name, _MOCK_IMPL, is_package)

            elif action == _ModuleProviderAction.INTERN:
                for hook in self._intern_hooks.values():
                    hook(self, module_name)

                # The node in the dependency graph contains metadata that tells us
                # how to intern the module.
                if "provided" not in attrs:
                    raise AssertionError(
                        f"Module was marked `intern` but not provided: {module_name}"
                    )

                if attrs.get("is_pickle") is True:
                    # This node came from save_source_pickle, we don't need to write any source for it.
                    continue

                is_package = attrs["is_package"]
                source = attrs["source"]
                self._write_source_string(module_name, source, is_package)

            else:
                raise AssertionError(
                    f"Invalid action: {module_name}, {action}. Please report a bug to PyTorch."
                )

        extern_file_contents = "\n".join(extern_modules) + "\n"
        self._write(".data/extern_modules", extern_file_contents)

    def close(self):
        """Write the package to the filesystem. Any calls after :meth:`close` are now invalid.
        It is preferable to use resource guard syntax instead::

            with PackageExporter("file.zip") as e:
                ...
        """
        if self.verbose:
            print(f"Dependency graph for exported package: \n{self._write_dep_graph()}")

        self._execute_dependency_graph()

        self.script_module_serializer.write_files()
        self._finalize_zip()

    def _finalize_zip(self):
        """Called at the very end of packaging to leave the zipfile in a closed but valid state."""
        del self.zip_file
        if self.buffer:
            self.buffer.flush()

    def _filename(self, package, resource):
        package_path = package.replace(".", "/")
        resource = _normalize_path(resource)
        return f"{package_path}/{resource}"

    def _can_implicitly_extern(self, module_name: str):
        top_level_package_name = module_name.partition(".")[0]
        return top_level_package_name == "torch" or (
            top_level_package_name not in _DISALLOWED_MODULES
            and is_stdlib_module(top_level_package_name)
        )


# even though these are in the standard library, we do not allow them to be
# automatically externed since they offer a lot of system level access
_DISALLOWED_MODULES = ["sys", "io"]

_MOCK_IMPL = """\
from _mock import MockedObject
def __getattr__(attr: str):
    return MockedObject(__name__ + '.' + attr, _suppress_err=True)
"""


def _read_file(filename: str) -> str:
    with open(filename, "rb") as f:
        b = f.read()
        return b.decode("utf-8")
