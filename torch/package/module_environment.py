import importlib
import sys
from contextlib import contextmanager
from pickle import _getattribute  # type: ignore
from types import ModuleType
from typing import Any, List, Optional, Tuple

from ._mangling import demangle, get_mangle_prefix, is_mangled
from .importer import BaseImporter


class ModuleEnvError(Exception):
    """Raised by ModuleEnv.check_obj."""

    pass


class _DefaultImporter(BaseImporter):
    """An importer that implements the default behavior of Python."""

    def __init__(self):
        self.modules = sys.modules

    def import_module(self, module_name: str):
        return importlib.import_module(module_name)


DefaultImporter = _DefaultImporter()


class ModuleEnv:
    """An environment that manages multiple module importers with overlapping names.

    By default, you can figure out what module an object belongs by checking
    __module__ and importing the result using __import__ or importlib.import_module.

    torch.package introduces module importers other than the default one.
    Each PackageImporter introduces a new namespace. Potentially a single
    name (e.g. 'foo.bar') is present in multiple namespaces.

    ModuleEnv provides a structured way to represent a collection of
    importers and ensuring that we are retrieving the correct modules for a
    given name.

    It supports two main operations:
        get_name: object -> (parent module name, name of obj within module)
        import_module: module_name -> module

    The guarantee is that following round-trip will succeed or throw a ModuleEnvError.
        module_name, obj_name = env.get_name(obj)
        module = env.import_module(module_name)
        obj2 = getattr(module, obj_name)
        assert obj1 is obj2
    """

    def __init__(self, importers: Optional[List[BaseImporter]] = None):
        if importers is None:
            importers = [DefaultImporter]
        self._importers = importers

    def import_module(self, module_name: str) -> ModuleType:
        """Import 'module_name' from this environment.

        Importers will be tried in order of `self.importers`, and the first match will be taken.
        """
        last_err = None
        for importer in self._importers:
            if not isinstance(importer, BaseImporter):
                raise TypeError(
                    f"{importer} is not a BaseImporter. "
                    "All importers in ModuleEnv must inherit from BaseImporter."
                )
            try:
                return importer.import_module(module_name)
            except ModuleNotFoundError as err:
                last_err = err

        if last_err is not None:
            raise last_err
        else:
            raise ModuleNotFoundError(module_name)

    def get_name(
        self, obj: Any, name: Optional[str] = None, check_obj: bool = True
    ) -> Tuple[str, str]:
        """Given an object, return a name that can be used to retrieve the
        object from this environment.

        Args:
            obj: An object to get the the module-environment-relative name for.
            name: If set, use this name instead of looking up __name__ or __qualname__ on `obj`.
                This is only here to match how Pickler handles __reduce__ functions that return a string,
                don't use otherwise.
            check_obj: If true, check that retrieving the returned name produces the same object as `obj`.
                This should only be false if you plan to call `check_obj()` somewhere else.

        Returns:
            A tuple (parent_module_name, attr_name) that can be used to retrieve `obj` from this environment.
            Use it like:
                mod = module_env.import_module(parent_module_name)
                obj = getattr(mod, attr_name)

        Raises:
            ModuleEnvError: `check_obj()` has failed.
        """
        if name is None:
            name = getattr(obj, "__qualname__", None)
        if name is None:
            name = obj.__name__

        orig_module_name = self._whichmodule(obj, name)
        # Demangle the module name before importing. If this obj came out of a
        # PackageImporter, `__module__` will be mangled. See mangling.md for
        # details.
        module_name = demangle(orig_module_name)

        if check_obj:
            module = self.import_module(module_name)
            obj2, _ = _getattribute(module, name)
            self.check_obj(obj, obj2, name)

        return module_name, name

    def is_default(self) -> bool:
        """Returns true if this ModuleEnv only represents the default Python environment."""
        return self._importers == [DefaultImporter]

    def check_obj(self, obj, obj2, name=None):
        """Checks that two objects are the same identity.

        Don't use this method, prefer to use the automatic check in `get_name`.
        It's only separate to help implement CustomImportPickler.

        Args:
            obj: The object that the user gave us. This will be treated as such in error messaging.
            obj2: The object retrieve by looking up a name in `self`.
            name: If set, use this name instead of looking up __name__ or __qualname__ in error reporting.
                This is only here to match how Pickler handles __reduce__ functions that return a string,
                don't use otherwise.

        Returns:
            None

        Raises:
            ModuleEnvError: The two objects are not the same identity. The exception message will contain
                a description and suggestions for fixing the issue.
        """
        if obj is obj2:
            return

        if name is None:
            name = getattr(obj, "__qualname__", None)
        if name is None:
            name = obj.__name__

        orig_module_name = self._whichmodule(obj, name)
        obj_module_name = getattr(obj, "__module__", orig_module_name)
        obj2_module_name = getattr(obj2, "__module__", self._whichmodule(obj2, name))

        is_obj_mangled = is_mangled(obj_module_name)
        is_obj2_mangled = is_mangled(obj2_module_name)

        msg = ""
        if is_obj_mangled or is_obj2_mangled:
            obj_location = (
                get_mangle_prefix(obj_module_name)
                if is_obj_mangled
                else "the current Python environment"
            )
            obj2_location = (
                get_mangle_prefix(obj2_module_name)
                if is_obj2_mangled
                else "the current Python environment"
            )

            obj_importer_name = (
                f"the importer for {get_mangle_prefix(obj_module_name)}"
                if is_obj_mangled
                else "'DefaultImporter'"
            )
            obj2_importer_name = (
                f"the importer for {get_mangle_prefix(obj2_module_name)}"
                if is_obj2_mangled
                else "'DefaultImporter'"
            )

            msg = (
                f"\n\nThe object provided is from '{orig_module_name}', "
                f"which is coming from {obj_location}."
                f"\nHowever, when we import '{obj2_module_name}', it's coming from {obj2_location}."
                "\nTo fix this, make sure 'PackageExporter.importers' lists "
                f"{obj_importer_name} before {obj2_importer_name}."
            )
        raise ModuleEnvError(msg)

    def _whichmodule(self, obj, name):
        """Find the module an object belongs to.

        Taken from pickle.py, but modified to search our importer module
        lists instead of sys.modules.
        """
        module_name = getattr(obj, "__module__", None)
        if module_name is not None:
            return module_name
        # Protect the iteration by using a list copy of importer.modules against dynamic
        # modules that trigger imports of other modules upon calls to getattr.
        for importer in self._importers:
            for module_name, module in importer.modules.copy().items():
                if (
                    module_name == "__main__"
                    or module_name == "__mp_main__"  # bpo-42406
                    or module is None
                ):
                    continue
                try:
                    if _getattribute(module, name)[0] is obj:
                        return module_name
                except AttributeError:
                    pass
        return "__main__"
