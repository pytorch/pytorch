# mypy: allow-untyped-defs
# pyrefly: ignore [missing-module-attribute]
import sys
from pickle import (  # type: ignore[attr-defined]
    _compat_pickle,
    _extension_registry,
    _getattribute,
    _Pickler,
    EXT1,
    EXT2,
    EXT4,
    GLOBAL,
    PicklingError,
    STACK_GLOBAL,
)
from struct import pack
from types import FunctionType

from .importer import Importer, ObjMismatchError, ObjNotFoundError, sys_importer


class _PyTorchLegacyPickler(_Pickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._persistent_id = None

    def persistent_id(self, obj):
        if self._persistent_id is None:
            return super().persistent_id(obj)
        return self._persistent_id(obj)


class PackagePickler(_PyTorchLegacyPickler):
    """Package-aware pickler.

    This behaves the same as a normal pickler, except it uses an `Importer`
    to find objects and modules to save.
    """

    def __init__(self, importer: Importer, *args, **kwargs):
        self.importer = importer
        super().__init__(*args, **kwargs)

        # Make sure the dispatch table copied from _Pickler is up-to-date.
        # Previous issues have been encountered where a library (e.g. dill)
        # mutate _Pickler.dispatch, PackagePickler makes a copy when this lib
        # is imported, then the offending library removes its dispatch entries,
        # leaving PackagePickler with a stale dispatch table that may cause
        # unwanted behavior.
        self.dispatch = _Pickler.dispatch.copy()  # type: ignore[misc]
        self.dispatch[FunctionType] = PackagePickler.save_global  # type: ignore[assignment]

    def save_global(self, obj, name=None):
        # ruff: noqa: F841
        # unfortunately the pickler code is factored in a way that
        # forces us to copy/paste this function. The only change is marked
        # CHANGED below.
        write = self.write  # type: ignore[attr-defined]
        memo = self.memo  # type: ignore[attr-defined]

        # CHANGED: import module from module environment instead of __import__
        try:
            module_name, name = self.importer.get_name(obj, name)
        except (ObjNotFoundError, ObjMismatchError) as err:
            raise PicklingError(f"Can't pickle {obj}: {str(err)}") from err

        module = self.importer.import_module(module_name)
        if sys.version_info >= (3, 14):
            # pickle._getattribute signature changes in 3.14
            # to take iterable and return just the object (not tuple)
            # We need to get the parent object that contains the attribute
            name_parts = name.split(".")
            if "<locals>" in name_parts:
                raise PicklingError(f"Can't pickle local object {obj!r}")
            if len(name_parts) == 1:
                parent = module
            else:
                parent = _getattribute(module, name_parts[:-1])
        else:
            _, parent = _getattribute(module, name)
        # END CHANGED

        if self.proto >= 2:  # type: ignore[attr-defined]
            code = _extension_registry.get((module_name, name))
            if code:
                if code <= 0:
                    raise AssertionError(
                        f"expected positive extension code, got {code}"
                    )
                if code <= 0xFF:
                    write(EXT1 + pack("<B", code))
                elif code <= 0xFFFF:
                    write(EXT2 + pack("<H", code))
                else:
                    write(EXT4 + pack("<i", code))
                return
        lastname = name.rpartition(".")[2]
        if parent is module:
            name = lastname
        # Non-ASCII identifiers are supported only with protocols >= 3.
        if self.proto >= 4:  # type: ignore[attr-defined]
            self.save(module_name)  # type: ignore[attr-defined]
            self.save(name)  # type: ignore[attr-defined]
            write(STACK_GLOBAL)
        elif parent is not module:
            self.save_reduce(getattr, (parent, lastname))  # type: ignore[attr-defined]
        elif self.proto >= 3:  # type: ignore[attr-defined]
            write(
                GLOBAL
                + bytes(module_name, "utf-8")
                + b"\n"
                + bytes(name, "utf-8")
                + b"\n"
            )
        else:
            if self.fix_imports:  # type: ignore[attr-defined]
                r_name_mapping = _compat_pickle.REVERSE_NAME_MAPPING
                r_import_mapping = _compat_pickle.REVERSE_IMPORT_MAPPING
                if (module_name, name) in r_name_mapping:
                    module_name, name = r_name_mapping[(module_name, name)]
                elif module_name in r_import_mapping:
                    module_name = r_import_mapping[module_name]
            try:
                write(
                    GLOBAL
                    + bytes(module_name, "ascii")
                    + b"\n"
                    + bytes(name, "ascii")
                    + b"\n"
                )
            except UnicodeEncodeError as exc:
                raise PicklingError(
                    f"can't pickle global identifier '{module}.{name}' using "
                    f"pickle protocol {self.proto:d}"  # type: ignore[attr-defined]
                ) from exc

        self.memoize(obj)  # type: ignore[attr-defined]


def create_pickler(data_buf, importer, protocol=4):
    if importer is sys_importer:
        # if we are using the normal import library system, then
        # we can use the C implementation of pickle which is faster
        return _PyTorchLegacyPickler(data_buf, protocol=protocol)
    else:
        return PackagePickler(importer, data_buf, protocol=protocol)
