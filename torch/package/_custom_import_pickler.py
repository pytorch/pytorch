from pickle import Pickler, whichmodule, _Pickler, _getattribute, _extension_registry, _compat_pickle  # type: ignore
from pickle import GLOBAL, STACK_GLOBAL, EXT1, EXT2, EXT4, PicklingError
from types import FunctionType
from struct import pack
from ._mangling import demangle, is_mangled, get_mangle_prefix
import importlib


class CustomImportPickler(_Pickler):
    dispatch = _Pickler.dispatch.copy()

    def __init__(self, import_module, *args, **kwargs):
        self.import_module = import_module
        super().__init__(*args, **kwargs)

    def save_global(self, obj, name=None):
        # unfortunately the pickler code is factored in a way that
        # forces us to copy/paste this function. The only change is marked
        # CHANGED below.
        write = self.write
        memo = self.memo

        if name is None:
            name = getattr(obj, '__qualname__', None)
        if name is None:
            name = obj.__name__

        orig_module_name = whichmodule(obj, name)
        # CHANGED: demangle the module name before importing. If this obj came
        # out of a PackageImporter, `__module__` will be mangled. See
        # mangling.md for details.
        module_name = demangle(orig_module_name)
        try:
            # CHANGED: self.import_module rather than
            # __import__
            module = self.import_module(module_name)
            obj2, parent = _getattribute(module, name)
        except (ImportError, KeyError, AttributeError):
            raise PicklingError(
                "Can't pickle %r: it's not found as %s.%s" %
                (obj, module_name, name)) from None
        else:
            if obj2 is not obj:
                # CHANGED: More specific error message in the case of mangling.
                obj_module_name = getattr(obj, "__module__", orig_module_name)
                obj2_module_name = getattr(obj2, "__module__", orig_module_name)

                msg = f"Can't pickle {obj}: it's not the same object as {obj2_module_name}.{name}."

                is_obj_mangled = is_mangled(obj_module_name)
                is_obj2_mangled = is_mangled(obj2_module_name)

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
                        else "'importlib.import_module'"
                    )
                    obj2_importer_name = (
                        f"the importer for {get_mangle_prefix(obj2_module_name)}"
                        if is_obj2_mangled
                        else "'importlib.import_module'"
                    )

                    msg += (f"\n\nThe object being pickled is from '{orig_module_name}', "
                            f"which is coming from {obj_location}."
                            f"\nHowever, when we import '{orig_module_name}', it's coming from {obj2_location}."
                            "\nTo fix this, make sure 'PackageExporter.importers' lists "
                            f"{obj_importer_name} before {obj2_importer_name}")
                raise PicklingError(msg)

        if self.proto >= 2:
            code = _extension_registry.get((module_name, name))
            if code:
                assert code > 0
                if code <= 0xff:
                    write(EXT1 + pack("<B", code))
                elif code <= 0xffff:
                    write(EXT2 + pack("<H", code))
                else:
                    write(EXT4 + pack("<i", code))
                return
        lastname = name.rpartition('.')[2]
        if parent is module:
            name = lastname
        # Non-ASCII identifiers are supported only with protocols >= 3.
        if self.proto >= 4:
            self.save(module_name)
            self.save(name)
            write(STACK_GLOBAL)
        elif parent is not module:
            self.save_reduce(getattr, (parent, lastname))
        elif self.proto >= 3:
            write(GLOBAL + bytes(module_name, "utf-8") + b'\n' +
                  bytes(name, "utf-8") + b'\n')
        else:
            if self.fix_imports:
                r_name_mapping = _compat_pickle.REVERSE_NAME_MAPPING
                r_import_mapping = _compat_pickle.REVERSE_IMPORT_MAPPING
                if (module_name, name) in r_name_mapping:
                    module_name, name = r_name_mapping[(module_name, name)]
                elif module_name in r_import_mapping:
                    module_name = r_import_mapping[module_name]
            try:
                write(GLOBAL + bytes(module_name, "ascii") + b'\n' +
                      bytes(name, "ascii") + b'\n')
            except UnicodeEncodeError:
                raise PicklingError(
                    "can't pickle global identifier '%s.%s' using "
                    "pickle protocol %i" % (module, name, self.proto)) from None

        self.memoize(obj)
    dispatch[FunctionType] = save_global

def import_module_from_importers(module_name, importers):
    last_err = None
    for import_module in importers:
        try:
            return import_module(module_name)
        except ModuleNotFoundError as err:
            last_err = err

    if last_err is not None:
        raise last_err
    else:
        raise ModuleNotFoundError(module_name)

def create_custom_import_pickler(data_buf, importers):
    if importers == [importlib.import_module]:
        # if we are using the normal import library system, then
        # we can use the C implementation of pickle which is faster
        return Pickler(data_buf, protocol=3)
    else:
        return CustomImportPickler(lambda mod: import_module_from_importers(mod, importers),
                                   data_buf, protocol=3)
