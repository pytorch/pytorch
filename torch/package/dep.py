from torch.package._mangling import is_mangled
from torch.package._package_pickler import create_pickler
from torch.package.find_file_dependencies import find_files_source_depends_on
from torch.package._stdlib import is_stdlib_module
from torch.package import sys_importer
import types
from typing import Optional, Any
import linecache

from _glob_group import GlobPattern, _GlobGroup

import pickletools

class Dep:
    def __init__(self, importer=sys_importer):
        self.importer = importer
        self.broken_modules = {}
        self.debug_deps = []
        self.external = []
        self.patterns = []
        self.matched_patterns = set()
        self.provided = {}

    def _persistent_id(self, obj):
        if torch.is_storage(obj):
            return "storage"
        # TODO what about __reduce_package__?
        return None

    def _import_module(self, module_name: str):
        if "caffe2.python.caffe2_pybind11_state" in module_name:
            raise ModuleNotFoundErr("get me out")
        try:
            print("importing, ", module_name)
            return self.importer.import_module(module_name)
        except Exception as e:
            print(e)

            if not is_mangled(module_name):
                return None
            msg = (
                f"Module not found: '{module_name}'. Modules imported "
                "from a torch.package cannot be re-exported directly."
            )
            return None

    def _get_source_of_module(self, module: types.ModuleType) -> Optional[str]:
        filename = getattr(module, "__file__", None)
        if filename is None or not filename.endswith(".py"):
            try:
                self.broken_modules[module.__name__] = filename
            except AttributeError:
                pass

            return None
        result = linecache.getlines(filename, module.__dict__)
        return "".join(result)

    def _module_is_already_provided(self, qualified_name: str) -> bool:
        return qualified_name in self.provided

    def require_module_if_not_provided(self, module_name: str):
        if self._module_is_already_provided(module_name):
            return
        self.require_module(module_name)

    def _can_implicitly_extern(self, module_name: str):
        return module_name == "torch" or (
            is_stdlib_module(module_name)
        )

    def extern(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
    ):
        self.patterns.append(
            (_GlobGroup(include, exclude), self.save_extern_module, allow_empty)
        )

    def save_extern_module(self, module_name: str):
        """Add `module_name` to the list of external modules, regardless of whether it is
        required by other modules.

        Prefer using `extern` to only mark modules extern if they are actually required by the packaged code.
        """
        if module_name not in self.external:
            self.external.append(module_name)

    def require_module(self, module_name: str):
        """This is called by dependencies resolution when it finds that something in the package
        depends on the module and it is not already present. It then decides how to provide that module.
        The default resolution rules will mark the module as extern if it is part of the standard library,
        and call `save_module` otherwise. Clients can subclass this object
        and override this method to provide other behavior, such as automatically mocking out a whole class
        of modules"""

        root_name = module_name.split(".", maxsplit=1)[0]
        if self._can_implicitly_extern(root_name):
            self.save_extern_module(root_name)
            return

        for i, (pattern, action, _) in enumerate(self.patterns):
            if pattern.matches(module_name):
                action(module_name)
                print(f"Extern glob: {module_name}")
                self.matched_patterns.add(i)
                return

        print(f"Trying to save {module_name}")
        self.save_module(module_name)

    def save_pickle(self, package: str, resource: str, obj: Any):
        data_buf = io.BytesIO()
        pickler = create_pickler(data_buf, self.importer)
        pickler.persistent_id = self._persistent_id
        pickler.dump(obj)

        data_value = data_buf.getvalue()
        all_dependencies = []
        for opcode, arg, pos in pickletools.genops(data_value):
            if opcode.name == "GLOBAL":  # a global reference
                assert isinstance(arg, str)
                module, field = arg.split(" ")
                if module not in all_dependencies:
                    all_dependencies.append(module)

        for dep in all_dependencies:
            self.debug_deps.append((package + "." + resource, dep))

        for module_name in all_dependencies:
            self.require_module_if_not_provided(module_name)

    def save_module(self, module_name):
        module = self._import_module(module_name)
        source = self._get_source_of_module(module)
        if source is not None:
            self.save_source_string(
                module_name,
                source,
                hasattr(module, "__path__"),
                module.__file__,
            )
    def save_source_string(
        self,
        module_name: str,
        src: str,
        is_package: bool = False,
        orig_file_name: str = None,
    ):
        print("saving", module_name)
        # if module_name =="google.protobuf.internal.well_known_types":
        #     import pdb; pdb.set_trace()
        self.provided[module_name] = True
        package = (
            module_name if is_package else module_name.rsplit(".", maxsplit=1)[0]
        )
        print("find source files", module_name)
        dep_pairs = find_files_source_depends_on(src, package)
        print("finished find source files", module_name)

        dep_list = {}
        for dep_module_name, dep_module_obj in dep_pairs:
            # handle the case where someone did something like `from pack import sub`
            # where `sub` is a submodule. In this case we don't have to save pack, just sub.
            # this ensures we don't pick up additional dependencies on pack.
            # However, in the case where `sub` is not a submodule but an object, then we do have
            # to save pack.
            if dep_module_obj is not None:
                possible_submodule = f"{dep_module_name}.{dep_module_obj}"
                if self._module_exists(possible_submodule):
                    dep_list[possible_submodule] = True
                    # we don't need to save `pack`
                    continue
            if self._module_exists(dep_module_name):
                dep_list[dep_module_name] = True

        for dep in dep_list.keys():
            self.debug_deps.append((module_name, dep))

        for dep in dep_list.keys():
            self.require_module_if_not_provided(dep)

    def _module_exists(self, module_name: str) -> bool:
        mod = self._import_module(module_name)
        return mod is not None
