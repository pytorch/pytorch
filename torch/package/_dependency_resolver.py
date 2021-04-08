import pickletools

from .find_file_dependencies import find_files_source_depends_on
from .importer import Importer, sys_importer

from ._utils import _import_module
from ._stdlib import is_stdlib_module

_DISALLOWED_MODULES = ["sys", "io"]


class _DependencyResolver:
    def __init__(self, importer: Importer=sys_importer, verbose=False, on_require=None):
        self.importer = importer
        self.verbose = verbose
        self.provided: Dict[str, bool] = {}
        self.debug_deps: List[Tuple[str, str]] = []
        self.extern_modules: List[str] = []
        self.on_require = on_require

    def _module_exists(self, module_name: str) -> bool:
        try:
            _import_module(module_name, self.importer)
            return True
        except Exception:
            return False

    def mark_provided(self, name: str):
        self.provided[name] = True

    def is_provided(self, name: str):
        return name in self.provided and self.provided[name] == True

    def add_extern(self, module_name: str):
        if module_name not in self.extern_modules:
            self.extern_modules.append(module_name)

    def _can_implicitly_extern(self, module_name: str):
        return module_name == "torch" or (
            module_name not in _DISALLOWED_MODULES and is_stdlib_module(module_name)
        )

    def _module_is_already_provided(self, qualified_name: str) -> bool:
        for mod in self.extern_modules:
            if qualified_name == mod or qualified_name.startswith(mod + "."):
                return True
        return self.is_provided(qualified_name)

    def require_module_if_not_provided(self, module_name: str, dependencies=True):
        if self._module_is_already_provided(module_name):
            return
        self.require_module(module_name, dependencies)

    def require_module(self, module_name: str, dependencies=True):
        """This is called by dependencies resolution when it finds that something in the package
        depends on the module and it is not already present. It then decides how to provide that module.
        The default resolution rules will mark the module as extern if it is part of the standard library,
        and call `save_module` otherwise. Clients can subclass this object
        and override this method to provide other behavior, such as automatically mocking out a whole class
        of modules"""
        root_name = module_name.split(".", maxsplit=1)[0]
        if self._can_implicitly_extern(root_name):
            if self.verbose:
                print(
                    f"implicitly adding {root_name} to external modules "
                    f"since it is part of the standard library and is a dependency."
                )
            self.add_extern(root_name)
            return

        self.on_require(module_name, dependencies)

    def scan_module_dependencies(self, module_name: str, src: str, is_package: bool, orig_file_name: str):
        package = (
            module_name if is_package else module_name.rsplit(".", maxsplit=1)[0]
        )
        dep_pairs = find_files_source_depends_on(src, package)
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

        if self.verbose:
            dep_str = "".join(f"  {dep}\n" for dep in dep_list.keys())
            file_info = (
                f"(from file {orig_file_name}) "
                if orig_file_name is not None
                else ""
            )
            print(f"{module_name} {file_info}depends on:\n{dep_str}\n")

        for dep in dep_list.keys():
            self.require_module_if_not_provided(dep)
