import argparse
import inspect
import importlib
import sys

from types import ModuleType
from collections import namedtuple
from typing import List, Any, Set, NamedTuple, Tuple, Dict
from pathlib import Path


BUILTIN_MODULES = set()

for name in sys.builtin_module_names:
    BUILTIN_MODULES.add(importlib.import_module(name))


def main(module: ModuleType, public: bool, private: bool, errors: bool) -> None:
    result = Crawler(module)

    for path in result.apis.values():
        is_private = False
        for item in path:
            if item.name.startswith("_"):
                is_private = True
                break

        if public and not is_private:
            print(".".join([a.name for a in path]))
        elif private and is_private:
            print(".".join([a.name for a in path]))

    if errors:
        for error in result.errors:
            print(error)


class Crawler:
    Error = namedtuple("Error", ("reason", "path"))
    Item = namedtuple("Item", ("obj", "name"))

    def __init__(self, module: ModuleType):
        self.module = module
        self.public = []
        self.private = []
        self.errors = []

        self.all_objects = []
        self.apis = {}

        self.crawl(obj=module, name=module.__name__, path=[])
        self.add_class_attributes()

    def add_class_attributes(self):
        """
        Add class attributes after we've already crawled through all the
        modules their recursive members. This has to be delayed since we can run
        into a class multiple times and we de-duplicate those based on the class'
        id(), but that doesn't work for attributes since they are just a string.
        So, instead we wait until all the classes are added and then add their
        attributes.
        """
        to_add = []
        for path in self.apis.values():
            item = path[-1]
            obj = item.obj
            if inspect.isclass(obj):
                # We don't want to re-add class attributes (these don't have an
                # associated id() to de-duplicate them with, so we have to do it
                # manually
                attrs = dir(obj)

                for attr in attrs:
                    to_add.append(path + [Crawler.Item(name=attr, obj=None)])

        for index, path in enumerate(to_add):
            self.apis[f"attr-{index}"] = path

    def get_submodules(self, module: ModuleType) -> List[ModuleType]:
        """
        Some modules aren't imported directly into the parent (i.e. torch.fx). So
        this fails

            import torch
            print(torch.fx)  # AttributeError: module 'torch' has no attribute 'fx'

        but this works

            import torch.fx
            print(torch.fx)

        This function gathers all submodules of a module by inspecting its search
        path (the folder) and importing all subdirectories it can
        """
        ignore_list = {
            "for_onnx"
        }
        try:
            spec = importlib.util.find_spec(module.__name__)
        except Exception as e:
            # print(e)
            return []

        if spec is None or spec.submodule_search_locations is None:
            return []

        search_paths = [Path(loc) for loc in spec.submodule_search_locations]
        submodules = []
        for path in search_paths:
            submodule_names = [subdir for subdir in path.glob("*") if subdir.is_dir()]
            submodule_names = [subdir.name for subdir in submodule_names]
            for submodule_name in submodule_names:
                if submodule_name in ignore_list:
                    continue

                qualified_name = f"{module.__name__}.{submodule_name}"
                try:
                    submodules.append(importlib.import_module(qualified_name))
                except Exception as e:
                    self.errors.append(Crawler.Error(reason=str(e), path=qualified_name))

        return submodules

    def get_module_attributes(self, module: ModuleType) -> Set[str]:
        """
        Return the combined list of attributes from dir and __all__
        """
        attrs = []

        try:
            attrs += dir(module)
            attrs += getattr(module, "__all__", [])
        except Exception as e:
            self.errors.append(Crawler.Error(reason=str(e), path=""))

        return set(attrs)

    def crawl(
        self,
        obj: Any,
        name: str,
        path: List[Item],
    ) -> None:
        if self.should_skip(obj, path):
            return

        # Keep a reference around to ensure id() calls are valid
        self.all_objects.append(obj)

        if id(obj) in self.apis:
            # Some things are listed twice, usually as a side-effect of importing
            # it directly while still in the library. This uses a heuristic of
            # shorter-path-is-better to pick the best one
            prev_len = len(self.apis[id(obj)])
            curr_len = len(path) + 1
            if curr_len >= prev_len:
                # It's a longer path and this already has an entry, so don't
                # both with it any further
                return

        self.apis[id(obj)] = path + [Crawler.Item(name=name, obj=obj)]

        # Drop down into modules and classes
        if isinstance(obj, ModuleType):
            # Explicit crawl through submodules that aren't directly reachable
            # (e.g. torch.fx) from the parent
            submodules = self.get_submodules(obj)
            for submodule in submodules:
                self.crawl(submodule, submodule.__name__.split(".")[-1], path + [Crawler.Item(name=name, obj=obj)])

            for attr in self.get_module_attributes(obj):
                item = Crawler.Item(name=name, obj=obj)
                attr_path = path + [item]
                try:
                    next_obj = getattr(obj, attr)
                except (ModuleNotFoundError, AttributeError) as e:
                    self.errors.append(Crawler.Error(path=attr_path, reason=str(e)))

                self.crawl(next_obj, attr, attr_path)

    def get_name(self, path: List[str], name: str) -> str:
        names = [item.name for item in path]
        return ".".join(names + [name])

    def should_skip(self, obj: Any, path: List[Item]) -> bool:
        """
        Return whether 'obj' is relevant to 'relevant_module'. If 'obj' is not a
        module, then get its module from the __module__ attribute. Then check if
        the module is part of the relevant_module by comparing their __file__ paths.

        We don't want to crawl or list attributes of system modules or other
        third-party modules that have been exposed via imports. Since submodules of
        torch all live in the same directory, the heuristic here checks that the
        module's __file__ contains /torch/.
        """
        # Don't look at this object if it is somewhere earlier in its own path
        for item in path:
            if id(obj) == id(item.obj):
                # We've hit a circular include (i.e torch.__config__.torch.__config...)
                # so give up
                return True

        module = obj
        # print(obj, path)

        if not isinstance(obj, ModuleType):
            # If the object isn't already a module, resolve it to one
            module_name = getattr(obj, "__module__", None)

            if module_name is None:
                return False

            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as e:
                # Likely we hit an incorrectly named module defined in C (so
                # the name given to the CPython API isn't actually importable)
                return True


        if module == self.module:
            # If the object is a direct child of the concered module, then don't
            # skip
            return False

        if module in BUILTIN_MODULES:
            # Skip built ins
            return True

        module_file = None
        try:
            # For some objects merely accessing an attribute causes an error, so catch
            # that here
            module_file = getattr(module, "__file__", None)
        except RuntimeError as e:
            return False

        if not isinstance(module_file, str):
            return False

        # Check if the module is a submodule of the relevant module via checking
        # their file paths
        parent = Path(self.module.__file__).resolve().parent
        maybe_child = Path(module_file).resolve()

        return parent not in maybe_child.parents



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively list all reachable objects in a Python module (library)."
        + " This is used in PyTorch releases to determine the API surface changes"
    )
    parser.add_argument("--module", help="module to crawl", required=True)

    def add_flag(name: str, default: bool, help: str) -> None:
        parser.add_argument(f"--{name}", dest=name, help=help, action="store_true")
        parser.add_argument(f"--no-{name}", dest=name, help=help, action="store_false")
        parser.set_defaults(**{name: default})

    add_flag(name="public", default=True, help="list public APIs")
    add_flag(
        name="private",
        default=False,
        help="list private APIs (those that start with a _)",
    )
    add_flag(name="errors", default=False, help="show errors (unreachable APIs)")

    args = parser.parse_args()

    main(
        module=importlib.import_module(args.module),
        public=args.public,
        private=args.private,
        errors=args.errors,
    )
