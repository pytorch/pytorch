import functools
import importlib
import re
import sys
from importlib.machinery import PathFinder
from types import ModuleType


# torch modules to replace with MockTorch
to_mock = [
    "torch",
    "torch._dynamo",
    "torch._inductor",
]

# modules to import
allowed_imports = [
    "torch._inductor.runtime",
    "torch._inductor.compile_worker",
    # "torch._dynamo.device_interface",
]

allowed_regexp = re.compile(
    r"^({})$".format(
        "|".join(
            [
                r"(?!torch[.]).*",  # not starting with torch
                *map("{}([.].*)?".format, map(re.escape, allowed_imports)),
                *map(re.escape, to_mock),
            ]
        )
    )
)

get_parent_module = functools.partial(re.compile("[.][^.]*").sub, "")


class MockTorch(ModuleType):
    def __getattr__(self, item):
        raise AttributeError(f"Compile workers cannot use: {self.__name__}.{item}")

    def __init__(self, spec):
        assert spec.name not in sys.modules
        super().__init__(spec.name)
        self.__spec__ = spec
        self.__file__ = spec.origin
        self.__path__ = spec.submodule_search_locations


def install():
    for name in to_mock:
        spec = PathFinder.find_spec(
            name,
            path=sys.modules[get_parent_module(name)].__path__
            if "." in name
            else sys.path,
        )
        sys.modules[name] = MockTorch(spec)
    for name in allowed_imports:
        importlib.import_module(name)
    verify()


def verify():
    for name in sys.modules.keys():
        assert allowed_regexp.match(
            name
        ), f"compile worker should not have imported {name}"
