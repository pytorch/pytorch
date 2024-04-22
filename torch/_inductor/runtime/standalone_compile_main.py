import functools
import importlib
import re
import sys
from importlib.machinery import PathFinder
from types import ModuleType


# torch modules to replace with MockTorch
to_mock = [
    "torch",
    "torch.version",
    "torch.cuda",
    "torch._dynamo",
    "torch._inductor",
    "torch._C",
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
        assert spec.name not in sys.modules, spec.name
        super().__init__(spec.name)
        self.__spec__ = spec
        self.__file__ = spec.origin
        self.__path__ = spec.submodule_search_locations


def install(device_type):
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

    # Fake out just enough stuff we can import and use `triton.compile()`
    import torch
    from torch import cuda, version

    torch.version = version
    torch.cuda = cuda
    cuda.is_available = lambda: device_type == "cuda"
    cuda.get_device_capability = not_implemented
    cuda.current_device = not_implemented
    cuda.set_device = not_implemented
    torch.version.hip = (device_type == "hip") or None


def verify():
    """Assert we didn't import stuff we shouldn't have"""
    for name in sys.modules.keys():
        assert allowed_regexp.match(
            name
        ), f"compile worker should not have imported {name}"


def not_implemented(*args, **kwargs):
    raise NotImplementedError("in compile worker with mock torch.*")


def main():
    _, device_type, key, path, kernel_name = sys.argv
    install(device_type)
    from torch._inductor.runtime.compile_tasks import _reload_triton_kernel_in_subproc

    kernel = _reload_triton_kernel_in_subproc(key, path, kernel_name)
    kernel.precompile(warm_cache_only=True)
    verify()


if __name__ == "__main__":
    main()
