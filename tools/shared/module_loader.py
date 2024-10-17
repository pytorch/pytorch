from importlib.abc import Loader
from types import ModuleType
from typing import cast


def import_module(name: str, path: str) -> ModuleType:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    cast(Loader, spec.loader).exec_module(module)
    return module
