# mypy: allow-untyped-defs
# We are exposing all subpackages to the end-user.
# Because of possible inter-dependency, we want to avoid
# the cyclic imports, thus implementing lazy version
# as per https://peps.python.org/pep-0562/

import importlib

__all__ = [
    "intrinsic",
    "qat",
    "quantizable",
    "quantized",
    "sparse",
]

def __getattr__(name):
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
