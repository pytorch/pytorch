import types

from .modules import *  # noqa: F403
from .modules.fused import _FusedModule


# # Subpackages
# from . import qat
# from . import quantized

__all__ = [
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearReLU",
    "BNReLU2d",
    "BNReLU3d",
    "LinearBn1d",
    "LinearLeakyReLU",
    "LinearTanh",
    "ConvAdd2d",
    "ConvAddReLU2d",
]


# We are exposing all subpackages to the end-user.
# Because of possible inter-dependency, we want to avoid
# the cyclic imports, thus implementing lazy version
# as per https://peps.python.org/pep-0562/
def __getattr__(name: str) -> types.ModuleType:
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
