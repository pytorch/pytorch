# We are exposing all subpackages to the end-user.
# Because of possible inter-dependency, we want to avoid
# the cyclic imports, thus implementing lazy version
# as per https://peps.python.org/pep-0562/

from typing import TYPE_CHECKING as _TYPE_CHECKING


if _TYPE_CHECKING:
    from types import ModuleType

    from torch.ao.nn import (  # noqa: TC004
        intrinsic as intrinsic,
        qat as qat,
        quantizable as quantizable,
        quantized as quantized,
        sparse as sparse,
    )


__all__ = [
    "intrinsic",
    "qat",
    "quantizable",
    "quantized",
    "sparse",
]


def __getattr__(name: str) -> "ModuleType":
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
