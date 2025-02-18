# torch.ao is a package with a lot of interdependencies.
# We will use lazy import to avoid cyclic dependencies here.

from typing import TYPE_CHECKING as _TYPE_CHECKING


if _TYPE_CHECKING:
    from types import ModuleType

    from torch.ao import (  # noqa: TC004
        nn as nn,
        ns as ns,
        pruning as pruning,
        quantization as quantization,
    )


__all__ = [
    "nn",
    "ns",
    "pruning",
    "quantization",
]


def __getattr__(name: str) -> "ModuleType":
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
