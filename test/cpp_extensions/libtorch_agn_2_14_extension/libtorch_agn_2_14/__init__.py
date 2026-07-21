import torch  # noqa: F401  ensure libtorch / libtorch_python are loaded first

# Importing _C both runs the extension's static initializers (registering the
# STABLE_TORCH_LIBRARY ops onto torch.ops) and, via PyInit__C, exposes the
# GIL-held PyObject<->Tensor interop helpers used by the interop tests.
from . import _C  # noqa: F401
from . import ops


__all__ = [
    "_C",
    "ops",
]
