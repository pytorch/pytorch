import torch

# Importing _C runs the extension's static initializers (which register its
# STABLE_TORCH_LIBRARY ops onto torch.ops) and, via PyInit__C, exposes the
# PyObject<->Tensor interop helpers used by the tests. This replaces a separate
# torch.ops.load_library call, which would load the same _C.so a second time.
from . import _C, ops


__all__ = [
    "_C",
    "ops",
]
