# No STABLE_TORCH_LIBRARY ops yet, only the importable _interop module; an op
# extension (_C) can be added when the first 2.15 op lands (see setup.py).
from . import _interop


__all__ = [
    "_interop",
]
