"""Compatibility imports for descriptors pickled before the package rename."""

import sys

from ..nixl import (
    _memory,
    _transport,
    _work,
    NIXLMemory,
    NIXLMemoryView,
    NIXLMutableMemoryView,
    NIXLRemoteBuffer,
    NIXLTransport,
)


for module in (_memory, _transport, _work):
    sys.modules[f"{__name__}.{module.__name__.rsplit('.', 1)[-1]}"] = module

__all__ = [
    "NIXLMemory",
    "NIXLMemoryView",
    "NIXLMutableMemoryView",
    "NIXLRemoteBuffer",
    "NIXLTransport",
]
