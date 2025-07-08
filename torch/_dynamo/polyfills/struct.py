"""
Python polyfills for builtins
"""

from __future__ import annotations

import struct
from typing import Any, Union

from ..decorators import substitute_in_graph


__all__ = [
    "pack",
    "unpack",
]


@substitute_in_graph(struct.pack, can_constant_fold_through=True)  # type: ignore[arg-type]
def pack(fmt: Union[bytes, str], /, *v: Any) -> bytes:
    return struct.pack(fmt, *v)


@substitute_in_graph(struct.unpack, can_constant_fold_through=True)  # type: ignore[arg-type]
def unpack(format: Union[bytes, str], buffer: bytes) -> tuple[Any, ...]:
    return struct.unpack(format, buffer)
