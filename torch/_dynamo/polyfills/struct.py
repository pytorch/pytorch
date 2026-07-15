"""
Python polyfills for struct
"""

from __future__ import annotations

import struct
from typing import Any
from typing_extensions import Buffer

from ..decorators import substitute_in_graph


__all__ = [
    "pack",
    "unpack",
]


@substitute_in_graph(struct.pack, can_constant_fold_through=True)  # type: ignore[arg-type]
def pack(fmt: bytes | str, /, *v: Any) -> bytes:
    return struct.pack(fmt, *v)


@substitute_in_graph(struct.unpack, can_constant_fold_through=True)  # type: ignore[arg-type]
def unpack(format: bytes | str, buffer: Buffer, /) -> tuple[Any, ...]:
    return struct.unpack(format, buffer)
