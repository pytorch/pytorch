# pyre-strict
"""Shared dtype constants and utilities for torchfuzz operator modules."""

from __future__ import annotations

import math
import random

import torch


FLOAT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
)


def is_float_dtype(dt: torch.dtype) -> bool:
    return dt in FLOAT_DTYPES


def contiguous_stride(size: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major contiguous strides for ``size``."""
    if not size:
        return ()
    strides: list[int] = [1]
    for dim in reversed(size[1:]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))


def scalar_repr(value: object) -> str:
    """Format a scalar for embedding in codegen, handling inf/nan."""
    if isinstance(value, float):
        if math.isinf(value):
            return f"float('{'-' if value < 0 else ''}inf')"
        if math.isnan(value):
            return "float('nan')"
    return repr(value)


def random_broadcast_shape(output_size: tuple[int, ...]) -> tuple[int, ...]:
    """Return a shape that broadcasts to ``output_size``.

    Each dimension has a 30% chance of shrinking to 1. Leading size-1
    dims may then be dropped (50% chance each) to produce a lower-rank
    input. Returns ``output_size`` unchanged if no dims were shrunk.
    """
    if not output_size:
        return ()
    result = list(output_size)
    changed = False
    for i in range(len(result)):
        if random.random() < 0.3:
            result[i] = 1
            changed = True
    if not changed:
        return output_size
    while len(result) > 1 and result[0] == 1 and random.random() < 0.5:
        result.pop(0)
    return tuple(result)
