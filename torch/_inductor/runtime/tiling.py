"""Tiled axis names and block-size limits.

Pointwise generation/filtering orders axes from outermost to innermost;
reduction axes use increasing numeric order. SUPPORTED_POINTWISE_PREFIXES
lists launch axes from innermost to outermost for template callers.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from torch.utils._sympy.symbol import prefix_str, SymT


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


TRITON_LAUNCH_GRID_NDIM = 3


def get_axis_prefix(name: str) -> str | None:
    """Extract an axis prefix from a generated index, block, or offset name."""
    match = re.match(r"[rz]\d+_|[xy]", name.lower())
    return match.group() if match else None


def _is_numbered_prefix(prefix: str, family: str) -> bool:
    return (
        prefix.startswith(family)
        and prefix.endswith("_")
        and prefix[len(family) : -1].isdigit()
    )


def is_z_prefix(prefix: str) -> bool:
    return _is_numbered_prefix(prefix, prefix_str[SymT.Z0_BLOCK].rstrip("0123456789_"))


def is_reduction_prefix(prefix: str) -> bool:
    return _is_numbered_prefix(prefix, prefix_str[SymT.R0_INDEX].rstrip("0123456789_"))


def _get_prefixes(prefixes: Iterable[str], symts: Iterable[SymT]) -> tuple[str, ...]:
    prefixes = tuple(prefixes)
    result = []
    for symt in symts:
        canonical = prefix_str[symt]
        if canonical.endswith("_"):
            family = canonical.rstrip("0123456789_")
            result.extend(
                sorted(
                    (p for p in prefixes if _is_numbered_prefix(p, family)),
                    key=lambda p: int(p[len(family) : -1]),
                )
            )
        elif canonical in prefixes:
            result.append(canonical)
    return tuple(result)


def reduction_prefixes(ndim: int) -> tuple[str, ...]:
    """Generate reduction axes in increasing numeric order."""
    if ndim < 0:
        raise ValueError(f"Invalid reduction rank: {ndim}")
    return tuple(f"r{i}_" for i in range(ndim))


def get_reduction_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    """Filter existing reduction axes and sort them numerically."""
    return _get_prefixes(prefixes, (SymT.R0_INDEX,))


def get_z_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    """Filter existing Z axes, ordered from innermost to outermost."""
    return _get_prefixes(prefixes, (SymT.Z0_BLOCK,))


def pointwise_prefixes(ndim: int) -> tuple[str, ...]:
    """Generate pointwise axes ordered from outermost to innermost."""
    if ndim < 0:
        raise ValueError(f"Invalid pointwise rank: {ndim}")
    if ndim == 0:
        return ()
    z_prefixes = tuple(f"z{i}_" for i in reversed(range(max(0, ndim - 2))))
    return z_prefixes + ("y", "x")[-ndim:]


def get_pointwise_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    """Filter existing pointwise axes, ordered from outermost to innermost."""
    prefixes = tuple(prefixes)
    return tuple(reversed(get_z_prefixes(prefixes))) + _get_prefixes(
        prefixes, (SymT.YBLOCK, SymT.XBLOCK)
    )


def get_pointwise_block_product(
    block_sizes: Mapping[str, int], *, exclude: str | None = None
) -> int:
    return math.prod(
        block_sizes[p] for p in get_pointwise_prefixes(block_sizes) if p != exclude
    )


def get_max_block(prefix: str, max_blocks: Mapping[str, int]) -> int:
    """Use the shared Z0_ limit for Z axes and R1_ for unlisted reductions."""
    key = "Z0_" if is_z_prefix(prefix.lower()) else prefix.upper()
    if is_reduction_prefix(prefix.lower()) and key not in max_blocks:
        key = "R1_"
    return max_blocks[key]


def block_to_prefix(block: str) -> str:
    return block.removesuffix("BLOCK").lower()
