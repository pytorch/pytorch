"""Tiled axis names and block-size limits.

Pointwise generation/filtering orders axes from outermost to innermost;
reduction axes use increasing numeric order.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from torch.utils._sympy.symbol import prefix_str, SymT, symt_to_block_prefix

from .runtime_utils import ceildiv


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def prefix_to_family(prefix: str) -> str:
    return prefix[0]


_Z_FAMILY = prefix_to_family(prefix_str[SymT.Z0_BLOCK])
_REDUCTION_FAMILY = prefix_to_family(prefix_str[SymT.R0_INDEX])
_AXIS_PREFIX = re.compile(
    rf"[{_REDUCTION_FAMILY}{_Z_FAMILY}]\d+_|"
    rf"{prefix_str[SymT.XBLOCK]}|{prefix_str[SymT.YBLOCK]}"
)


def get_axis_prefix(name: str) -> str | None:
    """Extract an axis prefix from a generated index, block, or offset name."""
    match = _AXIS_PREFIX.match(name.lower())
    return match.group() if match else None


def _is_numbered_prefix(prefix: str, family: str) -> bool:
    return (
        prefix.startswith(family)
        and prefix.endswith("_")
        and prefix[len(family) : -1].isdigit()
    )


def is_z_prefix(prefix: str) -> bool:
    return _is_numbered_prefix(prefix, _Z_FAMILY)


def is_reduction_prefix(prefix: str) -> bool:
    return _is_numbered_prefix(prefix, _REDUCTION_FAMILY)


def _get_prefixes(prefixes: Iterable[str], family: str) -> tuple[str, ...]:
    prefixes = tuple(dict.fromkeys(prefixes))
    return tuple(
        sorted(
            (p for p in prefixes if _is_numbered_prefix(p, family)),
            key=lambda p: int(p[len(family) : -1]),
        )
    )


def reduction_prefixes(ndim: int) -> tuple[str, ...]:
    """Generate reduction axes in increasing numeric order."""
    if ndim < 0:
        raise ValueError(f"Invalid reduction rank: {ndim}")
    return tuple(f"{_REDUCTION_FAMILY}{i}_" for i in range(ndim))


def get_reduction_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    """Filter existing reduction axes and sort them numerically."""
    return _get_prefixes(prefixes, _REDUCTION_FAMILY)


def get_z_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    """Filter existing Z axes, ordered from innermost to outermost."""
    return _get_prefixes(prefixes, _Z_FAMILY)


def has_packed_z_grid(prefixes: Iterable[str]) -> bool:
    """Multiple Z axes pack their grids and require strict hinted limits.

    A single Z axis uses Grid3D, where rounded hints may overestimate a
    launchable grid. Its filter uses the minimum unrounded size hint.
    """
    return len(get_z_prefixes(prefixes)) > 1


def pointwise_prefixes(ndim: int) -> tuple[str, ...]:
    """Generate pointwise axes ordered from outermost to innermost."""
    if ndim < 0:
        raise ValueError(f"Invalid pointwise rank: {ndim}")
    if ndim == 0:
        return ()
    z_prefixes = tuple(f"{_Z_FAMILY}{i}_" for i in reversed(range(max(0, ndim - 2))))
    return z_prefixes + (prefix_str[SymT.YBLOCK], prefix_str[SymT.XBLOCK])[-ndim:]


def get_pointwise_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    """Filter existing pointwise axes, ordered from outermost to innermost."""
    prefixes = tuple(prefixes)
    return tuple(reversed(get_z_prefixes(prefixes))) + tuple(
        prefix_str[symt]
        for symt in (SymT.YBLOCK, SymT.XBLOCK)
        if prefix_str[symt] in prefixes
    )


def get_yz_grid_extents(
    size_hints: Mapping[str, int],
    block_sizes: Mapping[str, int],
    *,
    lower_bound: bool = False,
) -> dict[tuple[str, ...], int]:
    """Compute Y and packed Z extents for groups with complete block sizes.

    Kernels without Z axes use a separate Y-overflow launch layout.
    With lower_bound, use the smallest numel that rounds up to each hint.
    """
    z_prefixes = get_z_prefixes(size_hints)
    if not z_prefixes:
        return {}
    return {
        group: math.prod(
            ceildiv(
                size_hints[p] // 2 + 1 if lower_bound else size_hints[p], block_sizes[p]
            )
            for p in group
        )
        for group in (("y",), z_prefixes)
        if all(p in size_hints and p in block_sizes for p in group)
    }


def get_max_block_prefix(prefix: str, max_blocks: Mapping[str, int]) -> str:
    key = (
        symt_to_block_prefix(SymT.Z0_BLOCK)
        if is_z_prefix(prefix.lower())
        else prefix.upper()
    )
    if is_reduction_prefix(prefix.lower()) and key not in max_blocks:
        key = symt_to_block_prefix(SymT.R1_INDEX)
    return key


def get_max_block(prefix: str, max_blocks: Mapping[str, int]) -> int:
    """Use the shared Z0_ limit for Z axes and R1_ for unlisted reductions."""
    return max_blocks[get_max_block_prefix(prefix, max_blocks)]


def get_z_blocks(config: Mapping[str, int]) -> list[int]:
    """List single-kernel Z blocks from innermost to outermost."""
    prefixes = get_z_prefixes(block_to_prefix(k) for k in config) or ("z0_",)
    return [config.get(f"{p.upper()}BLOCK", 1) for p in prefixes]


def block_to_prefix(block: str) -> str:
    return block.removesuffix("BLOCK").lower()
