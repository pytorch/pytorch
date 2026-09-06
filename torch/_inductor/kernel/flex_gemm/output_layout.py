# mypy: allow-untyped-defs
"""FlexGEMM physical output-layout descriptors and storage geometry."""

import enum
from collections.abc import Sequence
from typing import Any


class FlexGemmOutputStorageLayout(str, enum.Enum):
    """Physical storage layout applied independently to each returned value."""

    BLOCKED_128X4 = "blocked_128x4"
    TRANSPOSED = "transposed"


def blocked_128x4_numel(logical_shape: Sequence[Any]) -> Any:
    """Return the padded element count for a logical two-dimensional tensor."""
    rows, cols = logical_shape
    return 512 * ((rows + 127) // 128) * ((cols + 3) // 4)


def blocked_128x4_carrier_shape(logical_shape: Sequence[Any]) -> tuple[Any, ...]:
    """Return the host tensor shape carrying a blocked 128x4 output."""
    rows, cols = logical_shape
    return (1, (rows + 127) // 128, (cols + 3) // 4, 512)


def transposed_carrier_shape(logical_shape: Sequence[Any]) -> tuple[Any, ...]:
    """Return contiguous storage for the transpose of a logical matrix."""
    rows, cols = logical_shape
    return (cols, rows)


def output_layout_supports_config(
    layout: FlexGemmOutputStorageLayout | None, config: Any, geometry: Any | None
) -> bool:
    """Return whether a local-reduce tile composes with the layout atom."""
    if layout is None:
        return True
    if geometry is None:
        return False
    match layout:
        case FlexGemmOutputStorageLayout.BLOCKED_128X4:
            if geometry.axis != 1 or config.tile_m % config.cluster_m != 0:
                return False
            cta_tile_m = config.tile_m // config.cluster_m
            if config.swap_ab:
                if cta_tile_m % geometry.group != 0:
                    return False
                row_tile = config.tile_n
                column_tile = cta_tile_m // geometry.group
            else:
                if config.tile_n % geometry.group != 0:
                    return False
                row_tile = cta_tile_m
                column_tile = config.tile_n // geometry.group
            return (128 % row_tile == 0 or row_tile % 128 == 0) and (
                4 % column_tile == 0 or column_tile % 4 == 0
            )
        case FlexGemmOutputStorageLayout.TRANSPOSED:
            return not config.swap_ab
        case _:
            return False
