# mypy: allow-untyped-defs
"""DMA staging layouts for the Gluon flex-attention templates.

``buffer_load_to_shared`` writes LDS directly and needs exact register/lane/warp
bases saying which lane of which wave moves which element. These cannot be
improvised -- the plain Blocked/Swizzled layouts fail to lower -- so these tables
are the ones taken from the hand-tuned gfx950 kernel.

They are data rather than a branch chain in the template so that the hook and the
body cannot disagree: the hook offers the async body exactly when a key is
present, and the body renders its layouts from the value. Adding a combination
(BLOCK_N=32 and 128 are the unexplored ones) is a new entry and nothing else.
"""

from typing import NamedTuple


class DmaLayouts(NamedTuple):
    """Linear-layout bases for staging one K^T tile and one V tile into LDS.

    ``*_offset_bases`` map a shared-memory offset to tile coordinates; the
    ``reg``/``lane``/``warp`` bases say how far along each tile dimension to move
    when the corresponding bit of the register / lane / warp index flips.
    """

    kt_offset_bases: list[list[int]]
    kt_reg_bases: list[list[int]]
    kt_lane_bases: list[list[int]]
    kt_warp_bases: list[list[int]]
    v_offset_bases: list[list[int]]
    v_reg_bases: list[list[int]]
    v_lane_bases: list[list[int]]
    v_warp_bases: list[list[int]]


# K^T is staged [head_dim, BLOCK_N] and V is staged [BLOCK_N, head_dim], so their
# bases are mirror images of each other. The offset bases depend only on the tile
# shape; the reg/lane/warp split depends on how many waves share the transfer.
_KT_OFFSETS_128 = [
    [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
    [0, 16], [0, 32], [0, 1], [0, 2], [0, 4], [0, 8],
]  # fmt: skip
_V_OFFSETS_128 = [
    [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
    [16, 0], [32, 0], [1, 0], [2, 0], [4, 0], [8, 0],
]  # fmt: skip
_KT_OFFSETS_64 = [
    [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
    [0, 16], [0, 32], [0, 1], [0, 2], [0, 4], [0, 8],
]  # fmt: skip
_V_OFFSETS_64 = [
    [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32],
    [16, 0], [32, 0], [1, 0], [2, 0], [4, 0], [8, 0],
]  # fmt: skip


# (QK_HEAD_DIM_ROUNDED, BLOCK_N, num_warps) -> layouts
CDNA4_DMA_LADDER: dict[tuple[int, int, int], DmaLayouts] = {
    (128, 64, 8): DmaLayouts(
        kt_offset_bases=_KT_OFFSETS_128,
        kt_reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
        kt_lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
        kt_warp_bases=[[0, 1], [0, 2], [0, 4]],
        v_offset_bases=_V_OFFSETS_128,
        v_reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
        v_lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
        v_warp_bases=[[1, 0], [2, 0], [4, 0]],
    ),
    (128, 64, 4): DmaLayouts(
        kt_offset_bases=_KT_OFFSETS_128,
        kt_reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8], [0, 4]],
        kt_lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
        kt_warp_bases=[[0, 1], [0, 2]],
        v_offset_bases=_V_OFFSETS_128,
        v_reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [4, 0]],
        v_lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
        v_warp_bases=[[1, 0], [2, 0]],
    ),
    (64, 64, 8): DmaLayouts(
        kt_offset_bases=_KT_OFFSETS_64,
        kt_reg_bases=[[1, 0], [2, 0], [4, 0]],
        kt_lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
        kt_warp_bases=[[0, 2], [0, 4], [0, 8]],
        v_offset_bases=_V_OFFSETS_64,
        v_reg_bases=[[0, 1], [0, 2], [0, 4]],
        v_lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
        v_warp_bases=[[2, 0], [4, 0], [8, 0]],
    ),
    (64, 64, 4): DmaLayouts(
        kt_offset_bases=_KT_OFFSETS_64,
        kt_reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
        kt_lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
        kt_warp_bases=[[0, 2], [0, 4]],
        v_offset_bases=_V_OFFSETS_64,
        v_reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
        v_lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
        v_warp_bases=[[2, 0], [4, 0]],
    ),
}


def as_template_options(layouts: DmaLayouts) -> dict[str, list[list[int]]]:
    """Template options the async body renders its layout declarations from."""
    return {
        "GLUON_KT_OFFSET_BASES": layouts.kt_offset_bases,
        "GLUON_KT_REG_BASES": layouts.kt_reg_bases,
        "GLUON_KT_LANE_BASES": layouts.kt_lane_bases,
        "GLUON_KT_WARP_BASES": layouts.kt_warp_bases,
        "GLUON_V_OFFSET_BASES": layouts.v_offset_bases,
        "GLUON_V_REG_BASES": layouts.v_reg_bases,
        "GLUON_V_LANE_BASES": layouts.v_lane_bases,
        "GLUON_V_WARP_BASES": layouts.v_warp_bases,
    }
