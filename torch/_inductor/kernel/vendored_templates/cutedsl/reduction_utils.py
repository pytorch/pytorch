"""CuTeDSL geometry helpers shared by NVGEMM reduction kernels."""

import cutlass.cute as cute
from cutlass import const_expr, Int32
from cutlass.cutlass_dsl import dsl_user_op


__all__ = ["get_lane_warp_layouts", "partition_for_epilogue"]


def get_lane_warp_layouts(tiled_copy, reference_src=True):
    """Derive two-dimensional lane and warp layouts for an epilogue copy."""
    from torch._vendor.quack import layout_utils

    layout_tv = (
        tiled_copy.layout_src_tv_tiled
        if reference_src
        else tiled_copy.layout_dst_tv_tiled
    )
    ref_layout = cute.right_inverse(layout_tv)
    tile_m = cute.size(tiled_copy.tiler_mn[0])
    tile_n = cute.size(tiled_copy.tiler_mn[1])
    ref_layout_mn = cute.composition(ref_layout, cute.make_layout((tile_m, tile_n)))

    num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
    tv_to_lane = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(1, 0, 0))
    ref_to_lane = cute.composition(tv_to_lane, ref_layout_mn)
    lane_m = cute.filter(cute.select(ref_to_lane, [0]))
    lane_n = cute.filter(cute.select(ref_to_lane, [1]))
    lane_layout_mn = layout_utils.concat_layout(lane_m, lane_n)

    tv_to_warp = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(0, 1, 0))
    ref_to_warp = cute.composition(tv_to_warp, ref_layout_mn)
    warp_m = cute.filter(cute.select(ref_to_warp, [0]))
    warp_n = cute.filter(cute.select(ref_to_warp, [1]))
    warp_layout_mn = layout_utils.concat_layout(warp_m, warp_n)
    return lane_layout_mn, warp_layout_mn


@dsl_user_op
def partition_for_epilogue(
    cT: cute.Tensor,
    epi_tile: cute.Tile,
    tiled_copy: cute.TiledCopy,
    tidx: Int32,
    reference_src: bool,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    """Partition one epilogue tensor through the selected copy layout."""
    thread_copy = tiled_copy.get_slice(tidx)
    tiled_tensor = cute.flat_divide(cT, epi_tile)
    if const_expr(reference_src):
        return thread_copy.partition_S(tiled_tensor, loc=loc, ip=ip)
    return thread_copy.partition_D(tiled_tensor, loc=loc, ip=ip)
