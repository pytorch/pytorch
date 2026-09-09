"""CuTeDSL geometry helpers shared by NVGEMM reduction kernels."""

__all__ = ["get_lane_warp_layouts", "partition_for_epilogue"]


def get_lane_warp_layouts(tiled_copy, reference_src=True):
    """Derive two-dimensional lane and warp layouts for an epilogue copy."""
    import cutlass.cute as cute

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


def partition_for_epilogue(*args, **kwargs):
    from torch._vendor.quack.sm90_utils import partition_for_epilogue as partition

    return partition(*args, **kwargs)
