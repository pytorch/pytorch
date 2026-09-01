"""Build per-group metadata on device."""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as cute_testing
from cutlass import Int32, Int64

from torch._native.instrumentation import instrumented_cutedsl_cache

from ._common import _make_fake_1d_tensor, _make_fake_2d_tensor, read_only


class _BuildGroupMetadata:
    def __init__(
        self,
        elem_size_ab: int,
        elem_size_scale: int,
        elem_size_c: int,
        cluster_m: int,
        groups_split_k: bool = False,
    ):
        self.elem_size_ab = elem_size_ab
        self.elem_size_scale = elem_size_scale
        self.elem_size_c = elem_size_c
        self.cluster_m = cluster_m
        self.groups_split_k = groups_split_k

    @cute.jit
    def __call__(
        self,
        group_count: Int32,
        offs: cute.Tensor,
        base_a: Int64,
        base_b: Int64,
        base_c: Int64,
        base_scale_a: Int64,
        base_scale_b: Int64,
        stride_a_row: Int64,
        stride_b_group: Int64,
        stride_c_row: Int64,
        stride_scale_a_group: Int64,
        stride_scale_b_group: Int64,
        scale_a_rows_per_block: Int32,
        tile_m: Int32,
        tile_n: Int32,
        total_m: Int32,
        n: Int32,
        k: Int32,
        out_mnkl: cute.Tensor,
        out_ptrs_abc: cute.Tensor,
        out_ptrs_scale_ab: cute.Tensor,
        out_tile_offsets: cute.Tensor,
        out_total_tiles: cute.Tensor,
        num_blocks: Int32,
        threads_per_block: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        self.kernel(
            group_count,
            offs,
            base_a,
            base_b,
            base_c,
            base_scale_a,
            base_scale_b,
            stride_a_row,
            stride_b_group,
            stride_c_row,
            stride_scale_a_group,
            stride_scale_b_group,
            scale_a_rows_per_block,
            tile_m,
            tile_n,
            total_m,
            n,
            k,
            out_mnkl,
            out_ptrs_abc,
            out_ptrs_scale_ab,
            out_tile_offsets,
            out_total_tiles,
        ).launch(
            grid=[num_blocks, 1, 1],
            block=[threads_per_block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        group_count: Int32,
        offs: cute.Tensor,
        base_a: Int64,
        base_b: Int64,
        base_c: Int64,
        base_scale_a: Int64,
        base_scale_b: Int64,
        stride_a_row: Int64,
        stride_b_group: Int64,
        stride_c_row: Int64,
        stride_scale_a_group: Int64,
        stride_scale_b_group: Int64,
        scale_a_rows_per_block: Int32,
        tile_m: Int32,
        tile_n: Int32,
        total_m: Int32,
        n: Int32,
        k: Int32,
        out_mnkl: cute.Tensor,
        out_ptrs_abc: cute.Tensor,
        out_ptrs_scale_ab: cute.Tensor,
        out_tile_offsets: cute.Tensor,
        out_total_tiles: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdimx, _, _ = cute.arch.block_dim()
        g = bidx * bdimx + tidx

        if g < group_count:
            off_start = Int32(0)
            if g > 0:
                off_start = offs[g - 1]
            off_end = offs[g]
            cute_testing.assert_(
                off_end >= off_start, "group offsets must be nondecreasing"
            )
            elem_ab = Int64(cutlass.const_expr(self.elem_size_ab))
            elem_c = Int64(cutlass.const_expr(self.elem_size_c))
            elem_scale = Int64(cutlass.const_expr(self.elem_size_scale))

            if cutlass.const_expr(self.groups_split_k):
                cute_testing.assert_(
                    off_end <= k, "group offsets must not exceed mat_a.size(1)"
                )
                cute_testing.assert_(
                    off_start % 128 == 0,
                    "group offsets must be a multiple of 128 when offs splits K",
                )
                out_mnkl[g, 0] = total_m
                out_mnkl[g, 1] = n
                out_mnkl[g, 2] = off_end - off_start
                out_mnkl[g, 3] = off_start

                out_ptrs_abc[g, 0] = base_a + Int64(off_start) * elem_ab
                out_ptrs_abc[g, 1] = base_b + Int64(off_start) * elem_ab
                out_ptrs_abc[g, 2] = base_c + Int64(g) * stride_c_row * elem_c
            else:
                cute_testing.assert_(
                    off_end <= total_m, "group offsets must not exceed mat_a.size(0)"
                )
                out_mnkl[g, 0] = off_end - off_start
                out_mnkl[g, 1] = n
                out_mnkl[g, 2] = k
                out_mnkl[g, 3] = Int32(1)

                out_ptrs_abc[g, 0] = base_a + Int64(off_start) * stride_a_row * elem_ab
                out_ptrs_abc[g, 1] = base_b + Int64(g) * stride_b_group * elem_ab
                out_ptrs_abc[g, 2] = base_c + Int64(off_start) * stride_c_row * elem_c

            scale_a_start = off_start // scale_a_rows_per_block
            out_ptrs_scale_ab[g, 0] = (
                base_scale_a + Int64(scale_a_start) * stride_scale_a_group * elem_scale
            )
            scale_b_start = g
            if cutlass.const_expr(self.groups_split_k):
                scale_b_start = off_start // 128
            out_ptrs_scale_ab[g, 1] = (
                base_scale_b + Int64(scale_b_start) * stride_scale_b_group * elem_scale
            )

        if tidx == 0 and bidx == 0:
            total = Int32(0)
            out_tile_offsets[0] = total
            for i in cutlass.range(group_count):
                group_rows = total_m
                if cutlass.const_expr(not self.groups_split_k):
                    start = Int32(0)
                    if i > 0:
                        start = offs[i - 1]
                    group_rows = offs[i] - start
                tiles_m = cute.ceil_div(group_rows, tile_m)
                tiles_m = cute.ceil_div(tiles_m, self.cluster_m) * self.cluster_m
                total += tiles_m * cute.ceil_div(n, tile_n)
                out_tile_offsets[i + 1] = total
            out_total_tiles[0] = total


@instrumented_cutedsl_cache(
    "aten::_scaled_grouped_mm_v2",
    key_fn=lambda elem_size_ab,
    elem_size_scale,
    elem_size_c,
    cluster_m,
    groups_split_k: (
        f"build_group_metadata ab={elem_size_ab} "
        f"scale={elem_size_scale} c={elem_size_c} cluster_m={cluster_m} "
        f"groups_split_k={groups_split_k}"
    ),
)
def _compile_build_group_metadata(
    elem_size_ab: int,
    elem_size_scale: int,
    elem_size_c: int,
    cluster_m: int,
    groups_split_k: bool = False,
):
    from cutlass import Int32 as _Int32, Int64 as _Int64

    offs_fake = _make_fake_1d_tensor(_Int32)
    out_mnkl_fake = _make_fake_2d_tensor(_Int32, 4)
    out_ptrs_abc_fake = _make_fake_2d_tensor(_Int64, 3)
    out_ptrs_scale_ab_fake = _make_fake_2d_tensor(_Int64, 2)
    out_tile_offsets_fake = _make_fake_1d_tensor(_Int32)
    out_total_tiles_fake = _make_fake_1d_tensor(_Int32)
    zero_i64 = _Int64(0)
    zero_i32 = _Int32(0)
    return cute.compile(
        _BuildGroupMetadata(
            elem_size_ab, elem_size_scale, elem_size_c, cluster_m, groups_split_k
        ),
        zero_i32,
        offs_fake,
        zero_i64,
        zero_i64,
        zero_i64,
        zero_i64,
        zero_i64,
        zero_i64,
        zero_i64,
        zero_i64,
        zero_i64,
        zero_i64,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_i32,
        out_mnkl_fake,
        out_ptrs_abc_fake,
        out_ptrs_scale_ab_fake,
        out_tile_offsets_fake,
        out_total_tiles_fake,
        zero_i32,
        256,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --enable-assertions",
    )


def launch_build_group_metadata(
    offs,
    base_a: int,
    base_b: int,
    base_c: int,
    base_scale_a: int,
    base_scale_b: int,
    stride_a_row: int,
    stride_b_group: int,
    stride_c_row: int,
    stride_scale_a_group: int,
    stride_scale_b_group: int,
    scale_a_rows_per_block: int,
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    total_m: int,
    n: int,
    k: int,
    out_mnkl,
    out_ptrs_abc,
    out_ptrs_scale_ab,
    out_tile_offsets,
    out_total_tiles,
    elem_size_ab: int,
    elem_size_scale: int,
    elem_size_c: int,
    groups_split_k: bool = False,
) -> None:
    threads_per_block = 256
    num_blocks = (offs.numel() + threads_per_block - 1) // threads_per_block
    _compile_build_group_metadata(
        elem_size_ab, elem_size_scale, elem_size_c, cluster_m, groups_split_k
    )(
        offs.numel(),
        read_only(offs),
        base_a,
        base_b,
        base_c,
        base_scale_a,
        base_scale_b,
        stride_a_row,
        stride_b_group,
        stride_c_row,
        stride_scale_a_group,
        stride_scale_b_group,
        scale_a_rows_per_block,
        tile_m,
        tile_n,
        total_m,
        n,
        k,
        out_mnkl,
        out_ptrs_abc,
        out_ptrs_scale_ab,
        out_tile_offsets,
        out_total_tiles,
        num_blocks,
    )
