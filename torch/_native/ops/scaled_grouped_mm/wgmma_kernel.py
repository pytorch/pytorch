"""WGMMA kernels for SM90 DeepSeek grouped mm."""

import functools
import os

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as hopper
from cutlass import BFloat16, Float32, Float8E4M3FN, Int32
from cutlass.cute.nvgpu import warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache

from ._common import (
    _make_fake_1d_tensor,
    BLOCKWISE_1X128,
    SCALE_BULK_COPY_ALIGN as _SCALE_BULK_COPY_ALIGN,
    scale_stage_size,
)


_DEBUG_COORD_MAP = os.environ.get("TORCH_CUTEDSL_DEBUG_COORD_MAP") == "1"


def _make_fake_matmul_operand(dtype, dim0, dim1):
    return cute.runtime.make_fake_tensor(
        dtype,
        (dim0, dim1),
        stride=(cute.sym_int64(divisibility=16), 1),
        assumed_align=16,
    )


def _make_fake_mat_b_tensor(dtype, g, k, n):
    return cute.runtime.make_fake_tensor(
        dtype,
        (g, k, n),
        stride=(cute.sym_int64(divisibility=16), 1, cute.sym_int64(divisibility=16)),
        assumed_align=16,
    )


def _make_fake_scale_a_tensor(dtype, recipe_a, m):
    if recipe_a == BLOCKWISE_1X128:
        shape = (m, cute.sym_int())
    else:
        shape = (cute.sym_int(), cute.sym_int())
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=(1, cute.sym_int64()),
        assumed_align=16 if recipe_a == BLOCKWISE_1X128 else None,
    )


def _make_fake_scale_b_tensor(dtype, recipe_b, g, n):
    if recipe_b == BLOCKWISE_1X128:
        shape = (g, n, cute.sym_int())
    else:
        shape = (g, cute.sym_int(), cute.sym_int())
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=(cute.sym_int64(), 1, cute.sym_int64()),
        assumed_align=16 if recipe_b == BLOCKWISE_1X128 else None,
    )


def _make_fake_compact_2d_tensor(dtype, cols: int):
    rows = cute.sym_int()
    return cute.runtime.make_fake_tensor(dtype, (rows, cols), stride=(cols, 1))


def _make_fake_tensormaps(dtype, num_tensormaps: int, tensormap_stride: int):
    ctas = cute.sym_int()
    return cute.runtime.make_fake_tensor(
        dtype,
        (ctas, num_tensormaps, tensormap_stride),
        stride=(num_tensormaps * tensormap_stride, tensormap_stride, 1),
    )


class _DeepSeekWgmmaBase:
    def __init__(
        self, recipe_a: int, recipe_b: int, tile_m: int, tile_n: int, tile_k: int = 128
    ):
        self.recipe_a = recipe_a
        self.recipe_b = recipe_b
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k

    @cute.jit
    def _scale_weights(
        self,
        sScaleA,
        scale_b128_vals,
        scale_a_row_shift,
        row_base,
        scale_a=None,
        abs_row_base=None,
        k_tile=None,
    ):
        if cutlass.const_expr(self.small_scale_a):
            return self._scale_weights_global(
                scale_a, abs_row_base + row_base, k_tile, scale_b128_vals
            )
        valid = self.scale_a_stage_rows if self.a_scale_wide else self.tile_m
        row_lo = row_base + scale_a_row_shift
        row_hi = row_lo + 8
        if row_lo >= valid:
            row_lo = valid - 1
        elif row_lo < 0:
            row_lo = Int32(0)
        if row_hi >= valid:
            row_hi = valid - 1
        elif row_hi < 0:
            row_hi = Int32(0)
        sb = scale_b128_vals[0]
        return sScaleA[row_lo].to(Float32) * sb, sScaleA[row_hi].to(Float32) * sb

    @cute.jit
    def _scale_weights_global(self, scale_a, abs_row_base, k_tile, scale_b128_vals):
        last = cute.size(scale_a, mode=[0]) - 1
        row_lo = cutlass.min(abs_row_base, last)
        row_hi = cutlass.min(abs_row_base + 8, last)
        sb = scale_b128_vals[0]
        return (
            scale_a[row_lo, k_tile].to(Float32) * sb,
            scale_a[row_hi, k_tile].to(Float32) * sb,
        )

    @cute.jit
    def _apply_scaled(
        self, acc_epi, partial_epi, w_lo, w_hi, tRS_cC, epi_tile_num, epi_tile_layout
    ):
        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            coord_cur = tRS_cC[(None, None, None, epi_coord[0], epi_coord[1])]
            size_coord = cute.size(cute.filter_zeros(coord_cur))
            for i in cutlass.range_constexpr(size_coord):
                acc_idx = epi_idx * size_coord + i
                w = w_hi if ((i >> 1) & 1) else w_lo
                acc_epi[acc_idx] = acc_epi[acc_idx] + partial_epi[acc_idx] * w

    @cute.jit
    def accumulate_scaled(
        self,
        acc_epi: cute.Tensor,
        partial_epi: cute.Tensor,
        sScaleA: cute.Tensor,
        sScaleB: cute.Tensor,
        scale_a128_val: Float32,
        scale_b128_vals: cute.Tensor,
        scale_a_row_shift: Int32,
        scale_b_col_shift: Int32,
        n_tile: Int32,
        tRS_cC: cute.Tensor,
        epi_tile_num: cutlass.Constexpr[int],
        epi_tile_layout,
        row_base: Int32 = None,
        scale_a: cute.Tensor = None,
        abs_row_base: Int32 = None,
        k_tile: Int32 = None,
    ):
        scale_a_valid_rows = (
            self.scale_a_stage_rows if self.a_scale_wide else self.tile_m
        )
        scale_b_valid_cols = (
            self.scale_b_stage_cols if self.b_scale_wide else self.tile_n
        )
        if cutlass.const_expr(
            self.recipe_b != BLOCKWISE_1X128 and self.scale_b128_span > 1
        ):
            col_block0 = (n_tile * self.tile_n) // 128

        if cutlass.const_expr(self.can_hoist_scale):
            w_lo, w_hi = self._scale_weights(
                sScaleA,
                scale_b128_vals,
                scale_a_row_shift,
                row_base,
                scale_a,
                abs_row_base,
                k_tile,
            )
            self._apply_scaled(
                acc_epi, partial_epi, w_lo, w_hi, tRS_cC, epi_tile_num, epi_tile_layout
            )

        if cutlass.const_expr(not self.can_hoist_scale):
            for epi_idx in cutlass.range_constexpr(epi_tile_num):
                epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
                coord_cur = tRS_cC[(None, None, None, epi_coord[0], epi_coord[1])]
                coord_flt = cute.filter_zeros(coord_cur)
                size_coord = cute.size(coord_flt)
                for i in cutlass.range_constexpr(size_coord):
                    coord = coord_flt[i]
                    if cutlass.const_expr(_DEBUG_COORD_MAP):
                        dbg_tidx, _, _ = cute.arch.thread_idx()
                        if (
                            dbg_tidx == 128
                            or dbg_tidx == 129
                            or dbg_tidx == 132
                            or dbg_tidx == 160
                        ):
                            cute.printf(
                                "COORDMAP tid={} epi={} i={} row={} col={}",
                                dbg_tidx,
                                epi_idx,
                                i,
                                coord[0],
                                coord[1],
                            )
                    if cutlass.const_expr(
                        self.recipe_a == BLOCKWISE_1X128 and self.small_scale_a
                    ):
                        row = cutlass.min(
                            abs_row_base + coord[0], cute.size(scale_a, mode=[0]) - 1
                        )
                        scale_a_val = scale_a[row, k_tile].to(Float32)
                    elif cutlass.const_expr(self.recipe_a == BLOCKWISE_1X128):
                        scale_a_row = coord[0] + scale_a_row_shift
                        if scale_a_row >= scale_a_valid_rows:
                            scale_a_row = scale_a_valid_rows - 1
                        elif scale_a_row < 0:
                            scale_a_row = Int32(0)
                        scale_a_val = sScaleA[scale_a_row].to(Float32)
                    else:
                        scale_a_val = scale_a128_val
                    if cutlass.const_expr(self.recipe_b == BLOCKWISE_1X128):
                        scale_b_col = coord[1] + scale_b_col_shift
                        if scale_b_col >= scale_b_valid_cols:
                            scale_b_col = scale_b_valid_cols - 1
                        elif scale_b_col < 0:
                            scale_b_col = Int32(0)
                        scale_b_val = sScaleB[scale_b_col].to(Float32)
                    elif cutlass.const_expr(self.scale_b128_span == 1):
                        # tile_n <= 128 keeps the whole tile inside one scale block.
                        scale_b_val = scale_b128_vals[0]
                    else:
                        col_block = (n_tile * self.tile_n + coord[1]) // 128
                        scale_b_val = scale_b128_vals[col_block - col_block0]
                    acc_idx = epi_idx * size_coord + i
                    acc_epi[acc_idx] = (
                        acc_epi[acc_idx]
                        + partial_epi[acc_idx] * scale_a_val * scale_b_val
                    )


class _DeepSeekPersistentWgmma(_DeepSeekWgmmaBase):
    # A/C TMA descriptors are re-pointed per group (ragged M); B isn't (its
    # group axis is a plain TMA dimension already).
    bytes_per_tensormap = 128
    num_tensormaps = 2  # A, C
    _validated_register_budget_tiles = {(64, 64), (64, 128), (128, 64), (128, 128)}
    RASTER_GROUP_N = 8  # raster N-band width; must be a multiple of cluster_n

    def __init__(
        self,
        recipe_a: int,
        recipe_b: int,
        tile_m: int,
        tile_n: int,
        cluster_m: int = 1,
        cluster_n: int = 1,
        a_scale_wide: bool = True,
        b_scale_wide: bool = True,
        tile_k: int = 128,
        ab_stage: int = 4,
        epi_stage: int = 4,
        scale_k_aligned: bool = False,
        small_scale_a: bool = False,
    ):
        super().__init__(recipe_a, recipe_b, tile_m, tile_n, tile_k)
        self.a_scale_wide = a_scale_wide
        self.b_scale_wide = b_scale_wide
        self.scale_k_aligned = scale_k_aligned
        # total_m smaller than the bulk-copy width: cp.async.bulk has no bounds
        # check, so the A scales are read from global per thread instead.
        self.small_scale_a = small_scale_a
        if cluster_m != 1:
            raise ValueError(
                "cluster_m must be 1 (only N-direction clustering is implemented)"
            )
        if cluster_n not in (1, 2):
            raise ValueError("cluster_n must be 1 or 2")
        self.cluster_m = cluster_m
        self.cluster_n = cluster_n
        # The grid is 2D: x indexes independent CTAs/clusters, y indexes a
        # CTA's rank within its cluster (clusterDim=(cluster_m,cluster_n,1)
        # puts the clustered axis on y), matching grouped_gemm.py's own
        # convention -- see persistent_kernel for the block-index handling.
        self.num_mcast_ctas_a = cluster_n
        self.num_mcast_ctas_b = cluster_m
        self.is_a_mcast = cluster_n > 1
        self.is_b_mcast = cluster_m > 1
        self.ab_stage = ab_stage
        self.load_warp_id = 0
        self.num_dma_warp_groups = 1
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = 128
        if tile_m > 64 and tile_m % 128 != 0:
            raise ValueError(
                f"tile_m={tile_m} enters cooperative MMA mode (tile_m > 64) "
                "without being a multiple of 128; epi_tile's row tiling "
                "truncates output rows silently in that case"
            )
        if (tile_m, tile_n) not in self._validated_register_budget_tiles:
            raise ValueError(
                f"(tile_m={tile_m}, tile_n={tile_n}) has not been profiled "
                "for register spilling under the fixed "
                f"load_register_requirement=40/mma_register_requirement=232 "
                "budget; verify via ncu/cuobjdump before adding it to "
                "_validated_register_budget_tiles"
            )
        if tile_k != 128:
            raise ValueError(
                f"tile_k={tile_k} unsupported: DeepSeek blockwise scale "
                "loading assumes each k_tile is exactly one 128-element "
                "scale block (scale_a/scale_b indexed directly by k_tile)"
            )
        self.atom_layout_mnk = (2, 1, 1) if tile_m > 64 else (1, 1, 1)
        self.num_mma_warp_groups = (
            self.atom_layout_mnk[0] * self.atom_layout_mnk[1] * self.atom_layout_mnk[2]
        )
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.scale_b128_span = (self.tile_n + 127) // 128
        self.scale_a_stage_rows = scale_stage_size(self.tile_m)
        self.scale_b_stage_cols = scale_stage_size(self.tile_n)
        self.epi_stage = epi_stage
        self.epi_store_warp_id = (
            self.num_dma_warp_groups * self.num_warps_per_warp_group
        )
        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )
        self.epi_tile = (128, 32) if self.num_mma_warp_groups > 1 else (64, 32)
        self.can_hoist_scale = (
            recipe_a == BLOCKWISE_1X128
            and recipe_b != BLOCKWISE_1X128
            and self.scale_b128_span == 1
            and self.epi_tile[0] == self.tile_m
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )
        self.shared_storage = None

    @cute.jit
    def _clamp_tile_start(self, raw_start, tile_size, total_size, extra_offset, align):
        if cutlass.const_expr(self.scale_k_aligned):
            # extra_offset is a multiple of align, so alignment depends only on
            # raw_start and drops the whole per-k_tile chain.
            limit = cutlass.max(total_size - tile_size, Int32(0))
            limit = limit - (limit % align)
            start = raw_start - (raw_start % align)
            start = cutlass.min(start, limit)
            return start, raw_start - start
        safe_start = raw_start
        if raw_start + tile_size > total_size:
            safe_start = total_size - tile_size
            if safe_start < 0:
                safe_start = Int32(0)
        combined = safe_start + extra_offset
        residue = combined % align
        floor_combined = combined - residue
        aligned_combined = floor_combined
        if residue != 0:
            # Rounding down (the usual case) can't be used when it would
            # push the start negative -- e.g. safe_start==0 at the very
            # first tile of the whole array leaves no room to shift back.
            # Round up instead; this is always representable (>=0) since
            # safe_start>=0 and the increase is <=align.
            floor_start = (floor_combined - extra_offset).to(Int32)
            if floor_start < 0:
                aligned_combined = floor_combined + align
        aligned_start = (aligned_combined - extra_offset).to(Int32)
        if aligned_start < 0:
            aligned_start = Int32(0)
        elif aligned_start + tile_size > total_size:
            aligned_start = (total_size - tile_size).to(Int32)
            if aligned_start < 0:
                aligned_start = Int32(0)
        return aligned_start, raw_start - aligned_start

    @cute.jit
    def _locate_tile(self, tile, offs, tile_offsets, problem_sizes):
        lo = Int32(0)
        hi = Int32(offs.shape[0])
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if tile_offsets[mid] <= tile:
                lo = mid
            else:
                hi = mid
        group = lo
        local_tile = tile - tile_offsets[group]
        group_start = Int32(0)
        if group > 0:
            group_start = offs[group - 1]
        group_m = problem_sizes[group, 0]
        group_n = problem_sizes[group, 1]
        tiles_n = cute.ceil_div(group_n, self.tile_n)
        # Raster by N-band for L2 locality; falls back to one band if
        # tiles_n isn't divisible. band_width stays a multiple of cluster_n.
        band_width = tiles_n
        if tiles_n % self.RASTER_GROUP_N == 0:
            band_width = self.RASTER_GROUP_N
        tiles_per_band = band_width * cute.ceil_div(group_m, self.tile_m)
        band = local_tile // tiles_per_band
        tile_in_band = local_tile - band * tiles_per_band
        m_tile = tile_in_band // band_width
        n_tile = band * band_width + (tile_in_band - m_tile * band_width)
        return group, group_start, group_m, m_tile, n_tile

    @cute.jit
    def _compute_scale_indices(
        self,
        k_tile,
        group,
        group_start,
        m_tile,
        n_tile,
        scale_a,
        scale_b,
        scale_b128_vals,
    ):
        scale_a128_val = Float32(1.0)
        scale_a_row_shift = Int32(0)
        if cutlass.const_expr(self.recipe_a != BLOCKWISE_1X128):
            row_block = (group_start + m_tile * self.tile_m) // 128
            scale_a128_val = scale_a[k_tile, row_block].to(Float32)
        else:
            row_start_raw = group_start + m_tile * self.tile_m
            if cutlass.const_expr(self.a_scale_wide):
                _, scale_a_row_shift = self._clamp_tile_start(
                    row_start_raw,
                    self.scale_a_stage_rows,
                    cute.size(scale_a, mode=[0]),
                    k_tile * scale_a.stride[1],
                    _SCALE_BULK_COPY_ALIGN,
                )
            else:
                _, scale_a_row_shift = self._clamp_tile_start(
                    row_start_raw,
                    self.tile_m,
                    cute.size(scale_a, mode=[0]),
                    k_tile * scale_a.stride[1],
                    _SCALE_BULK_COPY_ALIGN,
                )
        scale_b_col_shift = Int32(0)
        if cutlass.const_expr(self.recipe_b != BLOCKWISE_1X128):
            col_block0 = (n_tile * self.tile_n) // 128
            for w in cutlass.range_constexpr(self.scale_b128_span):
                scale_b128_vals[w] = scale_b[group, k_tile, col_block0 + w].to(Float32)
        else:
            col_start_raw = n_tile * self.tile_n
            if cutlass.const_expr(self.b_scale_wide):
                _, scale_b_col_shift = self._clamp_tile_start(
                    col_start_raw,
                    self.scale_b_stage_cols,
                    cute.size(scale_b, mode=[1]),
                    group * scale_b.stride[0] + k_tile * scale_b.stride[2],
                    _SCALE_BULK_COPY_ALIGN,
                )
            else:
                _, scale_b_col_shift = self._clamp_tile_start(
                    col_start_raw,
                    self.tile_n,
                    cute.size(scale_b, mode=[1]),
                    group * scale_b.stride[0] + k_tile * scale_b.stride[2],
                    _SCALE_BULK_COPY_ALIGN,
                )
        return scale_a128_val, scale_a_row_shift, scale_b_col_shift

    @cute.jit
    def _barrier_wait(self, sync_object, state):
        # Spinning beats try_wait/suspend here by ~6%, at every timeout hint
        # from 2k to 200k ticks -- suspend/resume latency is the cost, not the
        # library's 10ms default.
        bar = sync_object.get_barrier(state.index)
        done = cute.arch.mbarrier_try_wait(bar, state.phase)
        while done == 0:
            done = cute.arch.mbarrier_try_wait(bar, state.phase)

    @cute.jit
    def _issue_wgmma(self, tiled_mma, partial, tCrA, tCrB, num_k_blocks, stage_index):
        partial.fill(0.0)
        warpgroup.fence()
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, False)
        for k_blk in cutlass.range_constexpr(num_k_blocks):
            coord = (None, None, k_blk, stage_index)
            cute.gemm(mma_atom, partial, tCrA[coord], tCrB[coord], partial)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()

    @cute.jit
    def _make_tensor_for_tensormap_update(
        self, ptr_i64, dtype, dim0, dim1, stride0, stride1
    ):
        ptr = cute.make_ptr(dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16)
        return cute.make_tensor(
            ptr,
            cute.make_layout(
                (dim0, dim1, Int32(1)), stride=(stride0, stride1, cutlass.Int64(0))
            ),
        )

    @cute.jit
    def __call__(
        self,
        mat_a: cute.Tensor,
        mat_b: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        offs: cute.Tensor,
        problem_sizes: cute.Tensor,
        tile_offsets: cute.Tensor,
        total_tiles: cute.Tensor,
        ptrs_abc: cute.Tensor,
        tensormaps: cute.Tensor,
        out: cute.Tensor,
        k: Int32,
        num_ctas: Int32,
        num_blocks: Int32,
        threads_per_block: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        stride_a_m, stride_a_k = mat_a.stride[0], mat_a.stride[1]
        stride_c_m, stride_c_n = out.stride[0], out.stride[1]
        tiled_mma = hopper.make_trivial_tiled_mma(
            Float8E4M3FN,
            Float8E4M3FN,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=self.atom_layout_mnk,
            tiler_mn=(64, self.tile_n),
        )
        tile_shape = (self.tile_m, self.tile_n, self.tile_k)
        sA_layout = hopper.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR, tile_shape, mat_a.element_type, self.ab_stage
        )
        sB_layout = hopper.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, tile_shape, mat_b.element_type, self.ab_stage
        )
        sC_layout = hopper.make_smem_layout_epi(
            out.element_type,
            LayoutEnum.ROW_MAJOR,
            self.epi_tile,
            self.epi_stage,
        )
        mat_a_lmk = cute.make_tensor(
            mat_a.iterator,
            cute.prepend(mat_a.layout, cute.make_layout(1), up_to_rank=3),
        )
        mat_a_mkl = cute.make_tensor(
            mat_a_lmk.iterator,
            cute.select(mat_a_lmk.layout, [1, 2, 0]),
        )
        mat_b_nkl = cute.make_tensor(
            mat_b.iterator,
            cute.select(mat_b.layout, [2, 1, 0]),
        )
        out_lmn = cute.make_tensor(
            out.iterator,
            cute.prepend(out.layout, cute.make_layout(1), up_to_rank=3),
        )
        out_mnl = cute.make_tensor(
            out_lmn.iterator,
            cute.select(out_lmn.layout, [1, 2, 0]),
        )
        sA_stage = cute.slice_(sA_layout, (None, None, 0))
        sB_stage = cute.slice_(sB_layout, (None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            (
                cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
                if cutlass.const_expr(self.is_a_mcast)
                else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            ),
            mat_a_mkl,
            sA_stage,
            (self.tile_m, self.tile_k),
            num_multicast=(
                self.cluster_n if cutlass.const_expr(self.is_a_mcast) else 1
            ),
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mat_b_nkl,
            sB_stage,
            (self.tile_n, self.tile_k),
        )
        sC_stage = cute.slice_(sC_layout, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            out_mnl,
            sC_stage,
            self.epi_tile,
        )
        copy_atom_r2s = hopper.sm90_get_smem_store_op(
            LayoutEnum.ROW_MAJOR,
            elem_ty_d=out.element_type,
            elem_ty_acc=Float32,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4),
            out.element_type,
        )
        tiled_copy_C = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            tensormap_buffer: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int64,
                    self.num_tensormaps * self.bytes_per_tensormap // 8,
                ],
                128,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[mat_a.element_type, cute.cosize(sA_layout)],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[mat_b.element_type, cute.cosize(sB_layout)],
                1024,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[out.element_type, cute.cosize(sC_layout)],
                1024,
            ]
            sScaleA: cute.struct.Align[
                cute.struct.MemRange[Float32, self.scale_a_stage_rows * self.ab_stage],
                16,
            ]
            sScaleB: cute.struct.Align[
                cute.struct.MemRange[Float32, self.scale_b_stage_cols * self.ab_stage],
                16,
            ]

        cta_layout_mnk = cute.make_layout((self.cluster_m, self.cluster_n, 1))

        self.shared_storage = SharedStorage
        kernel = self.persistent_kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            scale_a,
            scale_b,
            offs,
            problem_sizes,
            tile_offsets,
            total_tiles,
            ptrs_abc,
            tensormaps,
            stride_a_m,
            stride_a_k,
            stride_c_m,
            stride_c_n,
            k,
            num_ctas,
            tiled_mma,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_r2s,
            cta_layout_mnk,
        )
        if cutlass.const_expr(self.is_a_mcast):
            kernel.launch(
                grid=[num_blocks // self.cluster_n, self.cluster_n, 1],
                block=[threads_per_block, 1, 1],
                cluster=(self.cluster_m, self.cluster_n, 1),
                min_blocks_per_mp=1,
                stream=stream,
            )
        else:
            kernel.launch(
                grid=[num_blocks, 1, 1],
                block=[threads_per_block, 1, 1],
                min_blocks_per_mp=1,
                stream=stream,
            )

    @cute.kernel
    def persistent_kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        offs: cute.Tensor,
        problem_sizes: cute.Tensor,
        tile_offsets: cute.Tensor,
        total_tiles: cute.Tensor,
        ptrs_abc: cute.Tensor,
        tensormaps: cute.Tensor,
        stride_a_m: cutlass.Int64,
        stride_a_k: cutlass.Int64,
        stride_c_m: cutlass.Int64,
        stride_c_n: cutlass.Int64,
        k: Int32,
        num_ctas: Int32,
        tiled_mma: cute.TiledMma,
        sA_layout,
        sB_layout,
        sC_layout,
        tiled_copy_r2s: cute.TiledCopy,
        cta_layout_mnk: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = Int32(0)
        if cutlass.const_expr(self.is_a_mcast):
            cidx, _, _ = cute.arch.cluster_idx()
            bidx = cidx * self.cluster_n
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            bidx += cta_rank_in_cluster
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        a_mcast_mask = cutlass.Int16(0)
        b_mcast_mask = cutlass.Int16(0)
        if cutlass.const_expr(self.is_a_mcast):
            cta_layout_vmnk = cute.make_layout((1, self.cluster_m, self.cluster_n, 1))
            cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
            a_mcast_mask = cute.nvgpu.cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, cluster_coord_vmnk, 2
            )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
        sC = storage.sC.get_tensor(sC_layout.outer, swizzle=sC_layout.inner)
        sScaleA = storage.sScaleA.get_tensor(
            cute.make_layout(
                (self.scale_a_stage_rows, self.ab_stage),
                stride=(1, self.scale_a_stage_rows),
            )
        )
        # Narrower tile_m-sized view for the a_scale_wide=False copy.
        sScaleANarrow = cute.make_tensor(
            sScaleA.iterator,
            cute.make_layout(
                (self.tile_m, self.ab_stage),
                stride=(1, self.scale_a_stage_rows),
            ),
        )
        sScaleB = storage.sScaleB.get_tensor(
            cute.make_layout(
                (self.scale_b_stage_cols, self.ab_stage),
                stride=(1, self.scale_b_stage_cols),
            )
        )
        # Narrower tile_n-sized view for the b_scale_wide=False copy.
        sScaleBNarrow = cute.make_tensor(
            sScaleB.iterator,
            cute.make_layout(
                (self.tile_n, self.ab_stage),
                stride=(1, self.scale_b_stage_cols),
            ),
        )
        tensormap_manager = utils.TensorMapManager(
            utils.TensorMapUpdateMode.SMEM, self.bytes_per_tensormap
        )
        tensormap_i64_stride = self.bytes_per_tensormap // 8
        tensormap_a_smem_ptr = storage.tensormap_buffer.data_ptr()
        tensormap_c_smem_ptr = tensormap_a_smem_ptr + tensormap_i64_stride
        tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(bidx, 0, None)].iterator
        )
        tensormap_c_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(bidx, 1, None)].iterator
        )
        tma_desc_a = tensormap_manager.get_tensormap_ptr(
            tensormap_a_gmem_ptr, cute.AddressSpace.generic
        )
        tma_desc_c = tensormap_manager.get_tensormap_ptr(
            tensormap_c_gmem_ptr, cute.AddressSpace.generic
        )
        # Acquire-fence any tensormap left in this (reused) gmem slot by a
        # prior launch before this launch's first update_tensormap call.
        tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_c_gmem_ptr)

        sA_stage = cute.slice_(sA_layout, (None, None, 0))
        sB_stage = cute.slice_(sB_layout, (None, None, 0))
        scale_bytes = 0
        # a_scale_wide/b_scale_wide are compile-time choices (set by the
        # caller from total_m/N) since tx_count below must be a Python
        # value, not runtime.
        if cutlass.const_expr(
            self.recipe_a == BLOCKWISE_1X128 and not self.small_scale_a
        ):
            if cutlass.const_expr(self.a_scale_wide):
                scale_bytes += self.scale_a_stage_rows * 4
            else:
                scale_bytes += self.tile_m * 4
        if cutlass.const_expr(self.recipe_b == BLOCKWISE_1X128):
            if cutlass.const_expr(self.b_scale_wide):
                scale_bytes += self.scale_b_stage_cols * 4
            else:
                scale_bytes += self.tile_n * 4
        tma_copy_bytes = (
            cute.size_in_bytes(Float8E4M3FN, sA_stage)
            + cute.size_in_bytes(Float8E4M3FN, sB_stage)
            + scale_bytes
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group,
            ),
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )
        scale_copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyBulkG2SOp(), Float32
        )
        pipeline_init_arrive(
            cluster_shape_mn=(self.cluster_m, self.cluster_n), is_relaxed=True
        )

        gA = cute.local_tile(
            mA_mkl,
            cute.slice_((self.tile_m, self.tile_n, self.tile_k), (None, 0, None)),
            (None, None, None),
        )
        gB = cute.local_tile(
            mB_nkl,
            cute.slice_((self.tile_m, self.tile_n, self.tile_k), (0, None, None)),
            (None, None, None),
        )
        gC = cute.local_tile(
            mC_mnl,
            cute.slice_((self.tile_m, self.tile_n, self.tile_k), (None, None, 0)),
            (None, None, None),
        )
        if cutlass.const_expr(self.is_a_mcast):
            a_cta_layout = cute.make_layout(
                cute.slice_(cta_layout_mnk, (0, None, 0)).shape
            )
            a_cta_crd = cluster_coord_mnk[1]
        else:
            a_cta_layout = cute.make_layout(1)
            a_cta_crd = 0
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        pipeline_init_wait(cluster_shape_mn=(self.cluster_m, self.cluster_n))
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
        else:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

        tile = bidx
        chunks = k // self.tile_k
        if warp_idx == self.load_warp_id:
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_a, tensormap_a_smem_ptr, self.load_warp_id
            )
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            did_work = tile < total_tiles[0]
            last_group_idx = Int32(-1)
            while tile < total_tiles[0]:
                group, group_start, group_m, m_tile, n_tile = self._locate_tile(
                    tile, offs, tile_offsets, problem_sizes
                )
                is_group_changed = group != last_group_idx
                if is_group_changed:
                    real_tensor_a = self._make_tensor_for_tensormap_update(
                        ptrs_abc[group, 0],
                        Float8E4M3FN,
                        group_m,
                        k,
                        stride_a_m,
                        stride_a_k,
                    )
                    tensormap_manager.update_tensormap(
                        (real_tensor_a,),
                        (tma_atom_a,),
                        (tensormap_a_gmem_ptr,),
                        self.load_warp_id,
                        (tensormap_a_smem_ptr,),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
                    last_group_idx = group
                producer_state.reset_count()
                for k_tile in cutlass.range(chunks):
                    self._barrier_wait(
                        mainloop_pipeline.sync_object_empty, producer_state
                    )
                    mainloop_pipeline.sync_object_full.arrive(
                        producer_state.index, mainloop_pipeline.producer_mask
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, m_tile, k_tile, 0)],
                        tAsA[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            producer_state
                        ),
                        tma_desc_ptr=tma_desc_a,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, n_tile, k_tile, group)],
                        tBsB[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            producer_state
                        ),
                        mcast_mask=b_mcast_mask,
                    )
                    if cutlass.const_expr(
                        self.recipe_a == BLOCKWISE_1X128 and not self.small_scale_a
                    ):
                        row_start_raw = group_start + m_tile * self.tile_m
                        if cutlass.const_expr(self.a_scale_wide):
                            row_start, _ = self._clamp_tile_start(
                                row_start_raw,
                                self.scale_a_stage_rows,
                                cute.size(scale_a, mode=[0]),
                                k_tile * scale_a.stride[1],
                                _SCALE_BULK_COPY_ALIGN,
                            )
                            sfa_iter = scale_a.iterator + (
                                row_start * scale_a.stride[0]
                                + k_tile * scale_a.stride[1]
                            )
                            sfa_src = cute.make_tensor(
                                sfa_iter,
                                cute.make_layout(
                                    (self.scale_a_stage_rows,),
                                    stride=(scale_a.stride[0],),
                                ),
                            )
                            cute.copy(
                                scale_copy_atom,
                                sfa_src,
                                sScaleA[(None, producer_state.index)],
                                mbar_ptr=mainloop_pipeline.producer_get_barrier(
                                    producer_state
                                ),
                            )
                        else:
                            row_start, _ = self._clamp_tile_start(
                                row_start_raw,
                                self.tile_m,
                                cute.size(scale_a, mode=[0]),
                                k_tile * scale_a.stride[1],
                                _SCALE_BULK_COPY_ALIGN,
                            )
                            sfa_iter = scale_a.iterator + (
                                row_start * scale_a.stride[0]
                                + k_tile * scale_a.stride[1]
                            )
                            sfa_src = cute.make_tensor(
                                sfa_iter,
                                cute.make_layout(
                                    (self.tile_m,), stride=(scale_a.stride[0],)
                                ),
                            )
                            cute.copy(
                                scale_copy_atom,
                                sfa_src,
                                sScaleANarrow[(None, producer_state.index)],
                                mbar_ptr=mainloop_pipeline.producer_get_barrier(
                                    producer_state
                                ),
                            )
                    if cutlass.const_expr(self.recipe_b == BLOCKWISE_1X128):
                        col_start_raw = n_tile * self.tile_n
                        if cutlass.const_expr(self.b_scale_wide):
                            col_start, _ = self._clamp_tile_start(
                                col_start_raw,
                                self.scale_b_stage_cols,
                                cute.size(scale_b, mode=[1]),
                                group * scale_b.stride[0] + k_tile * scale_b.stride[2],
                                _SCALE_BULK_COPY_ALIGN,
                            )
                            sfb_iter = scale_b.iterator + (
                                group * scale_b.stride[0]
                                + col_start * scale_b.stride[1]
                                + k_tile * scale_b.stride[2]
                            )
                            sfb_src = cute.make_tensor(
                                sfb_iter,
                                cute.make_layout(
                                    (self.scale_b_stage_cols,),
                                    stride=(scale_b.stride[1],),
                                ),
                            )
                            cute.copy(
                                scale_copy_atom,
                                sfb_src,
                                sScaleB[(None, producer_state.index)],
                                mbar_ptr=mainloop_pipeline.producer_get_barrier(
                                    producer_state
                                ),
                            )
                        else:
                            col_start, _ = self._clamp_tile_start(
                                col_start_raw,
                                self.tile_n,
                                cute.size(scale_b, mode=[1]),
                                group * scale_b.stride[0] + k_tile * scale_b.stride[2],
                                _SCALE_BULK_COPY_ALIGN,
                            )
                            sfb_iter = scale_b.iterator + (
                                group * scale_b.stride[0]
                                + col_start * scale_b.stride[1]
                                + k_tile * scale_b.stride[2]
                            )
                            sfb_src = cute.make_tensor(
                                sfb_iter,
                                cute.make_layout(
                                    (self.tile_n,), stride=(scale_b.stride[1],)
                                ),
                            )
                            cute.copy(
                                scale_copy_atom,
                                sfb_src,
                                sScaleBNarrow[(None, producer_state.index)],
                                mbar_ptr=mainloop_pipeline.producer_get_barrier(
                                    producer_state
                                ),
                            )
                    mainloop_pipeline.producer_commit(producer_state)
                    producer_state.advance()
                tile += num_ctas
            if did_work:
                mainloop_pipeline.producer_tail(producer_state)
        elif not is_dma_warp_group:
            mma_warp_group_thread_layout = cute.make_layout(
                self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
            )
            mma_wg_id = warp_group_idx - self.num_dma_warp_groups
            thr_mma = tiled_mma.get_slice(mma_warp_group_thread_layout(mma_wg_id))
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            tCgC = thr_mma.partition_C(gC)
            acc = cute.make_rmem_tensor(tCgC.shape[:3], Float32)
            partial = cute.make_rmem_tensor(tCgC.shape[:3], Float32)
            scale_b128_vals = cute.make_rmem_tensor((self.scale_b128_span,), Float32)
            num_k_blocks = cute.size(tCrA, mode=[2])
            mma_tidx = tidx - self.num_threads_per_warp_group

            cC = cute.make_identity_tensor((self.tile_m, self.tile_n))
            cC_epi = cute.flat_divide(cC, self.epi_tile)
            thr_copy_r2s = tiled_copy_r2s.get_slice(mma_tidx)
            tRS_cC = thr_copy_r2s.partition_S(cC_epi)
            acc_row_base = cute.filter_zeros(tRS_cC[(None, None, None, 0, 0)])[0][0]
            epi_tile_shape = (cute.size(cC_epi, mode=[2]), cute.size(cC_epi, mode=[3]))
            epi_tile_num = cute.size(epi_tile_shape)
            epi_tile_layout = cute.make_layout(
                epi_tile_shape, stride=(epi_tile_shape[1], 1)
            )
            acc_epi = tiled_copy_r2s.retile(acc)
            partial_epi = tiled_copy_r2s.retile(partial)

            read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_threads,
                ),
            )
            tiles_executed = Int32(0)
            did_work = tile < total_tiles[0]
            last_group_idx = Int32(-1)
            if warp_idx == self.epi_store_warp_id:
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_c, tensormap_c_smem_ptr, self.epi_store_warp_id
                )
            while tile < total_tiles[0]:
                group, group_start, group_m, m_tile, n_tile = self._locate_tile(
                    tile, offs, tile_offsets, problem_sizes
                )
                group_n = problem_sizes[group, 1]
                is_group_changed = group != last_group_idx
                if is_group_changed:
                    if warp_idx == self.epi_store_warp_id:
                        real_tensor_c = self._make_tensor_for_tensormap_update(
                            ptrs_abc[group, 2],
                            BFloat16,
                            group_m,
                            group_n,
                            stride_c_m,
                            stride_c_n,
                        )
                        tensormap_manager.update_tensormap(
                            (real_tensor_c,),
                            (tma_atom_c,),
                            (tensormap_c_gmem_ptr,),
                            self.epi_store_warp_id,
                            (tensormap_c_smem_ptr,),
                        )
                        tensormap_manager.fence_tensormap_update(tensormap_c_gmem_ptr)
                    self.epilog_sync_barrier.arrive_and_wait()
                    last_group_idx = group
                acc.fill(0.0)
                read_state.reset_count()
                release_state.reset_count()
                for k_tile in cutlass.range(chunks):
                    self._barrier_wait(mainloop_pipeline.sync_object_full, read_state)
                    self._issue_wgmma(
                        tiled_mma,
                        partial,
                        tCrA,
                        tCrB,
                        num_k_blocks,
                        read_state.index,
                    )
                    scale_a128_val, scale_a_row_shift, scale_b_col_shift = (
                        self._compute_scale_indices(
                            k_tile,
                            group,
                            group_start,
                            m_tile,
                            n_tile,
                            scale_a,
                            scale_b,
                            scale_b128_vals,
                        )
                    )
                    warpgroup.wait_group(0)
                    self.accumulate_scaled(
                        acc_epi,
                        partial_epi,
                        sScaleA[(None, read_state.index)],
                        sScaleB[(None, read_state.index)],
                        scale_a128_val,
                        scale_b128_vals,
                        scale_a_row_shift,
                        scale_b_col_shift,
                        n_tile,
                        tRS_cC,
                        epi_tile_num,
                        epi_tile_layout,
                        acc_row_base,
                        scale_a,
                        group_start + m_tile * self.tile_m,
                        k_tile,
                    )
                    mainloop_pipeline.consumer_release(release_state)
                    read_state.advance()
                    release_state.advance()
                self.epilogue_persistent(
                    acc,
                    gC[(None, None, m_tile, n_tile, 0)],
                    sC,
                    tma_atom_c,
                    tma_desc_c,
                    tiled_copy_r2s,
                    mma_tidx,
                    warp_idx,
                    tma_store_pipeline,
                    tiles_executed,
                )
                tiles_executed += 1
                tile += num_ctas
            if did_work:
                tma_store_pipeline.producer_tail()

    @cute.jit
    def epilogue_persistent(
        self,
        acc: cute.Tensor,
        gC: cute.Tensor,
        sC: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        tma_desc_c: cute.Pointer,
        tiled_copy_r2s: cute.TiledCopy,
        tidx: Int32,
        warp_idx: Int32,
        tma_store_pipeline: pipeline.PipelineTmaStore,
        tiles_executed: Int32,
    ):
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile(acc)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, Float32)
        tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, BFloat16)
        size_tRS_rD = cute.size(tRS_rD)
        gC_epi = cute.zipped_divide(gC, self.epi_tile)
        epi_tile_num = cute.size(gC_epi, mode=[1])
        epi_tile_shape = gC_epi.shape[1]
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            gC_epi,
        )
        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
            tRS_rD_out.store(tRS_rD.load().to(BFloat16))
            epi_buffer = (tiles_executed * epi_tile_num + epi_idx) % cute.size(
                tRS_sD, mode=[3]
            )
            cute.copy(
                tiled_copy_r2s,
                tRS_rD_out,
                tRS_sD[(None, None, None, epi_buffer)],
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            self.epilog_sync_barrier.arrive_and_wait()
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            if warp_idx == self.epi_store_warp_id:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                    tma_desc_ptr=tma_desc_c,
                )
                tma_store_pipeline.producer_commit()
                tma_store_pipeline.producer_acquire()
            self.epilog_sync_barrier.arrive_and_wait()


@instrumented_cutedsl_cache(
    "aten::_scaled_grouped_mm_v2",
    key_fn=lambda recipe_a,
    recipe_b,
    tile_m,
    tile_n,
    cluster_m,
    cluster_n,
    a_scale_wide,
    b_scale_wide,
    tile_k,
    ab_stage,
    epi_stage,
    scale_k_aligned,
    small_scale_a: (
        f"deepseek_persistent_wgmma a={recipe_a} b={recipe_b} "
        f"tile={tile_m}x{tile_n}x{tile_k} cluster={cluster_m}x{cluster_n} "
        f"a_scale_wide={a_scale_wide} b_scale_wide={b_scale_wide} "
        f"ab_stage={ab_stage} epi_stage={epi_stage} "
        f"scale_k_aligned={scale_k_aligned} small_scale_a={small_scale_a}"
    ),
)
def _compile_deepseek_persistent_wgmma(
    recipe_a: int,
    recipe_b: int,
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    a_scale_wide: bool,
    b_scale_wide: bool,
    tile_k: int = 128,
    ab_stage: int = 4,
    epi_stage: int = 4,
    scale_k_aligned: bool = False,
    small_scale_a: bool = False,
):
    from ._compile_with_safe_names import _compile_with_safe_names

    kernel = _DeepSeekPersistentWgmma(
        recipe_a,
        recipe_b,
        tile_m,
        tile_n,
        cluster_m,
        cluster_n,
        a_scale_wide,
        b_scale_wide,
        tile_k=tile_k,
        ab_stage=ab_stage,
        epi_stage=epi_stage,
        scale_k_aligned=scale_k_aligned,
        small_scale_a=small_scale_a,
    )
    zero_i32 = Int32(0)

    m = cute.sym_int()
    k = cute.sym_int(divisibility=128)
    g = cute.sym_int()
    n = cute.sym_int()

    return _compile_with_safe_names(
        lambda: cute.compile(
            kernel,
            _make_fake_matmul_operand(Float8E4M3FN, m, k),
            _make_fake_mat_b_tensor(Float8E4M3FN, g, k, n),
            _make_fake_scale_a_tensor(Float32, recipe_a, m),
            _make_fake_scale_b_tensor(Float32, recipe_b, g, n),
            _make_fake_1d_tensor(Int32),
            _make_fake_compact_2d_tensor(Int32, 4),
            _make_fake_1d_tensor(Int32),
            _make_fake_1d_tensor(Int32),
            _make_fake_compact_2d_tensor(cutlass.Int64, 3),
            _make_fake_tensormaps(
                cutlass.Int64,
                _DeepSeekPersistentWgmma.num_tensormaps,
                _DeepSeekPersistentWgmma.bytes_per_tensormap // 8,
            ),
            _make_fake_matmul_operand(BFloat16, m, n),
            zero_i32,
            zero_i32,
            zero_i32,
            kernel.threads_per_cta,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    )


@functools.lru_cache(maxsize=32)
def _alloc_tensormaps(
    device_index: int, num_blocks: int, stream_ptr: int
) -> torch.Tensor:
    device = torch.device("cuda", device_index)
    return torch.empty(
        (
            num_blocks,
            _DeepSeekPersistentWgmma.num_tensormaps,
            _DeepSeekPersistentWgmma.bytes_per_tensormap // 8,
        ),
        dtype=torch.int64,
        device=device,
    )


def _get_tensormaps(num_blocks: int, device: torch.device) -> torch.Tensor:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_ptr = torch.cuda.current_stream(device_index).cuda_stream
    return _alloc_tensormaps(device_index, num_blocks, stream_ptr)


def launch_deepseek_grouped_wgmma(
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    recipe_a: int,
    recipe_b: int,
    offs,
    problem_sizes,
    tile_offsets,
    total_tiles,
    ptrs_abc,
    out,
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    num_sms: int,
    a_scale_wide: bool = True,
    b_scale_wide: bool = True,
    tile_k: int = 128,
    ab_stage: int = 4,
    epi_stage: int = 4,
) -> None:
    num_blocks = num_sms
    if cluster_n > 1:
        num_blocks = (num_blocks // cluster_n) * cluster_n
    tensormaps = _get_tensormaps(num_blocks, mat_a.device)
    # cp.async.bulk has a compile-time width and no bounds check, so a scale_a
    # shorter than one staged tile can't be staged at all -- read it from
    # global instead.
    small_scale_a = recipe_a == BLOCKWISE_1X128 and scale_a.size(0) < (
        scale_stage_size(tile_m) if a_scale_wide else tile_m
    )
    # Every stride feeding a bulk-copy offset must be align-multiple, else the
    # address alignment depends on k_tile and needs the full clamp chain.
    scale_k_aligned = (
        recipe_a != BLOCKWISE_1X128 or scale_a.stride(1) % _SCALE_BULK_COPY_ALIGN == 0
    ) and (
        recipe_b != BLOCKWISE_1X128
        or (
            scale_b.stride(0) % _SCALE_BULK_COPY_ALIGN == 0
            and scale_b.stride(2) % _SCALE_BULK_COPY_ALIGN == 0
        )
    )
    _compile_deepseek_persistent_wgmma(
        recipe_a,
        recipe_b,
        tile_m,
        tile_n,
        cluster_m,
        cluster_n,
        a_scale_wide,
        b_scale_wide,
        tile_k,
        ab_stage,
        epi_stage,
        scale_k_aligned,
        small_scale_a,
    )(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        offs,
        problem_sizes,
        tile_offsets,
        total_tiles,
        ptrs_abc,
        tensormaps,
        out,
        mat_a.size(1),
        num_blocks,
        num_blocks,
    )
