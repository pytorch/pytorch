# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl

from .gemm_gfx950 import (
    __barrier,
    __waitcnt,
    _elem_dtype,
    _make_gemm_gfx950_tiled_mma,
    BlockSwizzle,
    buffer_load_lds_inline,
    GEMM_DTYPE_FP16,
    GemmGfx950Param,
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
)


def _grouped_swizzle_tile(param, num_pid_m, num_pid_n, local_tile):
    bid_m = local_tile % num_pid_m
    bid_n = local_tile // num_pid_m
    if const_expr(param.group_m <= 0):
        return bid_m, bid_n

    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    swizzled_m, swizzled_n = block_swizzle.swizzle(num_pid_m, num_pid_n, local_tile)
    num_workgroups = num_pid_m * num_pid_n
    use_block_swizzle = (num_workgroups >= block_swizzle.NUM_PIDS_THRESHOLD) & (
        (num_workgroups % block_swizzle.NUM_XCDS) == 0
    )
    if const_expr(isinstance(use_block_swizzle, bool)):
        if const_expr(use_block_swizzle):
            return swizzled_m, swizzled_n
        return bid_m, bid_n
    return (
        use_block_swizzle.select(swizzled_m, bid_m),
        use_block_swizzle.select(swizzled_n, bid_n),
    )


def get_grouped_gemm_persistent_grid_size(
    param: GemmGfx950Param,
    total_m: int,
    n: int,
    group_count: int,
    device_properties,
) -> int:
    num_cus = int(getattr(device_properties, "multi_processor_count", 1) or 1)
    if total_m <= 0 or n <= 0 or group_count <= 0:
        return 1

    smem_bytes = (
        param.stages
        * (param.block_m + param.block_n)
        * param.block_k
        * param.in_data_bytes
    )
    smem_bytes = max(smem_bytes, param.block_m * param.block_n * param.out_data_bytes)
    shared_memory_per_cu = getattr(
        device_properties, "shared_memory_per_multiprocessor", None
    )
    max_threads_per_cu = getattr(
        device_properties, "max_threads_per_multi_processor", None
    )
    if shared_memory_per_cu is None or max_threads_per_cu is None:
        resource_blocks_per_cu = 1
    else:
        resource_blocks_per_cu = min(
            max(int(shared_memory_per_cu) // smem_bytes, 1),
            max(int(max_threads_per_cu) // param.block_threads, 1),
        )

    light_tile = param.block_m <= 64 and param.block_n <= 128
    n_tiles = (n - 1) // param.block_n + 1
    if light_tile:
        for blocks_per_cu in (8, 4, 2, 1):
            if blocks_per_cu <= resource_blocks_per_cu:
                break
        # The host only knows total M. This lower bound prevents empty CTAs
        # for uniformly small groups, even though ragged inputs have more work.
        task_floor = ((total_m - 1) // param.block_m + 1) * n_tiles
        return max(1, min(num_cus * blocks_per_cu, task_floor))

    blocks_per_cu = 2 if resource_blocks_per_cu >= 2 else 1
    nonempty_groups_upper = min(group_count, total_m)
    m_tiles_upper = (
        nonempty_groups_upper + (total_m - nonempty_groups_upper) // param.block_m
    )
    return max(1, min(num_cus * blocks_per_cu, m_tiles_upper * n_tiles))


def make_grouped_gemm_gfx950_kernel_name(param: GemmGfx950Param) -> str:
    dtype_str = "fp16" if param.dtype_id == GEMM_DTYPE_FP16 else "bf16"
    name = (
        f"grouped_gemm_{dtype_str}_"
        f"t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}"
    )
    name += f"_w{param.m_waves}x{param.n_waves}"
    name += f"_gm{param.group_m}"
    name += f"_ktail{int(param.has_k_tail)}"
    name += "_hti" if param.use_half_tile_interleaved else "_ft"
    return name


@dataclass
class _GroupedGemmGfx950Resources:
    """Buffer descriptors, copy atoms and thread slices shared by both kernels."""

    out_buf: Any
    offs_buf: Any
    a_rsrc: Any
    b_rsrc: Any
    s2r_copy_atom: Any
    r2g_copy_atom: Any
    thr_mma: Any
    thr_copy_A: Any
    b_s2r_copy_atom: Any
    thr_copy_B: Any


def _grouped_gemm_gfx950_resources(out, a, b, offs, tiled_mma, elem_dtype, tid):
    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    offs_buf = fx.rocdl.make_buffer_tensor(offs, max_size=True)
    a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(a_buf))
    b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(b_buf))
    s2r_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    g2r_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    r2g_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(g2r_copy_atom, tiled_mma).get_slice(tid)
    b_s2r_copy_atom = fx.make_copy_atom(rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype)
    thr_copy_B = fx.make_tiled_copy_B(b_s2r_copy_atom, tiled_mma).get_slice(tid)
    return _GroupedGemmGfx950Resources(
        out_buf=out_buf,
        offs_buf=offs_buf,
        a_rsrc=a_rsrc,
        b_rsrc=b_rsrc,
        s2r_copy_atom=s2r_copy_atom,
        r2g_copy_atom=r2g_copy_atom,
        thr_mma=thr_mma,
        thr_copy_A=thr_copy_A,
        b_s2r_copy_atom=b_s2r_copy_atom,
        thr_copy_B=thr_copy_B,
    )


def _grouped_gemm_gfx950_allocate_shared_storage(
    block_m, block_n, block_k, stages, elem_dtype
):
    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[elem_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    return storage.ab.a.peek().ptr, storage.ab.b.peek().ptr, storage.c.peek().ptr


@dataclass
class _GroupedGemmGfx950Ctx:
    """Per-workgroup state of the non-interleaved grouped kernel.

    Everything here is tile-independent, so it is built once before the
    persistent tile loop and reused by every tile the workgroup claims.
    """

    param: Any
    tid: Any
    tiled_mma: Any
    thr_mma: Any
    smem_a: Any
    smem_b: Any
    a_rsrc: Any
    b_rsrc: Any
    out_buf: Any
    offs_buf: Any
    s2r_copy_atom: Any
    r2g_copy_atom: Any
    thr_copy_A: Any
    thr_copy_B: Any
    a_lds_layout: Any
    b_lds_layout: Any
    b_lds_s2r_layout: Any
    sC: Any
    frag_A: Any
    frag_B: Any
    frag_C: Any
    frag_C_out: Any
    frag_A_retile: Any
    frag_B_retile: Any
    thr_mma_cRow: Any
    thr_mma_cCol: Any
    b_s2r_copy_atom: Any
    thr_copy_cshuffle: Any
    thr_sC: Any
    thr_cRow: Any
    thr_cCol: Any
    frag_C_cshuffle: Any
    pred_C: Any
    wave_offset: Any


def _grouped_gemm_gfx950_setup(out, a, b, offs, tiled_mma, param):
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    block_threads = param.block_threads
    elem_dtype = _elem_dtype(param)
    tid = fx.thread_idx.x

    smem_a, smem_b, smem_c = _grouped_gemm_gfx950_allocate_shared_storage(
        block_m, block_n, block_k, stages, elem_dtype
    )

    rs = _grouped_gemm_gfx950_resources(out, a, b, offs, tiled_mma, elem_dtype, tid)

    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))
    a_lds_layout = fx.make_composed_layout(
        swizzle,
        fx.make_ordered_layout((block_m, block_k), (1, 0)),
    )
    b_lds_layout = fx.make_composed_layout(
        swizzle,
        fx.make_ordered_layout((block_k, block_n), (1, 0)),
    )
    b_lds_s2r_layout = fx.make_composed_layout(
        swizzle,
        fx.make_layout((block_n, block_k), (1, block_n)),
    )
    c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))

    sA = fx.make_view(smem_a, a_lds_layout)
    sC = fx.make_view(smem_c, c_lds_layout)
    frag_A = rs.thr_mma.make_fragment_A(sA)
    sB = fx.make_view(smem_b, b_lds_s2r_layout)
    frag_B = rs.thr_mma.make_fragment_B(sB)
    frag_B_retile = rs.thr_copy_B.retile(frag_B)

    frag_C = rs.thr_mma.make_fragment_C(sC)
    frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
    frag_A_retile = rs.thr_copy_A.retile(frag_A)
    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
    thr_mma_cRow = rs.thr_mma.partition_C(row_coords)
    thr_mma_cCol = rs.thr_mma.partition_C(col_coords)

    cshuffle_vec_size = GFX950_DMA_BYTES // param.out_data_bytes
    cshuffle_x_threads = block_n // cshuffle_vec_size
    cshuffle_thr_layout = fx.make_layout(
        (block_threads // cshuffle_x_threads, cshuffle_x_threads),
        (cshuffle_x_threads, 1),
    )
    cshuffle_val_layout = fx.make_layout((1, cshuffle_vec_size), (1, 1))
    cshuffle_tile, cshuffle_tv_layout = fx.make_layout_tv(
        cshuffle_thr_layout,
        cshuffle_val_layout,
    )
    tiled_copy_cshuffle = fx.make_tiled_copy(
        rs.r2g_copy_atom,
        cshuffle_tv_layout,
        cshuffle_tile,
    )
    thr_copy_cshuffle = tiled_copy_cshuffle.get_slice(tid)
    thr_sC = thr_copy_cshuffle.partition_S(sC)
    thr_cRow = thr_copy_cshuffle.partition_S(row_coords)[(0, None), None, None]
    thr_cCol = thr_copy_cshuffle.partition_S(col_coords)[(0, None), None, None]
    frag_C_cshuffle = fx.make_fragment_like(thr_sC)
    pred_C = fx.make_fragment_like(thr_cRow, dtype=fx.Boolean)
    wave_offset = rocdl.readfirstlane(
        fx.Int64.ir_type,
        fx.Int64(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * param.async_load_bytes),
    )

    return _GroupedGemmGfx950Ctx(
        param=param,
        tid=tid,
        tiled_mma=tiled_mma,
        thr_mma=rs.thr_mma,
        smem_a=smem_a,
        smem_b=smem_b,
        a_rsrc=rs.a_rsrc,
        b_rsrc=rs.b_rsrc,
        out_buf=rs.out_buf,
        offs_buf=rs.offs_buf,
        s2r_copy_atom=rs.s2r_copy_atom,
        r2g_copy_atom=rs.r2g_copy_atom,
        thr_copy_A=rs.thr_copy_A,
        thr_copy_B=rs.thr_copy_B,
        a_lds_layout=a_lds_layout,
        b_lds_layout=b_lds_layout,
        b_lds_s2r_layout=b_lds_s2r_layout,
        sC=sC,
        frag_A=frag_A,
        frag_B=frag_B,
        frag_C=frag_C,
        frag_C_out=frag_C_out,
        frag_A_retile=frag_A_retile,
        frag_B_retile=frag_B_retile,
        thr_mma_cRow=thr_mma_cRow,
        thr_mma_cCol=thr_mma_cCol,
        b_s2r_copy_atom=rs.b_s2r_copy_atom,
        thr_copy_cshuffle=thr_copy_cshuffle,
        thr_sC=thr_sC,
        thr_cRow=thr_cRow,
        thr_cCol=thr_cCol,
        frag_C_cshuffle=frag_C_cshuffle,
        pred_C=pred_C,
        wave_offset=wave_offset,
    )


def _grouped_tile_init(ctx, bid_m, bid_n, m, n, row_base):
    param = ctx.param
    block_m = param.block_m
    block_n = param.block_n
    tile_base = (row_base + bid_m * block_m) * n + bid_n * block_n
    gC = fx.make_view(
        fx.add_offset(fx.get_iter(ctx.out_buf), tile_base),
        fx.make_layout((block_m, block_n), (n, 1)),
    )
    thr_gC = ctx.thr_copy_cshuffle.partition_D(gC)
    ctx.frag_C.fill(0.0)

    for i in range_constexpr(fx.size(ctx.pred_C.shape).unpack()):
        local_row = fx.get_scalar(ctx.thr_cRow[i])
        local_col = fx.get_scalar(ctx.thr_cCol[i])
        row_idx = bid_m * block_m + local_row
        col_idx = bid_n * block_n + local_col
        ctx.pred_C[i] = (
            (local_row < block_m)
            & (local_col < block_n)
            & (row_idx < m)
            & (col_idx < n)
        )
    return thr_gC


def _grouped_tile_store(ctx, thr_gC):
    frag_C_out = ctx.frag_C_out
    frag_C_out.store(ctx.frag_C.load().to(_elem_dtype(ctx.param)))

    fx.gpu.barrier()
    for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
        row = fx.get_scalar(ctx.thr_mma_cRow[i])
        col = fx.get_scalar(ctx.thr_mma_cCol[i])
        ctx.sC[row, col] = frag_C_out[i]

    fx.gpu.barrier()
    fx.copy(ctx.s2r_copy_atom, ctx.thr_sC, ctx.frag_C_cshuffle)
    fx.copy(ctx.r2g_copy_atom, ctx.frag_C_cshuffle, thr_gC, pred=ctx.pred_C)
    fx.gpu.barrier()


@flyc.jit
def _grouped_compute_stage_from_lds(ctx, read_stage, k_tile, k):
    param = ctx.param
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    sA_stage = fx.make_view(
        ctx.smem_a + read_stage * block_m * block_k,
        ctx.a_lds_layout,
    )
    sB_stage = fx.make_view(
        ctx.smem_b + read_stage * block_n * block_k,
        ctx.b_lds_s2r_layout,
    )
    thr_sA_s2r = ctx.thr_copy_A.partition_S(sA_stage)
    thr_sB_s2r = ctx.thr_copy_B.partition_S(sB_stage)

    def compute_k_chunk(block_k_iter):
        fx.copy(
            ctx.b_s2r_copy_atom,
            thr_sB_s2r[None, None, block_k_iter],
            ctx.frag_B_retile[None, None, block_k_iter],
        )
        fx.copy(
            ctx.s2r_copy_atom,
            thr_sA_s2r[None, None, block_k_iter],
            ctx.frag_A_retile[None, None, block_k_iter],
        )
        fx.gemm(
            ctx.tiled_mma,
            ctx.frag_C,
            ctx.frag_A[None, None, block_k_iter],
            ctx.frag_B[None, None, block_k_iter],
            ctx.frag_C,
            traversal_order=fx.GemmTraversalOrder.KNM,
        )

    for block_k_iter in range_constexpr(block_k // param.mma_k):
        if const_expr(param.has_k_tail):
            global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
            if global_k_iter < k:
                compute_k_chunk(block_k_iter)
        else:
            compute_k_chunk(block_k_iter)


def _grouped_load_a_tile_async(ctx, row_base, bid_m, m, k, k_tile, stage):
    param = ctx.param
    block_m = param.block_m
    block_k = param.block_k
    async_load_bytes = param.async_load_bytes
    in_data_bytes = param.in_data_bytes
    async_load_vec_size = async_load_bytes // in_data_bytes
    ldg_x_threads = param.ldg_x_threads
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    lds_ptr = fx.recast_iter(
        fx.Int8, ctx.smem_a + stage * block_m * block_k
    ) + fx.Int32(ctx.wave_offset)
    for i in range_constexpr(ldg_a_iters):
        global_tid = block_threads * i + ctx.tid
        m_local_idx = global_tid // ldg_x_threads
        k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
        in_bounds_m = bid_m * block_m + m_local_idx < m
        global_m_idx = row_base + bid_m * block_m + m_local_idx
        safe_global_m_idx = in_bounds_m.select(global_m_idx, 0)
        col = (
            fx.get_scalar(fx.crd2idx((m_local_idx, k_local_idx), ctx.a_lds_layout))
            % block_k
        )
        global_k_idx = k_tile * block_k + col
        if const_expr(param.has_k_tail):
            safe_global_k_idx = (global_k_idx < k).select(global_k_idx, 0)
        else:
            safe_global_k_idx = global_k_idx
        global_offset = (safe_global_m_idx * k + safe_global_k_idx) * in_data_bytes
        buffer_load_lds_inline(ctx.a_rsrc, lds_ptr, global_offset, async_load_bytes)
        if i < ldg_a_iters - 1:
            lds_ptr = lds_ptr + block_threads * async_load_bytes


def _grouped_load_b_tile_async(ctx, bid_n, n, k, group_idx, k_tile, stage):
    param = ctx.param
    block_n = param.block_n
    block_k = param.block_k
    async_load_bytes = param.async_load_bytes
    in_data_bytes = param.in_data_bytes
    async_load_vec_size = async_load_bytes // in_data_bytes
    block_threads = param.block_threads
    ldg_b_iters = param.ldg_b_iters
    n_vectors = block_n // async_load_vec_size
    lds_ptr = fx.recast_iter(
        fx.Int8, ctx.smem_b + stage * block_n * block_k
    ) + fx.Int32(ctx.wave_offset)

    for i in range_constexpr(ldg_b_iters):
        vector_idx = block_threads * i + ctx.tid
        k_local_idx = vector_idx // n_vectors
        n_local_idx = vector_idx % n_vectors * async_load_vec_size
        global_k_idx = k_tile * block_k + k_local_idx
        swizzled_n_idx = (
            fx.get_scalar(fx.crd2idx((k_local_idx, n_local_idx), ctx.b_lds_layout))
            % block_n
        )
        global_n_idx = bid_n * block_n + swizzled_n_idx
        if const_expr(param.has_k_tail):
            safe_global_k_idx = (global_k_idx < k).select(global_k_idx, 0)
        else:
            safe_global_k_idx = global_k_idx
        global_offset = (
            group_idx * k * n + safe_global_k_idx * n + global_n_idx
        ) * in_data_bytes
        buffer_load_lds_inline(ctx.b_rsrc, lds_ptr, global_offset, async_load_bytes)
        if i < ldg_b_iters - 1:
            lds_ptr = lds_ptr + block_threads * async_load_bytes


@flyc.kernel
def gemm_gfx950_grouped_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    offs: fx.Tensor,
    group_count: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmGfx950Param,
):
    ctx = _grouped_gemm_gfx950_setup(out, a, b, offs, tiled_mma, param)
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    has_k_tail = param.has_k_tail
    ldg_a_iters = param.ldg_a_iters
    num_pid_n = (n - 1) // block_n + 1
    k_tiles = (k - 1) // block_k + 1
    grid = fx.Int32(fx.grid_dim.x)
    work_idx = fx.Int32(fx.block_idx.x)
    tiles_before = fx.Int32(0)
    row_base = fx.Int32(0)
    for g in range(0, group_count, 1):
        row_end = fx.get_scalar(ctx.offs_buf[g])
        m_g = row_end - row_base
        num_pid_m = (m_g + block_m - 1) // block_m
        tiles_after = tiles_before + num_pid_m * num_pid_n

        for linear_tile in range(work_idx, tiles_after, grid):
            local_tile = linear_tile - tiles_before
            bid_m, bid_n = _grouped_swizzle_tile(
                param, num_pid_m, num_pid_n, local_tile
            )
            thr_gC = _grouped_tile_init(ctx, bid_m, bid_n, m_g, n, row_base)
            ldg_wait_count = ldg_a_iters + param.ldg_b_iters
            for stage in range_constexpr(stages - 1):
                _grouped_load_b_tile_async(ctx, bid_n, n, k, g, stage, stage)
                _grouped_load_a_tile_async(ctx, row_base, bid_m, m_g, k, stage, stage)
            rocdl.sched_barrier(0)
            if const_expr(has_k_tail):
                main_loop_end = (k_tiles > stages - 1).select(k_tiles - (stages - 1), 0)
            else:
                main_loop_end = k_tiles - (stages - 1)
            for k_tile in range(0, main_loop_end, 1):
                current_stage = k_tile % stages
                write_stage = (current_stage + stages - 1) % stages
                __barrier((stages - 2) * ldg_wait_count)
                _grouped_load_b_tile_async(
                    ctx, bid_n, n, k, g, k_tile + (stages - 1), write_stage
                )
                _grouped_load_a_tile_async(
                    ctx, row_base, bid_m, m_g, k, k_tile + (stages - 1), write_stage
                )
                _grouped_compute_stage_from_lds(ctx, current_stage, k_tile, k)
            current_stage = main_loop_end % stages
            for s in range_constexpr(0, stages - 1):
                __barrier((stages - 2 - s) * ldg_wait_count)
                _grouped_compute_stage_from_lds(
                    ctx, current_stage, main_loop_end + s, k
                )
                current_stage = (current_stage + 1) % stages
            _grouped_tile_store(ctx, thr_gC)

        remaining = (work_idx < tiles_after).select(tiles_after - work_idx, fx.Int32(0))
        work_idx = work_idx + (remaining + grid - 1) // grid * grid
        tiles_before = tiles_after
        row_base = row_end


@flyc.kernel
def gemm_hti_gfx950_grouped_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    offs: fx.Tensor,
    group_count: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmGfx950Param,
):
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    half_block_m = block_m // 2
    half_block_n = block_n // 2
    has_k_tail = param.has_k_tail
    async_load_bytes = param.async_load_bytes
    in_data_bytes = param.in_data_bytes
    async_load_vec_size = async_load_bytes // in_data_bytes
    ldg_x_threads = param.ldg_x_threads
    block_threads = param.block_threads
    half_ldg_a_iters = param.ldg_a_iters // 2
    half_ldg_b_iters = param.ldg_b_iters // 2
    elem_dtype = _elem_dtype(param)

    tid = fx.thread_idx.x
    num_pid_n = (n - 1) // block_n + 1
    k_tiles = (k - 1) // block_k + 1

    smem_a, smem_b, smem_c = _grouped_gemm_gfx950_allocate_shared_storage(
        block_m, block_n, block_k, 2, elem_dtype
    )

    rs = _grouped_gemm_gfx950_resources(out, a, b, offs, tiled_mma, elem_dtype, tid)

    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))
    a_lds_layout = fx.make_composed_layout(
        swizzle,
        fx.make_ordered_layout((half_block_m, block_k), (1, 0)),
    )
    b_half_lds_layout = fx.make_composed_layout(
        swizzle,
        fx.make_ordered_layout((block_k, half_block_n), (1, 0)),
    )
    b_half_lds_s2r_layout = fx.make_composed_layout(
        swizzle,
        fx.make_layout((half_block_n, block_k), (1, half_block_n)),
    )
    c_lds_layout = fx.make_layout((half_block_m, half_block_n), (half_block_n, 1))
    wave_offset = rocdl.readfirstlane(
        fx.Int64.ir_type,
        fx.Int64(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * async_load_bytes),
    )

    def make_wave_lds_ptr(ptr):
        return fx.recast_iter(fx.Int8, ptr) + fx.Int32(wave_offset)

    def swizzled_col_idx(row, col):
        return fx.get_scalar(fx.crd2idx((row, col), a_lds_layout)) % block_k

    def half_a_base(stage, m_part):
        return smem_a + (stage * block_m + m_part * half_block_m) * block_k

    def half_b_base(stage, n_part):
        return smem_b + (stage * block_n + n_part * half_block_n) * block_k

    def load_a_half(m_part, k_tile, stage, bid_m, m, row_base):
        lds_ptr = make_wave_lds_ptr(half_a_base(stage, m_part))
        for i in range_constexpr(half_ldg_a_iters):
            global_tid = block_threads * i + tid
            m_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            m_tile_idx = bid_m * block_m + m_part * half_block_m + m_local_idx
            in_bounds_m = m_tile_idx < m
            safe_global_m_idx = in_bounds_m.select(row_base + m_tile_idx, 0)
            global_k_idx = k_tile * block_k + swizzled_col_idx(m_local_idx, k_local_idx)
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < k).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset = (safe_global_m_idx * k + safe_global_k_idx) * in_data_bytes
            buffer_load_lds_inline(rs.a_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < half_ldg_a_iters - 1:
                lds_ptr = lds_ptr + block_threads * async_load_bytes

    def load_b_half(n_part, k_tile, stage, bid_n, group):
        lds_ptr = make_wave_lds_ptr(half_b_base(stage, n_part))
        n_vectors = half_block_n // async_load_vec_size
        for i in range_constexpr(half_ldg_b_iters):
            vector_idx = block_threads * i + tid
            k_local_idx = vector_idx // n_vectors
            n_local_idx = vector_idx % n_vectors * async_load_vec_size
            global_k_idx = k_tile * block_k + k_local_idx
            swizzled_n_idx = (
                fx.get_scalar(fx.crd2idx((k_local_idx, n_local_idx), b_half_lds_layout))
                % half_block_n
            )
            global_n_idx = bid_n * block_n + n_part * half_block_n + swizzled_n_idx
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < k).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset = (
                group * k * n + safe_global_k_idx * n + global_n_idx
            ) * in_data_bytes
            buffer_load_lds_inline(rs.b_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < half_ldg_b_iters - 1:
                lds_ptr = lds_ptr + block_threads * async_load_bytes

    def load_a_fragment(m_part, read_stage, k_tile):
        sA = fx.make_view(half_a_base(read_stage, m_part), a_lds_layout)
        frag_A = rs.thr_mma.make_fragment_A(sA)
        frag_A_retile = rs.thr_copy_A.retile(frag_A)
        thr_sA_s2r = rs.thr_copy_A.partition_S(sA)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    fx.copy(
                        rs.s2r_copy_atom,
                        thr_sA_s2r[None, None, block_k_iter],
                        frag_A_retile[None, None, block_k_iter],
                    )
            else:
                fx.copy(
                    rs.s2r_copy_atom,
                    thr_sA_s2r[None, None, block_k_iter],
                    frag_A_retile[None, None, block_k_iter],
                )
        return frag_A

    def load_b_fragment(n_part, read_stage, k_tile):
        sB = fx.make_view(
            half_b_base(read_stage, n_part),
            b_half_lds_s2r_layout,
        )
        frag_B = rs.thr_mma.make_fragment_B(sB)
        frag_B_retile = rs.thr_copy_B.retile(frag_B)
        thr_sB_s2r = rs.thr_copy_B.partition_S(sB)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    fx.copy(
                        rs.b_s2r_copy_atom,
                        thr_sB_s2r[None, None, block_k_iter],
                        frag_B_retile[None, None, block_k_iter],
                    )
            else:
                fx.copy(
                    rs.b_s2r_copy_atom,
                    thr_sB_s2r[None, None, block_k_iter],
                    frag_B_retile[None, None, block_k_iter],
                )
        return frag_B

    def consume(k_tile, frag_C, frag_A, frag_B):
        rocdl.sched_barrier(0)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    fx.gemm(
                        tiled_mma,
                        frag_C,
                        frag_A[None, None, block_k_iter],
                        frag_B[None, None, block_k_iter],
                        frag_C,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )
            else:
                fx.gemm(
                    tiled_mma,
                    frag_C,
                    frag_A[None, None, block_k_iter],
                    frag_B[None, None, block_k_iter],
                    frag_C,
                    traversal_order=fx.GemmTraversalOrder.KNM,
                )
        rocdl.sched_barrier(0)

    def half_gC(m_part, n_part, bid_m, bid_n, row_base):
        row = row_base + bid_m * block_m + m_part * half_block_m
        col = bid_n * block_n + n_part * half_block_n
        return fx.make_view(
            fx.add_offset(fx.get_iter(rs.out_buf), row * n + col),
            fx.make_layout((half_block_m, half_block_n), (n, 1)),
        )

    cshuffle_vec_size = GFX950_DMA_BYTES // param.out_data_bytes
    cshuffle_x_threads = half_block_n // cshuffle_vec_size
    cshuffle_thr_layout = fx.make_layout(
        (block_threads // cshuffle_x_threads, cshuffle_x_threads),
        (cshuffle_x_threads, 1),
    )
    cshuffle_val_layout = fx.make_layout((1, cshuffle_vec_size), (1, 1))
    cshuffle_tile, cshuffle_tv_layout = fx.make_layout_tv(
        cshuffle_thr_layout,
        cshuffle_val_layout,
    )
    tiled_copy_cshuffle = fx.make_tiled_copy(
        rs.r2g_copy_atom,
        cshuffle_tv_layout,
        cshuffle_tile,
    )
    thr_copy_cshuffle = tiled_copy_cshuffle.get_slice(tid)
    row_coords = fx.make_view(0, fx.make_layout((half_block_m, half_block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((half_block_m, half_block_n), (0, 1)))
    thr_mma_cRow = rs.thr_mma.partition_C(row_coords)
    thr_mma_cCol = rs.thr_mma.partition_C(col_coords)
    thr_cRow = thr_copy_cshuffle.partition_S(row_coords)[(0, None), None, None]
    thr_cCol = thr_copy_cshuffle.partition_S(col_coords)[(0, None), None, None]

    def store_half_tile(m_part, n_part, frag_C, bid_m, bid_n, m, row_base):
        gC = half_gC(m_part, n_part, bid_m, bid_n, row_base)
        sC = fx.make_view(smem_c, c_lds_layout)
        thr_sC = thr_copy_cshuffle.partition_S(sC)
        thr_gC = thr_copy_cshuffle.partition_D(gC)
        frag_C_cshuffle = fx.make_fragment_like(thr_sC)
        pred_C = fx.make_fragment_like(thr_cRow, dtype=fx.Boolean)

        for i in range_constexpr(fx.size(pred_C.shape).unpack()):
            local_row = fx.get_scalar(thr_cRow[i])
            local_col = fx.get_scalar(thr_cCol[i])
            row_idx = bid_m * block_m + m_part * half_block_m + local_row
            col_idx = bid_n * block_n + n_part * half_block_n + local_col
            pred_C[i] = (
                (local_row < half_block_m)
                & (local_col < half_block_n)
                & (row_idx < m)
                & (col_idx < n)
            )

        frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            frag_C_out[i] = frag_C[i].to(elem_dtype)

        fx.gpu.barrier()
        for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            sC[row, col] = frag_C_out[i]

        fx.gpu.barrier()
        fx.copy(rs.s2r_copy_atom, thr_sC, frag_C_cshuffle)
        fx.copy(rs.r2g_copy_atom, frag_C_cshuffle, thr_gC, pred=pred_C)
        fx.gpu.barrier()

    frag_shape = fx.make_view(
        fx.get_iter(rs.out_buf),
        fx.make_layout((half_block_m, half_block_n), (n, 1)),
    )
    c00 = rs.thr_mma.make_fragment_C(frag_shape)
    c01 = rs.thr_mma.make_fragment_C(frag_shape)
    c10 = rs.thr_mma.make_fragment_C(frag_shape)
    c11 = rs.thr_mma.make_fragment_C(frag_shape)

    def compute_double_tile(
        k_tile,
        prefetch_next,
        bid_m,
        bid_n,
        m,
        row_base,
        group,
    ):
        next_k_tile = k_tile + 2

        b0 = load_b_fragment(0, 0, k_tile)
        a0 = load_a_fragment(0, 0, k_tile)
        load_a_half(1, k_tile + 1, 1, bid_m, m, row_base)
        rocdl.s_barrier()
        consume(k_tile, c00, a0, b0)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            load_b_half(0, next_k_tile, 0, bid_n, group)
            rocdl.s_barrier()
        consume(k_tile, c01, a0, b1)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            load_a_half(0, next_k_tile, 0, bid_m, m, row_base)
            rocdl.s_barrier()
        consume(k_tile, c10, a1, b0)
        rocdl.s_barrier()

        b0 = load_b_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            load_b_half(1, next_k_tile, 0, bid_n, group)
            __barrier(2 * half_ldg_b_iters + half_ldg_a_iters)
        consume(k_tile, c11, a1, b1)
        if const_expr(not prefetch_next):
            __waitcnt(0)
        rocdl.s_barrier()

        a0 = load_a_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            load_a_half(1, next_k_tile, 0, bid_m, m, row_base)
            rocdl.s_barrier()
        consume(k_tile + 1, c00, a0, b0)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            load_b_half(0, next_k_tile + 1, 1, bid_n, group)
            rocdl.s_barrier()
        consume(k_tile + 1, c01, a0, b1)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            load_a_half(0, next_k_tile + 1, 1, bid_m, m, row_base)
            rocdl.s_barrier()
        consume(k_tile + 1, c10, a1, b0)
        rocdl.s_barrier()

        if const_expr(prefetch_next):
            load_b_half(1, next_k_tile + 1, 1, bid_n, group)
            __barrier(half_ldg_b_iters + half_ldg_a_iters)
        consume(k_tile + 1, c11, a1, b1)
        rocdl.s_barrier()

    grid = fx.Int32(fx.grid_dim.x)
    work_idx = fx.Int32(fx.block_idx.x)
    tiles_before = fx.Int32(0)
    row_base = fx.Int32(0)
    for g in range(0, group_count, 1):
        row_end = fx.get_scalar(rs.offs_buf[g])
        m_g = row_end - row_base
        num_pid_m = (m_g + block_m - 1) // block_m
        tiles_after = tiles_before + num_pid_m * num_pid_n

        for linear_tile in range(work_idx, tiles_after, grid):
            local_tile = linear_tile - tiles_before
            bid_m, bid_n = _grouped_swizzle_tile(
                param, num_pid_m, num_pid_n, local_tile
            )
            c00.fill(0.0)
            c01.fill(0.0)
            c10.fill(0.0)
            c11.fill(0.0)

            load_b_half(0, 0, 0, bid_n, g)
            load_a_half(0, 0, 0, bid_m, m_g, row_base)
            load_b_half(1, 0, 0, bid_n, g)
            load_a_half(1, 0, 0, bid_m, m_g, row_base)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            load_b_half(0, 1, 1, bid_n, g)
            load_a_half(0, 1, 1, bid_m, m_g, row_base)
            load_b_half(1, 1, 1, bid_n, g)
            __barrier(half_ldg_b_iters + half_ldg_a_iters)

            final_double_tile = ((k_tiles % 2) == 0).select(k_tiles - 2, k_tiles - 1)
            main_loop_end = (k_tiles > 2).select(final_double_tile, 0)
            for k_tile in range(0, main_loop_end, 2):
                compute_double_tile(k_tile, True, bid_m, bid_n, m_g, row_base, g)
            compute_double_tile(main_loop_end, False, bid_m, bid_n, m_g, row_base, g)

            store_half_tile(0, 0, c00, bid_m, bid_n, m_g, row_base)
            store_half_tile(0, 1, c01, bid_m, bid_n, m_g, row_base)
            store_half_tile(1, 0, c10, bid_m, bid_n, m_g, row_base)
            store_half_tile(1, 1, c11, bid_m, bid_n, m_g, row_base)

        remaining = (work_idx < tiles_after).select(tiles_after - work_idx, fx.Int32(0))
        work_idx = work_idx + (remaining + grid - 1) // grid * grid
        tiles_before = tiles_after
        row_base = row_end


@flyc.jit
def launch_gemm_gfx950_grouped(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    offs: fx.Tensor,
    group_count: int,
    n: fx.Int32,
    k: fx.Int32,
    grid_size: int,
    param: GemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    tiled_mma = _make_gemm_gfx950_tiled_mma(param)
    kernel_impl = (
        gemm_hti_gfx950_grouped_kernel
        if param.use_half_tile_interleaved
        else gemm_gfx950_grouped_kernel
    )
    kernel_impl._known_block_size = [param.block_threads, 1, 1]
    kernel_impl._func.__name__ = make_grouped_gemm_gfx950_kernel_name(param)
    kernel_impl(out, a, b, offs, group_count, n, k, tiled_mma, param).launch(
        grid=(grid_size, 1, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )
