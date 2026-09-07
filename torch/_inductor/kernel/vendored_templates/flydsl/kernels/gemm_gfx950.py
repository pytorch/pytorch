# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch


GFX950_DMA_BYTES = 16
GFX950_WAVE_SIZE = 64
GEMM_DTYPE_BF16 = 2
GEMM_DTYPE_FP16 = 3


@fx.struct
class GemmGfx950Param:
    dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    use_half_tile_interleaved: fx.Constexpr[bool]
    has_bias: fx.Constexpr[bool]
    has_k_tail: fx.Constexpr[bool]
    async_load_bytes: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    out_data_bytes: fx.Constexpr[int]
    ldg_x_threads: fx.Constexpr[int]
    block_threads: fx.Constexpr[int]
    ldg_a_iters: fx.Constexpr[int]
    ldg_b_iters: fx.Constexpr[int]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]


def make_gemm_gfx950_param(
    dtype_id: int = GEMM_DTYPE_BF16,
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 64,
    stages: int = 2,
    m_waves: int = 2,
    n_waves: int = 4,
    group_m: int = 0,
    use_half_tile_interleaved: bool = False,
    has_bias: bool = False,
    has_k_tail: bool = False,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
) -> GemmGfx950Param:
    if dtype_id not in (GEMM_DTYPE_BF16, GEMM_DTYPE_FP16):
        raise ValueError(f"unsupported dtype_id={dtype_id}")
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or stages <= 0:
        raise ValueError("block_m, block_n, block_k, and stages must be positive")
    if (mma_m, mma_n, mma_k) != (16, 16, 32):
        raise ValueError("the gfx950 layout kernel currently requires mma=16x16x32")
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves and n_waves must be positive")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")

    in_dbytes = out_dbytes = 2
    cshuffle_vec_size = GFX950_DMA_BYTES // out_dbytes
    if use_half_tile_interleaved:
        half_block_m = block_m // 2
        half_block_n = block_n // 2
        if stages != 2:
            raise ValueError("half-tile interleaved kernel requires stages=2")
        if m_waves != 2 or n_waves < 2:
            raise ValueError(
                "half-tile interleaved kernel requires m_waves=2 and n_waves>=2"
            )
        if half_block_m * 2 != block_m or half_block_n * 2 != block_n:
            raise ValueError(
                "half-tile interleaved kernel requires even block_m and block_n"
            )
        mma_m_half_repeat = half_block_m // m_waves // mma_m
        mma_n_half_repeat = half_block_n // n_waves // mma_n
        if mma_m_half_repeat * m_waves * mma_m != half_block_m:
            raise ValueError("half block_m must be divisible by m_waves * mma_m")
        if mma_n_half_repeat * n_waves * mma_n != half_block_n:
            raise ValueError("half block_n must be divisible by n_waves * mma_n")
        if mma_n_half_repeat != 2:
            raise ValueError(
                "half-tile interleaved kernel requires "
                "half_block_n / n_waves / mma_n == 2"
            )
        if half_block_n % cshuffle_vec_size != 0:
            raise ValueError(
                "half block_n must be divisible by the c-shuffle vector size"
            )
    elif block_n % cshuffle_vec_size != 0:
        raise ValueError("block_n must be divisible by the c-shuffle vector size")

    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
    smem_bytes = max(smem_bytes, block_m * block_n * out_dbytes)
    smem_capacity = {
        "gfx942": 65536,
        "gfx950": 163840,
    }.get(get_rocm_arch(), 65536)
    if smem_bytes > smem_capacity:
        raise ValueError(
            "staged LDS buffers exceed the device shared-memory capacity: "
            f"stages={stages}, block_m={block_m}, block_n={block_n}, "
            f"block_k={block_k}, smem_bytes={smem_bytes}, capacity={smem_capacity}"
        )

    async_load_vec_size = GFX950_DMA_BYTES // in_dbytes
    ldg_x_threads = block_k // async_load_vec_size
    if ldg_x_threads * async_load_vec_size != block_k:
        raise ValueError(
            "block_k must be divisible by the async load vector size: "
            f"block_k={block_k}, async_load_vec_size={async_load_vec_size}"
        )

    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    load_elems_per_iter = block_threads * async_load_vec_size
    if (block_m * block_k) % load_elems_per_iter != 0:
        raise ValueError(
            "A tile load schedule must exactly cover the LDS tile: "
            f"block_m={block_m}, block_k={block_k}, "
            f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}"
        )
    if (block_n * block_k) % load_elems_per_iter != 0:
        raise ValueError(
            "B tile load schedule must exactly cover the LDS tile: "
            f"block_n={block_n}, block_k={block_k}, "
            f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}"
        )
    ldg_a_iters = (block_m * block_k) // load_elems_per_iter
    ldg_b_iters = (block_n * block_k) // load_elems_per_iter
    if use_half_tile_interleaved:
        half_ldg_a_iters = ((block_m // 2) * block_k) // load_elems_per_iter
        half_ldg_b_iters = ((block_n // 2) * block_k) // load_elems_per_iter
        if half_ldg_a_iters * load_elems_per_iter != (block_m // 2) * block_k:
            raise ValueError(
                "half-tile A load schedule must exactly cover the LDS tile"
            )
        if half_ldg_b_iters * load_elems_per_iter != (block_n // 2) * block_k:
            raise ValueError(
                "half-tile B load schedule must exactly cover the LDS tile"
            )
    if (stages - 2) * (ldg_a_iters + ldg_b_iters) >= 63:
        raise ValueError("staged pipeline wait count exceeds supported range")

    mma_m_repeat = block_m // m_waves // mma_m
    mma_n_repeat = block_n // n_waves // mma_n
    if mma_m_repeat * m_waves * mma_m != block_m:
        raise ValueError(
            "block_m must be divisible by m_waves * mma_m: "
            f"block_m={block_m}, m_waves={m_waves}, mma_m={mma_m}"
        )
    if mma_n_repeat * n_waves * mma_n != block_n:
        raise ValueError(
            "block_n must be divisible by n_waves * mma_n: "
            f"block_n={block_n}, n_waves={n_waves}, mma_n={mma_n}"
        )

    return GemmGfx950Param(
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        use_half_tile_interleaved=use_half_tile_interleaved,
        has_bias=has_bias,
        has_k_tail=has_k_tail,
        async_load_bytes=GFX950_DMA_BYTES,
        in_data_bytes=in_dbytes,
        out_data_bytes=out_dbytes,
        ldg_x_threads=ldg_x_threads,
        block_threads=block_threads,
        ldg_a_iters=ldg_a_iters,
        ldg_b_iters=ldg_b_iters,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
    )


def make_gemm_gfx950_kernel_name(param: GemmGfx950Param) -> str:
    dtype_str = "fp16" if param.dtype_id == GEMM_DTYPE_FP16 else "bf16"
    name = f"gemm_{dtype_str}_t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}"
    name += f"_w{param.m_waves}x{param.n_waves}"
    name += f"_gm{param.group_m}"
    name += f"_bias{int(param.has_bias)}"
    name += f"_ktail{int(param.has_k_tail)}"
    name += "_nt"
    name += "_hti" if param.use_half_tile_interleaved else "_ft"
    return name


class BlockSwizzle:
    def __init__(self, NUM_XCDS, NUM_PIDS_THRESHOLD, GROUP_M):
        self.NUM_XCDS = NUM_XCDS
        self.NUM_PIDS_THRESHOLD = NUM_PIDS_THRESHOLD
        self.GROUP_M = GROUP_M

    @flyc.jit
    def swizzle(self, num_pid_m, num_pid_n, pid):
        simple_m = pid // num_pid_n
        simple_n = pid % num_pid_n
        if const_expr(self.GROUP_M <= 0):
            return simple_m, simple_n
        num_xcds = self.NUM_XCDS
        swizzle_threshold = self.NUM_PIDS_THRESHOLD
        num_wg = num_pid_m * num_pid_n
        linear_id = pid
        intra_xcd = linear_id // num_xcds
        xcd = linear_id % num_xcds
        wgid = xcd * (num_wg // num_xcds) + intra_xcd
        group_m = self.GROUP_M
        wgid_per_group = group_m * num_pid_n
        group_id = wgid // wgid_per_group
        intra_group = wgid % wgid_per_group
        first_pid_m = group_id * group_m
        remaining_m = num_pid_m - first_pid_m
        group_size_m = (remaining_m < group_m).select(remaining_m, group_m)
        swizzled_n = intra_group // group_size_m
        swizzled_m = first_pid_m + (intra_group % group_size_m)
        use_simple = (num_wg < swizzle_threshold) | ((num_wg % num_xcds) != 0)
        if const_expr(isinstance(use_simple, bool)):
            if const_expr(use_simple):
                return simple_m, simple_n
            return swizzled_m, swizzled_n
        return (
            use_simple.select(simple_m, swizzled_m),
            use_simple.select(simple_n, swizzled_n),
        )


# TODO: Move common ROCm synchronization and buffer-load helpers to FlyDSL.
def __barrier(vmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
        "",
        has_side_effects=True,
    )


def __waitcnt(vmcnt=0):
    llvm.InlineAsmOp(None, [], f"s_waitcnt vmcnt({vmcnt})", "", has_side_effects=True)


def buffer_load_lds_inline(rsrc, lds_ptr, global_offset, dma_bytes):
    buffer_load_asm_dict = {
        16: "buffer_load_dwordx4",
        8: "buffer_load_dwordx2",
        4: "buffer_load_dword",
    }
    llvm.InlineAsmOp(
        None,
        [
            llvm.IntToPtrOp(
                ir.Type.parse("!llvm.ptr<3>"),
                fx.as_ir_value(fx.ptrtoint(lds_ptr)),
            ).result,
            fx.as_ir_value(global_offset),
            fx.as_ir_value(rsrc),
        ],
        f"s_mov_b32 m0, $0\n\t{buffer_load_asm_dict[dma_bytes]} $1, $2, 0 offen sc0 lds",
        "s,v,s",
        has_side_effects=True,
    )


def _elem_dtype(param: GemmGfx950Param):
    return fx.Float16 if const_expr(param.dtype_id == GEMM_DTYPE_FP16) else fx.BFloat16


@flyc.kernel
def gemm_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    a_row_stride: fx.Int32,
    b_row_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmGfx950Param,
):
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    has_k_tail = param.has_k_tail
    async_load_bytes = param.async_load_bytes
    in_data_bytes = param.in_data_bytes
    async_load_vec_size = async_load_bytes // in_data_bytes
    ldg_x_threads = param.ldg_x_threads
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters
    ldg_wait_count = ldg_a_iters + ldg_b_iters
    elem_dtype = _elem_dtype(param)

    tid = fx.thread_idx.x
    num_pid_m = (m - 1) // block_m + 1
    num_pid_n = (n - 1) // block_n + 1
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    k_tiles = (k - 1) // block_k + 1

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[elem_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

    a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(a_buf))
    b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(b_buf))

    s2r_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    g2r_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    r2g_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]
    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(g2r_copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(g2r_copy_atom, tiled_mma).get_slice(tid)

    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))

    def make_lds_layout(rows):
        return fx.make_composed_layout(
            swizzle,
            fx.make_ordered_layout((rows, block_k), (1, 0)),
        )

    a_lds_layout = make_lds_layout(block_m)
    b_lds_layout = make_lds_layout(block_n)
    c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))

    sA = fx.make_view(smem_a, a_lds_layout)
    sB = fx.make_view(smem_b, b_lds_layout)
    sC = fx.make_view(smem_c, c_lds_layout)

    frag_A = thr_mma.make_fragment_A(sA)
    frag_B = thr_mma.make_fragment_B(sB)
    frag_C = thr_mma.make_fragment_C(gC)
    frag_A_retile = thr_copy_A.retile(frag_A)
    frag_B_retile = thr_copy_B.retile(frag_B)

    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)

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
        r2g_copy_atom,
        cshuffle_tv_layout,
        cshuffle_tile,
    )
    thr_copy_cshuffle = tiled_copy_cshuffle.get_slice(tid)
    thr_sC = thr_copy_cshuffle.partition_S(sC)
    thr_gC = thr_copy_cshuffle.partition_D(gC)
    thr_cRow = thr_copy_cshuffle.partition_S(row_coords)[(0, None), None, None]
    thr_cCol = thr_copy_cshuffle.partition_S(col_coords)[(0, None), None, None]
    frag_C_cshuffle = fx.make_fragment_like(thr_sC)
    pred_C = fx.make_fragment_like(thr_cRow, dtype=fx.Boolean)

    frag_C.fill(0.0)
    if const_expr(param.has_bias):
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            col_idx = fx.get_scalar(thr_mma_cCol[i])
            global_n_idx = bid_n * block_n + col_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            frag_C[i] = bias_buf[safe_global_n_idx].to(fx.Float32)

    for i in range_constexpr(fx.size(pred_C.shape).unpack()):
        local_row = fx.get_scalar(thr_cRow[i])
        local_col = fx.get_scalar(thr_cCol[i])
        row_idx = bid_m * block_m + local_row
        col_idx = bid_n * block_n + local_col
        pred_C[i] = (
            (local_row < block_m)
            & (local_col < block_n)
            & (row_idx < m)
            & (col_idx < n)
        )

    wave_offset = rocdl.readfirstlane(
        fx.Int64.ir_type,
        fx.Int64(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * async_load_bytes),
    )

    def make_wave_lds_ptr(ptr):
        return fx.recast_iter(fx.Int8, ptr) + fx.Int32(wave_offset)

    def swizzled_col_idx(row, col, layout):
        elem_offset = fx.get_scalar(fx.crd2idx((row, col), layout))
        return elem_offset % block_k

    def async_load_a_to_lds(k_tile, stage):
        lds_ptr = make_wave_lds_ptr(smem_a + stage * block_m * block_k)
        for i in range_constexpr(ldg_a_iters):
            global_tid = block_threads * i + tid
            m_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_m_idx = bid_m * block_m + m_local_idx
            safe_global_m_idx = (global_m_idx < m).select(global_m_idx, 0)
            global_k_idx = k_tile * block_k + swizzled_col_idx(
                m_local_idx,
                k_local_idx,
                a_lds_layout,
            )
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < k).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset = (
                safe_global_m_idx * a_row_stride + safe_global_k_idx
            ) * in_data_bytes
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < ldg_a_iters - 1:
                lds_ptr = lds_ptr + block_threads * async_load_bytes

    def async_load_b_to_lds(k_tile, stage):
        lds_ptr = make_wave_lds_ptr(smem_b + stage * block_n * block_k)
        for i in range_constexpr(ldg_b_iters):
            global_tid = block_threads * i + tid
            n_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_n_idx = bid_n * block_n + n_local_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            global_k_idx = k_tile * block_k + swizzled_col_idx(
                n_local_idx,
                k_local_idx,
                b_lds_layout,
            )
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < k).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset = (
                safe_global_n_idx * b_row_stride + safe_global_k_idx
            ) * in_data_bytes
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < ldg_b_iters - 1:
                lds_ptr = lds_ptr + block_threads * async_load_bytes

    def compute_stage(read_stage, k_tile):
        sA_stage = fx.make_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
        sB_stage = fx.make_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
        thr_sA_s2r = thr_copy_A.partition_S(sA_stage)
        thr_sB_s2r = thr_copy_B.partition_S(sB_stage)

        def compute_k_chunk(block_k_iter):
            fx.copy(
                s2r_copy_atom,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
            fx.copy(
                s2r_copy_atom,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, block_k_iter],
                frag_B[None, None, block_k_iter],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    compute_k_chunk(block_k_iter)
            else:
                compute_k_chunk(block_k_iter)

    for stage in range_constexpr(stages - 1):
        async_load_b_to_lds(stage, stage)
        async_load_a_to_lds(stage, stage)
    rocdl.sched_barrier(0)

    if const_expr(has_k_tail):
        main_loop_end = (k_tiles > stages - 1).select(k_tiles - (stages - 1), 0)
    else:
        main_loop_end = k_tiles - (stages - 1)
    for k_tile in range(0, main_loop_end, 1):
        current_stage = k_tile % stages
        write_stage = (current_stage + stages - 1) % stages
        __barrier((stages - 2) * ldg_wait_count)
        async_load_b_to_lds(k_tile + (stages - 1), write_stage)
        async_load_a_to_lds(k_tile + (stages - 1), write_stage)
        compute_stage(current_stage, k_tile)

    current_stage = main_loop_end % stages
    for s in range_constexpr(0, stages - 1):
        __barrier((stages - 2 - s) * ldg_wait_count)
        compute_stage(current_stage, main_loop_end + s)
        current_stage = (current_stage + 1) % stages

    frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
    frag_C_out.store(frag_C.load().to(elem_dtype))

    fx.gpu.barrier()
    for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
        row = fx.get_scalar(thr_mma_cRow[i])
        col = fx.get_scalar(thr_mma_cCol[i])
        sC[row, col] = frag_C_out[i]

    fx.gpu.barrier()
    fx.copy(s2r_copy_atom, thr_sC, frag_C_cshuffle)
    fx.copy(r2g_copy_atom, frag_C_cshuffle, thr_gC, pred=pred_C)


@flyc.kernel
def gemm_hti_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    a_row_stride: fx.Int32,
    b_row_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmGfx950Param,
):
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    half_block_m = block_m // 2
    half_block_n = block_n // 2
    stages = param.stages
    has_k_tail = param.has_k_tail
    async_load_bytes = param.async_load_bytes
    in_data_bytes = param.in_data_bytes
    async_load_vec_size = async_load_bytes // in_data_bytes
    ldg_x_threads = param.ldg_x_threads
    block_threads = param.block_threads
    n_waves = param.n_waves
    half_ldg_a_iters = param.ldg_a_iters // 2
    half_ldg_b_iters = param.ldg_b_iters // 2
    elem_dtype = _elem_dtype(param)

    tid = fx.thread_idx.x
    wid = tid // GFX950_WAVE_SIZE
    num_pid_m = (m - 1) // block_m + 1
    num_pid_n = (n - 1) // block_n + 1
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    k_tiles = (k - 1) // block_k + 1

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[elem_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out = fx.rocdl.make_buffer_tensor(out, max_size=False)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

    a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(a_buf))
    b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(b_buf))

    s2r_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    g2r_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    r2g_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)

    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(g2r_copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(g2r_copy_atom, tiled_mma).get_slice(tid)

    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))

    def make_lds_layout(rows):
        return fx.make_composed_layout(
            swizzle,
            fx.make_ordered_layout((rows, block_k), (1, 0)),
        )

    a_lds_layout = make_lds_layout(half_block_m)
    b_lds_layout = make_lds_layout(half_block_n)
    c_lds_layout = fx.make_layout((half_block_m, half_block_n), (half_block_n, 1))

    wave_offset = rocdl.readfirstlane(
        fx.Int64.ir_type,
        fx.Int64(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * async_load_bytes),
    )

    def make_wave_lds_ptr(ptr):
        return fx.recast_iter(fx.Int8, ptr) + fx.Int32(wave_offset)

    def swizzled_col_idx(row, col, layout):
        elem_offset = fx.get_scalar(fx.crd2idx((row, col), layout))
        return elem_offset % block_k

    def half_a_base(stage, m_part):
        return smem_a + (stage * block_m + m_part * half_block_m) * block_k

    def half_b_base(stage, n_part):
        return smem_b + (stage * block_n + n_part * half_block_n) * block_k

    def async_load_a_to_lds(m_part, k_tile, stage):
        lds_ptr = make_wave_lds_ptr(half_a_base(stage, m_part))
        for i in range_constexpr(half_ldg_a_iters):
            global_tid = block_threads * i + tid
            m_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_m_idx = bid_m * block_m + m_part * half_block_m + m_local_idx
            safe_global_m_idx = (global_m_idx < m).select(global_m_idx, 0)
            global_k_idx = k_tile * block_k + swizzled_col_idx(
                m_local_idx,
                k_local_idx,
                a_lds_layout,
            )
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < k).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset = (
                safe_global_m_idx * a_row_stride + safe_global_k_idx
            ) * in_data_bytes
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < half_ldg_a_iters - 1:
                lds_ptr = lds_ptr + block_threads * async_load_bytes

    def async_load_b_to_lds(n_part, k_tile, stage):
        lds_ptr = make_wave_lds_ptr(half_b_base(stage, n_part))
        for i in range_constexpr(half_ldg_b_iters):
            global_tid = block_threads * i + tid
            n_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_n_idx = bid_n * block_n + n_part * half_block_n + n_local_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            global_k_idx = k_tile * block_k + swizzled_col_idx(
                n_local_idx,
                k_local_idx,
                b_lds_layout,
            )
            if const_expr(has_k_tail):
                safe_global_k_idx = (global_k_idx < k).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset = (
                safe_global_n_idx * b_row_stride + safe_global_k_idx
            ) * in_data_bytes
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < half_ldg_b_iters - 1:
                lds_ptr = lds_ptr + block_threads * async_load_bytes

    def make_gC(m_part, n_part):
        return fx.flat_divide(out, (half_block_m, half_block_n))[
            None, None, bid_m * 2 + m_part, bid_n * 2 + n_part
        ]

    def make_c_fragment(m_part, n_part):
        gC = make_gC(m_part, n_part)
        frag_C = thr_mma.make_fragment_C(gC)
        frag_C.fill(0.0)
        return frag_C

    def load_a_fragment(m_part, read_stage, k_tile):
        sA = fx.make_view(half_a_base(read_stage, m_part), a_lds_layout)
        frag_A = thr_mma.make_fragment_A(sA)
        frag_A_retile = thr_copy_A.retile(frag_A)
        thr_sA_s2r = thr_copy_A.partition_S(sA)

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    fx.copy(
                        s2r_copy_atom,
                        thr_sA_s2r[None, None, block_k_iter],
                        frag_A_retile[None, None, block_k_iter],
                    )
            else:
                fx.copy(
                    s2r_copy_atom,
                    thr_sA_s2r[None, None, block_k_iter],
                    frag_A_retile[None, None, block_k_iter],
                )
        return frag_A

    def load_b_fragment(n_part, read_stage, k_tile):
        sB = fx.make_view(half_b_base(read_stage, n_part), b_lds_layout)
        frag_B = thr_mma.make_fragment_B(sB)
        frag_B_retile = thr_copy_B.retile(frag_B)
        thr_sB_s2r = thr_copy_B.partition_S(sB)

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    fx.copy(
                        s2r_copy_atom,
                        thr_sB_s2r[None, None, block_k_iter],
                        frag_B_retile[None, None, block_k_iter],
                    )
            else:
                fx.copy(
                    s2r_copy_atom,
                    thr_sB_s2r[None, None, block_k_iter],
                    frag_B_retile[None, None, block_k_iter],
                )
        return frag_B

    def consume(k_tile, frag_C, frag_A, frag_B, emit_sched_barrier):
        if const_expr(emit_sched_barrier):
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
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)

    def store_half_tile(m_part, n_part, frag_C):
        gC = fx.flat_divide(out, (half_block_m, half_block_n))[
            None, None, bid_m * 2 + m_part, bid_n * 2 + n_part
        ]
        sC = fx.make_view(smem_c, c_lds_layout)

        row_coords = fx.make_view(
            0, fx.make_layout((half_block_m, half_block_n), (1, 0))
        )
        col_coords = fx.make_view(
            0, fx.make_layout((half_block_m, half_block_n), (0, 1))
        )
        thr_mma_cRow = thr_mma.partition_C(row_coords)
        thr_mma_cCol = thr_mma.partition_C(col_coords)

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
            r2g_copy_atom,
            cshuffle_tv_layout,
            cshuffle_tile,
        )
        thr_copy_cshuffle = tiled_copy_cshuffle.get_slice(tid)
        thr_sC = thr_copy_cshuffle.partition_S(sC)
        thr_gC = thr_copy_cshuffle.partition_D(gC)
        thr_cRow = thr_copy_cshuffle.partition_S(row_coords)[(0, None), None, None]
        thr_cCol = thr_copy_cshuffle.partition_S(col_coords)[(0, None), None, None]
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
            val = frag_C[i]
            if const_expr(param.has_bias):
                col = fx.get_scalar(thr_mma_cCol[i])
                global_n_idx = bid_n * block_n + n_part * half_block_n + col
                safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
                val = val + bias_buf[safe_global_n_idx].to(fx.Float32)
            frag_C_out[i] = val.to(elem_dtype)

        fx.gpu.barrier()
        for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            sC[row, col] = frag_C_out[i]

        fx.gpu.barrier()
        fx.copy(s2r_copy_atom, thr_sC, frag_C_cshuffle)
        fx.copy(r2g_copy_atom, frag_C_cshuffle, thr_gC, pred=pred_C)
        fx.gpu.barrier()

    c00 = make_c_fragment(0, 0)
    c01 = make_c_fragment(0, 1)
    c10 = make_c_fragment(1, 0)
    c11 = make_c_fragment(1, 1)

    async_load_b_to_lds(0, 0, 0)
    async_load_a_to_lds(0, 0, 0)
    async_load_b_to_lds(1, 0, 0)
    async_load_a_to_lds(1, 0, 0)
    rocdl.sched_barrier(0)
    if wid // n_waves == 1:
        rocdl.s_barrier()
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    async_load_b_to_lds(0, 1, 1)
    async_load_a_to_lds(0, 1, 1)
    async_load_b_to_lds(1, 1, 1)
    __barrier(half_ldg_b_iters + half_ldg_a_iters)

    def compute_double_tile(k_tile, prefetch_next):
        next_k_tile = k_tile + 2

        b0 = load_b_fragment(0, 0, k_tile)
        a0 = load_a_fragment(0, 0, k_tile)
        async_load_a_to_lds(1, k_tile + 1, 1)
        rocdl.s_barrier()
        consume(k_tile, c00, a0, b0, True)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            async_load_b_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile, c01, a0, b1, True)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            async_load_a_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile, c10, a1, b0, True)
        rocdl.s_barrier()

        b0 = load_b_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_b_to_lds(1, next_k_tile, 0)
            __barrier(2 * half_ldg_b_iters + half_ldg_a_iters)
        consume(k_tile, c11, a1, b1, True)
        if const_expr(not prefetch_next):
            __waitcnt(0)
        rocdl.s_barrier()

        a0 = load_a_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_a_to_lds(1, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile + 1, c00, a0, b0, True)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_b_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
        consume(k_tile + 1, c01, a0, b1, True)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_a_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
        consume(k_tile + 1, c10, a1, b0, True)
        rocdl.s_barrier()

        if const_expr(prefetch_next):
            async_load_b_to_lds(1, next_k_tile + 1, 1)
            __barrier(half_ldg_b_iters + half_ldg_a_iters)
        consume(k_tile + 1, c11, a1, b1, True)
        rocdl.s_barrier()

    final_double_tile = ((k_tiles % 2) == 0).select(k_tiles - 2, k_tiles - 1)
    main_loop_end = (k_tiles > 2).select(final_double_tile, 0)
    for k_tile in range(0, main_loop_end, 2):
        compute_double_tile(k_tile, True)

    compute_double_tile(main_loop_end, False)

    store_half_tile(0, 0, c00)
    store_half_tile(0, 1, c01)
    store_half_tile(1, 0, c10)
    store_half_tile(1, 1, c11)


@flyc.jit
def gemm_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    param: GemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    n = fx.Int32(fx.get_scalar(b.shape[0]))
    k = fx.Int32(fx.get_scalar(a.shape[1]))
    a_row_stride = fx.Int32(fx.get_scalar(a.stride[0]))
    b_row_stride = fx.Int32(fx.get_scalar(b.stride[0]))
    elem_dtype = _elem_dtype(param)
    mma_atom = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
    )
    k_per_mfma_group = param.mma_k // 4
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout(
            (param.m_waves, param.n_waves, 1),
            (param.n_waves, 1, 0),
        ),
        fx.make_tile(
            None,
            None,
            fx.make_layout(
                (k_per_mfma_group, 4),
                (1, k_per_mfma_group),
            ),
        ),
    )
    num_pid_m = (m - 1) // param.block_m + 1
    num_pid_n = (n - 1) // param.block_n + 1
    kernel_impl = (
        gemm_hti_gfx950_kernel
        if param.use_half_tile_interleaved
        else gemm_gfx950_kernel
    )
    kernel_impl._known_block_size = [param.block_threads, 1, 1]
    kernel_impl._func.__name__ = make_gemm_gfx950_kernel_name(param)
    kernel_impl(
        out,
        a,
        b,
        out,
        m,
        n,
        k,
        a_row_stride,
        b_row_stride,
        tiled_mma,
        param,
    ).launch(
        grid=(num_pid_m * num_pid_n, 1, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


def infer_has_k_tail(k: int, block_k: int, stages: int):
    k_tiles = (k + block_k - 1) // block_k
    return (k % block_k != 0) or (k_tiles < stages - 1)


def make_gemm_param_and_validate(m, n, k, kwargs):
    result = None
    try:
        result = make_gemm_gfx950_param(**kwargs)
    except Exception:
        return None
    if not ((n % 32 == 0) and (k % result.mma_k == 0)):
        return None
    if result.use_half_tile_interleaved:
        k_tiles = (k + result.block_k - 1) // result.block_k
        if k_tiles < 2:
            return None
    return result
