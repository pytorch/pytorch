# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch

GFX950_DMA_BYTES = 16
GFX950_WAVE_SIZE = 64


@fx.struct
class GemmGfx950Param:
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
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
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 64,
    stages: int = 2,
    m_waves: int = 2,
    n_waves: int = 4,
    group_m: int = 0,
    has_bias: bool = False,
    has_k_tail: bool = False,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
) -> GemmGfx950Param:
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
    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
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
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
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
    name = f"gemm_t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}"
    name += f"_w{param.m_waves}x{param.n_waves}"
    name += f"_gm{param.group_m}"
    name += f"_bias{int(param.has_bias)}"
    name += f"_ktail{int(param.has_k_tail)}"
    name += "_nt"
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


def __barrier(vmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
        "",
        has_side_effects=True,
    )


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


@flyc.kernel
def gemm_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    m: fx.Constexpr[int],
    n: fx.Constexpr[int],
    k: fx.Constexpr[int],
    tiled_mma: fx.TiledMma,
    param: GemmGfx950Param,
):
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    has_k_tail = param.has_k_tail
    wave_size = GFX950_WAVE_SIZE
    async_load_bytes = param.async_load_bytes
    in_data_bytes = param.in_data_bytes
    async_load_vec_size = async_load_bytes // in_data_bytes
    ldg_x_threads = param.ldg_x_threads
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters

    tid = fx.thread_idx.x
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    k_tiles = (k + block_k - 1) // block_k

    @fx.struct
    class LayoutGemmSharedStorage:
        a: fx.Array[fx.BFloat16, stages * block_m * block_k, 16]
        b: fx.Array[fx.BFloat16, stages * block_n * block_k, 16]

    lds = fx.SharedAllocator().allocate(LayoutGemmSharedStorage).peek()
    smem_a = lds.a.ptr
    smem_b = lds.b.ptr

    wave_offset = rocdl.readfirstlane(
        fx.Int64.ir_type,
        fx.Int64(tid // wave_size * wave_size * async_load_bytes),
    )

    def make_wave_lds_ptr(ptr):
        return fx.recast_iter(fx.Int8, ptr) + fx.Int32(wave_offset)

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    a_rsrc = buffer_ops.create_buffer_resource(a, max_size=True)
    b_rsrc = buffer_ops.create_buffer_resource(b, max_size=True)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None
    out = fx.rocdl.make_buffer_tensor(out, max_size=False)
    gC = fx.flat_divide(out, (block_m, block_n))[None, None, bid_m, bid_n]
    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))

    thr_mma = tiled_mma.thr_slice(tid)
    uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    copy_ab = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    thr_copy_s2r_A = fx.make_tiled_copy_A(copy_ab, tiled_mma).get_slice(tid)
    thr_copy_s2r_B = fx.make_tiled_copy_B(copy_ab, tiled_mma).get_slice(tid)
    thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)

    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))

    def make_lds_layout(rows):
        return fx.make_composed_layout(
            swizzle,
            fx.make_ordered_layout((rows, block_k), (1, 0)),
        )

    a_lds_layout = make_lds_layout(block_m)
    b_lds_layout = make_lds_layout(block_n)

    sA0 = fx.make_view(smem_a, a_lds_layout)
    sB0 = fx.make_view(smem_b, b_lds_layout)
    thr_gC = thr_copy_C.partition_D(gC)
    thr_cRow = thr_copy_C.partition_S(row_coords)[(0, None), None, None]
    thr_cCol = thr_copy_C.partition_S(col_coords)[(0, None), None, None]
    thr_mma_cCol = thr_mma.partition_C(col_coords)

    frag_A = thr_mma.make_fragment_A(sA0)
    frag_B = thr_mma.make_fragment_B(sB0)
    frag_C = thr_mma.make_fragment_C(gC)
    frag_C_bf16 = fx.make_fragment_like(frag_C, fx.BFloat16)
    pred_C = fx.make_fragment_like(thr_gC, dtype=fx.Boolean)

    frag_A_retile = thr_copy_s2r_A.retile(frag_A)
    frag_B_retile = thr_copy_s2r_B.retile(frag_B)
    frag_C_retile = thr_copy_C.retile(frag_C_bf16)

    frag_C.fill(0.0)
    if const_expr(param.has_bias):
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            col_idx = fx.get_scalar(thr_mma_cCol[i])
            global_n_idx = bid_n * block_n + col_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            bias_val = bias_buf[safe_global_n_idx].to(fx.Float32)
            frag_C[i] = bias_val
    for i in range_constexpr(fx.size(pred_C.shape).unpack()):
        row_idx = bid_m * block_m + fx.get_scalar(thr_cRow[i])
        col_idx = bid_n * block_n + fx.get_scalar(thr_cCol[i])
        pred_C[i] = (row_idx < m) & (col_idx < n)

    def swizzled_col_idx(row, col, layout):
        elem_offset = fx.get_scalar(
            fx.crd2idx(
                (row, col),
                layout,
            )
        )
        return elem_offset % block_k

    def advance_lds_ptr(lds_ptr):
        return lds_ptr + block_threads * async_load_bytes

    def get_a_stage_ptr(stage):
        return smem_a + stage * block_m * block_k

    def get_b_stage_ptr(stage):
        return smem_b + stage * block_n * block_k

    def get_a_stage_view(stage):
        return fx.make_view(get_a_stage_ptr(stage), a_lds_layout)

    def get_b_stage_view(stage):
        return fx.make_view(get_b_stage_ptr(stage), b_lds_layout)

    def async_load_a_to_lds(k_tile, stage):
        lds_ptr = make_wave_lds_ptr(get_a_stage_ptr(stage))
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
            global_offset = (safe_global_m_idx * k + safe_global_k_idx) * in_data_bytes
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < ldg_a_iters - 1:
                lds_ptr = advance_lds_ptr(lds_ptr)

    def async_load_b_to_lds(k_tile, stage):
        lds_ptr = make_wave_lds_ptr(get_b_stage_ptr(stage))
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
            global_offset = (safe_global_n_idx * k + safe_global_k_idx) * in_data_bytes
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, async_load_bytes)
            if i < ldg_b_iters - 1:
                lds_ptr = advance_lds_ptr(lds_ptr)

    def compute_stage(read_stage, k_tile):
        thr_sA_s2r = thr_copy_s2r_A.partition_S(get_a_stage_view(read_stage))
        thr_sB_s2r = thr_copy_s2r_B.partition_S(get_b_stage_view(read_stage))

        def compute_k_chunk(block_k_iter):
            fx.copy(
                uni_copy,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
            fx.copy(
                uni_copy,
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

    LDG_WAIT_COUNT = ldg_a_iters + ldg_b_iters

    for stage in range_constexpr(stages - 1):
        async_load_b_to_lds(stage, stage)
        async_load_a_to_lds(stage, stage)
    rocdl.sched_barrier(0)

    if const_expr(has_k_tail):
        has_main_loop = k_tiles > stages - 1
        if const_expr(isinstance(has_main_loop, bool)):
            main_loop_end = k_tiles - (stages - 1) if has_main_loop else 0
        else:
            main_loop_end = has_main_loop.select(k_tiles - (stages - 1), 0)
    else:
        main_loop_end = k_tiles - (stages - 1)
    for k_tile in range(0, main_loop_end, 1):
        current_stage = k_tile % stages
        write_stage = (current_stage + stages - 1) % stages
        __barrier((stages - 2) * LDG_WAIT_COUNT)
        async_load_b_to_lds(k_tile + (stages - 1), write_stage)
        async_load_a_to_lds(k_tile + (stages - 1), write_stage)
        compute_stage(current_stage, k_tile)

    current_stage = main_loop_end % stages
    for s in range_constexpr(0, stages - 1):
        __barrier((stages - 2 - s) * LDG_WAIT_COUNT)
        compute_stage(current_stage, main_loop_end + s)
        current_stage = (current_stage + 1) % stages

    frag_C_bf16.store(frag_C.load().to(fx.BFloat16))
    fx.copy(copy_c, frag_C_retile, thr_gC, pred=pred_C)


@flyc.jit
def launch_gemm_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    m: fx.Constexpr[int],
    n: fx.Constexpr[int],
    k: fx.Constexpr[int],
    param: GemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    mma_atom = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, fx.BFloat16)
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
    num_pid_m = (m + param.block_m - 1) // param.block_m
    num_pid_n = (n + param.block_n - 1) // param.block_n
    gemm_gfx950_kernel._known_block_size = [param.block_threads, 1, 1]
    gemm_gfx950_kernel._func.__name__ = make_gemm_gfx950_kernel_name(param)
    # The kernel ignores the bias operand when param.has_bias is false. Reuse
    # out so the JIT host signature does not need a separate dummy bias tensor.
    gemm_gfx950_kernel(out, a, b, out, m, n, k, tiled_mma, param).launch(
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
    return result
