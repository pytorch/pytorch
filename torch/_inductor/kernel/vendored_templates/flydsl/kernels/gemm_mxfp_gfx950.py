# SPDX-License-Identifier: BSD-3-Clause

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl

from .gemm_gfx950 import (
    __barrier,
    async_load_operand,
    AsyncLoadOperand,
    BlockSwizzle,
    buffer_load_lds_inline,
    GEMM_DTYPE_BF16,
    GEMM_DTYPE_MXFP4,
    GemmABLoadContext,
    GemmGfx950Param,
    get_leading_stride,
    get_wave_lds_offset,
    GFX950_DMA_BYTES,
    GFX950_SCALE_DMA_BYTES,
    GFX950_WAVE_SIZE,
    make_gemm_gfx950_kernel_name,
    make_lds_layout,
    make_wave_lds_ptr,
    mxfp8_scale_stage_bytes,
    MXFP_SCALE_BLOCK_K,
)


_FP8_KCONTIG_SWIZZLE_MASK = 3
_FP8_KCONTIG_SWIZZLE_BASE = 4
_FP8_KCONTIG_SWIZZLE_SHIFT = 4


def _fp8_kcontig_swizzle(block_k):
    """K-contiguous ds_read_b128 XOR. Dest bits must stay inside K."""
    if const_expr(block_k >= 128):
        return fx.SwizzleType.get(
            _FP8_KCONTIG_SWIZZLE_MASK,
            _FP8_KCONTIG_SWIZZLE_BASE,
            _FP8_KCONTIG_SWIZZLE_SHIFT,
        )
    if const_expr(block_k == 64):
        return fx.SwizzleType.get(2, 4, 3)
    return None


def make_fp8_lds_layout(rows, block_k, is_k_major):
    """LDS layout for gfx950 FP8 / MXFP8.

    K-contiguous uses a block_k-dependent XOR swizzle. K-major uses the same
    16-element XOR groups as ``ds_read_tr16``: ``ds_read_tr8`` is still a
    16-lane cooperative op.
    """
    if const_expr(is_k_major):
        return make_lds_layout(rows, block_k, is_transposed=True)
    base_layout = fx.make_ordered_layout((rows, block_k), (1, 0))
    swizzle = _fp8_kcontig_swizzle(block_k)
    if swizzle is None:
        return base_layout
    return fx.make_composed_layout(fx.static(swizzle), base_layout)


def make_mxfp_ab_load_context(elem_dtype, tiled_mma, tid, k, param: GemmGfx950Param):
    uni_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)

    if const_expr(param.a_is_transposed):
        a_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans8_64b(), elem_dtype
        )
        a_tiled_copy_atom = a_s2r_copy_atom
    else:
        a_s2r_copy_atom = uni_copy_atom
        a_tiled_copy_atom = buffer_copy_atom
    if const_expr(not param.b_is_transposed):
        b_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans8_64b(), elem_dtype
        )
        b_tiled_copy_atom = b_s2r_copy_atom
    else:
        b_s2r_copy_atom = uni_copy_atom
        b_tiled_copy_atom = buffer_copy_atom

    return GemmABLoadContext(
        wave_offset=get_wave_lds_offset(tid, param.async_load_bytes),
        tid=tid,
        inner_bound=k,
        param=param,
        uni_copy_atom=uni_copy_atom,
        buffer_copy_atom=buffer_copy_atom,
        a_s2r_copy_atom=a_s2r_copy_atom,
        b_s2r_copy_atom=b_s2r_copy_atom,
        thr_copy_a=fx.make_tiled_copy_A(a_tiled_copy_atom, tiled_mma).get_slice(tid),
        thr_copy_b=fx.make_tiled_copy_B(b_tiled_copy_atom, tiled_mma).get_slice(tid),
    )


def async_load_mxfp8_scales(
    scale,
    lds_base,
    tid,
    outer_offset,
    outer_bound,
    k_begin,
    rows,
    block_k,
    block_threads,
    has_k_tail,
    k_bound,
):
    """Stage four E8M0 bytes/lane with the same waitcnt protocol as A/B."""
    scale_k = block_k // MXFP_SCALE_BLOCK_K
    words_per_row = scale_k // 4
    stage_bytes = mxfp8_scale_stage_bytes(rows, block_k, block_threads)
    load_iters = stage_bytes // (block_threads * GFX950_SCALE_DMA_BYTES)
    rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale))
    lds_ptr = make_wave_lds_ptr(
        lds_base, get_wave_lds_offset(tid, GFX950_SCALE_DMA_BYTES)
    )
    stride = fx.Int32(fx.get_scalar(scale.stride[0]))
    for i in range_constexpr(load_iters):
        word = i * block_threads + tid
        row = outer_offset + (word // words_per_row) % rows
        safe_row = (row < outer_bound).select(row, 0)
        sk = k_begin // MXFP_SCALE_BLOCK_K + word % words_per_row * 4
        if const_expr(has_k_tail):
            sk = (sk < k_bound // MXFP_SCALE_BLOCK_K).select(sk, 0)
        buffer_load_lds_inline(
            rsrc,
            lds_ptr,
            safe_row * stride + sk,
            GFX950_SCALE_DMA_BYTES,
        )
        if i < load_iters - 1:
            lds_ptr = lds_ptr + block_threads * GFX950_SCALE_DMA_BYTES


def mxfp8_gemm(
    frag_C,
    frag_A,
    frag_B,
    scale_a,
    scale_b,
    tid,
    m_offset,
    n_offset,
    k_offset,
    m,
    n,
    param: GemmGfx950Param,
    scales_are_fragments=False,
):
    """Apply one MMA K step with unshuffled [outer, K // 32] E8M0 scales.

    lane % mma_m selects the row/column and lane // mma_m selects its
    32-element K scale group. opsel=0 consumes the low byte of each
    lane's i32 scale operand.
    """
    mma_atom = fx.make_mma_atom(
        fx.rocdl.cdna4.MFMA_Scale(
            param.mma_m, param.mma_n, param.mma_k, fx.Float8E4M3FN
        )
    )
    lane = tid % GFX950_WAVE_SIZE
    wave = tid // GFX950_WAVE_SIZE
    wave_m = wave // param.n_waves
    wave_n = wave % param.n_waves
    scale_lane = lane % param.mma_m
    scale_group = lane // param.mma_m
    scale_k = k_offset // MXFP_SCALE_BLOCK_K + scale_group

    a_scales = []
    for mi in range_constexpr(fx.size(frag_A.shape[1]).unpack()):
        if const_expr(scales_are_fragments):
            a_scales.append(scale_a[mi])
        else:
            row = m_offset + (mi * param.m_waves + wave_m) * param.mma_m + scale_lane
            safe_row = (row < m).select(row, 0)
            a_scales.append(scale_a[safe_row, scale_k].to(fx.Int32))
    b_scales = []
    for ni in range_constexpr(fx.size(frag_B.shape[1]).unpack()):
        if const_expr(scales_are_fragments):
            b_scales.append(scale_b[ni])
        else:
            col = n_offset + (ni * param.n_waves + wave_n) * param.mma_n + scale_lane
            safe_col = (col < n).select(col, 0)
            b_scales.append(scale_b[safe_col, scale_k].to(fx.Int32))

    for ni in range_constexpr(fx.size(frag_B.shape[1]).unpack()):
        for mi in range_constexpr(fx.size(frag_A.shape[1]).unpack()):
            fx.gemm(
                mma_atom,
                fx.coalesce(frag_C[None, mi, ni]),
                fx.coalesce(frag_A[None, mi]),
                fx.coalesce(frag_B[None, ni]),
                fx.coalesce(frag_C[None, mi, ni]),
                scale_a=a_scales[mi],
                scale_b=b_scales[ni],
            )


@flyc.kernel
def gemm_mxfp_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    param: GemmGfx950Param,
):
    elem_dtype = fx.Float8E4M3FN
    shuffle_dtype = (
        fx.BFloat16 if const_expr(param.out_dtype_id == GEMM_DTYPE_BF16) else fx.Float16
    )
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    block_threads = param.block_threads
    cshuffle_r2g_vec_size = GFX950_DMA_BYTES // param.out_data_bytes
    scale_k = param.scale_row_bytes
    scale_a_stage_bytes = param.scale_a_bytes
    scale_b_stage_bytes = param.scale_b_bytes
    has_k_tail = param.has_k_tail
    ldg_wait_count = (
        param.ldg_a_iters
        + param.ldg_b_iters
        + param.scale_a_iters
        + param.scale_b_iters
    )

    tid = fx.thread_idx.x
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    bid_m, bid_n = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    ).swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    k_tiles = (k - 1) // block_k + 1
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    @fx.struct
    class SharedMainloopStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]
        sa: fx.Array[fx.Uint8, stages * scale_a_stage_bytes, 16]
        sb: fx.Array[fx.Uint8, stages * scale_b_stage_bytes, 16]

    @fx.union
    class SharedStorage:
        mainloop: SharedMainloopStorage
        c: fx.Array[shuffle_dtype, block_m * block_n, 16]

    allocator = fx.SharedAllocator()
    storage = allocator.allocate(SharedStorage)
    smem_a = storage.mainloop.a.peek().ptr
    smem_b = storage.mainloop.b.peek().ptr
    smem_sa = storage.mainloop.sa.peek().ptr
    smem_sb = storage.mainloop.sb.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    scale_a_buf = fx.rocdl.make_buffer_tensor(scale_a, max_size=True)
    scale_b_buf = fx.rocdl.make_buffer_tensor(scale_b, max_size=True)

    mma_atom = fx.make_mma_atom(
        fx.rocdl.cdna4.MFMA_Scale(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
    )
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((param.m_waves, param.n_waves, 1), (param.n_waves, 1, 0)),
        fx.make_tile(None, None, fx.make_layout((16, 2, 4), (1, 64, 16))),
    )
    ab_load_context = make_mxfp_ab_load_context(elem_dtype, tiled_mma, tid, k, param)
    a_s2r_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_atom = ab_load_context.b_s2r_copy_atom
    thr_copy_A = ab_load_context.thr_copy_a
    thr_copy_B = ab_load_context.thr_copy_b
    thr_mma = tiled_mma.thr_slice(tid)

    a_lds_layout = make_fp8_lds_layout(block_m, block_k, param.a_is_transposed)
    b_lds_layout = make_fp8_lds_layout(block_n, block_k, not param.b_is_transposed)
    a_load = AsyncLoadOperand(
        context=ab_load_context,
        rsrc=fx.rocdl.get_buffer_rsrc(fx.get_iter(a_buf)),
        lds_layout=a_lds_layout,
        outer_tile_size=block_m,
        outer_bound=m,
        leading_stride=a_leading_stride,
        load_iters=param.ldg_a_iters,
        is_k_major=param.a_is_transposed,
    )
    b_load = AsyncLoadOperand(
        context=ab_load_context,
        rsrc=fx.rocdl.get_buffer_rsrc(fx.get_iter(b_buf)),
        lds_layout=b_lds_layout,
        outer_tile_size=block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=param.ldg_b_iters,
        is_k_major=not param.b_is_transposed,
    )

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]
    sA = fx.make_view(smem_a, a_lds_layout)
    sB = fx.make_view(smem_b, b_lds_layout)
    sC = fx.make_view(smem_c, fx.make_layout((block_m, block_n), (block_n, 1)))
    frag_A = thr_mma.make_fragment_A(sA)
    frag_B = thr_mma.make_fragment_B(sB)
    frag_C = thr_mma.make_fragment_C(gC)
    frag_A_retile = thr_copy_A.retile(frag_A)
    frag_B_retile = thr_copy_B.retile(frag_B)
    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)

    cshuffle_s2r_atom = fx.make_copy_atom(fx.UniversalCopy128b(), shuffle_dtype)
    cshuffle_r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), shuffle_dtype)
    cshuffle_x_threads = block_n // cshuffle_r2g_vec_size
    cshuffle_tile, cshuffle_tv_layout = fx.make_layout_tv(
        fx.make_layout(
            (block_threads // cshuffle_x_threads, cshuffle_x_threads),
            (cshuffle_x_threads, 1),
        ),
        fx.make_layout((1, cshuffle_r2g_vec_size), (1, 1)),
    )
    thr_copy_cshuffle = fx.make_tiled_copy(
        cshuffle_r2g_atom, cshuffle_tv_layout, cshuffle_tile
    ).get_slice(tid)
    thr_sC = thr_copy_cshuffle.partition_S(sC)
    thr_gC = thr_copy_cshuffle.partition_D(gC)
    thr_cRow = thr_copy_cshuffle.partition_S(row_coords)[(0, None), None, None]
    thr_cCol = thr_copy_cshuffle.partition_S(col_coords)[(0, None), None, None]
    frag_C_cshuffle = fx.make_fragment_like(thr_sC)
    pred_C = fx.make_fragment_like(thr_cRow, dtype=fx.Boolean)
    for i in range_constexpr(fx.size(pred_C.shape).unpack()):
        local_row = fx.get_scalar(thr_cRow[i])
        local_col = fx.get_scalar(thr_cCol[i])
        pred_C[i] = (
            (local_row < block_m)
            & (local_col < block_n)
            & (block_m_offset + local_row < m)
            & (block_n_offset + local_col < n)
        )
    frag_C.fill(0.0)

    def async_load_a(k_tile, stage):
        async_load_operand(
            a_load, smem_a + stage * block_m * block_k, block_m_offset, k_tile
        )
        async_load_mxfp8_scales(
            scale_a_buf,
            smem_sa + stage * scale_a_stage_bytes,
            tid,
            block_m_offset,
            m,
            k_tile * block_k,
            block_m,
            block_k,
            block_threads,
            has_k_tail,
            k,
        )

    def async_load_b(k_tile, stage):
        async_load_operand(
            b_load, smem_b + stage * block_n * block_k, block_n_offset, k_tile
        )
        async_load_mxfp8_scales(
            scale_b_buf,
            smem_sb + stage * scale_b_stage_bytes,
            tid,
            block_n_offset,
            n,
            k_tile * block_k,
            block_n,
            block_k,
            block_threads,
            has_k_tail,
            k,
        )

    def compute_stage(read_stage, k_tile):
        sA_stage = fx.make_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
        sB_stage = fx.make_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
        thr_sA_s2r = thr_copy_A.partition_S(sA_stage)
        thr_sB_s2r = thr_copy_B.partition_S(sB_stage)
        scale_a_view = fx.make_view(
            smem_sa + read_stage * scale_a_stage_bytes,
            fx.make_layout((block_m, scale_k), (scale_k, 1)),
        )
        scale_b_view = fx.make_view(
            smem_sb + read_stage * scale_b_stage_bytes,
            fx.make_layout((block_n, scale_k), (scale_k, 1)),
        )

        def compute_k_chunk(block_k_iter):
            fx.copy(
                b_s2r_atom,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
            fx.copy(
                a_s2r_atom,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
            mxfp8_gemm(
                frag_C,
                frag_A[None, None, block_k_iter],
                frag_B[None, None, block_k_iter],
                scale_a_view,
                scale_b_view,
                tid,
                0,
                0,
                block_k_iter * param.mma_k,
                block_m,
                block_n,
                param,
            )

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    compute_k_chunk(block_k_iter)
            else:
                compute_k_chunk(block_k_iter)

    for stage in range_constexpr(stages - 1):
        async_load_b(stage, stage)
        async_load_a(stage, stage)
    rocdl.sched_barrier(0)

    if const_expr(has_k_tail):
        main_loop_end = (k_tiles > stages - 1).select(k_tiles - (stages - 1), 0)
    else:
        main_loop_end = k_tiles - (stages - 1)
    for k_tile in range(0, main_loop_end, 1):
        current_stage = k_tile % stages
        write_stage = (current_stage + stages - 1) % stages
        __barrier((stages - 2) * ldg_wait_count)
        async_load_b(k_tile + (stages - 1), write_stage)
        async_load_a(k_tile + (stages - 1), write_stage)
        compute_stage(current_stage, k_tile)

    current_stage = main_loop_end % stages
    for s in range_constexpr(0, stages - 1):
        __barrier((stages - 2 - s) * ldg_wait_count)
        compute_stage(current_stage, main_loop_end + s)
        current_stage = (current_stage + 1) % stages

    frag_C_out = fx.make_fragment_like(frag_C, shuffle_dtype)
    frag_C_out.store(frag_C.load().to(shuffle_dtype))
    fx.gpu.barrier()
    for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
        sC[fx.get_scalar(thr_mma_cRow[i]), fx.get_scalar(thr_mma_cCol[i])] = frag_C_out[
            i
        ]
    fx.gpu.barrier()
    fx.copy(cshuffle_s2r_atom, thr_sC, frag_C_cshuffle)
    fx.copy(cshuffle_r2g_atom, frag_C_cshuffle, thr_gC, pred=pred_C)


@flyc.jit
def gemm_mxfp_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
    param: GemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    elements_per_byte = 2 if const_expr(param.dtype_id == GEMM_DTYPE_MXFP4) else 1
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    n = fx.Int32(fx.get_scalar(b.shape[1]))
    k = fx.Int32(fx.get_scalar(a.shape[1])) * fx.Int32(elements_per_byte)
    a_leading_stride = get_leading_stride(a, param.a_is_transposed)
    b_leading_stride = get_leading_stride(b, param.b_is_transposed)
    num_pid_m = (m - 1) // param.block_m + 1
    num_pid_n = (n - 1) // param.block_n + 1
    kernel = gemm_mxfp_gfx950_kernel
    kernel._known_block_size = [param.block_threads, 1, 1]
    kernel._func.__name__ = make_gemm_gfx950_kernel_name(param)
    kernel(
        out,
        a,
        b,
        scale_a,
        scale_b,
        m,
        n,
        k,
        a_leading_stride,
        b_leading_stride,
        param,
    ).launch(
        grid=(num_pid_m * num_pid_n, 1, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )
