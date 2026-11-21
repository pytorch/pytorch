# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# mypy: ignore-errors
"""FlyDSL gfx950 kernels of the fused all-gather + matmul and matmul + reduce-scatter (_fused_flydsl.py), plus the
triton kernels the calls launch before them.

Both launch fused_hti_kernel: the half-tile-interleaved kernel of torch's vendored FlyDSL GEMM (gemm_hti_gfx950_kernel
in torch/_inductor/kernel/vendored_templates/flydsl/kernels/gemm_gfx950.py) for bf16, with each op's tile order, waits
and transfers added; rs_world, a compile-time constant, picks the op (0: the all-gather).

A and C are addressed from each workgroup's own tile (_rebased): the stock kernel's descriptors at the tensors' bases
take a 32-bit offset per lane, and C's is sized cosize * 2 truncated to 32 bits, so an A or C of 4 GiB or more would
read wrapped rows, drop stores and overwrite the first rows. B keeps one descriptor: its K x N must stay below 4 GiB.

Every peer wait compares a wrapping int32 difference (flag - target >= 0): flags only grow and pass 2^31, and a plain
flag < target is false for the one call whose target wraps. A wait still short after the timeout in word 2 of the
control block at arg_ep (int32 [epoch, timeout reporters, timeout ms, unused]) traps: the first block on the rank to
time out (word 1 counts them) prints one [FATAL] line, and the launch fails (torch raises at the next sync) instead of
hanging.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import triton
import triton.language as tl
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm, rocdl as rocdl_dialect
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from torch._inductor.kernel.vendored_templates.flydsl.kernels.gemm_gfx950 import (
    __barrier,
    __waitcnt,
    _operand_fragment_dtype,
    async_load_operand,
    AsyncLoadOperand,
    BlockSwizzle,
    GEMM_DTYPE_BF16,
    GemmGfx950Param,
    GFX950_WAVE_SIZE,
    infer_has_k_tail,
    make_ab_lds_layouts,
    make_gemm_ab_load_context,
    make_gemm_gfx950_kernel_name,
    make_gemm_param_and_validate,
    make_gemm_tiled_mma,
)


_COPY_UNROLL = 4  # AG copy blocks: 16-byte loads in flight per thread
_REDUCE_UNROLL = 2  # RS reducers: 16-byte vectors per source per thread per item


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _flag_load(addr_i64):
    """The flag's value: a monotonic atomic load at system scope (one load per wave)."""
    return fx.Int32(
        llvm.load(
            T.i32,
            _global_i32_ptr(addr_i64).llvm_ptr,
            alignment=4,
            ordering=llvm.AtomicOrdering.monotonic,
            syncscope="",
        )
    )


def _atomic_add(addr_i64, value, syncscope):
    """Monotonic atomic add at `syncscope` ("" for system); returns the old value."""
    return fx.Int32(
        llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            _global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(value).ir_value(),
            llvm.AtomicOrdering.monotonic,
            syncscope=syncscope,
            alignment=4,
        ).result
    )


def _rt():
    """s_memrealtime: the constant 100 MHz clock."""
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def _load_i64(addr_i64):
    return fx.Int64(llvm.load(T.i64, _global_i32_ptr(addr_i64).llvm_ptr, alignment=8))


def _load_v4(addr_i64):
    return llvm.load(
        ir.VectorType.get([4], T.i32), _global_i32_ptr(addr_i64).llvm_ptr, alignment=16
    )


def _store16(addr_i64, v):
    llvm.store(v, _global_i32_ptr(addr_i64).llvm_ptr, alignment=16)


def _bf16x8():
    return ir.VectorType.get([8], ir.BF16Type.get())


def _f32x8():
    return ir.VectorType.get([8], ir.F32Type.get())


def _load_bf16x8(addr_i64):
    return llvm.load(_bf16x8(), _global_i32_ptr(addr_i64).llvm_ptr, alignment=16)


def _ext_f32x8(v):
    return llvm.FPExtOp(_f32x8(), v).result


def _add_f32x8(x, y):
    return llvm.FAddOp(x, y).result


def _trunc_bf16x8(v):
    return llvm.FPTruncOp(_bf16x8(), v).result


# Buffer loads with an explicit cache policy (plain LLVM loads can't carry sc0 / sc1 / nt; volatile adds a vmcnt(0)
# wait per load). The resource spans 4 GiB from the base, so offsets are raw bytes below 2^32.
# The base comes through readfirstlane (it is the same in every lane): the descriptor is scalar, with no waterfall loop
# around each load.
def _brsrc(addr_i64):
    i16 = ir.IntegerType.get_signless(16)
    base = rocdl.readfirstlane(T.i64, fx.Int64(addr_i64).ir_value())
    return rocdl_dialect.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"),
        llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), base).result,
        arith.ConstantOp(i16, ir.IntegerAttr.get(i16, 0)).result,
        fx.Int64(0xFFFFFFFF).ir_value(),
        fx.Int32(0x20070).ir_value(),
    ).result


def _bload_v4(rsrc, off_i64, cm):
    off = arith.TruncIOp(
        ir.IntegerType.get_signless(32), fx.Int64(off_i64).ir_value()
    ).result
    return rocdl_dialect.RawPtrBufferLoadOp(
        ir.VectorType.get([4], T.i32),
        rsrc,
        off,
        fx.Int32(0).ir_value(),
        aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cm),
    ).result


def _load_source(srcs, rsrcs, p, off, w_off):
    """16 bytes of source p's chunk: a buffer load with sc1 through its descriptor if there are descriptors, else a
    global load."""
    if rsrcs:
        return llvm.bitcast(_bf16x8(), _bload_v4(rsrcs[p], off + w_off, 16))
    return _load_bf16x8(srcs[p] + off + w_off)


def _rebased(t, row0, col0):
    """t viewed from its element (row0, col0): the same layout on a base address computed in 64 bits."""
    rows = fx.Int64(row0) * fx.Int64(fx.get_scalar(t.stride[0]))
    off = rows + fx.Int64(col0) * fx.Int64(fx.get_scalar(t.stride[1]))
    ptr = fx.Int64(fx.ptrtoint(fx.get_iter(t)))
    base = ptr + off * fx.Int64(fx.get_iter(t).dtype.width // 8)
    return fx.make_view(fx.inttoptr(fx.get_iter(t).type, base), fx.get_layout(t))


@flyc.kernel
def fused_hti_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    comm,
    tiled_mma: fx.TiledMma,
    param: GemmGfx950Param,
    rs_world: fx.Constexpr[int],
):
    """out = a @ b (bias unused) in the op's tile order. rs_world 0: the all-gather, comm = (arg_flags, arg_ep,
    i32_mult, flag_stride, rank, world, nsub, arg_ptrs, arg_gat, kc, shard_b, ch_b); else the reduce-scatter at TP
    rs_world, comm = (arg_ptrs, arg_cnt_off, arg_ep, arg_out, rank, nred, nb, rpb, lm): nb steps (row groups x column
    slabs) of rpb tile columns per slab, and nred reducer blocks."""
    if const_expr(rs_world == 0):
        (
            arg_flags,
            arg_ep,
            i32_mult,
            flag_stride,
            rank,
            world,
            nsub,
            arg_ptrs,
            arg_gat,
            kc,
            shard_b,
            ch_b,
        ) = comm
    else:
        arg_ptrs, arg_cnt_off, arg_ep, arg_out, rank, nred, nb, rpb, lm = comm
    operand_fragment_dtype = _operand_fragment_dtype(param)
    cshuffle_dtype = fx.BFloat16
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    half_block_m = block_m // 2
    half_block_n = block_n // 2
    stages = param.stages
    has_k_tail = param.has_k_tail
    block_threads = param.block_threads
    n_waves = param.n_waves
    half_ldg_a_iters = param.ldg_a_iters // 2
    half_ldg_b_iters = param.ldg_b_iters // 2

    tid = fx.thread_idx.x
    wid = tid // GFX950_WAVE_SIZE
    num_pid_m = (m - 1) // block_m + 1
    num_pid_n = (n - 1) // block_n + 1
    if const_expr(rs_world == 0):
        # The first lead = kc rounded up to a multiple of 8 blocks are copy blocks, then the tile blocks (a tile block
        # keeps XCD = block id % 8, and its tile index is its block id - lead). A copy block pulls a peer's shard into
        # the gathered A (kc / (world - 1) blocks per peer, peer (rank + 1 + w % (world - 1)) % world, chunk by chunk,
        # each chunk striped over the group in steps of block_threads * 16 B * _COPY_UNROLL) and ends the wave; the
        # padding blocks (ids >= kc) end at once, and no block both copies and computes. Copy blocks never wait on a
        # GEMM tile and are the lowest block ids, so they are resident first, before tile blocks that wait on their
        # chunks can fill the CUs. After its stripe of chunk j a block releases and adds 1 to flags[peer, j], the
        # counting form of the per-chunk signal async_input_mm waits on (a_chunk_signals, AsyncMM.cu; one write per
        # chunk): chunk j of call e is in once the flag reads e * kc / (world - 1). The source is half e & 1 of the
        # peer's symmetric buffer (arg_ptrs: the peers' buffer addresses); a system-scope acquire first drops this
        # XCD's stale lines of it. The host's barrier before the launch orders the peers' staging before these loads.
        lead = (kc + 7) // 8 * 8
        if fx.block_idx.x < lead:
            cq = fx.block_idx.x % (world - 1)
            ci = fx.block_idx.x // (world - 1)
            cs = kc // (world - 1)
            cp = (rank + 1 + cq) % world
            ce = _flag_load(arg_ep)
            csrc = (
                _load_i64(arg_ptrs + fx.Int64(cp) * fx.Int64(8))
                + fx.Int64(ce & 1) * shard_b
            )
            cdst = arg_gat + fx.Int64(cp) * shard_b
            llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="")
            vec_bytes = block_threads * 16
            stripe = vec_bytes * _COPY_UNROLL
            cj = fx.Int32(0)
            # the padding blocks (id >= kc) copy nothing
            nsub_mv = (fx.block_idx.x < kc).select(nsub, fx.Int32(0))
            while cj < nsub_mv:
                cbase = fx.Int64(cj) * ch_b
                coff = fx.Int64(ci) * fx.Int64(stripe) + fx.Int64(tid) * fx.Int64(16)
                while coff < ch_b:
                    cv = []
                    for cu in range_constexpr(_COPY_UNROLL):
                        cv.append(
                            _load_v4(csrc + cbase + coff + fx.Int64(cu * vec_bytes))
                        )
                    for cu in range_constexpr(_COPY_UNROLL):
                        _store16(cdst + cbase + coff + fx.Int64(cu * vec_bytes), cv[cu])
                    coff = coff + fx.Int64(cs) * fx.Int64(stripe)
                llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="workgroup")
                # vmcnt(0): every wave's stores of the stripe are done before thread 0 releases the chunk.
                __barrier(0)
                if tid == 0:
                    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="agent")
                    _atomic_add(
                        arg_flags + fx.Int64(cp * nsub + cj) * fx.Int64(flag_stride),
                        1,
                        "agent",
                    )
                cj = cj + 1
            llvm.call_intrinsic(None, "llvm.amdgcn.endpgm", [], [], [])
        # Tile row v of the block swizzle goes to order position o: the swizzle gives XCD x rows [x * R, (x + 1) * R)
        # (R = num_pid_m / 8, groups of GROUP_M rows), so o = (g * 8 + x) * GROUP_M + r for v = x * R + g * GROUP_M
        # + r: every XCD walks the order from its start and a group stays on its XCD (needs num_pid_m % (8 * GROUP_M)
        # == 0 and the swizzle on: at least 256 tiles, a multiple of 8). o < tpr (tile rows per rank): this rank's
        # rows, no wait; else o - tpr = s * tps * (world - 1) + (j - 1) * tps + i: peer (rank + j) % world,
        # sub-chunk s, its row s * tps + i.
        bid_v, bid_n = BlockSwizzle(
            NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
        ).swizzle(num_pid_m, num_pid_n, fx.block_idx.x - lead)
        xr = num_pid_m // 8
        bid_o = (
            (bid_v % xr) // param.group_m * 8 + bid_v // xr
        ) * param.group_m + bid_v % param.group_m
        tpr = num_pid_m // world
        tps = tpr // nsub
        o2 = bid_o - tpr
        per_s = tps * (world - 1)
        s_sub = o2 // per_s
        r_sub = o2 % per_s
        peer = (rank + r_sub // tps + 1) % world
        bid_m = (bid_o < tpr).select(
            rank * tpr + bid_o, peer * tpr + s_sub * tps + r_sub % tps
        )
        # A peer tile's thread 0 polls flags[peer * nsub + s] (flag_stride bytes apart) until it reaches epoch *
        # i32_mult, the epoch being the int32 at arg_ep (the device's, so a replayed graph stays in step), then an
        # agent-scope acquire; a barrier and a workgroup-scope acquire in every thread before the first A load.
        if bid_o >= tpr:
            flag_addr = arg_flags + fx.Int64(peer * nsub + s_sub) * fx.Int64(
                flag_stride
            )
            if fx.thread_idx.x == 0:
                tgt = _flag_load(arg_ep) * i32_mult
                if _flag_load(flag_addr) - tgt < 0:
                    w0 = _rt()
                    wms = _flag_load(arg_ep + fx.Int64(8))
                    while _flag_load(flag_addr) - tgt < 0:
                        rocdl.s_sleep(1)
                        if _rt() - w0 > fx.Int64(wms) * fx.Int64(100000):
                            if _atomic_add(arg_ep + fx.Int64(4), 1, "agent") == 0:
                                fx.printf(
                                    "[FATAL] fly_agmm: rank {} block {} waited {} ms for chunk {} of peer rank {} "
                                    "(flag {}, target {}, epoch {})\n",
                                    rank,
                                    fx.block_idx.x - lead,
                                    wms,
                                    s_sub,
                                    peer,
                                    _flag_load(flag_addr),
                                    tgt,
                                    _flag_load(arg_ep),
                                )
                            llvm.intr_trap()
                llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="agent")
            rocdl.s_barrier()
            llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="workgroup")
    else:
        # Rank r's partial C_r = A_r B_r [M, N] goes to its symmetric staging buffer (`out`); chunk c (rows
        # [c lm, (c + 1) lm)) belongs to rank c. The call runs nb = G * C steps t = g * C + c: step t is row group g (rg
        # tile rows of EVERY chunk) x column slab c (rpb tile columns), so every owner gets data from the first step (one
        # xGMI link per GPU pair) and a step's rg * world tile rows of A (what the Infinity Cache holds) are read from
        # HBM once per row group, the C slabs re-reading them from the cache. A step is a virtual grid of world * rg rows
        # (the chunks in production order, (rank + 1 + q) % world for q = 0..world - 1, the own chunk last) x rpb
        # columns under the stock XCD swizzle. The first rs_pad = nred rounded up to a multiple of 8 blocks (so a tile
        # block keeps XCD = block id % 8) are reducers and never compute; tile block cid = block id - rs_pad.
        rs_pad = (nred + 7) // 8 * 8
        rs_nm = lm // block_m
        rs_ncs = num_pid_n // rpb
        rs_rg = rs_nm // (nb // rs_ncs)
        rs_tpb = rs_rg * rpb
        if fx.block_idx.x < rs_pad:
            # Reducer blocks [0, nred) reduce this rank's chunk step by step: thread 0 polls the step's counters, then
            # a system-scope acquire, the barrier and a workgroup-scope acquire; the step's block of the chunk (rg
            # tile rows x rpb tile columns, a strided region) is cut into items of block_threads * 8 * _REDUCE_UNROLL
            # elements, item i to reducer i % nred: each 16-byte vector is read at the same offset of every source's
            # chunk, accumulated in fp32 in source-rank order and converted to bf16 once, then stored to arg_out (the
            # owner-side form of reduce_partials / reduce_partials_first_dim_kernel, fed from the peers' staging as
            # steps complete). Each source's chunk is addressed from its own 64-bit base; at TP 8 through one
            # descriptor per chunk (lm x N must stay below 4 GiB) with sc1 loads. The padding blocks end at once.
            if fx.block_idx.x < nred:
                rs_cnt = (
                    _load_i64(arg_ptrs + fx.Int64(rank) * fx.Int64(8)) + arg_cnt_off
                )
                rs_tgt = _flag_load(arg_ep) * rs_tpb
                rs_src = []
                for rs_p in range_constexpr(rs_world):
                    rs_src.append(
                        _load_i64(arg_ptrs + fx.Int64(rs_p * 8))
                        + fx.Int64(rank) * fx.Int64(lm) * fx.Int64(n) * fx.Int64(2)
                    )
                rs_rsrc = []
                if const_expr(rs_world == 8):
                    for rs_p in range_constexpr(rs_world):
                        rs_rsrc.append(_brsrc(rs_src[rs_p]))
                rs_item = block_threads * 8 * _REDUCE_UNROLL
                rs_ipb = rs_tpb * block_m * block_n // rs_item
                rs_rowvec = rpb * block_n // 8
                vec_bytes = block_threads * 16
                rs_fatal = (
                    f"[FATAL] fly_rsmm TP{rs_world}: rank {{}} reducer block {{}} waited {{}} ms for step {{}} from "
                    "source rank {} (counter {}, target {}, epoch {})\n"
                )
                rs_step = fx.Int32(0)
                while rs_step < nb:
                    if tid == 0:
                        rs_wp = fx.Int32(0)
                        while rs_wp < rs_world:
                            rs_ca = rs_cnt + fx.Int64(rs_step * 32 + rs_wp) * fx.Int64(
                                4
                            )
                            if _flag_load(rs_ca) - rs_tgt < 0:
                                rs_w0 = _rt()
                                rs_ms = _flag_load(arg_ep + fx.Int64(8))
                                while _flag_load(rs_ca) - rs_tgt < 0:
                                    rocdl.s_sleep(1)
                                    if _rt() - rs_w0 > fx.Int64(rs_ms) * fx.Int64(
                                        100000
                                    ):
                                        if (
                                            _atomic_add(
                                                arg_ep + fx.Int64(4), 1, "agent"
                                            )
                                            == 0
                                        ):
                                            fx.printf(
                                                rs_fatal,
                                                rank,
                                                fx.block_idx.x,
                                                rs_ms,
                                                rs_step,
                                                rs_wp,
                                                _flag_load(rs_ca),
                                                rs_tgt,
                                                _flag_load(arg_ep),
                                            )
                                        llvm.intr_trap()
                            rs_wp = rs_wp + 1
                        llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="")
                    rocdl.s_barrier()
                    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="workgroup")
                    rs_i0 = rs_step * rs_ipb
                    # element offset of the step block's corner: row g * rg * block_m, column c * rpb * block_n
                    rs_boff = (rs_step // rs_ncs) * (rs_rg * block_m) * n + (
                        rs_step % rs_ncs
                    ) * rpb * block_n
                    rs_i = rs_i0 + ((fx.block_idx.x - rs_i0) % nred + nred) % nred
                    while rs_i < rs_i0 + rs_ipb:
                        rs_offs = []
                        for rs_w in range_constexpr(_REDUCE_UNROLL):
                            # vector rs_vv of the step block: row rs_vv // rs_rowvec, column vector rs_vv % rs_rowvec
                            rs_vv = (rs_i - rs_i0) * (
                                block_threads * _REDUCE_UNROLL
                            ) + (rs_w * block_threads + tid)
                            rs_offs.append(
                                fx.Int64(rs_vv // rs_rowvec) * fx.Int64(n) * fx.Int64(2)
                                + (
                                    fx.Int64(rs_boff)
                                    + fx.Int64(rs_vv % rs_rowvec) * fx.Int64(8)
                                )
                                * fx.Int64(2)
                            )
                        rs_v = []
                        for rs_p in range_constexpr(rs_world):
                            for rs_w in range_constexpr(_REDUCE_UNROLL):
                                rs_v.append(
                                    _load_source(
                                        rs_src,
                                        rs_rsrc,
                                        rs_p,
                                        rs_offs[rs_w],
                                        fx.Int64(0),
                                    )
                                )
                        for rs_w in range_constexpr(_REDUCE_UNROLL):
                            rs_acc = _ext_f32x8(rs_v[rs_w])
                            for rs_p in range_constexpr(1, rs_world):
                                rs_acc = _add_f32x8(
                                    rs_acc,
                                    _ext_f32x8(rs_v[rs_p * _REDUCE_UNROLL + rs_w]),
                                )
                            _store16(arg_out + rs_offs[rs_w], _trunc_bf16x8(rs_acc))
                        rs_i = rs_i + nred
                    rs_step = rs_step + 1
            llvm.call_intrinsic(None, "llvm.amdgcn.endpgm", [], [], [])
        rs_cid = fx.block_idx.x - rs_pad
        rs_s = rs_cid // (rs_world * rs_tpb)
        rs_sm, rs_sn = BlockSwizzle(
            NUM_XCDS=8, NUM_PIDS_THRESHOLD=0, GROUP_M=param.group_m
        ).swizzle(rs_world * rs_rg, rpb, rs_cid % (rs_world * rs_tpb))
        rs_c = (rank + 1 + rs_sm // rs_rg) % rs_world
        bid_m = rs_c * rs_nm + (rs_s // rs_ncs) * rs_rg + rs_sm % rs_rg
        bid_n = (rs_s % rs_ncs) * rpb + rs_sn
    k_tiles = (k - 1) // block_k + 1
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    @fx.struct
    class SharedABStorage:
        a: fx.Array[operand_fragment_dtype, stages * block_m * block_k, 16]
        b: fx.Array[operand_fragment_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[cshuffle_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(_rebased(a, block_m_offset, 0), max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=False)
    out_tile = fx.rocdl.make_buffer_tensor(
        _rebased(out, block_m_offset, block_n_offset), max_size=True
    )

    ab_load_context = make_gemm_ab_load_context(
        operand_fragment_dtype, tiled_mma, tid, k, param
    )
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom
    thr_copy_A = ab_load_context.thr_copy_a
    thr_copy_B = ab_load_context.thr_copy_b
    thr_mma = tiled_mma.thr_slice(tid)
    a_lds_layout, b_lds_layout = make_ab_lds_layouts(
        half_block_m, half_block_n, block_k, param
    )
    a_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(a_buf),
        lds_layout=a_lds_layout,
        outer_tile_size=half_block_m,
        outer_bound=m - block_m_offset,
        leading_stride=a_leading_stride,
        load_iters=half_ldg_a_iters,
        is_k_major=param.a_is_transposed,
    )
    b_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(b_buf),
        lds_layout=b_lds_layout,
        outer_tile_size=half_block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=half_ldg_b_iters,
        is_k_major=not param.b_is_transposed,
    )

    def half_a_base(stage, m_part):
        return smem_a + (stage * block_m + m_part * half_block_m) * block_k

    def half_b_base(stage, n_part):
        return smem_b + (stage * block_n + n_part * half_block_n) * block_k

    def async_load_half(part, k_tile, stage, is_a):
        operand, lds_base, outer_offset, rows = (
            (a_load_operand, half_a_base(stage, part), fx.Int32(0), half_block_m)
            if is_a
            else (
                b_load_operand,
                half_b_base(stage, part),
                block_n_offset,
                half_block_n,
            )
        )
        async_load_operand(
            operand,
            lds_base=lds_base,
            global_outer_offset=outer_offset + part * rows,
            k_tile=k_tile,
        )

    def async_load_a_to_lds(part, k_tile, stage):
        async_load_half(part, k_tile, stage, True)

    def async_load_b_to_lds(part, k_tile, stage):
        async_load_half(part, k_tile, stage, False)

    def make_c_fragment(m_part, n_part):
        gC = fx.flat_divide(out_buf, (half_block_m, half_block_n))[
            None, None, bid_m * 2 + m_part, bid_n * 2 + n_part
        ]
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
                        a_s2r_copy_atom,
                        thr_sA_s2r[None, None, block_k_iter],
                        frag_A_retile[None, None, block_k_iter],
                    )
            else:
                fx.copy(
                    a_s2r_copy_atom,
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
                        b_s2r_copy_atom,
                        thr_sB_s2r[None, None, block_k_iter],
                        frag_B_retile[None, None, block_k_iter],
                    )
            else:
                fx.copy(
                    b_s2r_copy_atom,
                    thr_sB_s2r[None, None, block_k_iter],
                    frag_B_retile[None, None, block_k_iter],
                )
        return frag_B

    def consume(k_tile, frag_C, frag_A, frag_B):
        rocdl.sched_barrier(0)

        def mma_k_chunk(block_k_iter):
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
                    mma_k_chunk(block_k_iter)
            else:
                mma_k_chunk(block_k_iter)
        rocdl.sched_barrier(0)

    cshuffle_s2r_atom = fx.make_copy_atom(fx.UniversalCopy128b(), cshuffle_dtype)
    # The RS stores write through (cache policy 17: sc0 | sc1), so the system-scope release before each tile's
    # counter add finds no dirty lines to write back.
    cshuffle_r2g_atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy128b(17 if rs_world else 0), cshuffle_dtype
    )

    def cshuffle_views():
        sC = fx.make_view(
            smem_c, fx.make_layout((half_block_m, half_block_n), (half_block_n, 1))
        )

        row_coords = fx.make_view(
            0, fx.make_layout((half_block_m, half_block_n), (1, 0))
        )
        col_coords = fx.make_view(
            0, fx.make_layout((half_block_m, half_block_n), (0, 1))
        )
        thr_mma_cRow = thr_mma.partition_C(row_coords)
        thr_mma_cCol = thr_mma.partition_C(col_coords)

        cshuffle_vec_size = param.cshuffle_r2g_vec_size
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
            cshuffle_r2g_atom,
            cshuffle_tv_layout,
            cshuffle_tile,
        )
        thr_copy_cshuffle = tiled_copy_cshuffle.get_slice(tid)
        thr_sC = thr_copy_cshuffle.partition_S(sC)
        thr_cRow = thr_copy_cshuffle.partition_S(row_coords)[(0, None), None, None]
        thr_cCol = thr_copy_cshuffle.partition_S(col_coords)[(0, None), None, None]
        return (
            sC,
            thr_mma_cRow,
            thr_mma_cCol,
            thr_sC,
            thr_cRow,
            thr_cCol,
            thr_copy_cshuffle,
        )

    def store_half_tile(m_part, n_part, frag_C):
        gC = fx.flat_divide(out_tile, (half_block_m, half_block_n))[
            None, None, m_part, n_part
        ]
        (
            sC,
            thr_mma_cRow,
            thr_mma_cCol,
            thr_sC,
            thr_cRow,
            thr_cCol,
            thr_copy_cshuffle,
        ) = cshuffle_views()
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

        frag_C_out = fx.make_fragment_like(frag_C, cshuffle_dtype)
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            frag_C_out[i] = frag_C[i].to(cshuffle_dtype)

        fx.gpu.barrier()
        for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            sC[row, col] = frag_C_out[i]

        fx.gpu.barrier()
        fx.copy(cshuffle_s2r_atom, thr_sC, frag_C_cshuffle)
        fx.copy(
            cshuffle_r2g_atom,
            frag_C_cshuffle,
            thr_gC,
            pred=pred_C,
        )
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
        # This group's next barrier releases the first group to read stage 0.
        __waitcnt(0)
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
        consume(k_tile, c00, a0, b0)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            async_load_b_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile, c01, a0, b1)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            async_load_a_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile, c10, a1, b0)
        if const_expr(not prefetch_next):
            # Retire the leading group's stage-1 A-half-0 copies before this
            # barrier releases the first group. B/A half 1 may stay in flight.
            # Both groups wait to avoid a branch that spills MXFP8 accumulators.
            __waitcnt(half_ldg_b_iters + half_ldg_a_iters)
        rocdl.s_barrier()

        b0 = load_b_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_b_to_lds(1, next_k_tile, 0)
            __barrier(2 * half_ldg_b_iters + half_ldg_a_iters)
        consume(k_tile, c11, a1, b1)
        if const_expr(not prefetch_next):
            __waitcnt(0)
        rocdl.s_barrier()

        a0 = load_a_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_a_to_lds(1, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile + 1, c00, a0, b0)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_b_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
        consume(k_tile + 1, c01, a0, b1)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_a_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
        consume(k_tile + 1, c10, a1, b0)
        rocdl.s_barrier()

        if const_expr(prefetch_next):
            async_load_b_to_lds(1, next_k_tile + 1, 1)
            __barrier(half_ldg_b_iters + half_ldg_a_iters)
        consume(k_tile + 1, c11, a1, b1)
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

    if const_expr(rs_world):
        # Once every wave's stores are acknowledged, thread 0 releases at system scope and adds 1 to the owner's
        # counter (step s, this rank; a step's world counters share one 128-byte line). Step s of chunk c is in from
        # every source once each of its counters has reached epoch * rs_tpb. The host's barrier before each launch
        # orders every peer's reads of the previous call's staging before these stores, so GEMM tiles never wait
        # and the reducers cannot deadlock.
        __barrier(0)
        if tid == 0:
            llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="")
            rs_owner = _load_i64(arg_ptrs + fx.Int64(rs_c) * fx.Int64(8))
            _atomic_add(
                rs_owner + arg_cnt_off + fx.Int64(rs_s * 32 + rank) * fx.Int64(4), 1, ""
            )


def _launch(out, a, b, comm, param, stream, rs_world, name):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    n = fx.Int32(fx.get_scalar(b.shape[1]))
    k = fx.Int32(fx.get_scalar(a.shape[1]))
    a_leading_stride = fx.Int32(
        fx.get_scalar(a.stride[1] if const_expr(param.a_is_transposed) else a.stride[0])
    )
    b_leading_stride = fx.Int32(
        fx.get_scalar(b.stride[1] if const_expr(param.b_is_transposed) else b.stride[0])
    )
    tiled_mma = make_gemm_tiled_mma(param)
    num_pid_m = (m - 1) // param.block_m + 1
    num_pid_n = (n - 1) // param.block_n + 1
    ntiles = num_pid_m * num_pid_n
    if const_expr(rs_world == 0):
        # the all-gather's copy blocks (kc rounded up to a multiple of 8) lead the tile blocks
        ntiles = ntiles + (comm[9] + 7) // 8 * 8
    else:
        # the reduce-scatter's reducers (nred rounded up to a multiple of 8) lead them
        ntiles = ntiles + (comm[5] + 7) // 8 * 8
    fused_hti_kernel._known_block_size = [param.block_threads, 1, 1]
    fused_hti_kernel._func.__name__ = name + make_gemm_gfx950_kernel_name(param)
    fused_hti_kernel(
        out,
        a,
        b,
        out,
        m,
        n,
        k,
        a_leading_stride,
        b_leading_stride,
        comm,
        tiled_mma,
        param,
        rs_world,
    ).launch(
        grid=(ntiles, 1, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


@flyc.jit
def agmm_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    arg_flags: fx.Int64,
    arg_ep: fx.Int64,
    i32_mult: fx.Int32,
    flag_stride: fx.Int32,
    rank: fx.Int32,
    world: fx.Int32,
    nsub: fx.Int32,
    arg_ptrs: fx.Int64,
    arg_gat: fx.Int64,
    kc: fx.Int32,
    shard_b: fx.Int64,
    ch_b: fx.Int64,
    param: GemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    """out[M, N] = a[M, K] @ b[K, N], tiles in all-gather order, peer tiles waiting on their sub-chunk flag; the first
    kc blocks (rounded up to a multiple of 8) only copy the peers' shards into a (arg_gat)."""
    comm = (
        arg_flags,
        arg_ep,
        i32_mult,
        flag_stride,
        rank,
        world,
        nsub,
        arg_ptrs,
        arg_gat,
        kc,
        shard_b,
        ch_b,
    )
    _launch(out, a, b, comm, param, stream, 0, "agmm_")


@flyc.jit
def rsmm_gfx950(
    stage: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    arg_ptrs: fx.Int64,
    arg_cnt_off: fx.Int64,
    arg_ep: fx.Int64,
    arg_out: fx.Int64,
    rank: fx.Int32,
    world: fx.Constexpr[int],
    nred: fx.Int32,
    nb: fx.Int32,
    rpb: fx.Int32,
    lm: fx.Int32,
    param: GemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    """stage[M, N] = a[M, K] @ b[K, N] in reduce-scatter order; the first nred blocks (rounded up to a multiple of 8)
    only sum this rank's chunk of every source's stage into arg_out [lm, N]."""
    comm = (arg_ptrs, arg_cnt_off, arg_ep, arg_out, rank, nred, nb, rpb, lm)
    _launch(stage, a, b, comm, param, stream, world, f"rsmm_w{world}_")


# fused_hti_kernel's tile config.
CFG = dict(
    tile_m=256,
    tile_n=256,
    tile_k=64,
    stages=2,
    m_waves=2,
    n_waves=4,
    group_m=4,
    use_half_tile_interleaved=True,
)


def arg(t):
    return flyc.from_torch_tensor(t).mark_layout_dynamic()


def gemm_param(m, n, k, b_t, group_m=CFG["group_m"]):
    prm = make_gemm_param_and_validate(
        m,
        n,
        k,
        {
            "dtype_id": GEMM_DTYPE_BF16,
            **CFG,
            "group_m": group_m,
            "a_is_transposed": False,
            "b_is_transposed": b_t,
            "has_bias": False,
            "has_k_tail": infer_has_k_tail(k, CFG["tile_k"], CFG["stages"], True),
        },
    )
    if prm is None:
        raise RuntimeError(f"FlyDSL GEMM: no valid config for M {m}, N {n}, K {k}")
    return prm


@triton.jit
def _bump(ep):
    tl.store(ep, tl.load(ep) + 1)


@triton.jit
def _stage3(src, mine, own, ep, n, BUFS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(src + offs, mask=mask)
    # BUFS is a power of two; & keeps the half in range once the int32 epoch wraps (% would go negative), and the
    # kernel's copy blocks pick their source half the same way.
    tl.store(mine + (tl.load(ep) & (BUFS - 1)) * n + offs, v, mask=mask)
    tl.store(own + offs, v, mask=mask)


@triton.jit
def _flush(sink):
    # Launched after _stage3: a system-scope release writes back its XCD's whole L2, which the peers' xGMI reads of
    # the staged half need (dirty lines are not written back at the kernel boundary).
    tl.atomic_add(sink, 0, sem="release", scope="sys")
