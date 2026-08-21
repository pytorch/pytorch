"""gfx950 BF16 FlexAttention backward kernels."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .flex_attn_mask import (
    evaluate_mask_program,
    is_causal_document_mask_program,
)
from .flex_attn_utils import (
    load_scalar,
    make_global_view,
    make_shared_view,
    store_scalar,
)

_LOG2E = 1.4426950408889634
_NEG_BIG = -1.0e30
_BATCHED_CAUSAL_DOCUMENT_MASK_PROGRAM = (
    ("const_bool", True),
    ("ge", 2, 3),
    ("and", 4, 5),
    ("load_i32", 0, (0, 3)),
    ("le", 2, 7),
    ("and", 6, 8),
)


def _f32(x):
    return fx.Float32(x)


def _exp2(x):
    return fx.math.exp2(_f32(x))


def build_flex_attn_bwd_module(
    B,
    H,
    S,
    DQK,
    DV,
    dtype_str,
    block_m,
    block_n,
    *,
    scale=None,
    max_partial_blocks=None,
    max_full_blocks=None,
    block_mask_batch=1,
    block_mask_heads=1,
    causal_partial_blocks=False,
    mask_program=(),
    mask_program_output=0,
    mask_buffer_shapes=(),
    mask_buffer_strides=(),
    q_stride=None,
    k_stride=None,
    v_stride=None,
    out_stride=None,
    do_stride=None,
    dq_stride=None,
    dk_stride=None,
    dv_stride=None,
    lse_in_log2=False,
):
    mask_buffer_shapes = tuple(tuple(shape) for shape in mask_buffer_shapes)
    mask_buffer_strides = tuple(tuple(stride) for stride in mask_buffer_strides)
    batched_document_mask = (
        tuple(mask_program) == _BATCHED_CAUSAL_DOCUMENT_MASK_PROGRAM
        and int(mask_program_output) == 9
        and mask_buffer_shapes == ((B, S),)
        and mask_buffer_strides == ((S, 1),)
        and block_mask_batch == B
        and block_mask_heads == 1
    )
    use_short_dq_tile = (
        S == 4096
        and DQK == 192
        and DV == 128
        and int(block_m) == 128
        and int(block_n) == 128
        and (
            is_causal_document_mask_program(
                tuple(mask_program),
                int(mask_program_output),
                mask_buffer_strides,
            )
            or batched_document_mask
        )
    )
    return _build_flex_attn_bwd_module(
        B,
        H,
        S,
        DQK,
        DV,
        dtype_str,
        block_m,
        block_n,
        scale=scale,
        max_partial_blocks=max_partial_blocks,
        max_full_blocks=max_full_blocks,
        block_mask_batch=block_mask_batch,
        block_mask_heads=block_mask_heads,
        causal_partial_blocks=causal_partial_blocks,
        mask_program=mask_program,
        mask_program_output=mask_program_output,
        mask_buffer_shapes=mask_buffer_shapes,
        mask_buffer_strides=mask_buffer_strides,
        q_stride=q_stride,
        k_stride=k_stride,
        v_stride=v_stride,
        out_stride=out_stride,
        do_stride=do_stride,
        dq_stride=dq_stride,
        dk_stride=dk_stride,
        dv_stride=dv_stride,
        lse_in_log2=lse_in_log2,
        dq_reduction_rows=32 if use_short_dq_tile else 64,
    )


def _build_flex_attn_bwd_module(
    B,
    H,
    S,
    DQK,
    DV,
    dtype_str,
    block_m,
    block_n,
    *,
    scale=None,
    max_partial_blocks=None,
    max_full_blocks=None,
    block_mask_batch=1,
    block_mask_heads=1,
    causal_partial_blocks=False,
    mask_program=(),
    mask_program_output=0,
    mask_buffer_shapes=(),
    mask_buffer_strides=(),
    q_stride=None,
    k_stride=None,
    v_stride=None,
    out_stride=None,
    do_stride=None,
    dq_stride=None,
    dk_stride=None,
    dv_stride=None,
    lse_in_log2=False,
    dq_reduction_rows=64,
):
    if dtype_str != "bf16":
        raise ValueError(f"unsupported dtype {dtype_str}")
    if (DQK, DV) not in ((128, 128), (192, 128)):
        raise ValueError(
            "FlyDSL backward requires (DQK, DV) to be (128, 128) or (192, 128)"
        )
    if int(block_m) != 128 or int(block_n) != 128:
        raise ValueError("FlyDSL backward requires sparse Q/KV block size 128")
    if S <= 0 or S % 128:
        raise ValueError(
            "FlyDSL backward requires a positive sequence length divisible by 128"
        )
    if S > 16384:
        raise ValueError("FlyDSL backward currently supports sequence length <= 16384")
    if block_mask_batch not in (1, B):
        raise ValueError("BlockMask batch dimension must be 1 or B")
    if block_mask_heads not in (1, H):
        raise ValueError("MHA backward BlockMask head dimension must be 1 or H")
    if len(mask_buffer_shapes) != len(mask_buffer_strides):
        raise ValueError("mask buffer shape/stride descriptors must match")
    if len(mask_buffer_shapes) > 4:
        raise ValueError("FlyDSL backward supports at most four mask buffers")

    SB = int(block_m)
    BM = 64  # q rows per tile
    BN = 64  # kv rows per tile
    DQ_BN = int(dq_reduction_rows)  # kv rows streamed by the dQ kernel
    NW = 4  # waves per workgroup
    NT = NW * 64  # threads per workgroup

    if DQ_BN not in (32, 64):
        raise ValueError("dQ reduction rows must be 32 or 64")

    NB = S // SB  # sparse blocks along each axis
    MC = S // BM  # q chunks
    NC = S // BN  # kv chunks
    CPB = SB // BM  # chunks per sparse block
    PNT = max(64, MC)  # one prologue thread per chunk, with at least one wave
    MAX_PARTIAL = NB if max_partial_blocks is None else int(max_partial_blocks)
    MAX_FULL = NB if max_full_blocks is None else int(max_full_blocks)

    DQKp = DQK + 8  # padded LDS row stride for Q/K
    DVp = DV + 8  # padded LDS row stride for V/dO

    SCALE = float(DQK) ** -0.5 if scale is None else float(scale)
    SC2 = SCALE * _LOG2E
    LSE_IN_LOG2 = bool(lse_in_log2)
    BMB = int(block_mask_batch)
    BMH = int(block_mask_heads)
    CAUSAL_PARTIAL = bool(causal_partial_blocks)
    MASK_PROGRAM = tuple(mask_program)
    MASK_PROGRAM_OUTPUT = int(mask_program_output)
    MASK_BUFFER_SHAPES = tuple(tuple(shape) for shape in mask_buffer_shapes)
    MASK_BUFFER_STRIDES = tuple(tuple(stride) for stride in mask_buffer_strides)
    MASK_BUFFER_COUNT = len(MASK_BUFFER_SHAPES)
    CAUSAL_DOCUMENT_MASK = is_causal_document_mask_program(
        MASK_PROGRAM,
        MASK_PROGRAM_OUTPUT,
        MASK_BUFFER_STRIDES,
    )
    MASK_BUFFER_SIZES = tuple(
        1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
        for shape, strides in zip(MASK_BUFFER_SHAPES, MASK_BUFFER_STRIDES)
    )

    def contiguous_stride(dim):
        return (H * S * dim, S * dim, dim, 1)

    Q_STRIDE = tuple(q_stride or contiguous_stride(DQK))
    K_STRIDE = tuple(k_stride or contiguous_stride(DQK))
    V_STRIDE = tuple(v_stride or contiguous_stride(DV))
    OUT_STRIDE = tuple(out_stride or contiguous_stride(DV))
    DO_STRIDE = tuple(do_stride or contiguous_stride(DV))
    DQ_STRIDE = tuple(dq_stride or contiguous_stride(DQK))
    DK_STRIDE = tuple(dk_stride or contiguous_stride(DQK))
    DV_STRIDE = tuple(dv_stride or contiguous_stride(DV))

    BH = B * H
    LSTRIDE = S

    gview = make_global_view
    sview = make_shared_view

    LPR = DV // 8  # lanes cooperating on one row
    DROWS = NT // LPR  # rows per block
    DGRID = S // DROWS
    DSH = []
    _s = 1
    while _s < LPR:
        DSH.append(_s)
        _s *= 2

    @flyc.kernel
    def delta_kernel(OUT: fx.Tensor, DO: fx.Tensor, DELTA: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        bh = fx.block_idx.y
        batch = bh // fx.Int32(H)
        head = bh % fx.Int32(H)
        out_off = batch * fx.Int32(OUT_STRIDE[0]) + head * fx.Int32(OUT_STRIDE[1])
        do_off = batch * fx.Int32(DO_STRIDE[0]) + head * fx.Int32(DO_STRIDE[1])
        gO = gview(OUT, out_off, (S, DV), (OUT_STRIDE[2], OUT_STRIDE[3]))
        gDo = gview(DO, do_off, (S, DV), (DO_STRIDE[2], DO_STRIDE[3]))
        gDl = gview(DELTA, bh * fx.Int32(S), S, 1)

        chunk = tid % fx.Int32(LPR)
        row = bid * fx.Int32(DROWS) + tid // fx.Int32(LPR)

        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        fo = fx.make_rmem_tensor(8, fx.BFloat16)
        fd = fx.make_rmem_tensor(8, fx.BFloat16)
        orow = fx.logical_divide(fx.slice(gO, (row, None)), fx.make_layout(8, 1))
        drow = fx.logical_divide(fx.slice(gDo, (row, None)), fx.make_layout(8, 1))
        fx.copy(atom, fx.slice(orow, (None, chunk)), fo)
        fx.copy(atom, fx.slice(drow, (None, chunk)), fd)
        prod = fx.Vector(fo.load()).to(fx.Float32) * fx.Vector(fd.load()).to(fx.Float32)
        acc = _f32(prod[0])
        for i in fx.range_constexpr(7):
            acc = acc + _f32(prod[i + 1])
        for sh in DSH:
            acc = acc + _f32(fx.gpu.shuffle_xor(acc, sh, 64))
        satom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        fs = fx.make_rmem_tensor(1, fx.Float32)
        fs.store(fx.Vector.from_elements([acc], fx.Float32).ir_value())
        if chunk == fx.Int32(0):
            fx.copy(
                satom,
                fs,
                fx.slice(fx.logical_divide(gDl, fx.make_layout(1, 1)), (None, row)),
            )

    FLAG_N = ((NB * NB + 1 + PNT - 1) // PNT) * PNT
    FILL_IT = (NB * NB + PNT - 1) // PNT

    @fx.struct
    class PSmem:
        flags: fx.Array[fx.Int32, FLAG_N, 16]

    @flyc.kernel
    def prologue(
        KVN: fx.Tensor,
        KVI: fx.Tensor,
        FKVN: fx.Tensor,
        FKVI: fx.Tensor,
        CP_Q: fx.Tensor,
        IP_Q: fx.Tensor,
        CF_Q: fx.Tensor,
        IF_Q: fx.Tensor,
        CP_K: fx.Tensor,
        IP_K: fx.Tensor,
        CF_K: fx.Tensor,
        IF_K: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        bh = fx.block_idx.x
        batch = bh // fx.Int32(H)
        head = bh % fx.Int32(H)
        mask_b = fx.Int32(0) if BMB == 1 else batch
        mask_h = fx.Int32(0) if BMH == 1 else head
        mask_group = mask_b * fx.Int32(BMH) + mask_h
        lds = fx.SharedAllocator().allocate(PSmem).peek()
        fp = lds.flags.ptr

        gKVN = gview(KVN, mask_group * fx.Int32(NB), NB, 1)
        gKVI = gview(
            KVI,
            mask_group * fx.Int32(NB * MAX_PARTIAL),
            NB * MAX_PARTIAL,
            1,
        )
        gFKVN = gview(FKVN, mask_group * fx.Int32(NB), NB, 1)
        gFKVI = gview(
            FKVI,
            mask_group * fx.Int32(NB * MAX_FULL),
            NB * MAX_FULL,
            1,
        )
        gCPQ = gview(CP_Q, bh * fx.Int32(MC), MC, 1)
        gCFQ = gview(CF_Q, bh * fx.Int32(MC), MC, 1)
        gIPQ = gview(IP_Q, bh * fx.Int32(MC * NC), MC * NC, 1)
        gIFQ = gview(IF_Q, bh * fx.Int32(MC * NC), MC * NC, 1)
        gCPK = gview(CP_K, bh * fx.Int32(NC), NC, 1)
        gCFK = gview(CF_K, bh * fx.Int32(NC), NC, 1)
        gIPK = gview(IP_K, bh * fx.Int32(NC * MC), NC * MC, 1)
        gIFK = gview(IF_K, bh * fx.Int32(NC * MC), NC * MC, 1)

        i32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)

        def gload_i32(view, idx):
            return load_scalar(i32atom, view, idx, fx.Int32)

        def gstore_i32(view, idx, val):
            store_scalar(i32atom, view, idx, val, fx.Int32)

        # zero flags (+1 dump slot)
        zero = fx.Int32(0)
        for l in fx.range_constexpr(FLAG_N // PNT):
            fx.ptr_store(zero, fp + (fx.Int32(l * PNT) + tid))
        fx.gpu.barrier()

        # scatter block flags: 1 = partially masked, 2 = fully unmasked
        one = fx.Int32(1)
        two = fx.Int32(2)
        dump = fx.Int32(NB * NB)
        for l in fx.range_constexpr(FILL_IT):
            i = fx.Int32(l * PNT) + tid
            mb_raw = i // fx.Int32(NB)
            mb = (mb_raw < fx.Int32(NB)).select(mb_raw, fx.Int32(NB - 1))
            t = i % fx.Int32(NB)
            cnt = gload_i32(gKVN, mb)
            tp = (t < fx.Int32(MAX_PARTIAL)).select(t, fx.Int32(MAX_PARTIAL - 1))
            nb = gload_i32(
                gKVI,
                mb * fx.Int32(MAX_PARTIAL) + tp,
            )
            partial_active = (t < cnt) & (t < fx.Int32(MAX_PARTIAL))
            addr = partial_active.select(mb * fx.Int32(NB) + nb, dump)
            fx.ptr_store(one, fp + addr)
            cntf = gload_i32(gFKVN, mb)
            tf = (t < fx.Int32(MAX_FULL)).select(t, fx.Int32(MAX_FULL - 1))
            nbf = gload_i32(
                gFKVI,
                mb * fx.Int32(MAX_FULL) + tf,
            )
            full_active = (t < cntf) & (t < fx.Int32(MAX_FULL))
            addrf = full_active.select(mb * fx.Int32(NB) + nbf, dump)
            fx.ptr_store(two, fp + addrf)
        fx.gpu.barrier()

        # compaction: thread `mc` owns q chunk mc and kv chunk mc (MC == NC)
        mc = (tid < fx.Int32(MC)).select(tid, fx.Int32(MC - 1))
        mb = mc // fx.Int32(CPB)
        c1 = fx.Int32(0)
        c2 = fx.Int32(0)
        d1 = fx.Int32(0)
        d2 = fx.Int32(0)
        rowq = mc * fx.Int32(NC)
        rowk = mc * fx.Int32(MC)
        for nb in fx.range_constexpr(NB):
            fq = fx.ptr_load(fp + (mb * fx.Int32(NB) + fx.Int32(nb)))
            fk = fx.ptr_load(fp + (fx.Int32(nb * NB) + mb))
            for s in fx.range_constexpr(CPB):
                ch = fx.Int32(nb * CPB + s)
                gstore_i32(gIPQ, rowq + c1, ch)
                c1 = c1 + (fq == one).select(one, zero)
                gstore_i32(gIFQ, rowq + c2, ch)
                c2 = c2 + (fq == two).select(one, zero)
                gstore_i32(gIPK, rowk + d1, ch)
                d1 = d1 + (fk == one).select(one, zero)
                gstore_i32(gIFK, rowk + d2, ch)
                d2 = d2 + (fk == two).select(one, zero)
        gstore_i32(gCPQ, mc, c1)
        gstore_i32(gCFQ, mc, c2)
        gstore_i32(gCPK, mc, d1)
        gstore_i32(gCFK, mc, d2)

    VPT = 8
    QK_DCH = DQK // VPT
    V_DCH = DV // VPT
    QK_LOAD_IT = (BM * DQK // VPT) // NT
    V_LOAD_IT = (BM * DV // VPT) // NT
    DQ_QK_LOAD_IT = (DQ_BN * DQK // VPT) // NT
    DQ_V_LOAD_IT = (DQ_BN * DV // VPT) // NT
    DQ_SUBTILES = BN // DQ_BN

    CE = [16 * (e // 4) + (e % 4) for e in range(16)]
    DQ_CE = CE[: DQ_BN // 4]

    @fx.struct
    class KvSmem:
        a: fx.Array[fx.BFloat16, BM * DQKp, 16]
        b: fx.Array[fx.BFloat16, BM * DVp, 16]
        ld: fx.Array[fx.Float32, 2 * BM, 16]

    @fx.struct
    class QSmem:
        a: fx.Array[fx.BFloat16, BN * DQKp, 16]
        b: fx.Array[fx.BFloat16, BN * DVp, 16]

    @flyc.kernel
    def dkdv_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DO: fx.Tensor,
        DK: fx.Tensor,
        DVALUE: fx.Tensor,
        CP: fx.Tensor,
        IP: fx.Tensor,
        CF: fx.Tensor,
        IF: fx.Tensor,
        MaskBuffer0: fx.Tensor,
        MaskBuffer1: fx.Tensor,
        MaskBuffer2: fx.Tensor,
        MaskBuffer3: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        nc = fx.block_idx.x
        bh = fx.block_idx.y
        batch = bh // fx.Int32(H)
        head = bh % fx.Int32(H)
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        g = lane // fx.Int32(16)

        loff = bh * fx.Int32(LSTRIDE)
        kvbase = nc * fx.Int32(BN)
        qoff = batch * fx.Int32(Q_STRIDE[0]) + head * fx.Int32(Q_STRIDE[1])
        koff = batch * fx.Int32(K_STRIDE[0]) + head * fx.Int32(K_STRIDE[1])
        voff = batch * fx.Int32(V_STRIDE[0]) + head * fx.Int32(V_STRIDE[1])
        dooff = batch * fx.Int32(DO_STRIDE[0]) + head * fx.Int32(DO_STRIDE[1])
        dkoff = batch * fx.Int32(DK_STRIDE[0]) + head * fx.Int32(DK_STRIDE[1])
        dvoff = batch * fx.Int32(DV_STRIDE[0]) + head * fx.Int32(DV_STRIDE[1])

        lds = fx.SharedAllocator().allocate(KvSmem).peek()
        pa, pb, pld = lds.a.ptr, lds.b.ptr, lds.ld.ptr

        gQ = gview(Q, qoff, (S, DQK), (Q_STRIDE[2], Q_STRIDE[3]))
        gDO = gview(DO, dooff, (S, DV), (DO_STRIDE[2], DO_STRIDE[3]))
        gK = gview(K, koff, (S, DQK), (K_STRIDE[2], K_STRIDE[3]))
        gV = gview(V, voff, (S, DV), (V_STRIDE[2], V_STRIDE[3]))
        gLSE = gview(LSE, loff, S, 1)
        gDEL = gview(DELTA, loff, S, 1)
        gCP = gview(CP, bh * fx.Int32(NC), NC, 1)
        gCF = gview(CF, bh * fx.Int32(NC), NC, 1)
        gIP = gview(IP, bh * fx.Int32(NC * MC), NC * MC, 1)
        gIF = gview(IF, bh * fx.Int32(NC * MC), NC * MC, 1)
        MaskBuffers = []
        if const_expr(MASK_BUFFER_COUNT >= 1):
            MaskBuffers.append(
                gview(
                    MaskBuffer0,
                    None,
                    MASK_BUFFER_SIZES[0],
                    1,
                )
            )
        if const_expr(MASK_BUFFER_COUNT >= 2):
            MaskBuffers.append(
                gview(
                    MaskBuffer1,
                    None,
                    MASK_BUFFER_SIZES[1],
                    1,
                )
            )
        if const_expr(MASK_BUFFER_COUNT >= 3):
            MaskBuffers.append(
                gview(
                    MaskBuffer2,
                    None,
                    MASK_BUFFER_SIZES[2],
                    1,
                )
            )
        if const_expr(MASK_BUFFER_COUNT >= 4):
            MaskBuffers.append(
                gview(
                    MaskBuffer3,
                    None,
                    MASK_BUFFER_SIZES[3],
                    1,
                )
            )

        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        mma1 = fx.make_tiled_mma(atom, fx.make_layout((1, NW, 1), (0, 1, 0)))
        mma2 = fx.make_tiled_mma(atom, fx.make_layout((NW, 1, 1), (1, 0, 0)))
        t1 = mma1.thr_slice(tid)
        t2 = mma2.thr_slice(tid)

        g128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        u64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        tr16 = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans(16, 64),
            fx.BFloat16,
        )
        i32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        f32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        o16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

        def gload_i32(view, idx):
            return load_scalar(i32atom, view, idx, fx.Int32)

        def gload_f32(view, idx):
            return load_scalar(f32atom, view, idx, fx.Float32)

        def load_qk_tile(gsrc, rbase, pn):
            for l in fx.range_constexpr(QK_LOAD_IT):
                i = fx.Int32(l * NT) + tid
                m = i // fx.Int32(QK_DCH)
                c = i % fx.Int32(QK_DCH)
                fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
                rw = fx.logical_divide(
                    fx.slice(gsrc, (rbase + m, None)), fx.make_layout(VPT, 1)
                )
                fx.copy(g128, fx.slice(rw, (None, c)), fr)
                v = fx.Vector(fr.load())
                fx.ptr_store(v, pn + (m * fx.Int32(DQKp) + c * fx.Int32(VPT)))

        def load_v_tile(gsrc, rbase, pn):
            for l in fx.range_constexpr(V_LOAD_IT):
                i = fx.Int32(l * NT) + tid
                m = i // fx.Int32(V_DCH)
                c = i % fx.Int32(V_DCH)
                fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
                rw = fx.logical_divide(
                    fx.slice(gsrc, (rbase + m, None)), fx.make_layout(VPT, 1)
                )
                fx.copy(g128, fx.slice(rw, (None, c)), fr)
                v = fx.Vector(fr.load())
                fx.ptr_store(v, pn + (m * fx.Int32(DVp) + c * fx.Int32(VPT)))

        sK = sview(pa, (BN, DQK), (DQKp, 1))
        sV = sview(pb, (BN, DV), (DVp, 1))
        for l in fx.range_constexpr(QK_LOAD_IT):
            i = fx.Int32(l * NT) + tid
            m = i // fx.Int32(QK_DCH)
            c = i % fx.Int32(QK_DCH)
            fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
            rw = fx.logical_divide(
                fx.slice(gK, (kvbase + m, None)), fx.make_layout(VPT, 1)
            )
            fx.copy(g128, fx.slice(rw, (None, c)), fr)
            fx.ptr_store(
                fx.Vector(fr.load()),
                pa + (m * fx.Int32(DQKp) + c * fx.Int32(VPT)),
            )
        for l in fx.range_constexpr(V_LOAD_IT):
            i = fx.Int32(l * NT) + tid
            m = i // fx.Int32(V_DCH)
            c = i % fx.Int32(V_DCH)
            fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
            rw2 = fx.logical_divide(
                fx.slice(gV, (kvbase + m, None)), fx.make_layout(VPT, 1)
            )
            fx.copy(g128, fx.slice(rw2, (None, c)), fr)
            fx.ptr_store(
                fx.Vector(fr.load()),
                pb + (m * fx.Int32(DVp) + c * fx.Int32(VPT)),
            )
        fx.gpu.barrier()

        tcB1 = fx.make_tiled_copy_B(u64, mma1).get_slice(tid)
        fBK = t1.make_fragment_B(sK)
        fBV = t1.make_fragment_B(sV)
        fx.copy(u64, tcB1.partition_S(sK), tcB1.retile(fBK))
        fx.copy(u64, tcB1.partition_S(sV), tcB1.retile(fBV))
        fx.gpu.barrier()

        gDKb = fx.Tensor(
            fx.make_view(
                fx.add_offset(
                    fx.get_iter(fx.rocdl.make_buffer_tensor(DK)),
                    dkoff + kvbase * fx.Int32(DK_STRIDE[2]),
                ),
                fx.make_layout((BN, DQK), (DK_STRIDE[2], DK_STRIDE[3])),
            )
        )
        gDVb = fx.Tensor(
            fx.make_view(
                fx.add_offset(
                    fx.get_iter(fx.rocdl.make_buffer_tensor(DVALUE)),
                    dvoff + kvbase * fx.Int32(DV_STRIDE[2]),
                ),
                fx.make_layout((BN, DV), (DV_STRIDE[2], DV_STRIDE[3])),
            )
        )

        fC3 = t2.make_fragment_C(gDVb)  # dV
        fC4 = t2.make_fragment_C(gDKb)  # dK
        fC3.fill(0)
        fC4.fill(0)
        DK_NACC = (BN // NW) * DQK // 64

        sQn = sview(pa, (BM, DQK), (DQKp, 1))
        sDn = sview(pb, (BM, DV), (DVp, 1))
        sQt = sview(pa, (DQK, BM), (1, DQKp))
        sDt = sview(pb, (DV, BM), (1, DVp))

        tcA1 = fx.make_tiled_copy_A(u64, mma1).get_slice(tid)
        tcB2 = fx.make_tiled_copy_B(tr16, mma2).get_slice(tid)

        fA1 = t1.make_fragment_A(sQn)
        fA2 = t1.make_fragment_A(sDn)
        fB3 = t2.make_fragment_B(sDt)
        fB4 = t2.make_fragment_B(sQt)
        fST = fx.Tensor(
            fx.make_view(fx.get_iter(sQn), fx.make_layout((BN, BM), (BM, 1)))
        )
        fAP = t2.make_fragment_A(fST)
        fAS = t2.make_fragment_A(fST)

        kcol = fx.Int32(16) * wave + lane % fx.Int32(16)
        kj = kvbase + kcol

        def evaluate_mask(q_pos, kv_pos):
            return evaluate_mask_program(
                mask_program=MASK_PROGRAM,
                mask_program_output=MASK_PROGRAM_OUTPUT,
                mask_buffer_strides=MASK_BUFFER_STRIDES,
                mask_buffers=MaskBuffers,
                load_i32=gload_i32,
                batch=batch,
                head=head,
                q_pos=q_pos,
                kv_pos=kv_pos,
            )

        def tile(mchunk, masked):
            qbase = mchunk * fx.Int32(BM)
            r = (tid < fx.Int32(BM)).select(tid, fx.Int32(BM - 1))
            lv = gload_f32(gLSE, qbase + r)
            dv_ = gload_f32(gDEL, qbase + r)
            l2v = lv if LSE_IN_LOG2 else lv * _f32(_LOG2E)
            l2 = (lv < _f32(_NEG_BIG)).select(_f32(0.0), l2v)
            fx.ptr_store(l2, pld + r)
            fx.ptr_store(dv_, pld + (fx.Int32(BM) + r))
            load_qk_tile(gQ, qbase, pa)
            load_v_tile(gDO, qbase, pb)
            fx.gpu.barrier()

            fC1 = t1.make_fragment_C(fST)
            fC2 = t1.make_fragment_C(fST)
            fx.copy(u64, tcA1.partition_S(sQn), tcA1.retile(fA1))
            fx.copy(u64, tcA1.partition_S(sDn), tcA1.retile(fA2))
            fC1.fill(0)
            fC2.fill(0)
            fx.gemm(atom, fC1, fA1, fBK, fC1)
            fx.gemm(atom, fC2, fA2, fBV, fC2)

            g4 = fx.Int32(4) * g
            qrow0 = qbase + g4

            vs = fx.Vector(fC1.load())
            vp = fx.Vector(fC2.load())
            pl = []
            dsl = []
            for e in fx.range_constexpr(16):
                rr = g4 + fx.Int32(CE[e])
                l2e = fx.ptr_load(pld + rr)
                de = fx.ptr_load(pld + (fx.Int32(BM) + rr))
                x = _f32(vs[e]) * _f32(SC2) - l2e
                p = _exp2(x)
                if masked:
                    q_pos = qrow0 + fx.Int32(CE[e])
                    if const_expr(CAUSAL_DOCUMENT_MASK):
                        document_id = gload_i32(
                            MaskBuffers[0],
                            q_pos * fx.Int32(MASK_BUFFER_STRIDES[0][0]),
                        )
                        document_start = gload_i32(
                            MaskBuffers[1],
                            document_id * fx.Int32(MASK_BUFFER_STRIDES[1][0]),
                        )
                        keep = (q_pos >= kj) & (kj >= document_start)
                        p = keep.select(p, _f32(0.0))
                    elif const_expr(bool(MASK_PROGRAM)):
                        keep = evaluate_mask(q_pos, kj)
                        p = keep.select(p, _f32(0.0))
                    elif const_expr(CAUSAL_PARTIAL):
                        keep = q_pos >= kj
                        p = keep.select(p, _f32(0.0))
                pl.append(p)
                dsl.append(p * (_f32(vp[e]) - de))
            fAP.store(
                fx.Vector.from_elements(pl, fx.Float32).to(fx.BFloat16).ir_value()
            )
            fAS.store(
                fx.Vector.from_elements(dsl, fx.Float32).to(fx.BFloat16).ir_value()
            )

            fx.copy(tr16, tcB2.partition_S(sDt), tcB2.retile(fB3))
            fx.copy(tr16, tcB2.partition_S(sQt), tcB2.retile(fB4))
            fx.gemm(atom, fC3, fAP, fB3, fC3)
            fx.gemm(atom, fC4, fAS, fB4, fC4)
            fx.gpu.barrier()

        cntp = gload_i32(gCP, nc)
        cntf = gload_i32(gCF, nc)
        rowk = nc * fx.Int32(MC)

        for ti in range(fx.Int32(0), cntp, fx.Int32(1)):
            tile(gload_i32(gIP, rowk + fx.Int32(ti)), True)
        for ti in range(fx.Int32(0), cntf, fx.Int32(1)):
            tile(gload_i32(gIF, rowk + fx.Int32(ti)), False)

        vdk = fx.Vector(fC4.load()) * fx.Vector.filled(
            DK_NACC, SCALE, fx.Float32
        )
        fC4.store(vdk.ir_value())
        tcC2 = fx.make_tiled_copy_C(o16, mma2).get_slice(tid)
        oK = fx.make_fragment_like(fC4, fx.BFloat16.ir_type)
        oV = fx.make_fragment_like(fC3, fx.BFloat16.ir_type)
        oK.store(fx.Vector(fC4.load()).to(fx.BFloat16).ir_value())
        oV.store(fx.Vector(fC3.load()).to(fx.BFloat16).ir_value())
        fx.copy(o16, tcC2.retile(oK), tcC2.partition_S(gDKb))
        fx.copy(o16, tcC2.retile(oV), tcC2.partition_S(gDVb))

    @flyc.kernel
    def dq_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DO: fx.Tensor,
        DQ: fx.Tensor,
        CP: fx.Tensor,
        IP: fx.Tensor,
        CF: fx.Tensor,
        IF: fx.Tensor,
        MaskBuffer0: fx.Tensor,
        MaskBuffer1: fx.Tensor,
        MaskBuffer2: fx.Tensor,
        MaskBuffer3: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        mc = fx.block_idx.x
        bh = fx.block_idx.y
        batch = bh // fx.Int32(H)
        head = bh % fx.Int32(H)
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        g = lane // fx.Int32(16)

        loff = bh * fx.Int32(LSTRIDE)
        qbase = mc * fx.Int32(BM)
        qoff = batch * fx.Int32(Q_STRIDE[0]) + head * fx.Int32(Q_STRIDE[1])
        koff = batch * fx.Int32(K_STRIDE[0]) + head * fx.Int32(K_STRIDE[1])
        voff = batch * fx.Int32(V_STRIDE[0]) + head * fx.Int32(V_STRIDE[1])
        dooff = batch * fx.Int32(DO_STRIDE[0]) + head * fx.Int32(DO_STRIDE[1])
        dqoff = batch * fx.Int32(DQ_STRIDE[0]) + head * fx.Int32(DQ_STRIDE[1])

        lds = fx.SharedAllocator().allocate(QSmem).peek()
        pa, pb = lds.a.ptr, lds.b.ptr

        gQ = gview(Q, qoff, (S, DQK), (Q_STRIDE[2], Q_STRIDE[3]))
        gDO = gview(DO, dooff, (S, DV), (DO_STRIDE[2], DO_STRIDE[3]))
        gK = gview(K, koff, (S, DQK), (K_STRIDE[2], K_STRIDE[3]))
        gV = gview(V, voff, (S, DV), (V_STRIDE[2], V_STRIDE[3]))
        gLSE = gview(LSE, loff, S, 1)
        gDEL = gview(DELTA, loff, S, 1)
        gCP = gview(CP, bh * fx.Int32(MC), MC, 1)
        gCF = gview(CF, bh * fx.Int32(MC), MC, 1)
        gIP = gview(IP, bh * fx.Int32(MC * NC), MC * NC, 1)
        gIF = gview(IF, bh * fx.Int32(MC * NC), MC * NC, 1)
        MaskBuffers = []
        if const_expr(MASK_BUFFER_COUNT >= 1):
            MaskBuffers.append(
                gview(
                    MaskBuffer0,
                    None,
                    MASK_BUFFER_SIZES[0],
                    1,
                )
            )
        if const_expr(MASK_BUFFER_COUNT >= 2):
            MaskBuffers.append(
                gview(
                    MaskBuffer1,
                    None,
                    MASK_BUFFER_SIZES[1],
                    1,
                )
            )
        if const_expr(MASK_BUFFER_COUNT >= 3):
            MaskBuffers.append(
                gview(
                    MaskBuffer2,
                    None,
                    MASK_BUFFER_SIZES[2],
                    1,
                )
            )
        if const_expr(MASK_BUFFER_COUNT >= 4):
            MaskBuffers.append(
                gview(
                    MaskBuffer3,
                    None,
                    MASK_BUFFER_SIZES[3],
                    1,
                )
            )

        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        mma1 = fx.make_tiled_mma(atom, fx.make_layout((1, NW, 1), (0, 1, 0)))
        mma2 = fx.make_tiled_mma(atom, fx.make_layout((NW, 1, 1), (1, 0, 0)))
        t1 = mma1.thr_slice(tid)
        t2 = mma2.thr_slice(tid)

        g128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        u64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        tr16 = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans(16, 64),
            fx.BFloat16,
        )
        i32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        f32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        o16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

        def gload_i32(view, idx):
            return load_scalar(i32atom, view, idx, fx.Int32)

        def gload_f32(view, idx):
            return load_scalar(f32atom, view, idx, fx.Float32)

        sQ = sview(pa, (BM, DQK), (DQKp, 1))
        sD = sview(pb, (BM, DV), (DVp, 1))
        for l in fx.range_constexpr(QK_LOAD_IT):
            i = fx.Int32(l * NT) + tid
            m = i // fx.Int32(QK_DCH)
            c = i % fx.Int32(QK_DCH)
            fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
            rw = fx.logical_divide(
                fx.slice(gQ, (qbase + m, None)), fx.make_layout(VPT, 1)
            )
            fx.copy(g128, fx.slice(rw, (None, c)), fr)
            fx.ptr_store(
                fx.Vector(fr.load()),
                pa + (m * fx.Int32(DQKp) + c * fx.Int32(VPT)),
            )
        for l in fx.range_constexpr(V_LOAD_IT):
            i = fx.Int32(l * NT) + tid
            m = i // fx.Int32(V_DCH)
            c = i % fx.Int32(V_DCH)
            fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
            rw2 = fx.logical_divide(
                fx.slice(gDO, (qbase + m, None)), fx.make_layout(VPT, 1)
            )
            fx.copy(g128, fx.slice(rw2, (None, c)), fr)
            fx.ptr_store(
                fx.Vector(fr.load()),
                pb + (m * fx.Int32(DVp) + c * fx.Int32(VPT)),
            )
        fx.gpu.barrier()

        tcB1 = fx.make_tiled_copy_B(u64, mma1).get_slice(tid)
        fBQ = t1.make_fragment_B(sQ)
        fBD = t1.make_fragment_B(sD)
        fx.copy(u64, tcB1.partition_S(sQ), tcB1.retile(fBQ))
        fx.copy(u64, tcB1.partition_S(sD), tcB1.retile(fBD))

        qcol = fx.Int32(16) * wave + lane % fx.Int32(16)
        qi = qbase + qcol
        lv = gload_f32(gLSE, qi)
        de = gload_f32(gDEL, qi)
        if const_expr(CAUSAL_DOCUMENT_MASK):
            document_id = gload_i32(
                MaskBuffers[0],
                qi * fx.Int32(MASK_BUFFER_STRIDES[0][0]),
            )
            document_start = gload_i32(
                MaskBuffers[1],
                document_id * fx.Int32(MASK_BUFFER_STRIDES[1][0]),
            )
        l2v = lv if LSE_IN_LOG2 else lv * _f32(_LOG2E)
        l2e = (lv < _f32(_NEG_BIG)).select(_f32(0.0), l2v)
        fx.gpu.barrier()

        gDQb = fx.Tensor(
            fx.make_view(
                fx.add_offset(
                    fx.get_iter(fx.rocdl.make_buffer_tensor(DQ)),
                    dqoff + qbase * fx.Int32(DQ_STRIDE[2]),
                ),
                fx.make_layout((BM, DQK), (DQ_STRIDE[2], DQ_STRIDE[3])),
            )
        )
        fC3 = t2.make_fragment_C(gDQb)
        fC3.fill(0)
        DQ_NACC = (BM // NW) * DQK // 64

        sKn = sview(pa, (DQ_BN, DQK), (DQKp, 1))
        sVn = sview(pb, (DQ_BN, DV), (DVp, 1))
        sKt = sview(pa, (DQK, DQ_BN), (1, DQKp))

        tcA1 = fx.make_tiled_copy_A(u64, mma1).get_slice(tid)
        tcB2 = fx.make_tiled_copy_B(tr16, mma2).get_slice(tid)
        fA1 = t1.make_fragment_A(sKn)
        fA2 = t1.make_fragment_A(sVn)
        fB3 = t2.make_fragment_B(sKt)
        fSTc = fx.Tensor(
            fx.make_view(
                fx.get_iter(sKn),
                fx.make_layout((DQ_BN, BM), (BM, 1)),
            )
        )
        fSTa = fx.Tensor(
            fx.make_view(
                fx.get_iter(sKn),
                fx.make_layout((BM, DQ_BN), (DQ_BN, 1)),
            )
        )
        fAS_template = t2.make_fragment_A(fSTa)
        fAS_layout = fx.get_layout(fAS_template)

        def evaluate_mask(q_pos, kv_pos):
            return evaluate_mask_program(
                mask_program=MASK_PROGRAM,
                mask_program_output=MASK_PROGRAM_OUTPUT,
                mask_buffer_strides=MASK_BUFFER_STRIDES,
                mask_buffers=MaskBuffers,
                load_i32=gload_i32,
                batch=batch,
                head=head,
                q_pos=q_pos,
                kv_pos=kv_pos,
            )

        def subtile(kchunk, sub, masked):
            kvbase = kchunk * fx.Int32(BN) + sub * fx.Int32(DQ_BN)
            for l in fx.range_constexpr(DQ_QK_LOAD_IT):
                i = fx.Int32(l * NT) + tid
                m = i // fx.Int32(QK_DCH)
                c = i % fx.Int32(QK_DCH)
                fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
                rw = fx.logical_divide(
                    fx.slice(gK, (kvbase + m, None)), fx.make_layout(VPT, 1)
                )
                fx.copy(g128, fx.slice(rw, (None, c)), fr)
                v = fx.Vector(fr.load())
                fx.ptr_store(v, pa + (m * fx.Int32(DQKp) + c * fx.Int32(VPT)))
            for l in fx.range_constexpr(DQ_V_LOAD_IT):
                i = fx.Int32(l * NT) + tid
                m = i // fx.Int32(V_DCH)
                c = i % fx.Int32(V_DCH)
                fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
                rw2 = fx.logical_divide(
                    fx.slice(gV, (kvbase + m, None)), fx.make_layout(VPT, 1)
                )
                fx.copy(g128, fx.slice(rw2, (None, c)), fr)
                fx.ptr_store(
                    fx.Vector(fr.load()),
                    pb + (m * fx.Int32(DVp) + c * fx.Int32(VPT)),
                )
            fx.gpu.barrier()

            fC1 = t1.make_fragment_C(fSTc)
            fC2 = t1.make_fragment_C(fSTc)
            fx.copy(u64, tcA1.partition_S(sKn), tcA1.retile(fA1))
            fx.copy(u64, tcA1.partition_S(sVn), tcA1.retile(fA2))
            fC1.fill(0)
            fC2.fill(0)
            fx.gemm(atom, fC1, fA1, fBQ, fC1)
            fx.gemm(atom, fC2, fA2, fBD, fC2)

            g4 = fx.Int32(4) * g
            krow0 = kvbase + g4

            vs = fx.Vector(fC1.load())
            vp = fx.Vector(fC2.load())
            dsl = []
            for e in fx.range_constexpr(len(DQ_CE)):
                x = _f32(vs[e]) * _f32(SC2) - l2e
                p = _exp2(x)
                if masked:
                    kv_pos = krow0 + fx.Int32(DQ_CE[e])
                    if const_expr(CAUSAL_DOCUMENT_MASK):
                        keep = (qi >= kv_pos) & (kv_pos >= document_start)
                        p = keep.select(p, _f32(0.0))
                    elif const_expr(bool(MASK_PROGRAM)):
                        keep = evaluate_mask(qi, kv_pos)
                        p = keep.select(p, _f32(0.0))
                    elif const_expr(CAUSAL_PARTIAL):
                        keep = qi >= kv_pos
                        p = keep.select(p, _f32(0.0))
                dsl.append(p * (_f32(vp[e]) - de))
            fAS = fAS_template
            if const_expr(DQ_BN == 32):
                fAS = fx.Tensor(
                    fx.make_view(
                        fx.recast_iter(fx.BFloat16, fx.get_iter(fC2)),
                        fAS_layout,
                    )
                )
            fAS.store(
                fx.Vector.from_elements(dsl, fx.Float32).to(fx.BFloat16).ir_value()
            )

            fx.copy(tr16, tcB2.partition_S(sKt), tcB2.retile(fB3))
            fx.gemm(atom, fC3, fAS, fB3, fC3)
            fx.gpu.barrier()

        def tile(kchunk, masked):
            for sub in range(
                fx.Int32(0),
                fx.Int32(DQ_SUBTILES),
                fx.Int32(1),
            ):
                subtile(kchunk, fx.Int32(sub), masked)

        cntp = gload_i32(gCP, mc)
        cntf = gload_i32(gCF, mc)
        rowq = mc * fx.Int32(NC)
        for ti in range(fx.Int32(0), cntp, fx.Int32(1)):
            tile(gload_i32(gIP, rowq + fx.Int32(ti)), True)
        for ti in range(fx.Int32(0), cntf, fx.Int32(1)):
            tile(gload_i32(gIF, rowq + fx.Int32(ti)), False)

        vq = fx.Vector(fC3.load()) * fx.Vector.filled(
            DQ_NACC, SCALE, fx.Float32
        )
        fC3.store(vq.ir_value())
        tcC2 = fx.make_tiled_copy_C(o16, mma2).get_slice(tid)
        oQ = fx.make_fragment_like(fC3, fx.BFloat16.ir_type)
        oQ.store(fx.Vector(fC3.load()).to(fx.BFloat16).ir_value())
        fx.copy(o16, tcC2.retile(oQ), tcC2.partition_S(gDQb))

    @flyc.jit
    def _launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        OUT: fx.Tensor,
        LSE: fx.Tensor,
        DO: fx.Tensor,
        DQ: fx.Tensor,
        DK: fx.Tensor,
        DVALUE: fx.Tensor,
        KVN: fx.Tensor,
        KVI: fx.Tensor,
        FKVN: fx.Tensor,
        FKVI: fx.Tensor,
        DELTA: fx.Tensor,
        CP_Q: fx.Tensor,
        IP_Q: fx.Tensor,
        CF_Q: fx.Tensor,
        IF_Q: fx.Tensor,
        CP_K: fx.Tensor,
        IP_K: fx.Tensor,
        CF_K: fx.Tensor,
        IF_K: fx.Tensor,
        MaskBuffer0: fx.Tensor,
        MaskBuffer1: fx.Tensor,
        MaskBuffer2: fx.Tensor,
        MaskBuffer3: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        delta_kernel(OUT, DO, DELTA).launch(
            grid=(DGRID, BH, 1), block=(NT, 1, 1), stream=stream
        )
        prologue(
            KVN, KVI, FKVN, FKVI, CP_Q, IP_Q, CF_Q, IF_Q, CP_K, IP_K, CF_K, IF_K
        ).launch(grid=(BH, 1, 1), block=(PNT, 1, 1), stream=stream)
        dkdv_kernel(
            Q,
            K,
            V,
            LSE,
            DELTA,
            DO,
            DK,
            DVALUE,
            CP_K,
            IP_K,
            CF_K,
            IF_K,
            MaskBuffer0,
            MaskBuffer1,
            MaskBuffer2,
            MaskBuffer3,
        ).launch(
            grid=(NC, BH, 1), block=(NT, 1, 1), stream=stream
        )
        dq_kernel(
            Q,
            K,
            V,
            LSE,
            DELTA,
            DO,
            DQ,
            CP_Q,
            IP_Q,
            CF_Q,
            IF_Q,
            MaskBuffer0,
            MaskBuffer1,
            MaskBuffer2,
            MaskBuffer3,
        ).launch(
            grid=(MC, BH, 1), block=(NT, 1, 1), stream=stream
        )

    return _launch
