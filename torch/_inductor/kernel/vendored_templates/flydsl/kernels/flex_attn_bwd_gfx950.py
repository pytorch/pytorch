"""FlexAttention backward implementation for AMD gfx950 GPUs.

``build_flex_attn_bwd_module`` returns a FlyDSL JIT launcher that computes the
query, key, and value gradients for BF16 attention. Query/key and value head
dimensions must be positive multiples of 32. BlockMask metadata uses 128 x 128
blocks, and tensor layouts are described by explicit strides.

The launcher submits the following kernels to the caller-provided stream:

1. ``delta_kernel`` computes the row-wise correction
   ``delta[b, h, i] = sum_d(out[b, h, i, d] * grad_out[b, h, i, d])``.
2. ``prologue`` is used only for arbitrary and document masks. It converts
   BlockMask metadata into compact query-major and key/value-major work lists.
3. ``compute_kernel`` dispatches workgroups to dQ owners or paired dK/dV
   owners. In a paired owner, producer waves compute ``P``, ``dS``, and dK;
   consumer waves reuse the staged ``P`` fragments for dV. Dense, causal, and
   fixed-window masks derive their tile ranges directly and skip the prologue.

For attention probabilities ``P`` and score gradients ``dS``, the kernel
computes:

    P  = exp2(Q @ K.T * (scale * log2(e)) - LSE * log2(e))
    dV = P.T @ dO
    dP = dO @ V.T
    dS = P * (dP - delta)
    dQ = (dS @ K) * scale
    dK = (dS.T @ Q) * scale

Masked probability entries are zero. Partial tiles evaluate the same bounded
mask bytecode as the forward kernel, including captured int32 mask buffers.

The MFMA32 pipeline uses the compatible accumulator and transposed-operand
fragment layouts of the 32 x 32 x 16 BF16 instruction. Probability and score
gradient fragments can therefore feed the gradient update directly from
registers without an intermediate LDS round trip.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .flex_attn_bwd_kv_owner import make_dkdv_mfma32_body
from .flex_attn_bwd_q_owner import make_dq_mfma32_body
from .flex_attn_utils import (
    fast_exp2,
    is_causal_document_mask_program,
    make_global_view,
)

_LOG2E = 1.4426950408889634
_BATCHED_CAUSAL_DOCUMENT_MASK_PROGRAM = (
    ("const_bool", True),
    ("ge", 2, 3),
    ("and", 4, 5),
    ("load_i32", 0, (0, 3)),
    ("le", 2, 7),
    ("and", 6, 8),
)
_DENSE_MASK_PROGRAM = (("const_i32", 0), ("ge", 2, 4))


def _f32(x):
    return fx.Float32(x)


def _causal_window_size(mask_program, mask_program_output):
    program = tuple(mask_program)
    if (
        len(program) == 5
        and program[0] == ("ge", 2, 3)
        and (program[1] == ("sub", 2, 3))
        and (len(program[2]) == 2)
        and (program[2][0] == "const_i32")
        and (program[3] == ("lt", 5, 6))
        and (program[4] == ("and", 4, 7))
        and (int(mask_program_output) == 8)
    ):
        window_size = int(program[2][1])
        return window_size if window_size > 0 else None
    return None


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
    mask_buffer_shapes = tuple((tuple(shape) for shape in mask_buffer_shapes))
    mask_buffer_strides = tuple((tuple(stride) for stride in mask_buffer_strides))
    batched_causal_document_mask = (
        tuple(mask_program) == _BATCHED_CAUSAL_DOCUMENT_MASK_PROGRAM
        and int(mask_program_output) == 9
        and (mask_buffer_shapes == ((B, S),))
        and (mask_buffer_strides == ((S, 1),))
        and (block_mask_batch == B)
        and (block_mask_heads == 1)
    )
    narrow_dq_reduction = (
        S == 4096
        and DQK == 192
        and (DV == 128)
        and (int(block_m) == 128)
        and (int(block_n) == 128)
        and (
            is_causal_document_mask_program(tuple(mask_program), int(mask_program_output), mask_buffer_strides)
            or batched_causal_document_mask
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
        dq_reduction_rows=32 if narrow_dq_reduction else 64,
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
    if DQK <= 0 or DQK % 32:
        raise ValueError("FlyDSL backward requires DQK to be a positive multiple of 32")
    if DV <= 0 or DV % 32:
        raise ValueError("FlyDSL backward requires DV to be a positive multiple of 32")
    BLOCK_M = int(block_m)
    BLOCK_N = int(block_n)
    if BLOCK_M != 128 or BLOCK_N != 128:
        raise ValueError("FlyDSL backward requires a 128x128 sparse block")
    if S <= 0 or S % BLOCK_M:
        raise ValueError("FlyDSL backward requires S to be divisible by BLOCK_M")
    if block_mask_batch not in (1, B):
        raise ValueError("BlockMask batch dimension must be 1 or B")
    if block_mask_heads not in (1, H):
        raise ValueError("MHA backward BlockMask head dimension must be 1 or H")
    if len(mask_buffer_shapes) != len(mask_buffer_strides):
        raise ValueError("mask buffer shape/stride descriptors must match")
    if len(mask_buffer_shapes) > 4:
        raise ValueError("FlyDSL backward supports at most four mask buffers")
    CAUSAL_PARTIAL = bool(causal_partial_blocks)
    MASK_PROGRAM = tuple(mask_program)
    MASK_PROGRAM_OUTPUT = int(mask_program_output)
    MASK_BUFFER_SHAPES = tuple((tuple(shape) for shape in mask_buffer_shapes))
    MASK_BUFFER_STRIDES = tuple((tuple(stride) for stride in mask_buffer_strides))
    MASK_BUFFER_COUNT = len(MASK_BUFFER_SHAPES)
    WINDOW_SIZE = _causal_window_size(MASK_PROGRAM, MASK_PROGRAM_OUTPUT)
    DENSE_MASK = (
        MASK_PROGRAM == _DENSE_MASK_PROGRAM
        and MASK_PROGRAM_OUTPUT == 5
        and (MASK_BUFFER_COUNT == 0)
        and (not CAUSAL_PARTIAL)
    )
    CAUSAL_MASK = not MASK_PROGRAM and MASK_BUFFER_COUNT == 0 and CAUSAL_PARTIAL
    WINDOW_MASK = WINDOW_SIZE is not None and MASK_BUFFER_COUNT == 0 and (not CAUSAL_PARTIAL)
    DIRECT_MASK = DENSE_MASK or CAUSAL_MASK or WINDOW_MASK
    INDIRECT_MASK = not DIRECT_MASK
    WIDE_OWNER = (
        DENSE_MASK
        and DQK + DV <= 256
        and (S % 256 == 0)
        and (64 * DQK // 8 % (8 * 64) == 0)
        and (64 * DV // 8 % (8 * 64) == 0)
    )
    EIGHT_WAVE_COMPUTE = WIDE_OWNER
    REDUCTION_BLOCK = 64 if WIDE_OWNER or INDIRECT_MASK else 32
    NUM_WAVES = 8 if EIGHT_WAVE_COMPUTE else 2
    OUTPUT_DK = 0
    OUTPUT_DV = 1
    OUTPUT_DQ = 2
    SB = BLOCK_M
    BM = 64 if INDIRECT_MASK else 32
    BN = 64 if INDIRECT_MASK else 32
    NW = 4 if WIDE_OWNER or INDIRECT_MASK else 2
    NT = NW * 64
    NB = S // SB
    MC = S // BM
    NC = S // BN
    CPB = SB // BM
    MAX_PARTIAL = NB if max_partial_blocks is None else int(max_partial_blocks)
    MAX_FULL = NB if max_full_blocks is None else int(max_full_blocks)
    SCALE = float(DQK) ** (-0.5) if scale is None else float(scale)
    SC2 = SCALE * _LOG2E
    LSE_IN_LOG2 = bool(lse_in_log2)
    BMB = int(block_mask_batch)
    BMH = int(block_mask_heads)
    MASK_BUFFER_SIZES = tuple(
        (
            1 + sum(((size - 1) * stride for (size, stride) in zip(shape, strides)))
            for (shape, strides) in zip(MASK_BUFFER_SHAPES, MASK_BUFFER_STRIDES)
        )
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
    DELTA_PACKS = DV // 8
    LPR = DELTA_PACKS if DELTA_PACKS <= 64 and DELTA_PACKS & DELTA_PACKS - 1 == 0 else 4
    DELTA_LOADS = DELTA_PACKS // LPR
    DROWS = NT // LPR
    DGRID = S // DROWS
    DSH = []
    _s = 1
    while _s < LPR:
        DSH.append(_s)
        _s *= 2

    @flyc.kernel
    def delta_kernel(OUT: fx.Tensor, DO: fx.Tensor, DELTA: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        bh = fx.Int32(fx.block_idx.y)
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
        acc = _f32(0.0)
        for load_step in fx.range_constexpr(DELTA_LOADS):
            pack = chunk + fx.Int32(load_step * LPR)
            fx.copy(atom, fx.slice(orow, (None, pack)), fo)
            fx.copy(atom, fx.slice(drow, (None, pack)), fd)
            prod = fx.Vector(fo.load()).to(fx.Float32) * fx.Vector(fd.load()).to(fx.Float32)
            for i in fx.range_constexpr(8):
                acc = acc + _f32(prod[i])
        for sh in DSH:
            acc = acc + _f32(fx.gpu.shuffle_xor(acc, sh, 64))
        if chunk == fx.Int32(0):
            fx.get_iter(gDl)[row] = acc

    FLAG_N = (NB * NB + 1 + NT - 1) // NT * NT
    FILL_IT = (NB * NB + NT - 1) // NT

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
        tid = fx.Int32(fx.thread_idx.x)
        bh = fx.Int32(fx.block_idx.x)
        batch = bh // fx.Int32(H)
        head = bh % fx.Int32(H)
        mask_b = fx.Int32(0) if BMB == 1 else batch
        mask_h = fx.Int32(0) if BMH == 1 else head
        mask_group = mask_b * fx.Int32(BMH) + mask_h
        lds = fx.SharedAllocator().allocate(PSmem).peek()
        fp = lds.flags.ptr
        gKVN = gview(KVN, mask_group * fx.Int32(NB), NB, 1)
        gKVI = gview(KVI, mask_group * fx.Int32(NB * MAX_PARTIAL), NB * MAX_PARTIAL, 1)
        gFKVN = gview(FKVN, mask_group * fx.Int32(NB), NB, 1)
        gFKVI = gview(FKVI, mask_group * fx.Int32(NB * MAX_FULL), NB * MAX_FULL, 1)
        gCPQ = gview(CP_Q, bh * fx.Int32(MC), MC, 1)
        gCFQ = gview(CF_Q, bh * fx.Int32(MC), MC, 1)
        gIPQ = gview(IP_Q, bh * fx.Int32(MC * NC), MC * NC, 1)
        gIFQ = gview(IF_Q, bh * fx.Int32(MC * NC), MC * NC, 1)
        gCPK = gview(CP_K, bh * fx.Int32(NC), NC, 1)
        gCFK = gview(CF_K, bh * fx.Int32(NC), NC, 1)
        gIPK = gview(IP_K, bh * fx.Int32(NC * MC), NC * MC, 1)
        gIFK = gview(IF_K, bh * fx.Int32(NC * MC), NC * MC, 1)

        def gload_i32(view, idx):
            return fx.Int32(fx.get_iter(view)[idx])

        def gstore_i32(view, idx, val):
            fx.get_iter(view)[idx] = val

        zero = fx.Int32(0)
        for l in fx.range_constexpr(FLAG_N // NT):
            fx.ptr_store(zero, fp + (fx.Int32(l * NT) + tid))
        fx.gpu.barrier()
        one = fx.Int32(1)
        two = fx.Int32(2)
        dump = fx.Int32(NB * NB)
        for l in fx.range_constexpr(FILL_IT):
            i = fx.Int32(l * NT) + tid
            mb_raw = i // fx.Int32(NB)
            mb = (mb_raw < fx.Int32(NB)).select(mb_raw, fx.Int32(NB - 1))
            t = i % fx.Int32(NB)
            cnt = gload_i32(gKVN, mb)
            tp = (t < fx.Int32(MAX_PARTIAL)).select(t, fx.Int32(MAX_PARTIAL - 1))
            nb = gload_i32(gKVI, mb * fx.Int32(MAX_PARTIAL) + tp)
            partial_active = (t < cnt) & (t < fx.Int32(MAX_PARTIAL))
            addr = partial_active.select(mb * fx.Int32(NB) + nb, dump)
            fx.ptr_store(one, fp + addr)
            cntf = gload_i32(gFKVN, mb)
            tf = (t < fx.Int32(MAX_FULL)).select(t, fx.Int32(MAX_FULL - 1))
            nbf = gload_i32(gFKVI, mb * fx.Int32(MAX_FULL) + tf)
            full_active = (t < cntf) & (t < fx.Int32(MAX_FULL))
            addrf = full_active.select(mb * fx.Int32(NB) + nbf, dump)
            fx.ptr_store(two, fp + addrf)
        fx.gpu.barrier()
        mc = (tid < fx.Int32(MC)).select(tid, fx.Int32(MC - 1))
        mb = mc // fx.Int32(CPB)
        c1 = fx.Int32(0)
        c2 = fx.Int32(0)
        d1 = fx.Int32(0)
        d2 = fx.Int32(0)
        rowq = mc * fx.Int32(NC)
        rowk = mc * fx.Int32(MC)
        for nb in fx.range_constexpr(NB):
            fq = fx.Int32(fx.ptr_load(fp + (mb * fx.Int32(NB) + fx.Int32(nb))))
            fk = fx.Int32(fx.ptr_load(fp + (fx.Int32(nb * NB) + mb)))
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
    DQ32_BM = 256 if EIGHT_WAVE_COMPUTE else 64
    DQ32_BN = REDUCTION_BLOCK
    DQ32_NW = NUM_WAVES
    DQ32_NT = DQ32_NW * 64
    DQ32_MC = S // DQ32_BM
    DQ32_NC = S // DQ32_BN
    DQ32_K_LOAD_IT = DQ32_BN * DQK // VPT // DQ32_NT
    DQ32_V_LOAD_IT = DQ32_BN * DV // VPT // DQ32_NT
    DQ32_CE = [8 * (e // 4) + e % 4 for e in range(16)]
    DQ32_CAUSAL_TILES = DQ32_BM // DQ32_BN
    PAIR_BM = 32
    PAIR_LOGICAL_WAVES = NUM_WAVES // 2
    PAIR_BN = PAIR_LOGICAL_WAVES * 32
    PAIR_MC = S // PAIR_BM
    PAIR_NC = S // PAIR_BN
    PAIR_Q_LOAD_IT = PAIR_BM * DQK // VPT // (PAIR_LOGICAL_WAVES * 64)
    PAIR_DO_LOAD_IT = PAIR_BM * DV // VPT // (PAIR_LOGICAL_WAVES * 64)
    PAIR_CAUSAL_TILES = PAIR_BN // PAIR_BM
    PAIR_WINDOW_TILES = (
        (WINDOW_SIZE + PAIR_BM - 1) // PAIR_BM + PAIR_CAUSAL_TILES if WINDOW_MASK else 0
    )
    PAIR_KEY_SPLIT = max(1, BN // PAIR_BN)
    PAIR_LIST_Q_SPLIT = max(1, BM // PAIR_BM)
    FUSED_GRID = PAIR_NC + DQ32_MC
    PAIR_QDO_ELEMS = 2 * PAIR_BM * (DQK + DV)
    PAIR_PROB_ELEMS = 2 * PAIR_LOGICAL_WAVES * 64 * 16
    COMPUTE_A_ELEMS = max(BM * (DQK + 8), 2 * DQ32_BN * DQK, PAIR_QDO_ELEMS)
    COMPUTE_B_ELEMS = max(BM * (DV + 8), DQ32_BN * DV, PAIR_PROB_ELEMS)
    COMPUTE_LD_ELEMS = max(2 * BM, 4 * PAIR_BM)
    COMPUTE_NT = DQ32_NT

    @fx.struct
    class ComputeSmem:
        a: fx.Array[fx.BFloat16, COMPUTE_A_ELEMS, 16]
        b: fx.Array[fx.BFloat16, COMPUTE_B_ELEMS, 16]
        ld: fx.Array[fx.Float32, COMPUTE_LD_ELEMS, 16]

    _emit_dq_mfma32_body = make_dq_mfma32_body(locals(), _f32, fast_exp2)
    _emit_dkdv_mfma32_body = make_dkdv_mfma32_body(locals(), _f32, fast_exp2)

    @flyc.jit
    def _emit_owner(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DO: fx.Tensor,
        DQ: fx.Tensor,
        DK: fx.Tensor,
        DVALUE: fx.Tensor,
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
        task: fx.Int32,
        bh: fx.Int32,
        pa,
        pb,
        pld,
    ):
        owner = task // fx.Int32(3)
        output_direction = task % fx.Int32(3)
        if output_direction < fx.Int32(OUTPUT_DQ):
            paired_nc = owner * fx.Int32(2) + output_direction
            _emit_dkdv_mfma32_body(
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
                paired_nc,
                bh,
                pa,
                pb,
                pld,
            )
        else:
            _emit_dq_mfma32_body(
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
                owner,
                bh,
                pa,
                pb,
            )

    @flyc.kernel
    def compute_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DO: fx.Tensor,
        DQ: fx.Tensor,
        DK: fx.Tensor,
        DVALUE: fx.Tensor,
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
    ):
        task = fx.Int32(fx.block_idx.x)
        bh = fx.Int32(fx.block_idx.y)
        lds = fx.SharedAllocator().allocate(ComputeSmem).peek()
        _emit_owner(
            Q,
            K,
            V,
            LSE,
            DELTA,
            DO,
            DQ,
            DK,
            DVALUE,
            CP_Q,
            IP_Q,
            CF_Q,
            IF_Q,
            CP_K,
            IP_K,
            CF_K,
            IF_K,
            MaskBuffer0,
            MaskBuffer1,
            MaskBuffer2,
            MaskBuffer3,
            task,
            bh,
            lds.a.ptr,
            lds.b.ptr,
            lds.ld.ptr,
        )

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
        delta_kernel(OUT, DO, DELTA).launch(grid=(DGRID, BH, 1), block=(NT, 1, 1), stream=stream)
        if const_expr(INDIRECT_MASK):
            prologue(KVN, KVI, FKVN, FKVI, CP_Q, IP_Q, CF_Q, IF_Q, CP_K, IP_K, CF_K, IF_K).launch(
                grid=(BH, 1, 1), block=(NT, 1, 1), stream=stream
            )
        compute_kernel(
            Q,
            K,
            V,
            LSE,
            DELTA,
            DO,
            DQ,
            DK,
            DVALUE,
            CP_Q,
            IP_Q,
            CF_Q,
            IF_Q,
            CP_K,
            IP_K,
            CF_K,
            IF_K,
            MaskBuffer0,
            MaskBuffer1,
            MaskBuffer2,
            MaskBuffer3,
        ).launch(grid=(FUSED_GRID, BH, 1), block=(COMPUTE_NT, 1, 1), stream=stream)

    return _launch
