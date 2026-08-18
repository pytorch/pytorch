"""FlyDSL port of the PyTorch FlexAttention backward (`flex_attn_bwd.py`).

Structure (all compute in FlyDSL / MFMA, no torch or triton kernels):

  1. ``delta_kernel``   -- delta[b,h,i] = sum_d out[b,h,i,d] * do[b,h,i,d]
  2. ``prologue``       -- turns the BlockMask metadata (kv_num_blocks / kv_indices
                           and their FULL counterparts) into flat per-chunk work
                           lists for both halves of the backward, including the
                           transposed (per-KV-block) lists the dK/dV pass needs.
  3. ``dkdv_kernel``    -- one workgroup per (b, h, kv-chunk): loops over the q
                           chunks that touch it and accumulates dK / dV.
  4. ``dq_kernel``      -- one workgroup per (b, h, q-chunk): loops over the kv
                           chunks it attends to and accumulates dQ.

Math mirrors the Inductor-generated Triton backward exactly:

    p   = exp2(q.k * (scale*log2e) - lse*log2e)      (masked entries -> 0)
    dv += p^T @ do
    dp  = do @ v^T
    ds  = p * (dp - delta)
    dq += (ds  @ k) * scale
    dk += (ds^T @ q) * scale

MFMA operand trick: for the 16x16x16 atom the *C* fragment of a tile X[R,S]
(with waves tiled along N) is bit-for-bit the *A* fragment of X^T[S,R] (with
waves tiled along M).  That lets ``p`` / ``ds`` feed the second GEMM straight
out of registers with no LDS round-trip.

Mask reconstruction: the launch signature only exposes *block* level sparsity, so
the element mask inside a partially-masked block is rebuilt from the block
coordinates -- a diagonal block uses ``q >= kv`` and an off-diagonal partial
block (the sliding-window boundary) uses ``q - kv <= 128*(qblk - kvblk)``.  That
reproduces causal, document and sliding-window masks exactly.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

_LOG2E = 1.4426950408889634
_NEG_BIG = -1.0e30


def _f32(x):
    return fx.Float32(x)


def _exp2(x):
    """Single-instruction v_exp_f32 (no denormal guard)."""
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


def build_flex_attn_bwd_module(B, H, S, D, dtype_str, block_m, block_n):
    assert dtype_str in ("bf16",), f"unsupported dtype {dtype_str}"

    SB = int(block_m)               # sparse block size (== block_n == 128)
    BM = 64                         # q rows per tile
    BN = 64                         # kv rows per tile
    NW = 4                          # waves per workgroup
    NT = NW * 64                    # threads per workgroup

    NB = S // SB                    # sparse blocks along each axis
    MC = S // BM                    # q chunks
    NC = S // BN                    # kv chunks
    CPB = SB // BM                  # chunks per sparse block

    Dp = D + 8                      # padded LDS row stride, natural layout
    Mp = BM + 8                     # padded LDS row stride, transposed layout
    Np = BN + 8

    SCALE = float(D) ** -0.5
    SC2 = SCALE * _LOG2E

    BH = B * H
    QSTRIDE = S * D
    LSTRIDE = S

    dev = "cuda"

    # ---------------------------------------------------------------- scratch
    delta_buf = torch.empty(B, H, S, dtype=torch.float32, device=dev)
    cntP_q = torch.empty(MC, dtype=torch.int32, device=dev)
    cntF_q = torch.empty(MC, dtype=torch.int32, device=dev)
    idxP_q = torch.empty(MC * NC, dtype=torch.int32, device=dev)
    idxF_q = torch.empty(MC * NC, dtype=torch.int32, device=dev)
    cntP_k = torch.empty(NC, dtype=torch.int32, device=dev)
    cntF_k = torch.empty(NC, dtype=torch.int32, device=dev)
    idxP_k = torch.empty(NC * MC, dtype=torch.int32, device=dev)
    idxF_k = torch.empty(NC * MC, dtype=torch.int32, device=dev)

    # ---------------------------------------------------------------- helpers
    def gview(t, off, shape, stride):
        it = fx.get_iter(fx.rocdl.make_buffer_tensor(t))
        if off is not None:
            it = fx.add_offset(it, off)
        return fx.Tensor(fx.make_view(it, fx.make_layout(shape, stride)))

    def sview(ptr, shape, stride):
        return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

    ATOM = None  # created inside kernels

    # ================================================================= delta
    LPR = D // 8                     # lanes cooperating on one row
    DROWS = NT // LPR                # rows per block
    DGRID = (B * H * S) // DROWS
    DSH = []
    _s = 1
    while _s < LPR:
        DSH.append(_s)
        _s *= 2

    @flyc.kernel
    def delta_kernel(OUT: fx.Tensor, DO: fx.Tensor, DELTA: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        R = B * H * S
        gO = gview(OUT, None, (R, D), (D, 1))
        gDo = gview(DO, None, (R, D), (D, 1))
        gDl = gview(DELTA, None, R, 1)

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
            acc = acc + _f32(acc.shuffle_xor(fx.Int32(sh), fx.Int32(64)))
        satom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        fs = fx.make_rmem_tensor(1, fx.Float32)
        fs.store(fx.Vector.from_elements([acc], fx.Float32).ir_value())
        if chunk == fx.Int32(0):
            fx.copy(satom, fs, fx.slice(fx.logical_divide(gDl, fx.make_layout(1, 1)),
                                        (None, row)))

    # ============================================================== prologue
    FLAG_N = ((NB * NB + 1 + NT - 1) // NT) * NT
    FILL_IT = (NB * NB + NT - 1) // NT

    @fx.struct
    class PSmem:
        flags: fx.Array[fx.Int32, FLAG_N, 16]

    @flyc.kernel
    def prologue(KVN: fx.Tensor, KVI: fx.Tensor, FKVN: fx.Tensor, FKVI: fx.Tensor,
                 CP_Q: fx.Tensor, IP_Q: fx.Tensor, CF_Q: fx.Tensor, IF_Q: fx.Tensor,
                 CP_K: fx.Tensor, IP_K: fx.Tensor, CF_K: fx.Tensor, IF_K: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        lds = fx.SharedAllocator().allocate(PSmem).peek()
        fp = lds.flags.ptr

        gKVN = gview(KVN, None, NB, 1)
        gKVI = gview(KVI, None, NB * NB, 1)
        gFKVN = gview(FKVN, None, NB, 1)
        gFKVI = gview(FKVI, None, NB * NB, 1)
        gCPQ = gview(CP_Q, None, MC, 1)
        gCFQ = gview(CF_Q, None, MC, 1)
        gIPQ = gview(IP_Q, None, MC * NC, 1)
        gIFQ = gview(IF_Q, None, MC * NC, 1)
        gCPK = gview(CP_K, None, NC, 1)
        gCFK = gview(CF_K, None, NC, 1)
        gIPK = gview(IP_K, None, NC * MC, 1)
        gIFK = gview(IF_K, None, NC * MC, 1)

        i32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)

        def gload_i32(view, n, idx):
            fr = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy(i32atom, fx.slice(fx.logical_divide(view, fx.make_layout(1, 1)),
                                      (None, idx)), fr)
            return fx.Int32(fx.Vector(fr.load())[0])

        def gstore_i32(view, idx, val):
            fr = fx.make_rmem_tensor(1, fx.Int32)
            fr.store(fx.Vector.from_elements([val], fx.Int32).ir_value())
            fx.copy(i32atom, fr, fx.slice(fx.logical_divide(view, fx.make_layout(1, 1)),
                                          (None, idx)))

        # zero flags (+1 dump slot)
        zero = fx.Int32(0)
        for l in fx.range_constexpr(FLAG_N // NT):
            fx.ptr_store(zero, fp + (fx.Int32(l * NT) + tid))
        fx.gpu.barrier()

        # scatter block flags: 1 = partially masked, 2 = fully unmasked
        one = fx.Int32(1)
        two = fx.Int32(2)
        dump = fx.Int32(NB * NB)
        for l in fx.range_constexpr(FILL_IT):
            i = fx.Int32(l * NT) + tid
            mb_raw = i // fx.Int32(NB)
            mb = (mb_raw < fx.Int32(NB)).select(mb_raw, fx.Int32(NB - 1))
            t = i % fx.Int32(NB)
            cnt = gload_i32(gKVN, NB, mb)
            nb = gload_i32(gKVI, NB * NB, mb * fx.Int32(NB) + t)
            addr = (t < cnt).select(mb * fx.Int32(NB) + nb, dump)
            fx.ptr_store(one, fp + addr)
            cntf = gload_i32(gFKVN, NB, mb)
            nbf = gload_i32(gFKVI, NB * NB, mb * fx.Int32(NB) + t)
            addrf = (t < cntf).select(mb * fx.Int32(NB) + nbf, dump)
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

    # ---------------------------------------------------------- shared bits
    VPT = 8
    DCH = D // VPT
    LOAD_IT = (BM * D // VPT) // NT

    CE = [16 * (e // 4) + (e % 4) for e in range(16)]

    @fx.struct
    class KvSmem:
        a: fx.Array[fx.BFloat16, BM * Dp, 16]
        b: fx.Array[fx.BFloat16, BM * Dp, 16]
        at: fx.Array[fx.BFloat16, D * Mp, 16]
        bt: fx.Array[fx.BFloat16, D * Mp, 16]
        ld: fx.Array[fx.Float32, 2 * BM, 16]

    @fx.struct
    class QSmem:
        a: fx.Array[fx.BFloat16, BN * Dp, 16]
        b: fx.Array[fx.BFloat16, BN * Dp, 16]
        at: fx.Array[fx.BFloat16, D * Np, 16]

    # ============================================================== dk / dv
    @flyc.kernel
    def dkdv_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, LSE: fx.Tensor,
                    DELTA: fx.Tensor, DO: fx.Tensor, DK: fx.Tensor, DV: fx.Tensor,
                    CP: fx.Tensor, IP: fx.Tensor, CF: fx.Tensor, IF: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        nc = fx.Int32(fx.block_idx.x)
        bh = fx.Int32(fx.block_idx.y)
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        g = lane // fx.Int32(16)

        boff = bh * fx.Int32(QSTRIDE)
        loff = bh * fx.Int32(LSTRIDE)
        kvbase = nc * fx.Int32(BN)

        lds = fx.SharedAllocator().allocate(KvSmem).peek()
        pa, pb, pat, pbt, pld = lds.a.ptr, lds.b.ptr, lds.at.ptr, lds.bt.ptr, lds.ld.ptr

        gQ = gview(Q, boff, (S, D), (D, 1))
        gDO = gview(DO, boff, (S, D), (D, 1))
        gK = gview(K, boff, (S, D), (D, 1))
        gV = gview(V, boff, (S, D), (D, 1))
        gLSE = gview(LSE, loff, S, 1)
        gDEL = gview(DELTA, loff, S, 1)
        gCP = gview(CP, None, NC, 1)
        gCF = gview(CF, None, NC, 1)
        gIP = gview(IP, None, NC * MC, 1)
        gIF = gview(IF, None, NC * MC, 1)

        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        mma1 = fx.make_tiled_mma(atom, fx.make_layout((1, NW, 1), (0, 1, 0)))
        mma2 = fx.make_tiled_mma(atom, fx.make_layout((NW, 1, 1), (1, 0, 0)))
        t1 = mma1.thr_slice(tid)
        t2 = mma2.thr_slice(tid)

        g128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        u64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        i32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        f32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        o16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

        def gload_i32(view, idx):
            fr = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy(i32atom, fx.slice(fx.logical_divide(view, fx.make_layout(1, 1)),
                                      (None, idx)), fr)
            return fx.Int32(fx.Vector(fr.load())[0])

        def gload_f32(view, idx):
            fr = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy(f32atom, fx.slice(fx.logical_divide(view, fx.make_layout(1, 1)),
                                      (None, idx)), fr)
            return fx.Float32(fx.Vector(fr.load())[0])

        def load_tile(gsrc, rbase, pn, pt):
            for l in fx.range_constexpr(LOAD_IT):
                i = fx.Int32(l * NT) + tid
                m = i // fx.Int32(DCH)
                c = i % fx.Int32(DCH)
                fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
                rw = fx.logical_divide(fx.slice(gsrc, (rbase + m, None)),
                                       fx.make_layout(VPT, 1))
                fx.copy(g128, fx.slice(rw, (None, c)), fr)
                v = fx.Vector(fr.load())
                fx.ptr_store(v, pn + (m * fx.Int32(Dp) + c * fx.Int32(VPT)))
                for j in fx.range_constexpr(VPT):
                    fx.ptr_store(v[j], pt + ((c * fx.Int32(VPT) + fx.Int32(j))
                                             * fx.Int32(Mp) + m))

        # ---- K, V staged through LDS then held as B fragments ----
        sKV = sview(pa, (BN, D), (Dp, 1))
        for l in fx.range_constexpr(LOAD_IT):
            i = fx.Int32(l * NT) + tid
            m = i // fx.Int32(DCH)
            c = i % fx.Int32(DCH)
            fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
            rw = fx.logical_divide(fx.slice(gK, (kvbase + m, None)), fx.make_layout(VPT, 1))
            fx.copy(g128, fx.slice(rw, (None, c)), fr)
            fx.ptr_store(fx.Vector(fr.load()), pa + (m * fx.Int32(Dp) + c * fx.Int32(VPT)))
            rw2 = fx.logical_divide(fx.slice(gV, (kvbase + m, None)), fx.make_layout(VPT, 1))
            fx.copy(g128, fx.slice(rw2, (None, c)), fr)
            fx.ptr_store(fx.Vector(fr.load()), pb + (m * fx.Int32(Dp) + c * fx.Int32(VPT)))
        fx.gpu.barrier()

        tcB1 = fx.make_tiled_copy_B(u64, mma1).get_slice(tid)
        fBK = t1.make_fragment_B(sKV)
        fBV = t1.make_fragment_B(sKV)
        fx.copy(u64, tcB1.partition_S(sview(pa, (BN, D), (Dp, 1))), tcB1.retile(fBK))
        fx.copy(u64, tcB1.partition_S(sview(pb, (BN, D), (Dp, 1))), tcB1.retile(fBV))
        fx.gpu.barrier()

        # ---- accumulators ----
        gDKt = gview(DK, boff, (S, D), (D, 1))
        gDVt = gview(DV, boff, (S, D), (D, 1))
        gDKb = fx.Tensor(fx.make_view(
            fx.add_offset(fx.get_iter(fx.rocdl.make_buffer_tensor(DK)),
                          boff + kvbase * fx.Int32(D)),
            fx.make_layout((BN, D), (D, 1))))
        gDVb = fx.Tensor(fx.make_view(
            fx.add_offset(fx.get_iter(fx.rocdl.make_buffer_tensor(DV)),
                          boff + kvbase * fx.Int32(D)),
            fx.make_layout((BN, D), (D, 1))))

        fC3 = t2.make_fragment_C(gDVb)      # dV
        fC4 = t2.make_fragment_C(gDKb)      # dK
        fC3.fill(0)
        fC4.fill(0)
        NACC = (BN // NW) * D // 64

        sQn = sview(pa, (BM, D), (Dp, 1))
        sDn = sview(pb, (BM, D), (Dp, 1))
        sQt = sview(pat, (D, BM), (Mp, 1))
        sDt = sview(pbt, (D, BM), (Mp, 1))

        tcA1 = fx.make_tiled_copy_A(u64, mma1).get_slice(tid)
        tcB2 = fx.make_tiled_copy_B(u64, mma2).get_slice(tid)

        fA1 = t1.make_fragment_A(sQn)
        fA2 = t1.make_fragment_A(sDn)
        fB3 = t2.make_fragment_B(sDt)
        fB4 = t2.make_fragment_B(sQt)
        fST = fx.Tensor(fx.make_view(fx.get_iter(sQn), fx.make_layout((BN, BM), (BM, 1))))
        fAP = t2.make_fragment_A(fST)
        fAS = t2.make_fragment_A(fST)

        kcol = fx.Int32(16) * wave + lane % fx.Int32(16)
        kj = kvbase + kcol

        def tile(mchunk, masked):
            qbase = mchunk * fx.Int32(BM)
            # lse / delta staging
            r = (tid < fx.Int32(BM)).select(tid, fx.Int32(BM - 1))
            lv = gload_f32(gLSE, qbase + r)
            dv_ = gload_f32(gDEL, qbase + r)
            l2 = (lv < _f32(_NEG_BIG)).select(_f32(0.0), lv * _f32(_LOG2E))
            fx.ptr_store(l2, pld + r)
            fx.ptr_store(dv_, pld + (fx.Int32(BM) + r))
            load_tile(gQ, qbase, pa, pat)
            load_tile(gDO, qbase, pb, pbt)
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
            mb = qbase // fx.Int32(SB)
            nbk = kj // fx.Int32(SB)
            isdiag = (mb == nbk)
            dd = fx.Int32(SB) * (mb - nbk)
            tbase = qrow0 - kj

            vs = fx.Vector(fC1.load())
            vp = fx.Vector(fC2.load())
            pl = []
            dsl = []
            for e in fx.range_constexpr(16):
                rr = g4 + fx.Int32(CE[e])
                l2e = fx.Float32(fx.ptr_load(pld + rr))
                de = fx.Float32(fx.ptr_load(pld + (fx.Int32(BM) + rr)))
                x = _f32(vs[e]) * _f32(SC2) - l2e
                p = _exp2(x)
                if masked:
                    tt = tbase + fx.Int32(CE[e])
                    keep = isdiag.select(tt >= fx.Int32(0), tt <= dd)
                    p = keep.select(p, _f32(0.0))
                pl.append(p)
                dsl.append(p * (_f32(vp[e]) - de))
            fAP.store(fx.Vector.from_elements(pl, fx.Float32).to(fx.BFloat16).ir_value())
            fAS.store(fx.Vector.from_elements(dsl, fx.Float32).to(fx.BFloat16).ir_value())

            fx.copy(u64, tcB2.partition_S(sDt), tcB2.retile(fB3))
            fx.copy(u64, tcB2.partition_S(sQt), tcB2.retile(fB4))
            fx.gemm(atom, fC3, fAP, fB3, fC3)
            fx.gemm(atom, fC4, fAS, fB4, fC4)
            fx.gpu.barrier()

        cntp = gload_i32(gCP, nc)
        cntf = gload_i32(gCF, nc)
        rowk = nc * fx.Int32(MC)

        for ti in range(fx.Index(0), fx.Index(cntp), fx.Index(1)):
            tile(gload_i32(gIP, rowk + fx.Int32(ti)), True)
        for ti in range(fx.Index(0), fx.Index(cntf), fx.Index(1)):
            tile(gload_i32(gIF, rowk + fx.Int32(ti)), False)

        # ---- epilogue ----
        vdk = fx.Vector(fC4.load()) * fx.Vector.filled(NACC, SCALE, fx.Float32)
        fC4.store(vdk.ir_value())
        tcC2 = fx.make_tiled_copy_C(o16, mma2).get_slice(tid)
        oK = fx.make_fragment_like(fC4, fx.BFloat16.ir_type)
        oV = fx.make_fragment_like(fC3, fx.BFloat16.ir_type)
        oK.store(fx.Vector(fC4.load()).to(fx.BFloat16).ir_value())
        oV.store(fx.Vector(fC3.load()).to(fx.BFloat16).ir_value())
        fx.copy(o16, tcC2.retile(oK), tcC2.partition_S(gDKb))
        fx.copy(o16, tcC2.retile(oV), tcC2.partition_S(gDVb))

    # ================================================================== dq
    @flyc.kernel
    def dq_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, LSE: fx.Tensor,
                  DELTA: fx.Tensor, DO: fx.Tensor, DQ: fx.Tensor,
                  CP: fx.Tensor, IP: fx.Tensor, CF: fx.Tensor, IF: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        mc = fx.Int32(fx.block_idx.x)
        bh = fx.Int32(fx.block_idx.y)
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        g = lane // fx.Int32(16)

        boff = bh * fx.Int32(QSTRIDE)
        loff = bh * fx.Int32(LSTRIDE)
        qbase = mc * fx.Int32(BM)

        lds = fx.SharedAllocator().allocate(QSmem).peek()
        pa, pb, pat = lds.a.ptr, lds.b.ptr, lds.at.ptr

        gQ = gview(Q, boff, (S, D), (D, 1))
        gDO = gview(DO, boff, (S, D), (D, 1))
        gK = gview(K, boff, (S, D), (D, 1))
        gV = gview(V, boff, (S, D), (D, 1))
        gLSE = gview(LSE, loff, S, 1)
        gDEL = gview(DELTA, loff, S, 1)
        gCP = gview(CP, None, MC, 1)
        gCF = gview(CF, None, MC, 1)
        gIP = gview(IP, None, MC * NC, 1)
        gIF = gview(IF, None, MC * NC, 1)

        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        mma1 = fx.make_tiled_mma(atom, fx.make_layout((1, NW, 1), (0, 1, 0)))
        mma2 = fx.make_tiled_mma(atom, fx.make_layout((NW, 1, 1), (1, 0, 0)))
        t1 = mma1.thr_slice(tid)
        t2 = mma2.thr_slice(tid)

        g128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        u64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        i32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        f32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        o16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

        def gload_i32(view, idx):
            fr = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy(i32atom, fx.slice(fx.logical_divide(view, fx.make_layout(1, 1)),
                                      (None, idx)), fr)
            return fx.Int32(fx.Vector(fr.load())[0])

        def gload_f32(view, idx):
            fr = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy(f32atom, fx.slice(fx.logical_divide(view, fx.make_layout(1, 1)),
                                      (None, idx)), fr)
            return fx.Float32(fx.Vector(fr.load())[0])

        # ---- Q, dO staged through LDS then held as B fragments ----
        sT = sview(pa, (BM, D), (Dp, 1))
        for l in fx.range_constexpr(LOAD_IT):
            i = fx.Int32(l * NT) + tid
            m = i // fx.Int32(DCH)
            c = i % fx.Int32(DCH)
            fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
            rw = fx.logical_divide(fx.slice(gQ, (qbase + m, None)), fx.make_layout(VPT, 1))
            fx.copy(g128, fx.slice(rw, (None, c)), fr)
            fx.ptr_store(fx.Vector(fr.load()), pa + (m * fx.Int32(Dp) + c * fx.Int32(VPT)))
            rw2 = fx.logical_divide(fx.slice(gDO, (qbase + m, None)), fx.make_layout(VPT, 1))
            fx.copy(g128, fx.slice(rw2, (None, c)), fr)
            fx.ptr_store(fx.Vector(fr.load()), pb + (m * fx.Int32(Dp) + c * fx.Int32(VPT)))
        fx.gpu.barrier()

        tcB1 = fx.make_tiled_copy_B(u64, mma1).get_slice(tid)
        fBQ = t1.make_fragment_B(sT)
        fBD = t1.make_fragment_B(sT)
        fx.copy(u64, tcB1.partition_S(sview(pa, (BM, D), (Dp, 1))), tcB1.retile(fBQ))
        fx.copy(u64, tcB1.partition_S(sview(pb, (BM, D), (Dp, 1))), tcB1.retile(fBD))

        # per-lane q row (uniform across the C fragment columns)
        qcol = fx.Int32(16) * wave + lane % fx.Int32(16)
        qi = qbase + qcol
        lv = gload_f32(gLSE, qi)
        de = gload_f32(gDEL, qi)
        l2e = (lv < _f32(_NEG_BIG)).select(_f32(0.0), lv * _f32(_LOG2E))
        fx.gpu.barrier()

        gDQb = fx.Tensor(fx.make_view(
            fx.add_offset(fx.get_iter(fx.rocdl.make_buffer_tensor(DQ)),
                          boff + qbase * fx.Int32(D)),
            fx.make_layout((BM, D), (D, 1))))
        fC3 = t2.make_fragment_C(gDQb)
        fC3.fill(0)
        NACC = (BM // NW) * D // 64

        sKn = sview(pa, (BN, D), (Dp, 1))
        sVn = sview(pb, (BN, D), (Dp, 1))
        sKt = sview(pat, (D, BN), (Np, 1))

        tcA1 = fx.make_tiled_copy_A(u64, mma1).get_slice(tid)
        tcB2 = fx.make_tiled_copy_B(u64, mma2).get_slice(tid)
        fA1 = t1.make_fragment_A(sKn)
        fA2 = t1.make_fragment_A(sVn)
        fB3 = t2.make_fragment_B(sKt)
        fST = fx.Tensor(fx.make_view(fx.get_iter(sKn), fx.make_layout((BM, BN), (BN, 1))))
        fAS = t2.make_fragment_A(fST)

        def tile(kchunk, masked):
            kvbase = kchunk * fx.Int32(BN)
            for l in fx.range_constexpr(LOAD_IT):
                i = fx.Int32(l * NT) + tid
                m = i // fx.Int32(DCH)
                c = i % fx.Int32(DCH)
                fr = fx.make_rmem_tensor(VPT, fx.BFloat16)
                rw = fx.logical_divide(fx.slice(gK, (kvbase + m, None)),
                                       fx.make_layout(VPT, 1))
                fx.copy(g128, fx.slice(rw, (None, c)), fr)
                v = fx.Vector(fr.load())
                fx.ptr_store(v, pa + (m * fx.Int32(Dp) + c * fx.Int32(VPT)))
                for j in fx.range_constexpr(VPT):
                    fx.ptr_store(v[j], pat + ((c * fx.Int32(VPT) + fx.Int32(j))
                                              * fx.Int32(Np) + m))
                rw2 = fx.logical_divide(fx.slice(gV, (kvbase + m, None)),
                                        fx.make_layout(VPT, 1))
                fx.copy(g128, fx.slice(rw2, (None, c)), fr)
                fx.ptr_store(fx.Vector(fr.load()), pb + (m * fx.Int32(Dp) + c * fx.Int32(VPT)))
            fx.gpu.barrier()

            fC1 = t1.make_fragment_C(fST)
            fC2 = t1.make_fragment_C(fST)
            fx.copy(u64, tcA1.partition_S(sKn), tcA1.retile(fA1))
            fx.copy(u64, tcA1.partition_S(sVn), tcA1.retile(fA2))
            fC1.fill(0)
            fC2.fill(0)
            fx.gemm(atom, fC1, fA1, fBQ, fC1)
            fx.gemm(atom, fC2, fA2, fBD, fC2)

            g4 = fx.Int32(4) * g
            krow0 = kvbase + g4
            mb = qi // fx.Int32(SB)
            nbk = krow0 // fx.Int32(SB)
            isdiag = (mb == nbk)
            dd = fx.Int32(SB) * (mb - nbk)
            tbase = qi - krow0

            vs = fx.Vector(fC1.load())
            vp = fx.Vector(fC2.load())
            dsl = []
            for e in fx.range_constexpr(16):
                x = _f32(vs[e]) * _f32(SC2) - l2e
                p = _exp2(x)
                if masked:
                    tt = tbase - fx.Int32(CE[e])
                    keep = isdiag.select(tt >= fx.Int32(0), tt <= dd)
                    p = keep.select(p, _f32(0.0))
                dsl.append(p * (_f32(vp[e]) - de))
            fAS.store(fx.Vector.from_elements(dsl, fx.Float32).to(fx.BFloat16).ir_value())

            fx.copy(u64, tcB2.partition_S(sKt), tcB2.retile(fB3))
            fx.gemm(atom, fC3, fAS, fB3, fC3)
            fx.gpu.barrier()

        cntp = gload_i32(gCP, mc)
        cntf = gload_i32(gCF, mc)
        rowq = mc * fx.Int32(NC)
        for ti in range(fx.Index(0), fx.Index(cntp), fx.Index(1)):
            tile(gload_i32(gIP, rowq + fx.Int32(ti)), True)
        for ti in range(fx.Index(0), fx.Index(cntf), fx.Index(1)):
            tile(gload_i32(gIF, rowq + fx.Int32(ti)), False)

        vq = fx.Vector(fC3.load()) * fx.Vector.filled(NACC, SCALE, fx.Float32)
        fC3.store(vq.ir_value())
        tcC2 = fx.make_tiled_copy_C(o16, mma2).get_slice(tid)
        oQ = fx.make_fragment_like(fC3, fx.BFloat16.ir_type)
        oQ.store(fx.Vector(fC3.load()).to(fx.BFloat16).ir_value())
        fx.copy(o16, tcC2.retile(oQ), tcC2.partition_S(gDQb))

    # ================================================================ launch
    @flyc.jit
    def _launch(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, OUT: fx.Tensor,
                LSE: fx.Tensor, DO: fx.Tensor, DQ: fx.Tensor, DK: fx.Tensor,
                DV: fx.Tensor, KVN: fx.Tensor, KVI: fx.Tensor, FKVN: fx.Tensor,
                FKVI: fx.Tensor, DELTA: fx.Tensor,
                CP_Q: fx.Tensor, IP_Q: fx.Tensor, CF_Q: fx.Tensor, IF_Q: fx.Tensor,
                CP_K: fx.Tensor, IP_K: fx.Tensor, CF_K: fx.Tensor, IF_K: fx.Tensor,
                stream: fx.Stream = fx.Stream(None)):
        delta_kernel(OUT, DO, DELTA).launch(grid=(DGRID, 1, 1), block=(NT, 1, 1),
                                            stream=stream)
        prologue(KVN, KVI, FKVN, FKVI, CP_Q, IP_Q, CF_Q, IF_Q,
                 CP_K, IP_K, CF_K, IF_K).launch(grid=(1, 1, 1), block=(NT, 1, 1),
                                                stream=stream)
        dkdv_kernel(Q, K, V, LSE, DELTA, DO, DK, DV,
                    CP_K, IP_K, CF_K, IF_K).launch(grid=(NC, BH, 1), block=(NT, 1, 1),
                                                   stream=stream)
        dq_kernel(Q, K, V, LSE, DELTA, DO, DQ,
                  CP_Q, IP_Q, CF_Q, IF_Q).launch(grid=(MC, BH, 1), block=(NT, 1, 1),
                                                 stream=stream)

    def launch_fn(q, k, v, out, lse, do, dq, dk, dv,
                  kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices,
                  stream=None):
        if stream is None:
            stream = fx.Stream(None)
        _launch(q, k, v, out, lse, do, dq, dk, dv,
                kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices,
                delta_buf,
                cntP_q, idxP_q, cntF_q, idxF_q,
                cntP_k, idxP_k, cntF_k, idxF_k,
                stream=stream)

    return launch_fn
