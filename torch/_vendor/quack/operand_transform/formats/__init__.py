# Copyright (c) 2026, Tri Dao.
"""Decode formats for TransformAW4: each format is a self-contained bundle of

  * the register-level decode math (``decode_k16``: one k16 block for one
    MMA_M atom of one thread — raw packed words + scale words in, 4 packed
    bf16x2 fragment registers out),
  * its blob geometry (4-bit vs 8-bit raw words, SF strip width), and
  * the HOST side that must stay consistent with it: quantize/dequant
    references and the offline repack (``prepare``).

Adding a format = one class here + registration; no kernel edits. The
kernel-facing plumbing (TMA, smem, Machete interleave, fragment buffering)
lives in :class:`quack.operand_transform.transform.TransformAW4` and never
branches on the format. tests/test_gemm_w4.py's roundtrip fixture exercises every
registered format against its own dequant reference, so fn/repack
consistency is pinned for free.
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

from torch._vendor.quack.blockscaled import nvfp4_utils as U
from torch._vendor.quack.operand_transform.formats import qtip as Q


@cute.jit
def _mul4(r0, r1, r2, r3, h):
    """Scale 4 packed bf16x2 regs by the (row0, row1) scale pair ``h``:
    regs (r0, r2) hold fragment rows v1=0, (r1, r3) rows v1=1."""
    return (
        U.mul_bf16x2_bcast(r0, h, False),
        U.mul_bf16x2_bcast(r1, h, True),
        U.mul_bf16x2_bcast(r2, h, False),
        U.mul_bf16x2_bcast(r3, h, True),
    )


def _pad_n128(q, sf):
    """Pad packed weights (and their SF rows) to a multiple of 128 N rows
    (tile granularity) with zero-decoding values."""
    n, kb = q.shape
    n_pad = (128 - n % 128) % 128
    if n_pad:
        q = torch.cat([q, q.new_zeros(n_pad, kb)])
        if sf is not None:
            if sf.dtype == torch.bfloat16:
                sf = torch.cat([sf, sf.new_zeros(n_pad, sf.shape[1])])
            else:
                sf_u8 = sf.view(torch.uint8)
                sf = torch.cat([sf_u8, sf_u8.new_zeros(n_pad, sf.shape[1])])
    return q, sf


class DecodeFormat:
    """One W4/W8 decode format. Device contract:

    ``decode_k16(xw, sfw, b, consts)``: xw is the thread's raw ``Int32`` word
    view for one m-atom ((4,) for 4-bit, (8,) for 8-bit formats), sfw its
    ``(sf_words,)`` scale words, b the k16 block index (0..3), consts the
    ``make_consts()`` result hoisted once per kernel. Returns the block's 4
    packed bf16x2 registers in fragment slot order
    ((v1=0, v2=0), (v1=1, v2=0), (v1=0, v2=1), (v1=1, v2=1)).

    ``w8``: raw words are 32 B/thread (8 i32) instead of 16 B (4 i32).
    ``tile_k``: k-tile the repack format is built around (64 default; 128
    doubles the per-thread raw words and the k16 blocks per tile — the b
    index then runs 0..7).
    ``sf_words``: i32 scale words per thread slot per k-tile (0 = no strip);
    the strip is 128 * sf_words bytes per m64 block.
    ``promote``: slow-accum format (W4A8): the group scale is NOT folded in
    the decode — the mainloop WGMMAs each k-tile into a zero-init wave
    accumulator and the transform promotes ``acc += scale_row * wave`` in
    fp32 at tile end, reading the strip word via ``promote_scale_pair``.
    ``tile_state_words``: per-(m-atom, k-tile) register state derived from
    the strip words once per tile — ``build_tile_state(sfw, consts, ts)``
    fills the (tile_state_words,) i32 slice at each tile's block-0 produce,
    and ``decode_k16`` receives ``ts`` in place of ``sfw`` (e.g. the folded
    W4A8's per-row scaled magnitude tables).

    Host contract: ``quantize_reference(w) -> (q, sf)``,
    ``dequant_reference(q, sf) -> w'`` (fp32), and ``prepare(q, sf) ->
    (blob, sf_blob)`` — the offline repack whose word order ``decode_k16``
    assumes. The framework never interprets the blob; the roundtrip test
    fixture is what keeps the two sides honest.
    """

    name: str
    w8 = False
    tile_k = 64
    # False: the format's dequant needs operands the W4A16 roundtrip harness
    # doesn't supply (e.g. an epilogue-side per-channel scale). Opt OUT
    # explicitly — a new format is covered by tests/test_gemm_w4.py's
    # roundtrip fixture by default, never silently skipped.
    roundtrip = True
    sf_words = 0
    promote = False
    tile_state_words = 0
    # the MMA compute dtype decode_k16 produces (the gemm ctor a_dtype);
    # bf16 for the whole W4A16 family, e4m3 for w4a8 (int4sm)
    mma_dtype = cutlass.BFloat16

    @property
    def sf_bytes(self):
        return 128 * self.sf_words

    def make_consts(self):
        return None

    # host side ---------------------------------------------------------------

    def quantize_reference(self, w):
        raise NotImplementedError

    def dequant_reference(self, q, sf):
        raise NotImplementedError

    def prepare(self, q, sf):
        raise NotImplementedError


class Nvfp4(DecodeFormat):
    """e2m1 nibbles + e4m3 scale per 16 k; scale folded in the decode (exact:
    2-significand-bit e2m1 x 4-bit e4m3 needs <= 6 < bf16's 8 bits)."""

    name = "nvfp4"
    sf_words = 2

    def make_consts(self):
        return U.make_decode_luts()

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        r0, r1, r2, r3 = U.decode_e2m1x8_to_bf16x8(xw[b], consts)
        h = U.sf_pair_to_bf16x2(sfw[b // 2], (b % 2) * 2)
        return _mul4(r0, r1, r2, r3, h)

    def quantize_reference(self, w):
        return U.quantize_nvfp4_reference(w)

    def dequant_reference(self, q, sf):
        return U.dequant_nvfp4_reference(q, sf)

    def prepare(self, q, sf):
        q, sf = _pad_n128(q, sf)
        return U.repack_nvfp4_weight(q), U.repack_nvfp4_sf(sf)


class Int4(DecodeFormat):
    """u4b8 nibbles + bf16 scale per group of `group` k columns. group is a
    format parameter (registered as int4 / int4_g64 / int4_g32): 32, 64, or
    any multiple of 64. Groups of >= 64 ride one strip word per k-tile
    (duplicated across the group's tiles); group 32 rides two words and the
    decode selects by k16 block — no kernel changes, only the strip repack
    and this class."""

    def __init__(self, group: int = 128):
        assert group == 32 or group % 64 == 0, f"group {group} must be 32 or a multiple of 64"
        self.group = group
        self.name = "int4" if group == 128 else f"int4_g{group}"
        self.sf_words = 2 if group == 32 else 1

    def make_consts(self):
        return U.make_i4_decode_consts()

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        r0, r1, r2, r3 = U.decode_u4b8x8_to_bf16x8(xw[b], consts)
        if const_expr(self.group == 32):
            h = sfw[b // 2]  # word per 32-column group (b is a static int)
        else:
            h = sfw[0]  # already a (sf_r, sf_r8) bf16 pair
        return _mul4(r0, r1, r2, r3, h)

    def quantize_reference(self, w):
        return U.quantize_int4_reference(w, group=self.group)

    def dequant_reference(self, q, sf):
        return U.dequant_int4_reference(q, sf, group=self.group)

    def prepare(self, q, sf):
        q, sf = _pad_n128(q, sf)
        # blob word order tracks the decode branch: s2f6 hw-cvt on Blackwell
        blob = U.repack_int4_weight(q, s2f6=U.use_s2f6_int_cvt())
        return blob, U.repack_int4_sf(sf, q.shape[1] * 2, group=self.group)


class Int4Awq(DecodeFormat):
    """AWQ asymmetric int4: (q - z) * s computed as HADD2(magic, c) * s with
    c = -(128 + z) — magic + c = q - z is an EXACT small integer, so the
    result rounds ONCE (no pre-rounded bias constant); sf is (scales, zeros).
    Same op count as symmetric int4 (the zero rides the bias-add's addend),
    and HMUL2 issues at ~2x the rate of the HFMA2 it replaced (which
    occupied both fma-pipe halves). The rejected alternative — folding
    everything into one HFMA2 with b' = -s*(z + 128) — pre-rounds b' at
    ~0.5s: see memory project_transform_a.

    Stays on the magic route on Blackwell too: the s2f6 hw-cvt decode would
    yield q-8 instead of magic form, which needs the strip's add constant
    rebuilt as 8-z in place of -(128+z) — an arch-dependent SF blob for a
    2-op saving. Revisit only if sm120 W4A16-AWQ shows decode-bound."""

    name = "int4awq"
    sf_words = 2

    def make_consts(self):
        return U.pin_b32_vreg(0x43004300)  # magic only; the bias rides the strip

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        t0, t1, t2, t3 = U.decode_u4x8_to_magicx8(xw[b], consts)
        sp, cp = sfw[0], sfw[1]
        return (
            U.mul_bf16x2_bcast(U.add_bf16x2_bcast(t0, cp, False), sp, False),
            U.mul_bf16x2_bcast(U.add_bf16x2_bcast(t1, cp, True), sp, True),
            U.mul_bf16x2_bcast(U.add_bf16x2_bcast(t2, cp, False), sp, False),
            U.mul_bf16x2_bcast(U.add_bf16x2_bcast(t3, cp, True), sp, True),
        )

    def quantize_reference(self, w):
        q, scales, zeros = U.quantize_int4_awq_reference(w)
        return q, (scales, zeros)

    def dequant_reference(self, q, sf):
        scales, zeros = sf
        return U.dequant_int4_awq_reference(q, scales, zeros)

    def prepare(self, q, sf):
        scales, zeros = sf
        q, _ = _pad_n128(q, None)
        n_p = q.shape[0]
        if scales.shape[0] < n_p:
            pad = n_p - scales.shape[0]
            scales = torch.cat([scales, scales.new_ones(pad, scales.shape[1])])
            zeros = torch.cat([zeros, zeros.new_full((pad, zeros.shape[1]), 8)])
        return U.repack_int4_weight(q), U.repack_int4_awq_sf(scales, zeros, q.shape[1] * 2)


class Int4Sm(DecodeFormat):
    """W4A8: sign-magnitude int4 g128 weights x e4m3 per-token activations.
    Decodes raw integers -8..7 to e4m3 (prmt LUT + sign planes, EXACT — no
    scale fold: the 8-bit product grid can't hold it), fp8 QGMMA per k32
    block, and the group scale promotes in fp32 at k-tile end (tile_k 128 ==
    group, so it's constant per tile). ``b`` indexes k32 blocks: two raw
    words each, four e4m3x4 fragment registers out — the same slot order as
    the bf16 formats' k16 blocks. The per-token activation scale is an
    output-column factor applied in the epilogue (see gemm_w4.gemm_w4a8)."""

    name = "int4sm"
    tile_k = 128
    sf_words = 1
    mma_dtype = cutlass.Float8E4M3FN
    promote = True

    def make_consts(self):
        return U.make_e4m3_luts()

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        r0, r1 = U.decode_i4smx8_to_e4m3x8(xw[2 * b], consts)
        r2, r3 = U.decode_i4smx8_to_e4m3x8(xw[2 * b + 1], consts)
        return r0, r1, r2, r3

    @cute.jit
    def promote_scale_pair(self, sfp):
        """Strip word = the (sf_r, sf_r8) bf16 pair -> two exact f32."""
        sf0 = U.i32_as_f32(sfp << 16)
        sf1 = U.i32_as_f32(sfp & cutlass.Int32(U._s32(0xFFFF0000)))
        return sf0, sf1

    def quantize_reference(self, w):
        return U.quantize_int4sm_reference(w)

    def dequant_reference(self, q, sf):
        return U.dequant_int4sm_reference(q, sf)

    def prepare(self, q, sf):
        q, sf = _pad_n128(q, sf)
        return U.repack_w4a8_weight(q), U.repack_w4a8_sf(sf, q.shape[1] * 2)


class Int4SmFold(DecodeFormat):
    """Folded (no-drain) W4A8: same blob as int4sm, but the group scale is
    folded INTO the e4m3 decode via per-(row, k-tile) scaled magnitude
    tables rebuilt from the exact bf16 strip scale at each tile's block 0
    (build_tile_state: 6 FMUL + 2 packed cvts per row, amortized over 64
    values) — the per-word decode cost is identical to int4sm. No promote:
    the kernel runs the plain fast-accum fp8 mainloop at full pipelining.
    Accuracy: ONE e4m3 rounding of m * sf per weight (~2^-4 rel worst case)
    + the fast-accum fp8 chain; the exact alternative is int4sm. The
    channel scale from fold_int4sm_scales (|7 sf| <= 448 normalization) and
    the per-token act scale are epilogue vec factors (gemm_w4.gemm_w4a8)."""

    name = "int4smf"
    tile_k = 128
    sf_words = 1
    tile_state_words = 4
    mma_dtype = cutlass.Float8E4M3FN

    def make_consts(self):
        return None

    @cute.jit
    def build_tile_state(self, sfw, consts, ts):
        sf0 = U.i32_as_f32(sfw[0] << 16)
        sf1 = U.i32_as_f32(sfw[0] & cutlass.Int32(U._s32(0xFFFF0000)))
        ts[0], ts[1] = U.build_i4sm_scaled_luts(sf0)
        ts[2], ts[3] = U.build_i4sm_scaled_luts(sf1)

    @cute.jit
    def decode_k16(self, xw, ts, b, consts):
        r0, r1 = U.decode_i4smx8_scaled_e4m3x8(xw[2 * b], ts[0], ts[1], ts[2], ts[3])
        r2, r3 = U.decode_i4smx8_scaled_e4m3x8(xw[2 * b + 1], ts[0], ts[1], ts[2], ts[3])
        return r0, r1, r2, r3

    def quantize_reference(self, w):
        q, sf = U.quantize_int4sm_reference(w)
        return q, U.fold_int4sm_scales(sf)

    def dequant_reference(self, q, sf):
        sf_folded, chan = sf
        return U.dequant_int4smf_reference(q, sf_folded, chan)

    def prepare(self, q, sf):
        sf_folded, chan = sf
        q, sf_folded = _pad_n128(q, sf_folded)
        return U.repack_w4a8_weight(q), U.repack_w4a8_sf(sf_folded, q.shape[1] * 2)


class Int4B8(DecodeFormat):
    """EXPERIMENTAL biased W4A8 (cutlass PR #3432's mixed-input trick on the
    int4sm promote seam): weights stored offset-binary (code = q+8, same
    [-7, 7] grid as int4sm) decode by pure unpack — e4m3 bytes 0x00-0x0F are
    the linear denormal grid (q+8)*2^-9, so the 5-op decode is EXACT (no
    int4smf fold rounding). The promote scale carries the 2^9 (sf * 512),
    leaving acc = sum_blk sf * ((q+8) @ b); the bias is removed pre-store by
    an epilogue TileLoad correction c[m,n] = -8 * sum_blk rowsum_act[m,blk] *
    sf[n,blk] — a rank-K/128 term (small host GEMM feeding a TileLoad, NOT a
    vec op): D = (acc + c) * token_scale. roundtrip=False: the fixture has no
    correction seam; AI/accuracy_w4a8_biased.py is the harness.

    MEASURED (H100, 2026-08-04, M=512 N=1024 K=512..8192): QGMMA takes the
    denormal e4m3 operands exactly (one-hot gate bit-exact). Zero-mean acts:
    rel_rms 1.68e-3 = int4sm = the bf16 store floor, 15x better than
    int4smf. Positive-mean acts (relu, the down-proj regime): 0.8-2.9e-2
    growing with K — at/below int4smf, 5-17x worse than int4sm. The
    truncating fast-accum commits a systematic downward error proportional
    to the (q+8)-inflated positive wave that the exact fp32 correction
    cannot see. VERDICT: dominated by int4sm (same promote drain, exact,
    regime-independent) — kept as a documented negative result."""

    name = "int4b8"
    tile_k = 128
    sf_words = 1
    mma_dtype = cutlass.Float8E4M3FN
    promote = True
    roundtrip = False  # gemm output is biased; the harness applies the correction

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        r0, r1 = U.decode_u4b8x8_to_e4m3x8_biased(xw[2 * b])
        r2, r3 = U.decode_u4b8x8_to_e4m3x8_biased(xw[2 * b + 1])
        return r0, r1, r2, r3

    @cute.jit
    def promote_scale_pair(self, sfp):
        """int4sm's exact bf16-pair unpack x 512 (exact pow2 in f32): the
        decode's 2^-9 cancels here, so acc accumulates sf * (q+8) @ b."""
        sf0 = U.i32_as_f32(sfp << 16) * 512.0
        sf1 = U.i32_as_f32(sfp & cutlass.Int32(U._s32(0xFFFF0000))) * 512.0
        return sf0, sf1

    def quantize_reference(self, w):
        return U.quantize_int4b8_reference(w)

    def dequant_reference(self, q, sf):
        return U.dequant_int4_reference(q, sf, group=128)

    def prepare(self, q, sf):
        q, sf = _pad_n128(q, sf)
        return U.repack_w4a8_weight(q), U.repack_w4a8_sf(sf, q.shape[1] * 2)


class Mxfp4(DecodeFormat):
    """e2m1 nibbles + e8m0 scale per 32 k (exact: power-of-2 scale)."""

    name = "mxfp4"
    sf_words = 1

    def make_consts(self):
        return U.make_decode_luts()

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        if const_expr(U._arch_has_bf16_narrow_cvt()):
            # pow2 scale rides the converter's fused ue8m0 operand: decode +
            # scale + strip unpack in 5 SASS per 8 values
            return U.decode_e2m1x8_mul_e8m0pair_cvt(xw[b], sfw[0], (b // 2) * 2)
        r0, r1, r2, r3 = U.decode_e2m1x8_to_bf16x8(xw[b], consts)
        h = U.sf_e8m0_pair_to_bf16x2(sfw[0], (b // 2) * 2)
        return _mul4(r0, r1, r2, r3, h)

    def quantize_reference(self, w):
        return U.quantize_mxfp4_reference(w)

    def dequant_reference(self, q, sf):
        return U.dequant_mxfp4_reference(q, sf)

    def prepare(self, q, sf):
        q, sf = _pad_n128(q, sf)
        return U.repack_nvfp4_weight(q), U.repack_mxfp4_sf(sf)


class Int8(DecodeFormat):
    """int8 weights; dp4a sign-extract decode (s2f6 hw cvt on Blackwell);
    the per-channel scale is NOT applied here (epilogue's job — needs the
    multiplicative colvec op)."""

    name = "int8"
    w8 = True
    roundtrip = False  # per-channel scale lives in the epilogue, not the bundle

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        ra, rb = U.decode_i8x4_to_bf16x4(xw[2 * b])
        rc, rd = U.decode_i8x4_to_bf16x4(xw[2 * b + 1])
        return ra, rb, rc, rd

    def quantize_reference(self, w):
        return U.quantize_int8_reference(w)

    def prepare(self, q, sf):
        return U.repack_w8a16_weight(q), None


class Fp8(DecodeFormat):
    """e4m3 weights; per-channel scale applied in the epilogue (see Int8)."""

    name = "fp8"
    w8 = True
    roundtrip = False  # per-channel scale lives in the epilogue (see Int8)

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        ra, rb = U.decode_e4m3x4_to_bf16x4(xw[2 * b])
        rc, rd = U.decode_e4m3x4_to_bf16x4(xw[2 * b + 1])
        return ra, rb, rc, rd

    def quantize_reference(self, w):
        return U.quantize_fp8_reference(w)

    def prepare(self, q, sf):
        return U.repack_w8a16_weight(q.view(torch.int8)), None


class Mxfp8(DecodeFormat):
    """e4m3 weights + e8m0 scale per 32 k, folded in the decode (exact)."""

    name = "mxfp8"
    w8 = True
    sf_words = 1

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        ra, rb = U.decode_e4m3x4_to_bf16x4(xw[2 * b])
        rc, rd = U.decode_e4m3x4_to_bf16x4(xw[2 * b + 1])
        h = U.sf_e8m0_pair_to_bf16x2(sfw[0], (b // 2) * 2)
        return _mul4(ra, rb, rc, rd, h)

    def quantize_reference(self, w):
        return U.quantize_mxfp8_reference(w)

    def dequant_reference(self, q, sf):
        return U.dequant_mxfp8_reference(q, sf)

    def prepare(self, q, sf):
        return U.repack_w8a16_weight(q.view(torch.int8)), U.repack_mxfp4_sf(sf)


class Qtip(DecodeFormat):
    """QTIP trellis-coded 4-bit (arXiv 2406.11235), lookup-free bf16 "3INST"
    decode; one thread's 16 raw bytes = one tail-biting bitshift-trellis
    stream for its 32 fragment values (see formats/qtip.py). No SF strip:
    values decode at codebook scale, the per-tensor weight scale rides the
    epilogue alpha."""

    name = "qtip"

    def make_consts(self):
        return Q.make_qtip_consts()

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        return Q.decode_qtip_k16(xw, b, consts)

    def quantize_reference(self, w):
        return Q.quantize_qtip_reference(w)

    def dequant_reference(self, q, sf):
        return Q.dequant_qtip_reference(q, sf)

    def prepare(self, q, sf):
        g = q.shape[0]
        g_pad = (2 - g % 2) % 2  # pad N to a 128 multiple (tile granularity);
        if g_pad:  # padded rows decode to (finite) garbage, callers slice
            q = torch.cat([q, q.new_zeros(g_pad, *q.shape[1:])])
        return q.contiguous(), None


class Qtip2(DecodeFormat):
    """QTIP bitshift trellis, V=2 variant: one 16-bit state decodes a whole
    bf16x2 pair, streams are 32 B / 64 values per thread (tile_k=128), and
    the hash is a pure 64-bit multiply with the addend folded into the mask
    lop3s — 6 SASS per pair vs qtip's 11, and slightly LOWER quantization
    MSE (the V=2 step pins 8 tail-biting boundary bits instead of 12, and
    T=64 halves the short-sequence tax). See formats/qtip.py."""

    name = "qtip2"
    tile_k = 128

    def make_consts(self):
        return Q.make_qtip2_consts()

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        return Q.decode_qtip2_k16(xw, b, consts)

    def quantize_reference(self, w):
        return Q.quantize_qtip2_reference(w)

    def dequant_reference(self, q, sf):
        return Q.dequant_qtip2_reference(q, sf)

    def prepare(self, q, sf):
        g = q.shape[0]
        g_pad = (2 - g % 2) % 2  # pad N to a 128 multiple (tile granularity);
        if g_pad:  # padded rows decode to (finite) garbage, callers slice
            q = torch.cat([q, q.new_zeros(g_pad, *q.shape[1:])])
        return q.contiguous(), None


class Qtip2S(Qtip2):
    """qtip2's 6-SASS V=2 decode on qtip's SHORT (16-byte, 32-value,
    tile_k=64) streams: same smem/pipeline profile as qtip — the drop-in
    replacement, 1.06x qtip at m=4096 — at slightly better MSE (0.00540 vs
    0.00544; the full qtip2's T=64 gets 0.00495 but its 2x k-tile halves
    the pipeline stages, which costs compute-bound shapes more than the
    decode ALU saves). Hash is 2x mad.lo (IMAD.WIDE's half-rate 64-bit path
    is what made the wide hash lose prefill)."""

    name = "qtip2s"
    tile_k = 64

    def make_consts(self):
        return Q.make_qtip_consts()

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        return Q.decode_qtip2s_k16(xw, b, consts)

    def quantize_reference(self, w):
        return Q.quantize_qtip2_reference(w, tile_k=64, values=Q.qtip2s_state_values(w.device))

    def dequant_reference(self, q, sf):
        return Q.dequant_qtip2_reference(q, sf, values=Q.qtip2s_state_values(q.device))


W4_FORMATS = {
    f.name: f
    for f in (
        Nvfp4(),
        Int4(),
        Int4(group=64),
        Int4(group=32),
        Int4Awq(),
        Int4Sm(),
        Int4SmFold(),
        Int4B8(),
        Mxfp4(),
        Int8(),
        Fp8(),
        Mxfp8(),
        Qtip(),
        Qtip2(),
        Qtip2S(),
    )
}


def decode_format(fmt):
    """Resolve a format name or instance to a DecodeFormat."""
    if isinstance(fmt, DecodeFormat):
        return fmt
    return W4_FORMATS[fmt]
