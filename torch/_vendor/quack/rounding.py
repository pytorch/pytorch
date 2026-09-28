# Copyright (c) 2025-2026, Vijay Thakkar, Tri Dao.
"""Rounding mode control and stochastic rounding primitives for GEMM epilogues.

Provides a RoundingMode enum for configuring how epilogues downconvert the
accumulator dtype (typically FP32) to the output dtype before storing to gmem.
Stochastic rounding (RS) uses the hardware cvt.rs.satfinite.bf16x2.f32 PTX
instruction on SM100/SM103 (the .rs rounding mode only exists on
sm_100a/sm_103a — ptxas rejects it even on sm_110a/sm_120a); everywhere else
it uses a bit-exact software emulation (see cvt_f32x2_bf16x2_rs_sw). Per the
PTX ISA docs the bf16/f16 emulations assemble from sm_80 up (verified for
sm_80/90/110/120), the fp8 ones from sm_89 up (their
cvt.rn.satfinite.f8x2.f16x2 closer does not exist on sm_80), and the e2m1
(fp4) one from sm_80 up (it needs no f8 closer — the fp4 grid is reached
through f16, see cvt_f32x4_e2m1x4_rs_sw); on sm_100+ targets without hw .rs
(sm_100f/110/120/121) the fp4 emulation swaps its integer nibble extraction
for the direct cvt.rn.satfinite.e2m1x2.f32 closer, bit-identically
(cvt_f32x4_e2m1x4_rs_direct). Both paths consume the same
Philox-generated random bits, so 16-bit RS output is bitwise identical across
architectures for a given seed (bf16/f16 probed straight on both sm_100a and
sm_103a). The x4 narrow converts are NOT: sm_103a's f8x4/e2m1x4 instructions
permute the rand bytes across values and consume the odd values' bytes
bit-reversed (see cvt_f32x4_e2m1x4_rs), so fp8 hw output matches the sw
emulation bitwise on sm_100a only (fp4's sm_100a rand order is unprobed) —
still unbiased and statistically identical everywhere, but 8/4-bit seeds are
not bit-portable across sm_100a/sm_103a/sw.
"""

from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32, Uint64
from cutlass.base_dsl.arch import Arch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass._mlir_helpers.arith import bitcast as _bitcast
from cutlass.cutlass_dsl import dsl_user_op, Int32, T


def _f32_bits(x: Float32) -> Uint32:
    """Reinterpret an f32 as its u32 bit pattern (no instruction generated)."""
    return Uint32(_bitcast(Float32(x).ir_value(), T.i32()))


def _bits_f32(x: Uint32) -> Float32:
    """Reinterpret a u32 bit pattern as f32 (no instruction generated)."""
    return Float32(_bitcast(Uint32(x).ir_value(), T.f32()))


def _asm(res_type, ptx: str, constraints: str, args):
    """llvm.inline_asm with has_side_effects=False, shared by every PTX op here.

    All snippets in this module are PURE value computations (converts, fma,
    predicated adds). Declaring them side-effect-free is load-bearing: it
    lets LLVM CSE the Philox key chains shared across batches, DCE converts
    for unused lanes, and schedule/software-pipeline the SR stream across
    the epilogue barriers. Do NOT switch these to cute.arch.inline_ptx — it
    lowers to nvvm.inline_ptx, which is modeled as effectful (no CSE/DCE,
    ordered against other effectful ops incl. barriers); that is right for
    mbarrier/tensormap PTX, a pessimization for pure math.
    """
    return llvm.inline_asm(
        res_type, args, ptx, constraints, has_side_effects=False, is_align_stack=False
    )


class RoundingMode(IntEnum):
    """Rounding modes for epilogue dtype downconversion.

    RN — Round to nearest even (default hardware behavior)
    RS — Stochastic rounding (BF16/FP16 output; hw cvt.rs on SM100/SM103,
         bit-exact software emulation elsewhere)
    """

    RN = 0
    RS = 1


# Odd strides used to derive distinct Philox counters across output tiles and subtiles.
EPILOGUE_SR_SEED_M_STRIDE = 65537
EPILOGUE_SR_SEED_N_STRIDE = 257
EPILOGUE_SR_SEED_BATCH_STRIDE = 17
EPILOGUE_SR_SEED_SUBTILE_STRIDE = 7
EPILOGUE_SR_SEED_AUX_OUT_SALT = 0x9E3779B1

# 7 rounds is the Random123 BigCrush minimum (their margin recommendation is
# 10). The SM90 GEMM epilogue cost of RS vs RN (H100, 128x192 pingpong) is set
# by the Philox->cvt DEPENDENCY CHAIN, not instruction count: ablations
# measure the converter alone and the entropy consumption alone at ~0%, yet
# full SR costs +3% at 8192^3 (hidden under the mainloop) and +29% at 8Kx8K
# K=512 (epilogue-exposed: one epilogue warp per SMSP under the 168-register
# ceiling keeps only ~3-4 of the ~12 independent Philox batches per subtile
# in flight; the rest serialize in ~70-clk waves). n_rounds is the only big
# lever: 5 -> +21%, 3 -> +5% at K=512 (changes same-seed outputs; cross-arch
# parity is unaffected since SM100 runs the same Philox). Offloading entropy
# generation to the idle producer-WG warps through an smem ring was built,
# verified bitwise-identical, and measured NET-NEGATIVE at every shape (+55%
# vs +29% at K=512): Philox costs issue slots wherever it runs — inline
# spreads them over 8 math warps while the producer WG has only 2-3 idle
# ones, and ring backpressure confines production to the epilogue-locked
# windows where issue is already saturated. Don't rebuild it. When
# re-measuring any of this: interleave RN/RS (sequential 700W runs drift
# 8-15% thermally) and run one monkeypatched variant per process (the
# compile cache does not key on patched internals like n_rounds or a stubbed
# converter, so in-process A/B silently re-times the first kernel).
PHILOX_N_ROUNDS_DEFAULT = 7

PHILOX_ROUND_A = 0xD2511F53
PHILOX_ROUND_B = 0xCD9E8D57
PHILOX_KEY_A = 0x9E3779B9
PHILOX_KEY_B = 0xBB67AE85


@dsl_user_op
def epilogue_sr_seed(
    base_seed: Int32,
    tile_coord_mnkl: cute.Coord,
    subtile_idx,
    *,
    loc=None,
    ip=None,
) -> Int32:
    return base_seed + (
        tile_coord_mnkl[0] * EPILOGUE_SR_SEED_M_STRIDE
        + tile_coord_mnkl[1] * EPILOGUE_SR_SEED_N_STRIDE
        + tile_coord_mnkl[3] * EPILOGUE_SR_SEED_BATCH_STRIDE
        + subtile_idx * EPILOGUE_SR_SEED_SUBTILE_STRIDE
    )


@dsl_user_op
def epilogue_aux_out_sr_seed(
    base_seed: Int32,
    tile_coord_mnkl: cute.Coord,
    subtile_idx,
    *,
    loc=None,
    ip=None,
) -> Int32:
    return epilogue_sr_seed(
        base_seed + EPILOGUE_SR_SEED_AUX_OUT_SALT,
        tile_coord_mnkl,
        subtile_idx,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mul_wide_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> tuple:
    """Unsigned 32b x 32b -> 64 wide multiply via PTX `mul.wide.u32`.

    Returns (hi, lo) as a pair of Uint32 values.
    """
    prod = cute.arch.mul_wide(Uint32(a), Uint32(b), loc=loc, ip=ip)
    # ptxas folds the shift and to(Uint32) into the same IMAD.WIDE.U32 register pair
    # that the previous inline PTX mov.b64 split produced.
    hi = (prod >> 32).to(Uint32, loc=loc, ip=ip)
    lo = prod.to(Uint32, loc=loc, ip=ip)
    return hi, lo


@dsl_user_op
def cvt_f32x2_bf16x2_rs(
    a: Float32,
    b: Float32,
    rand_bits: Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Convert 2 FP32 values to packed BF16x2 using stochastic rounding.

    Uses Blackwell PTX instruction: cvt.rs.satfinite.bf16x2.f32 dst, src_hi, src_lo, rand
    """
    return cutlass.Int32(
        _asm(
            T.i32(),
            "cvt.rs.satfinite.bf16x2.f32 $0, $2, $1, $3;",
            "=r,f,f,r",
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Uint32(rand_bits).ir_value(loc=loc, ip=ip),
            ],
        )
    )


BF16_MAX_NORM = 3.3895313892515355e38  # 0x7F7F0000 as f32: largest finite bf16


@dsl_user_op
def add_u32_if(
    x: Uint32,
    incr: Uint32,
    pred: cutlass.Boolean,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Return ``x + incr if pred else x`` as a single predicated SASS add.

    The "b" constraint passes the DSL Boolean as a PTX .pred register and the
    "0" tied constraint makes the add in-place, so ptxas emits one predicated
    IADD3 instead of the SEL + IADD3 (or a branch) that a Python if/select
    lowers to.
    """
    return Uint32(
        _asm(
            T.i32(),
            "@$2 add.u32 $0, $1, $3;",
            "=r,0,b,r",
            [
                Uint32(x).ir_value(loc=loc, ip=ip),
                cutlass.Boolean(pred).ir_value(loc=loc, ip=ip),
                Uint32(incr).ir_value(loc=loc, ip=ip),
            ],
        )
    )


@dsl_user_op
def cvt_f32x2_bf16x2_rz_satfinite(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int32:
    """Pack cvt(a) into d[15:0] and cvt(b) into d[31:16] with RZ + satfinite.

    Single F2FP instruction on SM90/SM120. RZ truncates the 16 discarded
    mantissa bits; satfinite maps NaN -> 0x7FFF and |x| > BF16_MAX_NORM
    (incl. Inf) -> sign|0x7F7F, the same special-value rules as the SM100
    cvt.rs.satfinite instruction.
    """
    return cutlass.Int32(
        _asm(
            T.i32(),
            "cvt.rz.satfinite.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b)],
        )
    )


@dsl_user_op
def cvt_f32x2_bf16x2_rs_sw(
    a: Float32,
    b: Float32,
    rand_bits: Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Software emulation of cvt.rs.satfinite.bf16x2.f32 (any sm_80+ target).

    Bit-exact with the SM100 instruction. Per PTX ISA, .rs rounds toward/away
    from zero based on the carry out of the integer addition of the 16 random
    bits to the 16 discarded mantissa bits, which in sign-magnitude is simply
    ``bits(x) + r16`` followed by truncation of the low 16 bits. The adds are
    predicated off for special inputs (NaN, or |x| > BF16_MAX_NORM incl. Inf,
    whose rounding is input-determined) so the trailing cvt.rz.satfinite
    reproduces the hardware special-value results.

    7 SASS instructions on SM90/SM120 (2 FSETP.GTU, LOP3, PRMT, 2 predicated
    IADD3-class adds, F2FP.SATFINITE.BF16.F32.PACK_AB.RZ) vs 1 for the SM100
    instruction. Every PTX op here is sm_80+ (the binding constraint is the
    bf16x2 cvt destination format; .satfinite on it needs PTX ISA 8.1 =
    CUDA >= 12.1).

    7 is the practical floor for the EXACT emulation (explored exhaustively
    on sm_90a; every variant below was bit-checked vs a reference model on
    4M vectors and benched in isolation). Each element needs its own
    MAX_NORM compare (no packed-f32 compare exists, and sharing one
    predicate across the pair is wrong for mixed special/normal pairs) and
    its own conditional 16-bit-positioned add (IADD3 has no shifted-operand
    form), plus one rbits extraction and the F2FP. That sums to 6 only if
    ptxas would emit a PREDICATED LEA.HI (fused shift+add) — it refuses
    (builds IMAD.HI + 2 staging MOVs for the 64-bit addend pair) and
    likewise refuses to predicate IDP (emits IDP+SEL), so 7, reached by
    three independent formulations (this one, shr-based, dp2a-based).
    Under GEMM scheduling pressure ptxas sometimes de-predicates add_u32_if
    into IMAD.IADD + SEL (9/pair); harmless — it shortens the pred->use
    chain — don't fight it. Rejected routes, don't retry:
      - 3-instr dp2a form (dp2a.lo(r,0x0001,ua) / dp2a.lo(r,0x0100,ub) +
        F2FP): runs at RN-baseline speed but corrupts specials — Inf+r and
        the (MAX_NORM, Inf) sliver land in NaN bit patterns, and NaN
        payloads >= 0x7FFF0001 wrap into the sign bit.
      - every packed-16-bit-lane reformulation (SWAR 32b add, add.u16x2 +
        carry vector, HSET2/VIMNMX ulp tricks, guard-as-dp2a-multiplier):
        all die on the same wall — sm_90a has no per-lane carry-out
        primitive, and the carry of (discarded bits + r16) IS the SR signal.
      - FMNMX.NAN pre-clamp: canonicalizes NaN to 0x7FFFFFFF, then +r16
        wraps into the sign bit (NaN comes out -0/-subnormal).
      - batch-guard (FMNMX.NAN specials tree per 8 elements, branch to the
        dp2a fast path): bit-exact and asymptotically 5/pair, but
        BSSY/BSYNC branch overhead loses (1.38 vs 2.37 Tpairs/s).
      - llm.c-style threshold-compare: same distribution, different
        rand->decision mapping — not bit-compatible with cvt.rs.
    Isolated H100 throughput 2.37 Tpairs/s (plain-RN F2FP baseline 5.42);
    even the slowest exact variant tried (1.28) outruns HBM store
    bandwidth, so the converter never bottlenecks an epilogue — Philox
    does (see PHILOX_N_ROUNDS_DEFAULT).
    """
    a, b, rand_bits = Float32(a), Float32(b), Uint32(rand_bits)
    # False for NaN (unordered compare), +-Inf, and finite |x| > MAX_NORM;
    # ptxas folds |x| and the immediate into a single FSETP per element.
    a_in_range = abs(a) <= BF16_MAX_NORM
    b_in_range = abs(b) <= BF16_MAX_NORM
    r_lo = rand_bits & 0xFFFF
    # r >> 16, computed via prmt (each selector nibble picks a source byte:
    # "2,3" move r's top two bytes to the bottom, "4,4" fill the top with
    # zeros from the second operand). prmt costs the same as a shift (one ALU
    # instruction); we use it because it is OPAQUE to a ptxas peephole that
    # backfires here: written as shr, ptxas fuses shift+add into
    # IMAD.HI ub, r, 0x10000, c (= (r*2^16 + c) >> 32). Unpredicated that
    # fusion is a win (it becomes a single LEA.HI), but IMAD.HI's addend c is
    # a 64-bit aligned register PAIR, so feeding the scalar ub in as ub<<32
    # requires staging {lo=0, hi=ub} with two MOVs — all predicated:
    #     @!P1 MOV R2, RZ
    #     @!P1 MOV R3, R13
    #     @!P1 IMAD.HI.U32 R13, R7, 0x10000, R2
    # i.e. 3 instructions where prmt + a single predicated IADD3 need 2 (the
    # ideal @P1 LEA.HI exists in SASS but ptxas never emits LEA.HI
    # predicated). r_lo has no such trap: LOP3 has no and+add fusion, so its
    # predicated add is always a single IADD3.
    r_hi = Uint32(cute.arch.prmt(rand_bits, 0, 0x4432, loc=loc, ip=ip))
    ua = add_u32_if(_f32_bits(a), r_lo, a_in_range, loc=loc, ip=ip)
    ub = add_u32_if(_f32_bits(b), r_hi, b_in_range, loc=loc, ip=ip)
    return cvt_f32x2_bf16x2_rz_satfinite(_bits_f32(ua), _bits_f32(ub), loc=loc, ip=ip)


# =============================================================================
# FP16 / FP8 stochastic rounding (sw emulation of SM100 cvt.rs; f16 needs
# sm_80+, f8 needs sm_89+ for the cvt.rn.satfinite.f8x2.f16x2 closer)
#
# PTX .rs semantics: add the K random bits to the mantissa bits discarded by
# the truncation, as integers; the CARRY OUT of that add rounds the kept
# value away from zero (P(round away) = discarded/2^K — unbiased, magnitude
# proportional to how far x sits past the grid point). bf16 shares f32's
# exponent width, so for it this is a plain integer add on the f32 bit
# pattern at a FIXED position (bits + r16, truncate). f16/f8 have narrower
# exponents, and that breaks in exactly one place: subnormal results, where
# the target grid stops scaling with e_x and becomes fixed-point, so the
# kept-LSB position inside the f32 word varies per element (a per-element
# variable shift, ill-defined below half-min-subnormal).
#
# Instead, restate the integer semantics in floating point. Adding r at the
# top of the discarded field is identical to
#     y  = x + r * (+-2^(max(e_x, emin) - S))   exactly, floored toward zero
#     out = cvt.rz.satfinite(y)
# because for normal results ulp(x) = 2^(e_x - m) and 2^(e_x - S) =
# ulp * 2^-K, while for subnormal results the emin clamp pins the increment
# to the fixed-point grid's ulp * 2^-K — in both cases r lands on the top K
# discarded bits, where the hw aligns its rand too (B200-probed). x's bits
# BELOW those K (the sub-r residue) cannot change the outcome: D_top + r is
# an integer and the residue < 1 in its units, so it never bridges
# D_top + r = 2^K - 1 to a carry.
#
# So we compute  y = fma.rz(rf, sc, x)  with rf = (float)r (exact, K <= 13
# bits) and sc = +-2^(max(e_x, emin) - S) built integer-side (K = 13 for
# f16, 8 for f8). fma.rz is a single-rounding magnitude-floor of the exact
# sum onto the f32 grid; the closer then floors onto the f16/f8 grid.
# Composing floors onto NESTED grids is exact (every f16/f8 point is an f32
# point), and a floor preserves the carry decision: if the exact sum reached
# the next grid point G, fl_rz(sum) >= G (G representable, RZ cannot drop
# below it); if not, RZ cannot round UP to G — which is why .rz and not .rn
# (RN could manufacture a false carry by rounding the intermediate up across
# a grid point).
#
# No guards are needed — the specials are fixed points of the arithmetic
# (unlike bf16's raw integer add, which would wrap NaN payloads/Inf through
# the sign bit and needs its adds predicated off): sc is always finite (even
# e_field 255 minus S stays a normal float), so NaN/Inf flow through the fma
# unchanged; fma.rz never rounds a finite x up to Inf (saturates); +-0 keeps
# its sign (delta < target min subnormal, floors back to zero); and the
# satfinite closer applies the same special-value rules as the SM100
# instruction (probed on H100: f16 NaN -> 0x7FFF, |x| > 65504 ->
# sign|0x7BFF; e4m3 NaN -> 0x7F, clamp -> sign|0x7E; e5m2 NaN -> 0x7F,
# clamp -> sign|0x7B).
# =============================================================================


@dsl_user_op
def fma_rz_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """fl_rz(a*b + c): single FFMA.RZ.

    Round-toward-zero makes this an exact magnitude-floor onto the f32 grid
    (never crosses a representable value upward, saturates instead of
    producing Inf) — the keystone of the f16/f8 SR emulations.
    """
    return Float32(
        _asm(
            T.f32(),
            "fma.rz.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
        )
    )


@dsl_user_op
def u16x2_to_f32(x: Uint32, half: int, *, loc=None, ip=None) -> Float32:
    """(float)(x.h0) or (x.h1): one I2F.U16 with a half-register source selector.

    The mov.b32 unpack is virtual; ptxas encodes the half select directly in
    the I2F operand (R.H0/R.H1), so 16-bit rand extraction costs zero extra
    instructions. NB: the sub-word I2F runs at 1/8 rate on the XU pipe, but
    it hides under the ALU wall of these converters — measured strictly
    faster than I2F.U32 + explicit extraction (see tools/sass/pipe_rates.cu).
    """
    assert half in (0, 1)
    reg = "lo" if half == 0 else "hi"
    return Float32(
        _asm(
            T.f32(),
            "{\n\t.reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, $1;\n\t"
            f"cvt.rn.f32.u16 $0, {reg};\n\t"
            "}",
            "=f,r",
            [Uint32(x).ir_value(loc=loc, ip=ip)],
        )
    )


def _sr_scale_fma(x: Float32, rf: Float32, min_scale_bits: int, exp_shift: int) -> Float32:
    """y = fl_rz(x + rf * (+-2^(max(e_x, emin) - exp_shift))).

    The scale +-2^(e-S) is built from x's exponent field, restructured for
    sm_120a's second ALU (which takes only 2-input ops — IADD/FMNMX/... but
    not LOP3/VIMNMX): the S << 23 subtraction commutes with the max, so it
    is applied FIRST as a 2-input IADD, and the subnormal-boundary clamp
    becomes a 2-input immediate-operand max.f32 (FMNMX) on the sign-free
    magnitude — when m - (S << 23) wraps (e_x < S - 127), the wrapped
    pattern is a negative float (never NaN: the masked exponent field has no
    mantissa bits, so the wrap lands on +-Inf / +-2^k patterns only) and
    loses the float max to the bound, which is exactly the sub-emin clamp
    case. x's sign then merges with one 3-input LOP3. Bit-identical
    sc to the previous umax-on-signed-patterns formulation (the sign bit
    participated in that unsigned compare only as a shared MSB), with per
    value 2 int-pipe ops instead of 4 on sm_120a (and one fewer ALU op on
    sm_90a, where FMNMX rates like VIMNMX). The sign riding along in sc
    makes the fma add the random increment away from zero (sign-magnitude
    .rs semantics without ever touching |x|). sc is always finite: e_field
    255 (Inf/NaN) minus S stays a normal float, and fma(rf, finite, NaN/Inf)
    propagates the special to the satfinite closer.
    """
    ux = _f32_bits(x)
    # NB: mask constants above 0x7FFFFFFF must be wrapped in Uint32, else the
    # DSL promotes the expression to i64.
    m = ux & Uint32(0x7F800000)  # exponent field as a float: +2^e
    t = _bits_f32(m - Uint32(exp_shift << 23))
    bound = _bits_f32(Uint32(min_scale_bits - (exp_shift << 23)))
    u = cutlass.max(t, bound)  # FMNMX, 2-input + immediate: ALU2-eligible
    sc = _f32_bits(u) | (ux & Uint32(0x80000000))  # one 3-input LOP3
    return fma_rz_f32(rf, _bits_f32(sc), x)


F16_RAND_MASK = 0x1FFF1FFF  # 13 rbits per value (PTX: 3 MSBs of each half must be 0)


@dsl_user_op
def cvt_f32x2_f16x2_rz_satfinite(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int32:
    """Pack cvt(a) into d[15:0], cvt(b) into d[31:16] with RZ + satfinite.

    Single F2FP; NaN -> 0x7FFF, |x| > 65504 (incl Inf) -> sign|0x7BFF.
    """
    return cutlass.Int32(
        _asm(
            T.i32(),
            "cvt.rz.satfinite.f16x2.f32 $0, $2, $1;",
            "=r,f,f",
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b)],
        )
    )


@dsl_user_op
def cvt_f32x2_f16x2_rs(
    a: Float32,
    b: Float32,
    rand_bits: Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Hardware f32x2 -> f16x2 stochastic rounding (sm_100a/sm_103a only).

    a -> d[15:0] using rand_bits[12:0], b -> d[31:16] using rand_bits[28:16].
    rand_bits is passed UNMASKED: the PTX ISA asks for the 3 MSBs of each
    16-bit half to be zero, but the hw ignores bits 15:13 of each half
    (probed on B300 sm_103a: raw vs masked rbits bit-identical on 2M random
    + targeted cases — each MSB isolated, specials, subnormal results), so
    skipping the mask LOP3 still consumes the same 13 rbits the sw
    emulation masks off explicitly, and hw/sw parity holds. Re-probe if a
    new .rs-capable architecture appears.
    """
    return cutlass.Int32(
        _asm(
            T.i32(),
            "cvt.rs.satfinite.f16x2.f32 $0, $2, $1, $3;",
            "=r,f,f,r",
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Uint32(rand_bits).ir_value(loc=loc, ip=ip),
            ],
        )
    )


@dsl_user_op
def cvt_f32x2_f16x2_rs_sw(
    a: Float32,
    b: Float32,
    rand_bits: Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Software emulation of cvt.rs.satfinite.f16x2.f32 (any sm_80+ target).

    16 SASS instructions per pair (vs 1 on SM100): shared 13-bit rand mask
    (LOP3), 2x I2F.U16 with half selectors, per element LOP3 x2 + IMNMX +
    IADD + FFMA.RZ for the scaled exact-floor add, and one
    F2FP.SATFINITE.F16.F32.PACK_AB.RZ closer. Bit-exact incl. f16 subnormal
    results, +-0, NaN and satfinite clamping (verified vs a reference model
    on 4M vectors, and bitwise against the hw cvt.rs on B200 — incl. the
    subnormal-result rand alignment).
    """
    a, b, rand_bits = Float32(a), Float32(b), Uint32(rand_bits)
    rm = rand_bits & F16_RAND_MASK
    rf_a = u16x2_to_f32(rm, 0, loc=loc, ip=ip)
    rf_b = u16x2_to_f32(rm, 1, loc=loc, ip=ip)
    # emin = 2^-14 (below it the f16 grid is fixed-point 2^-24); S = 23 so
    # delta = r * 2^(e-23) = r * f16_ulp(x) * 2^-13.
    ya = _sr_scale_fma(a, rf_a, 0x38800000, 23)
    yb = _sr_scale_fma(b, rf_b, 0x38800000, 23)
    return cvt_f32x2_f16x2_rz_satfinite(ya, yb, loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# FP8 (e4m3 / e5m2), via the f16 grid. e5m2 is exactly f16 truncated to 2
# mantissa bits — same exponent bias, and the subnormal grids align (e5m2
# subnormal LSB 2^-16 sits at f16 mantissa bit 8) — so flooring y onto the
# e5m2 grid is cvt.rz.satfinite.f16x2 followed by ONE packed mask (0xFF00FF00)
# for the pair, in every region incl. subnormals, NaN (quiet bit is mantissa
# bit 9, survives the mask), Inf/clamp (0x7BFF -> 0x7B00 = 57344 = e5m2 max)
# and +-0. e4m3's subnormal knee is 2^-6, not f16's 2^-14: pre-scaling x by
# 2^-8 aligns the knees, making the e4m3*2^-8 grid exactly "f16 keep-3-
# mantissa-bits" (mask 0xFF80FF80; f16-subnormal masking = fixed-point 2^-17 =
# the scaled 2^-9 grid). The scale is exact in f32 (only |x| < 2^-118 rounds,
# far below the 2^-17 always-to-zero threshold, and the fma's clamped
# delta < 2^-17 - 2^-25 cannot bridge the gap), and one packed HMUL2 by 2^8
# unscales exactly (on-grid values x 2^8 are e4m3 values, all exact in f16;
# the 0x7B80 clamp remnant overflows to Inf which the closer maps to
# sign|0x7E, same as hw). The f16x2 -> f8x2 cvt.rn.satfinite closer converts
# exactly-representable values exactly and supplies the NaN/clamp rules.
#
# Rejected variants (all bit-checked and benched on sm_90a; don't retry):
# magic-number rand floats (as_float(0x3F800000 | r) - 1.0 instead of
# I2F.U16) need an extra clamp to keep sc finite — 22 vs 16 SASS/pair for
# f16; a unified magic-C floor built from sc (C = sc * 2^31) overflows C's
# exponent at e_x >= 108 (Inf - Inf = NaN) unless re-guarded; e4m3
# unscaling in the f16 domain BEFORE the mask rounds across grid points at
# the 2^-17 boundary (the prescale must happen in f32); the same
# f16-truncation trick applied to f32->f16 SR itself (prescale 2^-112)
# rounds away the entropy bits; predication-split converters (bf16-style
# guarded int-add for normal results + fma for subnormals) lose because
# ptxas refuses predicated FFMA (materializes FFMA+SEL, 21 vs 16/pair).
# These converters are ISSUE-bound in isolation (~1 IPC/SMSP): minimize
# instruction count, use pipe spread only as a tiebreak.
# ---------------------------------------------------------------------------


def _cvt_f32x2_f8x2_rs_sw(a, b, rand_bits, f8_kind, loc, ip):
    kind = _F8_KINDS[f8_kind]
    a, b, rand_bits = Float32(a), Float32(b), Uint32(rand_bits)
    # place rand bytes 0/1 into u16 halves: {0, r.b1, 0, r.b0}, then I2F.U16
    p01 = Uint32(cute.arch.prmt(rand_bits, 0, 0x4140, loc=loc, ip=ip))
    rf_a = u16x2_to_f32(p01, 0, loc=loc, ip=ip)
    rf_b = u16x2_to_f32(p01, 1, loc=loc, ip=ip)
    if kind["prescale"] is not None:
        a, b = a * kind["prescale"], b * kind["prescale"]  # FMUL x2, exact
    ya = _sr_scale_fma(a, rf_a, kind["min_scale"], kind["exp_shift"])
    yb = _sr_scale_fma(b, rf_b, kind["min_scale"], kind["exp_shift"])
    h = cvt_f32x2_f16x2_rz_satfinite(ya, yb, loc=loc, ip=ip)
    h = Uint32(h) & Uint32(kind["mask"])  # packed floor onto the f8 grid
    return cvt_f16x2_f8x2_rn_satfinite(
        h, f8_kind, unscale=kind["prescale"] is not None, loc=loc, ip=ip
    )


# Both kinds run in a domain whose subnormal knee is f16's 2^-14, so they
# share min_scale = sign|2^-14. e4m3 (scaled by 2^-8): grid ulp' = 2^(e'-3),
# 8 rbits -> S = 11, delta' = r * 2^(max(e',-14) - 11), clamped floor 2^-25.
# e5m2 (natural domain, ulp 2^(e-2)) -> S = 10, clamped floor 2^-24.
_F8_KINDS = {
    "e4m3": dict(min_scale=0x38800000, exp_shift=11, mask=0xFF80FF80, prescale=2.0**-8),
    "e5m2": dict(min_scale=0x38800000, exp_shift=10, mask=0xFF00FF00, prescale=None),
}


@dsl_user_op
def cvt_f16x2_f8x2_rn_satfinite(
    h: Uint32, f8_kind: str, unscale: bool = False, *, loc=None, ip=None
) -> cutlass.Uint16:
    """Convert packed f16x2 -> packed f8x2 with RN + satfinite.

    h[15:0] -> d[7:0], h[31:16] -> d[15:8]. NaN -> 0x7F; |x| > MAX_NORM (incl
    Inf) -> sign|0x7E (e4m3) / sign|0x7B (e5m2). Inputs must already lie on
    the f8 grid for exactness (RN of an exactly-representable value is exact).
    With unscale=True both halves are first multiplied by 2^8 (one HMUL2;
    ptxas folds the constant into an immediate-form HFMA2/HMUL2).
    """
    assert f8_kind in ("e4m3", "e5m2")
    body = (
        "{\n\t.reg .b32 c, u;\n\t"
        "mov.b32 c, 0x5C005C00;\n\t"  # 2^8 in both f16 halves
        "mul.rn.f16x2 u, $1, c;\n\t"
        f"cvt.rn.satfinite.{f8_kind}x2.f16x2 $0, u;\n\t}}"
        if unscale
        else f"cvt.rn.satfinite.{f8_kind}x2.f16x2 $0, $1;"
    )
    return cutlass.Uint16(_asm(T.i16(), body, "=h,r", [Uint32(h).ir_value(loc=loc, ip=ip)]))


@dsl_user_op
def cvt_f32x2_e4m3x2_rs_sw(
    a: Float32, b: Float32, rand_bits: Uint32, *, loc=None, ip=None
) -> cutlass.Uint16:
    """Software f32x2 -> e4m3x2 stochastic rounding (any sm_89+ target).

    a -> d[7:0] using rand_bits[7:0], b -> d[15:8] using rand_bits[15:8]
    (8 rbits per value, matching the SM100 cvt.rs.satfinite.e4m3x4.f32).
    22 SASS per pair (~20 in unrolled loops where the two-immediate LOP3
    constants hoist to UR): PRMT + 2 I2F.U16 + per element [FMUL prescale,
    LOP3 x2, IMNMX, IADD, FFMA.RZ] + F2FP f16x2 pack + packed-mask LOP3 +
    HMUL2 unscale + F2FP e4m3x2.f16x2 closer. Bit-exact vs the reference
    model on 4M vectors, and bitwise against the hw cvt.rs on B200 (which
    also confirmed the rand byte mapping and subnormal alignment).
    """
    return _cvt_f32x2_f8x2_rs_sw(a, b, rand_bits, "e4m3", loc, ip)


@dsl_user_op
def cvt_f32x2_e5m2x2_rs_sw(
    a: Float32, b: Float32, rand_bits: Uint32, *, loc=None, ip=None
) -> cutlass.Uint16:
    """Software f32x2 -> e5m2x2 stochastic rounding (any sm_89+ target; see e4m3).

    19 SASS per pair (~17 in unrolled loops): as e4m3 but no prescale FMULs
    and no HMUL2 unscale (e5m2 = f16 truncated to 2 mantissa bits, so the
    packed mask floors directly in the natural domain).
    """
    return _cvt_f32x2_f8x2_rs_sw(a, b, rand_bits, "e5m2", loc, ip)


@dsl_user_op
def cvt_f32x4_f8x4_rs(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    rand_bits: Uint32,
    f8_kind: str,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Hardware f32x4 -> f8x4 stochastic rounding (sm_100a/sm_103a only).

    Packing on both arches: v0 -> d[7:0], ..., v3 -> d[31:24]. The rand
    mapping is arch-specific: on sm_100a (B200-probed) value i consumes
    rand byte i straight, bitwise identical to the sw pair emulation; on
    sm_103a (B300-probed) v0 <- byte1, v1 <- byte0, v2 <- byte3, v3 <- byte2
    with the odd values' bytes consumed BIT-REVERSED (same quirk as
    cvt_f32x4_e2m1x4_rs — exhaustive per-byte sweeps still show round-up
    count == D_top on every slot, so uniform rbits stay unbiased; only
    bitwise seed portability is lost).
    """
    assert f8_kind in ("e4m3", "e5m2")
    return cutlass.Int32(
        _asm(
            T.i32(),
            f"cvt.rs.satfinite.{f8_kind}x4.f32 $0, {{$4, $3, $2, $1}}, $5;",
            "=r,f,f,f,f,r",
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)]
            + [Uint32(rand_bits).ir_value(loc=loc, ip=ip)],
        )
    )


@dsl_user_op
def cvt_f32x4_e2m1x4_rs(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    rand_bits: Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint16:
    """Hardware f32x4 -> e2m1x4 stochastic rounding (sm_100a/sm_103a only).

    Packing: v0 -> d[3:0], v1 -> d[7:4], v2 -> d[11:8], v3 -> d[15:12] (same
    brace-operand convention as the f8x4 wrapper). The rand mapping is NOT the
    f8x4 byte order (probed exhaustively on B300 sm_103a): v0 <- rand[23:16],
    v1 <- rand[7:0], v2 <- rand[31:24], v3 <- rand[15:8], and the odd values
    consume their byte BIT-REVERSED (carry iff bitrev(r8) + D_top >= 256,
    where D_top is the top 8 discarded mantissa bits). Full 256-value sweeps
    show round-up count == D_top on every slot, so uniform rbits give
    unbiased 8-rbit SR on all four values; the permutation/bit-reversal only
    matters if a sw emulation ever needs bitwise parity. Specials follow
    satfinite (NaN -> 0x7 = +6.0, |x| > 6 incl Inf -> sign|0x7).
    """
    return cutlass.Uint16(
        _asm(
            T.i16(),
            "cvt.rs.satfinite.e2m1x4.f32 $0, {$4, $3, $2, $1}, $5;",
            "=h,f,f,f,f,r",
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)]
            + [Uint32(rand_bits).ir_value(loc=loc, ip=ip)],
        )
    )


# ---------------------------------------------------------------------------
# FP4 (e2m1), via the f16 grid — same recipe as fp8, but the closer's role is
# split because no f32/f16 -> e2m1 cvt exists before sm_100a. e2m1's subnormal
# knee sits at 1.0 (min normal 1.0, subnormal step 0.5); pre-scaling x by
# 2^-14 drops that knee exactly onto f16's own (2^-14), and on the scaled
# domain the e2m1 grid IS the f16 grid truncated to mantissa bit 9: the eight
# magnitude codes 0..7 land on the f16 bit patterns 0x0000, 0x0200, ...,
# 0x0E00, i.e. h[14:9] == the e2m1 magnitude code, with f16's own gradual
# underflow reproducing the fixed-point subnormal region. So the existing
# cvt.rz.satfinite.f16x2.f32 F2FP performs the grid floor and every special:
# NaN -> 0x7FFF (sign dropped) clamps to code 7 = +6.0, the hw rule; |y| >=
# 8*2^-14 (incl. the satfinite 0x7BFF clamp remnant and past-6 inputs, whose
# rounding is input-determined) has h[14:9] >= 8 and clamps to sign|7. The
# nibble then falls out integer-side as min(h[14:9], 7) | sign — no e2m1 or
# f8 cvt instruction anywhere, so unlike the fp8 emulations this one runs
# from sm_80 up.
#
# The prescale FMUL is exact for |x| >= 2^-112 (the scaled value stays f32-
# normal); below that it may round (or flush under ftz), harmlessly: every
# such x sits at D_top = 0 on the scaled 2^-15 subnormal grid (D_top =
# floor(|x| * 2^9) = 0 already for |x| < 2^-9), so the round-up probability
# is 0 at 8-rbit granularity with or without the rounding, and the max
# delta 255 * 2^-23 keeps y below the first grid point. The fma stage is
# _sr_scale_fma with S = m + K = 1 + 8 = 9 and min_scale = sign|2^-14 (the
# scaled knee) — literally the f16 machinery with fp4's K = 8.
# ---------------------------------------------------------------------------

E2M1_PRESCALE = 2.0**-14  # aligns the e2m1 subnormal knee (1.0) with f16's (2^-14)


def _has_e2m1_cvt() -> bool:
    """True iff the compile target has cvt.rn.satfinite.e2m1x2.f32.

    Every sm_100-or-newer target at PTX ISA >= 9.0 (probed with ptxas 13.3:
    sm_100a/f, sm_103a, sm_110a, sm_120a/f, sm_121a all assemble it; at ISA
    8.7 only sm_100a/sm_120a did — a PTX version gate, not hardware). The
    pre-100 targets (sm_80..sm_90a) have no f32 -> e2m1 cvt at all.
    """
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    return int(arch.name.split("_")[1].rstrip("af")) >= 100


@dsl_user_op
def cvt_scaled_f16x2x2_e2m1x4(h01: Uint32, h23: Uint32, *, loc=None, ip=None) -> Uint32:
    """e2m1x4 from two packed f16x2 on the 2^-14-scaled grid, via the direct
    e2m1 cvt (sm_100+ targets; see cvt_f32x4_e2m1x4_rs_direct).

    Replaces the 2 x 12-op integer nibble extraction of
    :func:`_e2m1_pair_from_f16x2`: floor-mask onto the scaled fp4 grid (keep
    f16 mantissa bit 9), one exact HMUL2 unscale by 2^14 per pair (on-grid
    values x 2^14 are exact e2m1 values in f16), per-half F2F widening
    (HADD2.F32 — fp16 pipe, off the int wall), and
    cvt.rn.satfinite.e2m1x2.f32 — RN of an exactly-representable value is
    exact, and satfinite reproduces the specials: the mask keeps NaN a NaN
    (the F2FP closer's 0x7FFF quiet bit is mantissa bit 9) which the cvt maps
    to +6.0 code 7 (probed on sm_120a, same as the hw cvt.rs rule), and the
    satfinite clamp remnant sign|0x7BFF masks to sign|0x7A00 whose unscale
    overflows to sign|Inf -> sign|6.0. The e2m1 cvt lowers to
    F2FP.E2M1.PACK_AB_MERGE_C, and ONE PRMT (selector 0x0040: byte0 of each
    cvt result into the low half; bytes 2-3 are don't-care duplicates the
    caller's Uint16 truncation discards) assembles the final u16. The two
    byte zero-extensions still cost one LOP3 each (that IS cvt.u32.u8's
    lowering; ptxas does not track MERGE_C's zero bytes through the PRMT to
    dead-code them — probed). Returns quad nibbles packed as
    v0 -> [3:0] .. v3 -> [15:12] in the low 16 bits (high 16 undefined).
    """
    return Uint32(
        _asm(
            T.i32(),
            "{\n\t.reg .b32 u0, u1, c, r0, r1;\n\t"
            ".reg .b16 l0, h0, l1, h1;\n\t"
            ".reg .f32 f0, f1, f2, f3;\n\t"
            ".reg .b8 b0, b1;\n\t"
            "and.b32 u0, $1, 0xFE00FE00;\n\t"
            "and.b32 u1, $2, 0xFE00FE00;\n\t"
            "mov.b32 c, 0x74007400;\n\t"  # 2^14 in both f16 halves
            "mul.rn.f16x2 u0, u0, c;\n\t"
            "mul.rn.f16x2 u1, u1, c;\n\t"
            "mov.b32 {l0, h0}, u0;\n\t"
            "mov.b32 {l1, h1}, u1;\n\t"
            "cvt.f32.f16 f0, l0;\n\t"
            "cvt.f32.f16 f1, h0;\n\t"
            "cvt.f32.f16 f2, l1;\n\t"
            "cvt.f32.f16 f3, h1;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b0, f1, f0;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b1, f3, f2;\n\t"
            "cvt.u32.u8 r0, b0;\n\t"  # virtual: MERGE_C already wrote a b32
            "cvt.u32.u8 r1, b1;\n\t"
            "prmt.b32 $0, r0, r1, 0x0040;\n\t}",
            "=r,r,r",
            [Uint32(h01).ir_value(loc=loc, ip=ip), Uint32(h23).ir_value(loc=loc, ip=ip)],
        )
    )


@dsl_user_op
def cvt_f32x4_e2m1x4_rs_direct(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    rand_bits: Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint16:
    """Software f32x4 -> e2m1x4 stochastic rounding using the direct e2m1 cvt
    closer (any sm_100+ target without the hw cvt.rs, i.e. sm_100f/110/120/121).

    BIT-IDENTICAL to :func:`cvt_f32x4_e2m1x4_rs_sw` — same prescale, same
    _sr_scale_fma, same f16-grid floor, same straight rand byte order — only
    the nibble extraction differs (see cvt_scaled_f16x2x2_e2m1x4). Pipe
    accounting on sm_120a (see skills/sass-micro-opt): with the _sr_scale_fma
    restructure the int-pipe load is 20 ops/quad (vs 22 int-side for the
    HMNMX2-extraction sw path, 43 for the retired original) — measured in a
    compute-bound feedback-chain harness on RTX 5090 (interleaved medians,
    bitwise-verified chain): 1.53x the original emulation and 1.043x the
    current sw path, so this stays the sm_100+ selection.
    """
    vals = [Float32(v) for v in (v0, v1, v2, v3)]
    rand_bits = Uint32(rand_bits)
    p01 = Uint32(cute.arch.prmt(rand_bits, 0, 0x4140, loc=loc, ip=ip))
    p23 = Uint32(cute.arch.prmt(rand_bits, 0, 0x4342, loc=loc, ip=ip))
    rfs = [
        u16x2_to_f32(p01, 0, loc=loc, ip=ip),
        u16x2_to_f32(p01, 1, loc=loc, ip=ip),
        u16x2_to_f32(p23, 0, loc=loc, ip=ip),
        u16x2_to_f32(p23, 1, loc=loc, ip=ip),
    ]
    ys = [_sr_scale_fma(x * E2M1_PRESCALE, rf, 0x38800000, 9) for x, rf in zip(vals, rfs)]
    h01 = cvt_f32x2_f16x2_rz_satfinite(ys[0], ys[1], loc=loc, ip=ip)
    h23 = cvt_f32x2_f16x2_rz_satfinite(ys[2], ys[3], loc=loc, ip=ip)
    return cutlass.Uint16(cvt_scaled_f16x2x2_e2m1x4(h01, h23, loc=loc, ip=ip))


@dsl_user_op
def min_f16x2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Packed f16x2 minimum: one HMNMX2 (alu pipe, 0.5/clk; sm_80+).

    Number-favoring NaN semantics (min(NaN, y) = y) — load-bearing for the
    e2m1 emulation's NaN -> +6.0 rule. Honors f16 subnormals (no .ftz)."""
    return Uint32(
        _asm(
            T.i32(),
            "min.f16x2 $0, $1, $2;",
            "=r,r,r",
            [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a, b)],
        )
    )


@dsl_user_op
def max_f16x2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Packed f16x2 maximum: one HMNMX2 (alu pipe, 0.5/clk; sm_80+); see min_f16x2."""
    return Uint32(
        _asm(
            T.i32(),
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a, b)],
        )
    )


@dsl_user_op
def cvt_f32x4_e2m1x4_rs_sw(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    rand_bits: Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint16:
    """Software f32x4 -> e2m1x4 stochastic rounding (any sm_80+ target).

    Packing matches the hw wrapper (v0 -> d[3:0], ..., v3 -> d[15:12]). The
    rand mapping is the STRAIGHT byte order v_i <- rand_bits[8i+7 : 8i] — the
    sm_100a f8x4 convention, NOT sm_103a's permuted/bit-reversed e2m1 order
    (see cvt_f32x4_e2m1x4_rs): vs sm_103a the streams differ by construction;
    sm_100a's e2m1 rand order is unprobed, so bit parity there is plausible
    but unverified. Either way the output is unbiased 8-rbit SR and the sw
    stream is identical on every non-.rs target.

    43 SASS per quad on sm_90a (~10.75/value, vs the f8 pairs' 19-22/pair;
    was 62 with the retired per-pair shift/mask/clamp extraction — measured
    1.26x in a compute-bound feedback-chain harness, bitwise identical): 2
    PRMT + 4 I2F.U16 rand extraction, per value FMUL prescale + 5-op
    _sr_scale_fma (2 LOP3, VIADD, FMNMX, FFMA.RZ — restructured for
    sm_120a's 2-input second ALU; on sm_90a the IADD rides ptxas's VIADD
    substitution onto the fma pipe, so it costs one fewer ALU op here too),
    2 F2FP closers, and an
    11-op quad-wide nibble extract/pack. The extraction leans on two facts:
    (1) a two-sided packed-f16 clamp (2x HMNMX2 per pair) bounds every half
    to sign|mag<=0x0FFF while preserving in-range bits exactly — NaN loses
    the min against 0x0FFF (number-favoring), landing on +0x0FFF = code 7
    with the sign dropped, the hw rule, and +-Inf/satfinite remnants clamp
    to sign|0x0FFF; (2) after the clamp, byte 1 of every half is
    sign<<7 | code<<1 | junk, so ONE PRMT gathers all four values' bytes,
    two shift+LOP3 pairs build all four nibbles at once
    (((r>>1) & 0x07070707) | ((r>>4) & 0x08080808), the second AND+OR fusing
    into a 3-input LOP3), and nib | nib>>4 (a single LEA.HI) + a
    byte-picking PRMT pack the u16. Pipe walls per quad: alu 27 ops -> 54
    clk (was 43 -> 86 originally, 31 -> 62 pre-restructure), fma 12, xu 4 —
    still alu-bound, so every op removed was an alu op; do not "save" ops by
    moving work to IMAD forms that add instructions. Bit-exact vs the
    reference model (tests/test_rounding.py:
    12M random vectors incl. pure-bit-random NaN/Inf/denormal patterns,
    specials in every slot, exhaustive 256-rand per-cell sweeps showing
    round-up count == D_top on every slot, and the Philox frag wiring).
    """
    vals = [Float32(v) for v in (v0, v1, v2, v3)]
    rand_bits = Uint32(rand_bits)
    # place rand bytes into u16 halves for I2F.U16: {0, b1, 0, b0} / {0, b3, 0, b2}
    p01 = Uint32(cute.arch.prmt(rand_bits, 0, 0x4140, loc=loc, ip=ip))
    p23 = Uint32(cute.arch.prmt(rand_bits, 0, 0x4342, loc=loc, ip=ip))
    rfs = [
        u16x2_to_f32(p01, 0, loc=loc, ip=ip),
        u16x2_to_f32(p01, 1, loc=loc, ip=ip),
        u16x2_to_f32(p23, 0, loc=loc, ip=ip),
        u16x2_to_f32(p23, 1, loc=loc, ip=ip),
    ]
    ys = [_sr_scale_fma(x * E2M1_PRESCALE, rf, 0x38800000, 9) for x, rf in zip(vals, rfs)]
    h01 = Uint32(cvt_f32x2_f16x2_rz_satfinite(ys[0], ys[1], loc=loc, ip=ip))
    h23 = Uint32(cvt_f32x2_f16x2_rz_satfinite(ys[2], ys[3], loc=loc, ip=ip))
    # Two-sided f16 clamp into [-val(0x0FFF), val(0x0FFF)]: every half becomes
    # sign | mag<=0x0FFF (bits 14:12 zero) with in-range values untouched.
    hi_bound, lo_bound = Uint32(0x0FFF0FFF), Uint32(0x8FFF8FFF)
    hs01 = max_f16x2(min_f16x2(h01, hi_bound, loc=loc, ip=ip), lo_bound, loc=loc, ip=ip)
    hs23 = max_f16x2(min_f16x2(h23, hi_bound, loc=loc, ip=ip), lo_bound, loc=loc, ip=ip)
    # byte 1 of each half = sign<<7 | code<<1 | junk; gather all four values
    r = Uint32(cute.arch.prmt(hs01, hs23, 0x7531, loc=loc, ip=ip))
    nib = ((r >> 1) & Uint32(0x07070707)) | ((r >> 4) & Uint32(0x08080808))
    t = nib | (nib >> 4)  # byte0 = nib1<<4|nib0, byte2 = nib3<<4|nib2
    return cutlass.Uint16(Uint32(cute.arch.prmt(t, 0, 0x4420, loc=loc, ip=ip)))


@dsl_user_op
def philox(
    counter,
    key,
    n_rounds: int = PHILOX_N_ROUNDS_DEFAULT,
    *,
    loc=None,
    ip=None,
) -> tuple:
    """Philox 4x32b counter-based random number generator.

    ``counter`` and ``key`` each accept a 32b or 64b unsigned value: a 64b
    counter fills both counter words (c0, c1) and a 64b key fills both key
    words (k0, k1), widening the usable counter/key space to 2^64. A 32b
    input zeroes the corresponding high word, which reproduces the original
    32b-only behavior bit-exactly. The parameters are intentionally
    unannotated so no width coercion happens ahead of the dispatch below.

    Returns four pseudo-random uint32 words produced by running n_rounds of
    the Philox 4x32 bijection. Each round performs two wide 32x32->64
    multiplies with the Philox constants.
    """
    if cutlass.const_expr(isinstance(counter, Uint64)):
        c0 = (counter & Uint64(0xFFFFFFFF)).to(Uint32)
        c1 = (counter >> Uint64(32)).to(Uint32)
    else:
        c0 = Uint32(counter)
        c1 = Uint32(0)
    c2 = Uint32(0)
    c3 = Uint32(0)
    if cutlass.const_expr(isinstance(key, Uint64)):
        k0 = (key & Uint64(0xFFFFFFFF)).to(Uint32)
        k1 = (key >> Uint64(32)).to(Uint32)
    else:
        k0 = Uint32(key)
        k1 = Uint32(0)

    round_a = Uint32(PHILOX_ROUND_A)
    round_b = Uint32(PHILOX_ROUND_B)
    key_a = Uint32(PHILOX_KEY_A)
    key_b = Uint32(PHILOX_KEY_B)

    for _ in range(n_rounds):
        hi_b, lo_b = mul_wide_u32(c2, round_b, loc=loc, ip=ip)
        hi_a, lo_a = mul_wide_u32(c0, round_a, loc=loc, ip=ip)
        c0 = hi_b ^ c1 ^ k0
        c2 = hi_a ^ c3 ^ k1
        c1 = lo_b
        c3 = lo_a
        k0 = k0 + key_a
        k1 = k1 + key_b

    return c0, c1, c2, c3


def _use_hw_cvt() -> bool:
    """True iff the current DSL compile target has the hw cvt.rs instruction.

    Derived from the DSL's arch enum (compile_options / CUTE_DSL_ARCH /
    detection), NOT from the torch device: .rs legality is a property of the
    ptxas target, and the two diverge under QUACK_ARCH/CUTE_DSL_ARCH
    overrides and in GPU-blind async-compile workers (which must not touch
    torch.cuda at all). Exact membership, no >=: ptxas rejects .rs on
    everything but sm_100a/sm_103a — including sm_100f/sm_103f.
    """
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    return arch in (Arch.sm_100a, Arch.sm_103a)


@dsl_user_op
def convert_f32_to_f8_sr(
    src_vec,
    seed: Int32,
    tid: Int32,
    f8_kind: str,
    *,
    loc=None,
    ip=None,
):
    """Convert an MLIR FP32 vector to e4m3/e5m2 with stochastic rounding.

    Quad-based: 8 rbits per value, one Philox word per 4 values. Uses the
    hardware cvt.rs.satfinite.f8x4.f32 on sm_100a/sm_103a, or the pair-wise
    software emulation (sm_89+) elsewhere. The sw pair consumes
    rand[7:0]/rand[15:8], so splitting the quad's word as (r, r >> 16)
    matches sm_100a's straight byte order bitwise; sm_103a's hw permutes and
    bit-reverses its rand consumption (see cvt_f32x4_f8x4_rs), so there the
    output is statistically identical but not bit-identical to sw.
    """
    assert f8_kind in ("e4m3", "e5m2")
    dst_numeric = cutlass.Float8E4M3FN if f8_kind == "e4m3" else cutlass.Float8E5M2
    if _use_hw_cvt():

        def cvt_quad(v0, v1, v2, v3, rand, *, loc=None, ip=None):
            return cvt_f32x4_f8x4_rs(v0, v1, v2, v3, rand, f8_kind, loc=loc, ip=ip)

    else:
        pair_fn = cvt_f32x2_e4m3x2_rs_sw if f8_kind == "e4m3" else cvt_f32x2_e5m2x2_rs_sw

        def cvt_quad(v0, v1, v2, v3, rand, *, loc=None, ip=None):
            lo = pair_fn(v0, v1, rand, loc=loc, ip=ip)
            hi = pair_fn(v2, v3, Uint32(rand) >> 16, loc=loc, ip=ip)
            return cutlass.Int32(Uint32(lo).to(Uint32) | (Uint32(hi).to(Uint32) << 16))

    return _convert_f32_sr_quad_impl(src_vec, seed, tid, cvt_quad, dst_numeric, loc, ip)


@dsl_user_op
def convert_f32_to_e2m1_sr(
    src_vec,
    seed: Int32,
    tid: Int32,
    *,
    loc=None,
    ip=None,
):
    """Convert an MLIR FP32 vector to e2m1 (fp4) with stochastic rounding.

    Hardware cvt.rs.satfinite.e2m1x4.f32 on sm_100a/sm_103a; on other
    sm_100+ targets (sm_100f/110/120/121) the software emulation with the
    direct e2m1-cvt closer (cvt_f32x4_e2m1x4_rs_direct, bit-identical to the
    generic emulation, ~25% fewer SASS); the generic f16-grid emulation
    (cvt_f32x4_e2m1x4_rs_sw) on any other sm_80+ target. All consume one
    Philox word per 4 values, but the hw rand byte mappings differ (sw is
    straight bytes, sm_103a permutes + bit-reverses), so 4-bit seeds are not
    bit-portable hw<->sw — see the module docstring.
    """
    if _use_hw_cvt():
        cvt_fn = cvt_f32x4_e2m1x4_rs
    elif _has_e2m1_cvt():
        cvt_fn = cvt_f32x4_e2m1x4_rs_direct
    else:
        cvt_fn = cvt_f32x4_e2m1x4_rs_sw
    return _convert_f32_sr_quad_impl(src_vec, seed, tid, cvt_fn, cutlass.Float4E2M1FN, loc, ip)


def _convert_f32_sr_quad_impl(src_vec, seed, tid, cvt_quad_fn, dst_numeric, loc, ip):
    """Quad-based sibling of _convert_f32_sr_impl for 8/4-bit outputs: the
    converters are x4-only (hw f8x4/e2m1x4 instructions), each value consumes
    8 rbits, so one Philox word covers a quad and one batch covers 4 quads
    (16 values). Packs each quad's result word (i32 for f8, i16 for e2m1)
    into a vector and bitcasts to the destination element type."""
    src_vec_type = ir.VectorType(src_vec.type)
    num_elems = src_vec_type.shape[0]
    assert num_elems % 4 == 0, (
        f"8/4-bit stochastic rounding requires num_elems % 4 == 0, got {num_elems}"
    )
    num_quads = num_elems // 4

    dst_vec_type = ir.VectorType.get([num_elems], dst_numeric.mlir_type, loc=loc)
    word_numeric = Int32 if dst_numeric.width == 8 else cutlass.Uint16
    word_vec_type = ir.VectorType.get([num_quads], word_numeric.mlir_type, loc=loc)
    word_vec = llvm.mlir_undef(word_vec_type, loc=loc, ip=ip)

    for quad_idx in range(num_quads):
        vals = [
            Float32(
                vector.extract(
                    src_vec,
                    dynamic_position=[],
                    static_position=[quad_idx * 4 + j],
                    loc=loc,
                    ip=ip,
                )
            )
            for j in range(4)
        ]
        group_idx = quad_idx // 4
        intra_idx = quad_idx % 4
        if intra_idx == 0:
            counter = cutlass.Uint32(group_idx << 16) | cutlass.Uint32(tid)
            rand_batch = philox(counter, cutlass.Uint32(seed))
        entropy = rand_batch[intra_idx]
        packed = cvt_quad_fn(*vals, entropy, loc=loc, ip=ip)
        word_vec = vector.insert(
            word_numeric(packed).ir_value(loc=loc, ip=ip),
            word_vec,
            dynamic_position=[],
            static_position=[quad_idx],
            loc=loc,
            ip=ip,
        )

    return llvm.bitcast(dst_vec_type, word_vec, loc=loc, ip=ip)


@dsl_user_op
def convert_f32_to_bf16_sr(
    src_vec,
    seed: Int32,
    tid: Int32,
    *,
    loc=None,
    ip=None,
):
    """Convert an MLIR FP32 vector to BF16 with stochastic rounding.

    Processes elements in pairs using Philox PRNG for entropy and the hardware
    cvt.rs.satfinite.bf16x2.f32 instruction when compiling for sm_100a/sm_103a,
    or a bit-exact software emulation on any other sm_80+ target.
    """
    cvt_fn = cvt_f32x2_bf16x2_rs if _use_hw_cvt() else cvt_f32x2_bf16x2_rs_sw
    return _convert_f32_sr_impl(src_vec, seed, tid, cvt_fn, cutlass.BFloat16, loc, ip)


@dsl_user_op
def convert_f32_to_f16_sr(
    src_vec,
    seed: Int32,
    tid: Int32,
    *,
    loc=None,
    ip=None,
):
    """Convert an MLIR FP32 vector to FP16 with stochastic rounding.

    Same structure as convert_f32_to_bf16_sr; both hw (sm_100a/sm_103a cvt.rs)
    and sw (any other sm_80+ target) paths consume 13 rbits per value from
    the same Philox words.
    """
    cvt_fn = cvt_f32x2_f16x2_rs if _use_hw_cvt() else cvt_f32x2_f16x2_rs_sw
    return _convert_f32_sr_impl(src_vec, seed, tid, cvt_fn, cutlass.Float16, loc, ip)


# Storage dtypes the epilogue SR convert supports. bf16/f16/e2m1 run
# everywhere (sm_80+ sw emulation); e4m3/e5m2 run on sm_89+ (sw); all five
# use the hw cvt.rs when compiling for sm_100a/103a.
SR_STORE_DTYPES = (
    cutlass.BFloat16,
    cutlass.Float16,
    cutlass.Float8E4M3FN,
    cutlass.Float8E5M2,
    cutlass.Float4E2M1FN,
)


def convert_f32_frag_sr(frag: cute.Tensor, dtype, seed: Int32, tid: Int32) -> cute.Tensor:
    """Convert an f32 rmem fragment to ``dtype`` with stochastic rounding.

    Same-layout sibling of ``frag.to(dtype)`` for every dtype in
    SR_STORE_DTYPES; dtype is a trace-time constant. Callers derive ``seed``
    per (tile, subtile) via epilogue_sr_seed so rand streams never repeat
    across the output.
    """
    from cutlass.cute.tensor import TensorSSA

    assert dtype in SR_STORE_DTYPES, f"no stochastic-rounding converter for {dtype}"
    src_vec = frag.load()
    if dtype is cutlass.BFloat16:
        raw_vec = convert_f32_to_bf16_sr(src_vec, seed, tid)
    elif dtype is cutlass.Float16:
        raw_vec = convert_f32_to_f16_sr(src_vec, seed, tid)
    elif dtype is cutlass.Float8E4M3FN:
        raw_vec = convert_f32_to_f8_sr(src_vec, seed, tid, "e4m3")
    elif dtype is cutlass.Float8E5M2:
        raw_vec = convert_f32_to_f8_sr(src_vec, seed, tid, "e5m2")
    else:
        raw_vec = convert_f32_to_e2m1_sr(src_vec, seed, tid)
    out = cute.make_rmem_tensor_like(frag, dtype)
    out.store(TensorSSA(raw_vec, src_vec.shape, dtype))
    return out


def _convert_f32_sr_impl(src_vec, seed, tid, cvt_fn, dst_numeric, loc, ip):
    src_vec_type = ir.VectorType(src_vec.type)
    num_elems = src_vec_type.shape[0]
    assert num_elems % 2 == 0, f"requires even number of elements, got {num_elems}"
    num_pairs = num_elems // 2
    # No num_pairs % 4 requirement for 16-bit outputs.
    # For 8-bit outputs the QUAD is a hard requirement (the hw
    # instruction is f8x4-only)
    if dst_numeric.width == 8:
        assert num_elems % 4 == 0, (
            f"8-bit stochastic rounding requires num_elems % 4 == 0, got {num_elems}"
        )

    dst_mlir_type = dst_numeric.mlir_type
    dst_vec_type = ir.VectorType.get([num_elems], dst_mlir_type, loc=loc)

    i32_vec_type = ir.VectorType.get([num_pairs], Int32.mlir_type, loc=loc)
    i32_vec = llvm.mlir_undef(i32_vec_type, loc=loc, ip=ip)

    for pair_idx in range(num_pairs):
        lo_idx = pair_idx * 2
        hi_idx = pair_idx * 2 + 1

        src_lo = vector.extract(
            src_vec,
            dynamic_position=[],
            static_position=[lo_idx],
            loc=loc,
            ip=ip,
        )
        src_hi = vector.extract(
            src_vec,
            dynamic_position=[],
            static_position=[hi_idx],
            loc=loc,
            ip=ip,
        )

        group_idx = pair_idx // 4
        intra_idx = pair_idx % 4
        if intra_idx == 0:
            counter = cutlass.Uint32(group_idx << 16) | cutlass.Uint32(tid)
            rand_batch = philox(counter, cutlass.Uint32(seed))

        entropy = rand_batch[intra_idx]
        packed_i32 = cvt_fn(Float32(src_lo), Float32(src_hi), entropy, loc=loc, ip=ip)

        packed_i32_val = cutlass.Int32(packed_i32).ir_value(loc=loc, ip=ip)
        i32_vec = vector.insert(
            packed_i32_val,
            i32_vec,
            dynamic_position=[],
            static_position=[pair_idx],
            loc=loc,
            ip=ip,
        )

    dst_vec = llvm.bitcast(dst_vec_type, i32_vec, loc=loc, ip=ip)
    return dst_vec
