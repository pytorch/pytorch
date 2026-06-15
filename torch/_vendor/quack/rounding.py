# Copyright (c) 2025-2026, Vijay Thakkar, Tri Dao.
"""Rounding mode control and stochastic rounding primitives for GEMM epilogues.

Provides a RoundingMode enum for configuring how epilogues downconvert the
accumulator dtype (typically FP32) to the output dtype before storing to gmem.
Stochastic rounding (RS) uses the hardware cvt.rs.satfinite.bf16x2.f32 PTX
instruction and is only supported on Blackwell (SM100+) GPUs.
"""

from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cutlass_dsl import dsl_user_op, Int32, T


class RoundingMode(IntEnum):
    """Rounding modes for epilogue dtype downconversion.

    RN — Round to nearest even (default hardware behavior)
    RS — Stochastic rounding (SM100+ only, BF16 output only)
    """

    RN = 0
    RS = 1


# Odd strides used to derive distinct Philox counters across output tiles and subtiles.
EPILOGUE_SR_SEED_M_STRIDE = 65537
EPILOGUE_SR_SEED_N_STRIDE = 257
EPILOGUE_SR_SEED_BATCH_STRIDE = 17
EPILOGUE_SR_SEED_SUBTILE_STRIDE = 7
EPILOGUE_SR_SEED_AUX_OUT_SALT = 0x9E3779B1

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
    struct_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    result = llvm.inline_asm(
        struct_ty,
        [
            Uint32(a).ir_value(loc=loc, ip=ip),
            Uint32(b).ir_value(loc=loc, ip=ip),
        ],
        "{\n  .reg .u64 prod;\n  mul.wide.u32 prod, $2, $3;\n  mov.b64 {$1, $0}, prod;\n}",
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    i32_ty = T.i32()
    hi = cutlass.Uint32(llvm.extractvalue(i32_ty, result, [0], loc=loc, ip=ip))
    lo = cutlass.Uint32(llvm.extractvalue(i32_ty, result, [1], loc=loc, ip=ip))
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
        llvm.inline_asm(
            T.i32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Uint32(rand_bits).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rs.satfinite.bf16x2.f32 $0, $2, $1, $3;",
            "=r,f,f,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def philox(
    counter: Uint32,
    key: Uint32,
    n_rounds: int = PHILOX_N_ROUNDS_DEFAULT,
    *,
    loc=None,
    ip=None,
) -> tuple:
    """Philox 4x32b counter-based random number generator.

    Given a 32b counter and a 32b key, returns four pseudo-random uint32 words
    produced by running n_rounds of the Philox 4x32 bijection. Each round
    performs two wide 32x32->64 multiplies with the Philox constants.
    """
    c0 = Uint32(counter)
    c1 = Uint32(0)
    c2 = Uint32(0)
    c3 = Uint32(0)
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
    cvt.rs.satfinite.bf16x2.f32 instruction.
    """
    src_vec_type = ir.VectorType(src_vec.type)
    num_elems = src_vec_type.shape[0]
    assert num_elems % 2 == 0, f"requires even number of elements, got {num_elems}"
    num_pairs = num_elems // 2
    assert num_pairs % 4 == 0, (
        f"num_pairs must be divisible by 4 for stochastic rounding, got {num_pairs}"
    )

    dst_mlir_type = cutlass.BFloat16.mlir_type
    dst_vec_type = ir.VectorType.get([num_elems], dst_mlir_type, loc=loc)

    i32_vec_type = ir.VectorType.get([num_pairs], Int32.mlir_type, loc=loc)
    i32_vec = llvm.mlir_undef(i32_vec_type, loc=loc, ip=ip)

    for pair_idx in range(num_pairs):
        lo_idx = pair_idx * 2
        hi_idx = pair_idx * 2 + 1

        src_lo = vector.extractelement(
            src_vec,
            position=arith.constant(Int32.mlir_type, lo_idx, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        src_hi = vector.extractelement(
            src_vec,
            position=arith.constant(Int32.mlir_type, hi_idx, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

        group_idx = pair_idx // 4
        intra_idx = pair_idx % 4
        if intra_idx == 0:
            counter = cutlass.Uint32(group_idx << 16) | cutlass.Uint32(tid)
            rand_batch = philox(counter, cutlass.Uint32(seed))

        entropy = rand_batch[intra_idx]
        packed_i32 = cvt_f32x2_bf16x2_rs(Float32(src_lo), Float32(src_hi), entropy, loc=loc, ip=ip)

        packed_i32_val = cutlass.Int32(packed_i32).ir_value(loc=loc, ip=ip)
        i32_vec = vector.insertelement(
            packed_i32_val,
            i32_vec,
            position=arith.constant(Int32.mlir_type, pair_idx, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    dst_vec = llvm.bitcast(dst_vec_type, i32_vec, loc=loc, ip=ip)
    return dst_vec
