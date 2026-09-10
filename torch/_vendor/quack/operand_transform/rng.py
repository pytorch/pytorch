# Copyright (c) 2026, Tri Dao.
"""Packed-mask helpers for RNG operand transforms (dropout). The generator is
quack.rounding.philox (4x32, 7 rounds — same engine as the SR epilogue).

The dropout scheme: one philox
call per canonical GROUP — group(m, k) = (row-pair (m//16, m%8), k//32,
quad-class (k%8)//2), 16 elements at 8 bits/decision — chosen so a lane's
WGMMA fragment ownership is exactly whole groups (thread-local generation, no
shuffles, no redundancy) while staying a pure function of (m, k): any kernel
regenerates the same mask from (seed, offset).

Counter/key layout: counter64 = group_m | group_k << 32 with
group_m = (m//16)*8 + m%8 and group_k = ((k//32) << 2) | (k%8)//2;
key64 = seed + offset * PHILOX_OFFSET_MIX (odd), mixing the offset stream
into the key since the counter words are fully spent on coordinates.

Masking is integer-only: random bytes are PRMT'd into the low byte of a
constant-exponent 16-bit float pattern (bf16: 0x3F00 | b, f16: 0x3C00 | b —
positive normals whose order equals the byte's integer order), then one
``set.ge.u32.{bf16x2,f16x2}`` yields a 0xFFFF/0x0000 mask pair ANDed onto
both b16 elements with one LOP3. The keep-threshold ``BASE | int(p*256)`` is
a host-side constant; the mainloop never multiplies — fold 1/(1-p) into the
epilogue alpha or a later scale.
"""

import cutlass
from cutlass import Int32
from cutlass.cutlass_dsl import dsl_user_op

from torch._vendor.quack.blockscaled.nvfp4_utils import asm_i32
from torch._vendor.quack.rounding import philox  # noqa: F401  (re-exported for transforms)

# Odd 64-bit mix folding the offset stream into the philox key.
PHILOX_OFFSET_MIX = 0x9E3779B97F4A7C15

# PRMT selectors placing a word's random byte pairs into the low bytes of
# 16-bit lanes whose high byte comes from operand b (the constant base):
#   lanes = [bytes[2h], BASE, bytes[2h+1], BASE]
PRMT_SEL_H0 = 0x4140  # word bytes 0, 1 (k-half h = 0)
PRMT_SEL_H1 = 0x4342  # word bytes 2, 3 (k-half h = 1)


def b16_base_pattern(dtype) -> int:
    """High byte making ``BASE | random_byte`` a positive normal b16 whose
    order equals the byte's integer order (bf16: [1, 2); f16: [1.0, 1.25))."""
    return 0x3F00 if dtype is cutlass.BFloat16 else 0x3C00


@dsl_user_op
def set_ge_u32_b16x2(a: Int32, b: Int32, dtype, *, loc=None, ip=None) -> Int32:
    """Packed 16-bit-lane compare: 0xFFFF per lane where a >= b (as bf16x2 /
    f16x2 — exact unsigned integer order on the raw lanes given both are
    positive normals)."""
    suffix = "bf16x2" if dtype is cutlass.BFloat16 else "f16x2"
    return Int32(
        asm_i32(
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            f"set.ge.u32.{suffix} $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )
