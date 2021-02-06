/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>
#include <requantization/runtime-assembly.h>

# r0 mr
# r1 k
# r2 a
# r3 a_stride

.syntax unified

#  Args passed via stack.
#  TOS
#  |----------------|
#  |packed_a        | 0
#  |----------------|
#  

#  After loading w pointer in ip reg.
#  And after pushing r4-r9 and d8-d15 on stack
#  |----------------|
#  |r4 - r11        | 0 
#  |packed_a        | 32
#  |----------------|
#  

# Packed A format.
# 4kx4m blocks for alls blocks given 4 rows (4m) are placed in contiguous memory.
# Original A
# --------- K -----------          -- (K + 4 - 1) / 4 --
# |                     |          |                   |
# |                     |        (M + 4 - 1)/4         |
# |                     | Packed   |                   |
# M                     |  =>      |-------------------|
# |                     |        Thus Packed A has (K + 4 - 1)/4 * (M + 4 -1)/4 blocks
# |                     |
# |---------------------|
# 
# Each 4 x 4 blocks is transposed and stored.
# Each of the (K + 4 - 1)/4 blocks for a given group of 4 m blocks
# are stored adjacent in memory
# Thus, each block:
# |----4m-----|----4m-----|
# 4k          |           | ..... (K + 4 - 1)/4 blocks
# |-----------|-----------|
# This locality helps in loading 8kx4m blocks of activations
# Note when M is not multiple of 4, the rest can contain arbitrary
# data in packed A as we will not be writing those out.
# This wil be taken care by just copying the appropriate valid data

# void pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon(
#     size_t mr,
#     size_t K,
#     const uint8_t* a,
#     size_t a_stride,
#     uint8_t* packed_a,
BEGIN_FUNCTION pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon
    .arm
#ifndef __APPLE__
    .arch armv7-a
    .fpu neon
#endif

    PUSH {r4, r5, r6, r7, r8, r9, r10, r11}

    # r4 = a0 = a pointer
    MOV r4, r2
    # r2 = packed_a pointer
    LDR r2, [sp, 32]

    CMP r0, 2
    # r5 = a1
    ADD r5, r4, r3
    MOVLO r5, r4

    # r6 = a2
    ADD r6, r5, r3
    MOVLS r6, r5

    CMP r0, 4
    # r7 = a3
    ADD r7, r6, r3
    MOVNE r7, r6

    # num_k_blocks = (k + (4 - 1)) / 4
    ADD r1, r1, 3
    LSR r1, r1, 2

    SUBS r1, r1, 2
    BLO 1f

    .p2align 5
k_loop:
    VLD1.8 {d0}, [r4]!
    VLD1.8 {d1}, [r5]!
    VLD1.8 {d2}, [r6]!
    VLD1.8 {d3}, [r7]!

    #  Now we have 4x8 block of values that we will tranpose
    #  A matrix
    #  --------------------------------
    #  |                              |
    #  |a0-----a3 a4-----a7....|
    #  |b0 B00 b3 b4 B01 b7....|
    #  |c0     c3 c4     c7....|
    #  |d0-----d3 d4-----d7....|
    #  |                              |
    #  |                              |
    #  -------------------------------
    #  {va01, va23} = B00 + B01 = 2 uint8x16_t
    #  Sequence:
    #  VTRN.8 d0, d1 // low(va01), high(va01)
    #  VTRN.8 d2, d3 // low(va23), high(va23)
    #  VTRN.16 q0, q1 // va01, va23
    #  Now we have
    #  d0 = d4, c4, b4, a4 : d0, c0, b0, a0
    #  d1 = d5, c5, b5, a5 : d1, c1, b1, a1
    #  d2 = d6, c6, b6, a6 : d2, c2, b2, a2
    #  d3 = d7, c7, b7, a7 : d3, c3, b3, a3
    #  Thus 2 4x4 blocks are transposed.
    #  Now we have all 2 B00, B01 transposed.

    VTRN.8 d0, d1
    VTRN.8 d2, d3
    VTRN.16 q0, q1

    # Now VTRN.32 d0, d1
    # Now VTRN.32 d2, d3
    # Thus we have
    #  d0 = d1, c1, b1, a1 : d0, c0, b0, a0
    #  d1 = d5, c5, b5, a5 : d4, c4, b4, a4
    #  d2 = d3, c3, b3, a3 : d2, c2, b2, a2
    #  d3 = d7, c7, b7, a7 : d6, c6, b6, a6
    #  Then we can do
    # VSWP d1, d2
    #  d0 = d1, c1, b1, a1 : d0, c0, b0, a0
    #  d1 = d3, c3, b3, a3 : d2, c2, b2, a2
    #  d2 = d5, c5, b5, a5 : d4, c4, b4, a4
    #  d3 = d7, c7, b7, a7 : d6, c6, b6, a6
    # Now we can store q0 contiguously followed
    VTRN.32 d0, d1
    VTRN.32 d2, d3
    VSWP d1, d2

    # Now store the tranposed values
    # d0, d1, d2, d3
    VST1.8 {q0}, [r2]!
    VST1.8 {q1}, [r2]!

    SUBS r1, r1, 2

    BHS k_loop
1:
    CMP r1, -2
    BEQ 2f

    VLD1.32 {d0[]}, [r4]
    VLD1.32 {d1[]}, [r5]
    VLD1.32 {d2[]}, [r6]
    VLD1.32 {d3[]}, [r7]

    #  Now we have 4x8 block of values that we will tranpose
    #  _d{0-3} are arm neon vector registers
    #  va0 = _d0 = a0 a1 a2 a3
    #  va1 = _d1 = b0 b1 b2 b3
    #  va2 = _d2 = c0 c1 c2 c3
    #  va3 = _d3 = d0 d1 d2 d3
    #  A matrix
    #  ----------------------------
    #  |                          |
    #  |                 a0-----a3|
    #  |                 b0 B00 b3|
    #  |   last block    c0     c3|
    #  |                 d0-----d3|
    #  |                          |
    #  |                          |
    #  ---------------------------
    #  Sequence:
    #  VTRN.8 d0, d1 // va0, va1
    #  VTRN.8 d2, d3 // va2, va3
    #  Now we have
    #  d0 = b2, a2, b0, a0
    #  d1 = b3, a3, b1, a1
    #  d2 = d2, c2, d0, c0
    #  d3 = d3, c3, d1, c1
    #  Sequence:
    #  VTRN.16 d0, d2
    #  VTRN.16 d1, d3
    #  Now we have
    #  d0 = d0, c0, b0, a0
    #  d1 = d1, c1, b1, a1
    #  d2 = d2, c2, b2, a2
    #  d3 = d3, c3, b3, a3

    VTRN.8 d0, d1
    VTRN.8 d2, d3
    VTRN.16 d0, d2
    VTRN.16 d1, d3

    # Since upper half of d0 just contains duplicate values
    # We dont want to store those
    # So let's combine upper half of d0 to the lower part of d0
    # And lower half of d1 to upper half of d0
    # Same for d2, d3
    VEXT.8 d0, d0, d1, #4
    VEXT.8 d1, d2, d3, #4

    # Now store the tranposed values
    # d0, d1, d2, d3
    VST1.8 {q0}, [r2]
    .p2align 4
2:
    POP {r4, r5, r6, r7, r8, r9, r10, r11}
    BX lr

END_FUNCTION pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
