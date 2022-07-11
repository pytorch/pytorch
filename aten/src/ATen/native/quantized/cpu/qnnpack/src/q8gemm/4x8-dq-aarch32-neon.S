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
# r1 nr
# r2 k
# r3 a
# r6 a_stride

# d14 a_zero_point
# d15 b_zero_point

## Stack
# 4     quantization_params
# 4     c_stride
# 4     c
# 4     b
# 4     w
# 4     a_stride
# --
# 16    r4-r7
# 64    d8-d18

.syntax unified

#  Args passed via stack.
#  TOS
#  |-----------|
#  |a_stride   | 0
#  |w          | 4
#  |c          | 8
#  |c_stride   | 12
#  |out ch indx| 16
#  |params     | 20
#  |-----------|
#

#  After loading w pointer in ip reg.
#  And after pushing r4-r8 and d8-d15 on stack
#  |-----------|
#  |d8 - d15   | 0
#  |r4 - r7    | 64
#  |a_stride   | 80
#  |w          | 84
#  |b          | 88
#  |c          | 92
#  |c_stride   | 96
#  |out ch indx| 100
#  |params     | 104
#  |-----------|
#

# void pytorch_q8gemm_ukernel_4x8__aarch32_neon(
#     size_t mr,
#     size_t nr,
#     size_t k,
#     const uint8_t* restrict a,
#     size_t a_stride,
#     const void* restrict w,
#     const float* restrict b,
#     uint8_t* restrict c,
#     size_t c_stride,
#     size_t output_channel_index,
#     const union pytorch_qnnp_conv_dynamic_quantization_params quantization_params[restrict static 1])
BEGIN_FUNCTION pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon
    .arm
#ifndef __APPLE__
    .arch armv7-a
    .fpu neon
#endif
    # Load w
    # - ip = w
    LDR ip, [sp, 4]
    ADD ip, ip, 32

    PUSH {r4, r5, r6, r7}
    VPUSH {d8-d15}

    # Load output channel index
    LDR r5, [sp, 100]
    # Load quantization params
    # - r7 = quantization_params
    LDR r7, [sp, 104]
    # Load input_zero_point
    VLD1.8 {d14[]}, [r7]
    ADD r7, r7, 4
    # Load pointer to per channel zero points array
    # Post-index: After load increment r7 by 4
    LDR r4, [r7], #4
    # Byte offset of output channel index for requant scale.
    LSL r6, r5, 2

    VEOR q8,  q8,  q8
    VEOR q9,  q9,  q9

    # Load pointer to per channel requant scale
    LDR r7, [r7]
    # Add output_channel_index to the b_zero_point pointer
    ADD r4, r4, r5
    # Now r7 has the base_addr + offset for multipliers
    ADD r7, r7, r6

    # Load a_stride
    # - r6 = a_stride
    LDR r6, [sp, 80]

    VEOR q10, q10, q10
    VEOR q11, q11, q11

    VLD1.8 {d15}, [r4]

    CMP r0, 2
    ADD r4, r3, r6
    MOVLO r4, r3

    ADD r5, r4, r6
    MOVLS r5, r4

    CMP r0, 4
    ADD r6, r5, r6
    MOVNE r6, r5

    VEOR q12, q12, q12
    VEOR q13, q13, q13
    VEOR q14, q14, q14
    VEOR q15, q15, q15

    SUBS r2, r2, 8
    BLO 1f

    .p2align 5
0:
    # Load a0
    # - d1 = a0
    VLD1.8 {d1}, [r3]!

    # Load a1
    # - d3 = a1
    VLD1.8 {d3}, [r4]!

    # Load b0-b7 (channel 0)
    # - d9 = b0-b7
    VLD1.8 {d9}, [ip:64]!

    # Load a2
    # - d5 = a2
    VLD1.8 {d5}, [r5]!

    # q0 = va0 = a0
    SUB_ZERO_POINT q0, d1, d14

    # Load a3
    # - d7 = a3
    VLD1.8 {d7}, [r6]!

    # q1 = va1 = a1
    SUB_ZERO_POINT q1, d3, d14

    # q4 = b0:7 - b_zero_point
    # - d8 = vb0123 (channel 0)
    # - d9 = vb4567 (channel 0)
    VSUBL.U8 q4, d9, d15

    # q2 = va2 = a2
    SUB_ZERO_POINT q2, d5, d14
    # q3 = va3 = a3
    SUB_ZERO_POINT q3, d7, d14

    ### Channel 0 ###

    # Load b0-b7 (channel 1)
    # - d11 = b0-b7
    VLD1.8 {d11}, [ip:64]!

    # vacc0x0123 += vb0123 * va0[0]
    VMLAL.S16 q8, d8, d0[0]
    # vacc0x4567 += vb4567 * va0[0]
    VMLAL.S16 q9, d9, d0[0]

    # vacc1x0123 += vb0123 * va1[0]
    VMLAL.S16 q10, d8, d2[0]
    # vacc1x4567 += vb4567 * va1[0]
    VMLAL.S16 q11, d9, d2[0]

    # vacc2x0123 += vb0123 * va2[0]
    VMLAL.S16 q12, d8, d4[0]
    # vacc2x4567 += vb4567 * va2[0]
    VMLAL.S16 q13, d9, d4[0]

    # q5 = b0:7 - b_zero_point
    # - d10 = vb0123 (channel 1)
    # - d11 = vb4567 (channel 1)
    VSUBL.U8 q5, d11, d15

    # vacc3x0123 += vb0123 * va3[0]
    VMLAL.S16 q14, d8, d6[0]
    # vacc3x4567 += vb4567 * va3[0]
    VMLAL.S16 q15, d9, d6[0]

    ### Channel 1 ###

    # Load b0-b7 (channel 2)
    # - d9 = b0-b7
    VLD1.8 {d9}, [ip:64]!

    # vacc0x0123 += vb0123 * va0[1]
    VMLAL.S16 q8, d10, d0[1]
    # vacc0x4567 += vb4567 * va0[1]
    VMLAL.S16 q9, d11, d0[1]

    # vacc1x0123 += vb0123 * va1[1]
    VMLAL.S16 q10, d10, d2[1]
    # vacc1x4567 += vb4567 * va1[1]
    VMLAL.S16 q11, d11, d2[1]

    # vacc2x0123 += vb0123 * va2[1]
    VMLAL.S16 q12, d10, d4[1]
    # vacc2x4567 += vb4567 * va2[1]
    VMLAL.S16 q13, d11, d4[1]

    # q4 = b0:7 - b_zero_point
    # - d8 = vb0123 (channel 2)
    # - d9 = vb4567 (channel 2)
    VSUBL.U8 q4, d9, d15

    # vacc3x0123 += vb0123 * va3[1]
    VMLAL.S16 q14, d10, d6[1]
    # vacc3x4567 += vb4567 * va3[1]
    VMLAL.S16 q15, d11, d6[1]

    ### Channel 2 ###

    # Load b0-b7 (channel 3)
    # - d11 = b0-b7
    VLD1.8 {d11}, [ip:64]!

    # vacc0x0123 += vb0123 * va0[2]
    VMLAL.S16 q8, d8, d0[2]
    # vacc0x4567 += vb4567 * va0[2]
    VMLAL.S16 q9, d9, d0[2]

    # vacc1x0123 += vb0123 * va1[2]
    VMLAL.S16 q10, d8, d2[2]
    # vacc1x4567 += vb4567 * va1[2]
    VMLAL.S16 q11, d9, d2[2]

    # vacc2x0123 += vb0123 * va2[2]
    VMLAL.S16 q12, d8, d4[2]
    # vacc2x4567 += vb4567 * va2[2]
    VMLAL.S16 q13, d9, d4[2]

    # q5 = b0:7 - b_zero_point
    # - d10 = vb0123 (channel 3)
    # - d11 = vb4567 (channel 3)
    VSUBL.U8 q5, d11, d15

    # vacc3x0123 += vb0123 * va3[2]
    VMLAL.S16 q14, d8, d6[2]
    # vacc3x4567 += vb4567 * va3[2]
    VMLAL.S16 q15, d9, d6[2]

    ### Channel 3 ###

    # Load b0-b7 (channel 4)
    # - d9 = b0-b7
    VLD1.8 {d9}, [ip:64]!

    # vacc0x0123 += vb0123 * va0[3]
    VMLAL.S16 q8, d10, d0[3]
    # vacc0x4567 += vb4567 * va0[3]
    VMLAL.S16 q9, d11, d0[3]

    # vacc1x0123 += vb0123 * va1[3]
    VMLAL.S16 q10, d10, d2[3]
    # vacc1x4567 += vb4567 * va1[3]
    VMLAL.S16 q11, d11, d2[3]

    # vacc2x0123 += vb0123 * va2[3]
    VMLAL.S16 q12, d10, d4[3]
    # vacc2x4567 += vb4567 * va2[3]
    VMLAL.S16 q13, d11, d4[3]

    # q5 = b0:7 - b_zero_point
    # - d10 = vb0123 (channel 4)
    # - d11 = vb4567 (channel 4)
    VSUBL.U8 q4, d9, d15

    # vacc3x0123 += vb0123 * va3[3]
    VMLAL.S16 q14, d10, d6[3]
    # vacc3x4567 += vb4567 * va3[3]
    VMLAL.S16 q15, d11, d6[3]

    ### Channel 4 ###

    # Load b0-b7 (channel 5)
    # - d11 = b0-b7
    VLD1.8 {d11}, [ip:64]!

    # vacc0x0123 += vb0123 * va0[4]
    VMLAL.S16 q8, d8, d1[0]
    # vacc0x4567 += vb4567 * va0[4]
    VMLAL.S16 q9, d9, d1[0]

    # vacc1x0123 += vb0123 * va1[4]
    VMLAL.S16 q10, d8, d3[0]
    # vacc1x4567 += vb4567 * va1[4]
    VMLAL.S16 q11, d9, d3[0]

    # vacc2x0123 += vb0123 * va2[4]
    VMLAL.S16 q12, d8, d5[0]
    # vacc2x4567 += vb4567 * va2[4]
    VMLAL.S16 q13, d9, d5[0]

    # q4 = b0:7 - b_zero_point
    # - d8 = vb0123 (channel 5)
    # - d9 = vb4567 (channel 5)
    VSUBL.U8 q5, d11, d15

    # vacc3x0123 += vb0123 * va3[4]
    VMLAL.S16 q14, d8, d7[0]
    # vacc3x4567 += vb4567 * va3[4]
    VMLAL.S16 q15, d9, d7[0]

    ### Channel 5 ###

    # Load b0-b7 (channel 6)
    # - d9 = b0-b7
    VLD1.8 {d9}, [ip:64]!

    # vacc0x0123 += vb0123 * va0[5]
    VMLAL.S16 q8, d10, d1[1]
    # vacc0x4567 += vb4567 * va0[5]
    VMLAL.S16 q9, d11, d1[1]

    # vacc1x0123 += vb0123 * va1[5]
    VMLAL.S16 q10, d10, d3[1]
    # vacc1x4567 += vb4567 * va1[5]
    VMLAL.S16 q11, d11, d3[1]

    # vacc2x0123 += vb0123 * va2[5]
    VMLAL.S16 q12, d10, d5[1]
    # vacc2x4567 += vb4567 * va2[5]
    VMLAL.S16 q13, d11, d5[1]

    # q4 = b0:7 - b_zero_point
    # - d8 = vb0123 (channel 6)
    # - d9 = vb4567 (channel 6)
    VSUBL.U8 q4, d9, d15

    # vacc3x0123 += vb0123 * va3[5]
    VMLAL.S16 q14, d10, d7[1]
    # vacc3x4567 += vb4567 * va3[5]
    VMLAL.S16 q15, d11, d7[1]

    ### Channel 6 ###

    # Load b0-b7 (channel 7)
    # - d11 = b0-b7
    VLD1.8 {d11}, [ip:64]!

    # vacc0x0123 += vb0123 * va0[6]
    VMLAL.S16 q8, d8, d1[2]
    # vacc0x4567 += vb4567 * va0[6]
    VMLAL.S16 q9, d9, d1[2]

    # vacc1x0123 += vb0123 * va1[6]
    VMLAL.S16 q10, d8, d3[2]
    # vacc1x4567 += vb4567 * va1[6]
    VMLAL.S16 q11, d9, d3[2]

    # vacc2x0123 += vb0123 * va2[6]
    VMLAL.S16 q12, d8, d5[2]

    # q5 = b0:7 - b_zero_point
    # - d10 = vb0123 (channel 7)
    # - d11 = vb4567 (channel 7)
    VSUBL.U8 q5, d11, d15

    # vacc2x4567 += vb4567 * va2[6]
    VMLAL.S16 q13, d9, d5[2]

    # vacc3x0123 += vb0123 * va3[6]
    VMLAL.S16 q14, d8, d7[2]
    # vacc3x4567 += vb4567 * va3[6]
    VMLAL.S16 q15, d9, d7[2]

    ### Channel 8 ###
    SUBS r2, r2, 8

    # vacc0x0123 += vb0123 * va0[7]
    VMLAL.S16 q8, d10, d1[3]
    # vacc0x4567 += vb4567 * va0[7]
    VMLAL.S16 q9, d11, d1[3]

    # vacc1x0123 += vb0123 * va1[7]
    VMLAL.S16 q10, d10, d3[3]
    # vacc1x4567 += vb4567 * va1[7]
    VMLAL.S16 q11, d11, d3[3]

    # vacc2x0123 += vb0123 * va2[7]
    VMLAL.S16 q12, d10, d5[3]
    # vacc2x4567 += vb4567 * va2[7]
    VMLAL.S16 q13, d11, d5[3]

    # vacc3x0123 += vb0123 * va3[7]
    VMLAL.S16 q14, d10, d7[3]
    # vacc3x4567 += vb4567 * va3[7]
    VMLAL.S16 q15, d11, d7[3]

    BHS 0b

1:
    CMP r2, -8
    BEQ 2f

    # Adjust a0, a1, a2, a3
    ADD r3, r2
    ADD r4, r2
    ADD r5, r2
    ADD r6, r2

    # a_shift = 8 * k - 64
    LSL r2, r2, 3
    VDUP.32 d13, r2

    # Load a0
    # - d1 = a0
    VLD1.8 {d1}, [r3]

    # Load a1
    # - d3 = a1
    VLD1.8 {d3}, [r4]

    # Load b0-b7 (channel 0)
    # - d9 = b0-b7
    VLD1.8 {d9}, [ip:64]!

    # Load a2
    # - d5 = a2
    VLD1.8 {d5}, [r5]

    # q0 = va0 = a0
    VSHL.U64 d1, d1, d13
    SUB_ZERO_POINT q0, d1, d14

    # Load a3
    # - d7 = a3
    VLD1.8 {d7}, [r6]

    # q1 = va1 = a1
    VSHL.U64 d3, d3, d13
    SUB_ZERO_POINT q1, d3, d14

    # q4 = b0:7 - b_zero_point
    # - d8 = vb0123 (channel 0)
    # - d9 = vb4567 (channel 0)
    VSUBL.U8 q4, d9, d15

    # q2 = va2 = a2
    VSHL.U64 d5, d5, d13
    SUB_ZERO_POINT q2, d5, d14
    # q3 = va3 = a3
    VSHL.U64 d7, d7, d13
    SUB_ZERO_POINT q3, d7, d14

    ### Channel 0 ###

    # vacc0x0123 += vb0123 * va0[0]
    VMLAL.S16 q8, d8, d0[0]
    # vacc0x4567 += vb4567 * va0[0]
    VMLAL.S16 q9, d9, d0[0]

    # vacc1x0123 += vb0123 * va1[0]
    VMLAL.S16 q10, d8, d2[0]
    # vacc1x4567 += vb4567 * va1[0]
    VMLAL.S16 q11, d9, d2[0]

    # vacc2x0123 += vb0123 * va2[0]
    VMLAL.S16 q12, d8, d4[0]
    # vacc2x4567 += vb4567 * va2[0]
    VMLAL.S16 q13, d9, d4[0]

    # vacc3x0123 += vb0123 * va3[0]
    VMLAL.S16 q14, d8, d6[0]
    # vacc3x4567 += vb4567 * va3[0]
    VMLAL.S16 q15, d9, d6[0]

    CMP r2, -48
    BLO 2f

    ### Channel 1 ###

    # Load b0-b7 (channel 1)
    # - d11 = b0-b7
    VLD1.8 {d11}, [ip:64]!

    # q5 = b0:7 - b_zero_point
    # - d10 = vb0123 (channel 1)
    # - d11 = vb4567 (channel 1)
    VSUBL.U8 q5, d11, d15

    # vacc0x0123 += vb0123 * va0[1]
    VMLAL.S16 q8, d10, d0[1]
    # vacc0x4567 += vb4567 * va0[1]
    VMLAL.S16 q9, d11, d0[1]

    # vacc1x0123 += vb0123 * va1[1]
    VMLAL.S16 q10, d10, d2[1]
    # vacc1x4567 += vb4567 * va1[1]
    VMLAL.S16 q11, d11, d2[1]

    # vacc2x0123 += vb0123 * va2[1]
    VMLAL.S16 q12, d10, d4[1]
    # vacc2x4567 += vb4567 * va2[1]
    VMLAL.S16 q13, d11, d4[1]

    # vacc3x0123 += vb0123 * va3[1]
    VMLAL.S16 q14, d10, d6[1]
    # vacc3x4567 += vb4567 * va3[1]
    VMLAL.S16 q15, d11, d6[1]

    ### Channel 2 ###
    BLS 2f

    # Load b0-b7 (channel 2)
    # - d9 = b0-b7
    VLD1.8 {d9}, [ip:64]!

    # q4 = b0:7 - b_zero_point
    # - d8 = vb0123 (channel 2)
    # - d9 = vb4567 (channel 2)
    VSUBL.U8 q4, d9, d15

    # vacc0x0123 += vb0123 * va0[2]
    VMLAL.S16 q8, d8, d0[2]
    # vacc0x4567 += vb4567 * va0[2]
    VMLAL.S16 q9, d9, d0[2]

    # vacc1x0123 += vb0123 * va1[2]
    VMLAL.S16 q10, d8, d2[2]
    # vacc1x4567 += vb4567 * va1[2]
    VMLAL.S16 q11, d9, d2[2]

    # vacc2x0123 += vb0123 * va2[2]
    VMLAL.S16 q12, d8, d4[2]
    # vacc2x4567 += vb4567 * va2[2]
    VMLAL.S16 q13, d9, d4[2]

    # vacc3x0123 += vb0123 * va3[2]
    VMLAL.S16 q14, d8, d6[2]
    # vacc3x4567 += vb4567 * va3[2]
    VMLAL.S16 q15, d9, d6[2]

    ### Channel 3 ###
    CMP r2, -32
    BLO 2f

    # Load b0-b7 (channel 3)
    # - d9 = b0-b7
    VLD1.8 {d11}, [ip:64]!

    # q4 = b0:7 - b_zero_point
    # - d8 = vb0123 (channel 3)
    # - d9 = vb4567 (channel 3)
    VSUBL.U8 q5, d11, d15

    # vacc0x0123 += vb0123 * va0[3]
    VMLAL.S16 q8, d10, d0[3]
    # vacc0x4567 += vb4567 * va0[3]
    VMLAL.S16 q9, d11, d0[3]

    # vacc1x0123 += vb0123 * va1[3]
    VMLAL.S16 q10, d10, d2[3]
    # vacc1x4567 += vb4567 * va1[3]
    VMLAL.S16 q11, d11, d2[3]

    # vacc2x0123 += vb0123 * va2[3]
    VMLAL.S16 q12, d10, d4[3]
    # vacc2x4567 += vb4567 * va2[3]
    VMLAL.S16 q13, d11, d4[3]

    # vacc3x0123 += vb0123 * va3[3]
    VMLAL.S16 q14, d10, d6[3]
    # vacc3x4567 += vb4567 * va3[3]
    VMLAL.S16 q15, d11, d6[3]

    ### Channel 4 ###
    BLS 2f

    # Load b0-b7 (channel 4)
    # - d11 = b0-b7
    VLD1.8 {d9}, [ip:64]!

    # q5 = b0:7 - b_zero_point
    # - d10 = vb0123 (channel 4)
    # - d11 = vb4567 (channel 4)
    VSUBL.U8 q4, d9, d15

    # vacc0x0123 += vb0123 * va0[4]
    VMLAL.S16 q8, d8, d1[0]
    # vacc0x4567 += vb4567 * va0[4]
    VMLAL.S16 q9, d9, d1[0]

    # vacc1x0123 += vb0123 * va1[4]
    VMLAL.S16 q10, d8, d3[0]
    # vacc1x4567 += vb4567 * va1[4]
    VMLAL.S16 q11, d9, d3[0]

    # vacc2x0123 += vb0123 * va2[4]
    VMLAL.S16 q12, d8, d5[0]
    # vacc2x4567 += vb4567 * va2[4]
    VMLAL.S16 q13, d9, d5[0]

    # vacc3x0123 += vb0123 * va3[4]
    VMLAL.S16 q14, d8, d7[0]
    # vacc3x4567 += vb4567 * va3[4]
    VMLAL.S16 q15, d9, d7[0]

    ### Channel 5 ###
    CMP r2, -16
    BLO 2f

    # Load b0-b7 (channel 5)
    # - d13 = b0-b7
    VLD1.8 {d11}, [ip:64]!

    # q5 = b0:7 - b_zero_point
    # - d10 = vb0123 (channel 5)
    # - d11 = vb4567 (channel 5)
    VSUBL.U8 q5, d11, d15

    # vacc0x0123 += vb0123 * va0[5]
    VMLAL.S16 q8, d10, d1[1]
    # vacc0x4567 += vb4567 * va0[5]
    VMLAL.S16 q9, d11, d1[1]

    # vacc1x0123 += vb0123 * va1[5]
    VMLAL.S16 q10, d10, d3[1]
    # vacc1x4567 += vb4567 * va1[5]
    VMLAL.S16 q11, d11, d3[1]

    # vacc2x0123 += vb0123 * va2[5]
    VMLAL.S16 q12, d10, d5[1]
    # vacc2x4567 += vb4567 * va2[5]
    VMLAL.S16 q13, d11, d5[1]

    # vacc3x0123 += vb0123 * va3[5]
    VMLAL.S16 q14, d10, d7[1]
    # vacc3x4567 += vb4567 * va3[5]
    VMLAL.S16 q15, d11, d7[1]

    ### Channel 6 ###
    BLS 2f

    # Load b0-b7 (channel 6)
    # - d9 = b0-b7
    VLD1.8 {d9}, [ip:64]

    # q4 = b0:7 - b_zero_point
    # - d8 = vb0123 (channel 6)
    # - d9 = vb4567 (channel 6)
    VSUBL.U8 q4, d9, d15

    # vacc0x0123 += vb0123 * va0[6]
    VMLAL.S16 q8, d8, d1[2]
    # vacc0x4567 += vb4567 * va0[6]
    VMLAL.S16 q9, d9, d1[2]

    # vacc1x0123 += vb0123 * va1[6]
    VMLAL.S16 q10, d8, d3[2]
    # vacc1x4567 += vb4567 * va1[6]
    VMLAL.S16 q11, d9, d3[2]

    # vacc2x0123 += vb0123 * va2[6]
    VMLAL.S16 q12, d8, d5[2]

    # vacc2x4567 += vb4567 * va2[6]
    VMLAL.S16 q13, d9, d5[2]

    # vacc3x0123 += vb0123 * va3[6]
    VMLAL.S16 q14, d8, d7[2]
    # vacc3x4567 += vb4567 * va3[6]
    VMLAL.S16 q15, d9, d7[2]

    .p2align 4
2:
    LDR r6, [sp, 88]
    # Load q6: vmultiplier_c0123
    VLD1.32 {d12, d13}, [r7]!
    VCVT.F32.S32 q8, q8
    VCVT.F32.S32 q9, q9
    VCVT.F32.S32 q10, q10
    # Load q7: vmultiplier_c4567
    VLD1.32 {d14, d15}, [r7]
    VLD1.32 {q0, q1}, [r6]

    VCVT.F32.S32 q11, q11
    VCVT.F32.S32 q12, q12
    VCVT.F32.S32 q13, q13
    VCVT.F32.S32 q14, q14
    VCVT.F32.S32 q15, q15

    VMUL.F32 q8, q8, q6
    VMUL.F32 q9, q9, q7
    VMUL.F32 q10, q10, q6
    VMUL.F32 q11, q11, q7
    VMUL.F32 q12, q12, q6
    VMUL.F32 q13, q13, q7
    VMUL.F32 q14, q14, q6
    VMUL.F32 q15, q15, q7

    VADD.F32 q8, q8, q0
    VADD.F32 q9, q9, q1
    VADD.F32 q10, q10, q0
    VADD.F32 q11, q11, q1
    VADD.F32 q12, q12, q0
    VADD.F32 q13, q13, q1
    VADD.F32 q14, q14, q0
    VADD.F32 q15, q15, q1

    # Load c, c_stride:
    # - r2 = c
    # - r3 = c_stride
    LDRD r2, r3, [sp, 92]
    LSL r3, r3, 2
    ADD r4, r2, r3

    CMP r0, 2
    MOVLO r4, r2

    ADD r5, r4, r3
    MOVLS r5, r4

    CMP r0, 4
    ADD r3, r5, r3
    MOVNE r3, r5

    CMP r1, 8
    BNE 4f

    VST1.32 {q8}, [r2]!
    VST1.32 {q10}, [r4]!
    VST1.32 {q12}, [r5]!
    VST1.32 {q14}, [r3]!

    VST1.32 {q9}, [r2]
    VST1.32 {q11}, [r4]
    VST1.32 {q13}, [r5]
    VST1.32 {q15}, [r3]

    VPOP {d8-d15}
    POP {r4, r5, r6, r7}
    BX lr

    .p2align 3
4:
    CMP r1, 4
    BLO 5f

    VST1.32 {q8}, [r2]!
    VST1.32 {q10}, [r4]!
    VST1.32 {q12}, [r5]!
    VST1.32 {q14}, [r3]!

    SUB r1, 4

    VMOV.32 q8, q9
    VMOV.32 q10, q11
    VMOV.32 q12, q13
    VMOV.32 q14, q15

5:
    CMP r1, 2
    BLO 6f

    VST1.32 {d16}, [r2]!
    VST1.32 {d20}, [r4]!
    VST1.32 {d24}, [r5]!
    VST1.32 {d28}, [r3]!

    SUB r1, 2

    VEXT.32 q8, q8, 2
    VEXT.32 q10, q10, 2
    VEXT.32 q12, q12, 2
    VEXT.32 q14, q14, 2

6:
    TEQ r1, 0
    BEQ 7f

    VST1.32 {d16[0]}, [r2]!
    VST1.32 {d20[0]}, [r4]!
    VST1.32 {d24[0]}, [r5]!
    VST1.32 {d28[0]}, [r3]!

7:
    VPOP {d8-d15}
    POP {r4, r5, r6, r7}
    BX lr
END_FUNCTION pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
