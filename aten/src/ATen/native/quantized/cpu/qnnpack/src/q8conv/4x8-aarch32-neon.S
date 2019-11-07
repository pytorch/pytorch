/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>
#include <requantization/runtime-assembly.h>

.syntax unified

# void pytorch_q8conv_ukernel_4x8__aarch32_neon(
#     size_t mr,
#     size_t nr,
#     size_t kc,
#     size_t ks,
#     const uint8_t**restrict a,
#     const void*restrict w,
#     uint8_t*restrict c,
#     size_t c_stride,
#     const union pytorch_qnnp_conv_quantization_params quantization_params[restrict static 1])
BEGIN_FUNCTION pytorch_q8conv_ukernel_4x8__aarch32_neon
    .arm
#ifndef __APPLE__
    .arch armv7-a
    .fpu neon
#endif
    # Load w
    # - ip = w
    LDR ip, [sp, 4]
    PUSH {r4, r5, r6, r7, r8, r9, r10, r11}

    VPUSH {d8-d15}

    # Load bias0123, bias4567
    VLDM ip!, {d16-d19}
    # Load params:
    # - r9 = params
    LDR r9, [sp, 112]

    # q10 := vacc1x0123
    VMOV.I32 q10, q8
    MOV r4, 2
    # q11 := vacc1x4567
    VMOV.I32 q11, q9
    # Load a
    # - r8 = a
    LDR r8, [sp, 96]
    # q12 := vacc2x0123
    VMOV.I32 q12, q8
    # q13 := vacc2x4567
    VMOV.I32 q13, q9
    # q14 := vacc3x0123
    VMOV.I32 q14, q8
    # Load b_zero_point:
    # - d15 = b_zero_point
    VLD1.8 {d15[]}, [r9], r4
    # Load a_zero_point:
    # - d14 = a_zero_point
    VLD1.8 {d14[]}, [r9], r4
    # q15 := vacc3x4567
    VMOV.I32 q15, q9
    # Load multiplier:
    # - d12 = vmultiplier
    VLD1.32 {d12[]}, [r9]!

    .p2align 5
0:
    SUBS r10, r2, 8

    # Load a0, a1, a2, a3
    # - r4 = a0
    # - r5 = a1
    # - r6 = a2
    # - r7 = a3
    LDM r8!, {r4-r7}

    BLO 2f

1:
    # Load va0
    # - d1 = va0
    VLD1.8 {d1}, [r4]!

    # Load va1
    # - d3 = va1
    VLD1.8 {d3}, [r5]!

    # Load vb0-vb7 (channel 0)
    # - d9 = vb0-vb7
    VLD1.8 {d9}, [ip:64]!

    # Load va2
    # - d5 = va2
    VLD1.8 {d5}, [r6]!

    # q0 = va0 = a0
    SUB_ZERO_POINT q0, d1, d14

    # Load va3
    # - d7 = va3
    VLD1.8 {d7}, [r7]!

    # q1 = va1 = a1
    SUB_ZERO_POINT q1, d3, d14

    # q4 = b0:7 - vb_zero_point
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

    # q5 = b0:7 - vb_zero_point
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

    # q4 = b0:7 - vb_zero_point
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

    # q5 = b0:7 - vb_zero_point
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

    # q5 = b0:7 - vb_zero_point
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

    # q4 = b0:7 - vb_zero_point
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

    # q4 = b0:7 - vb_zero_point
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

    # q5 = b0:7 - vb_zero_point
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
    SUBS r10, r10, 8

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

    BHS 1b

2:
    CMP r10, -8
    BEQ 3f

    # Adjust a0, a1, a2, a3
    ADD r4, r10
    ADD r5, r10
    ADD r6, r10
    ADD r7, r10

    # a_shift = 8 * k - 64
    LSL r10, r10, 3
    VDUP.32 d13, r10

    # Load va0
    # - d1 = va0
    VLD1.8 {d1}, [r4]

    # Load va1
    # - d3 = va1
    VLD1.8 {d3}, [r5]

    # Load b0-b7 (channel 0)
    # - d9 = b0-b7
    VLD1.8 {d9}, [ip:64]!

    # Load a2
    # - d5 = a2
    VLD1.8 {d5}, [r6]

    # q0 = va0 = a0
    VSHL.U64 d1, d1, d13
    SUB_ZERO_POINT q0, d1, d14

    # Load a3
    # - d7 = a3
    VLD1.8 {d7}, [r7]

    # q1 = va1 = a1
    VSHL.U64 d3, d3, d13
    SUB_ZERO_POINT q1, d3, d14

    # q4 = b0:7 - vb_zero_point
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

    CMP r10, -48
    BLO 3f

    ### Channel 1 ###

    # Load b0-b7 (channel 1)
    # - d11 = b0-b7
    VLD1.8 {d11}, [ip:64]!

    # q5 = b0:7 - vb_zero_point
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
    BLS 3f

    # Load b0-b7 (channel 2)
    # - d9 = b0-b7
    VLD1.8 {d9}, [ip:64]!

    # q4 = b0:7 - vb_zero_point
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
    CMP r10, -32
    BLO 3f

    # Load b0-b7 (channel 3)
    # - d9 = b0-b7
    VLD1.8 {d11}, [ip:64]!

    # q4 = b0:7 - vb_zero_point
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
    BLS 3f

    # Load b0-b7 (channel 4)
    # - d11 = b0-b7
    VLD1.8 {d9}, [ip:64]!

    # q5 = b0:7 - vb_zero_point
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
    CMP r10, -16
    BLO 3f

    # Load b0-b7 (channel 5)
    # - d13 = b0-b7
    VLD1.8 {d11}, [ip:64]!

    # q5 = b0:7 - vb_zero_point
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
    BLS 3f

    # Load b0-b7 (channel 6)
    # - d9 = b0-b7
    VLD1.8 {d9}, [ip:64]!

    # q4 = b0:7 - vb_zero_point
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
3:
    SUBS r3, r3, 1
    BNE 0b

    # Load right_shift
    # - q4 = d8:d9 = vright_shift
    VLD1.32 {d8[], d9[]}, [r9]!

    VQRDMULH.S32  q8, q8, d12[0]
    VQRDMULH.S32  q9, q9, d12[0]
    VQRDMULH.S32 q10, q10, d12[0]
    VQRDMULH.S32 q11, q11, d12[0]

    # Compute vzero_shift_mask
    # - q5 = vzero_shift_mask
    VCEQ.S32 q5, q4, 0

    VQRDMULH.S32 q12, q12, d12[0]
    VQRDMULH.S32 q13, q13, d12[0]
    VQRDMULH.S32 q14, q14, d12[0]
    VQRDMULH.S32 q15, q15, d12[0]

    VBIC q0,  q8, q5
    VBIC q1,  q9, q5
    VBIC q2, q10, q5
    VBIC q3, q11, q5

    VSRA.S32  q8, q0, 31
    VSRA.S32  q9, q1, 31
    VSRA.S32 q10, q2, 31
    VSRA.S32 q11, q3, 31

    # Load output_zero_point
    # - q7 = d14:d15 = voutput_zero_point
    VLD1.16 {d14[], d15[]}, [r9]!

    VBIC q0, q12, q5
    VBIC q1, q13, q5
    VBIC q2, q14, q5
    VBIC q3, q15, q5

    VSRA.S32 q12, q0, 31
    VSRA.S32 q13, q1, 31
    VSRA.S32 q14, q2, 31
    VSRA.S32 q15, q3, 31

    # Load max:
    # - q5 = d10:d11 = voutput_max
    VLD1.8 {d10[], d11[]}, [r9]!

    VRSHL.S32  q8,  q8, q4
    VRSHL.S32  q9,  q9, q4
    VRSHL.S32 q10, q10, q4
    VRSHL.S32 q11, q11, q4
    VRSHL.S32 q12, q12, q4
    VRSHL.S32 q13, q13, q4
    VRSHL.S32 q14, q14, q4
    VRSHL.S32 q15, q15, q4

    # Load c, c_stride:
    # - r2 = c
    # - r3 = c_stride
    LDRD r2, r3, [sp, 104]

    VQMOVN.S32 d16,  q8
    VQMOVN.S32 d17,  q9
    VQMOVN.S32 d18, q10
    VQMOVN.S32 d19, q11
    VQMOVN.S32 d20, q12
    VQMOVN.S32 d21, q13
    VQMOVN.S32 d22, q14
    VQMOVN.S32 d23, q15

    # Load min:
    # - q4 = q8:q9 = voutput_min
    VLD1.8 {d8[], d9[]}, [r9]!
    ADD r4, r2, r3

    VQADD.S16  q8,  q8, q7
    VQADD.S16  q9,  q9, q7
    CMP r0, 2
    VQADD.S16 q10, q10, q7
    VQADD.S16 q11, q11, q7
    MOVLO r4, r2

    VQMOVUN.S16 d16,  q8
    VQMOVUN.S16 d17,  q9
    ADD r5, r4, r3
    VQMOVUN.S16 d18, q10
    VQMOVUN.S16 d19, q11
    MOVLS r5, r4

    VMIN.U8 q8, q8, q5
    CMP r0, 4
    VMIN.U8 q9, q9, q5
    ADD r3, r5, r3

    VMAX.U8 q8, q8, q4
    MOVNE r3, r5
    CMP r1, 8
    VMAX.U8 q9, q9, q4

    BNE 5f

    VST1.8 {d16}, [r2]
    VST1.8 {d17}, [r4]
    VST1.8 {d18}, [r5]
    VST1.8 {d19}, [r3]

    VPOP {d8-d15}
    POP {r4, r5, r6, r7, r8, r9, r10, r11}
    BX lr

    .p2align 3
5:
    CMP r1, 4
    BLO 6f

    VST1.32 {d16[0]}, [r2]!
    VST1.32 {d17[0]}, [r4]!
    VST1.32 {d18[0]}, [r5]!
    VST1.32 {d19[0]}, [r3]!

    SUB r1, 4
    VEXT.8 q8, q8, q8, 4
    VEXT.8 q9, q9, q9, 4

6:
    CMP r1, 2
    BLO 7f

    VST1.16 {d16[0]}, [r2]!
    VST1.16 {d17[0]}, [r4]!
    VST1.16 {d18[0]}, [r5]!
    VST1.16 {d19[0]}, [r3]!

    SUB r1, 2
    VEXT.8 q8, q8, q8, 2
    VEXT.8 q9, q9, q9, 2

7:
    TEQ r1, 0
    BEQ 8f

    VST1.8 {d16[0]}, [r2]
    VST1.8 {d17[0]}, [r4]
    VST1.8 {d18[0]}, [r5]
    VST1.8 {d19[0]}, [r3]

8:
    VPOP {d8-d15}
    POP {r4, r5, r6, r7, r8, r9, r10, r11}
    BX lr
END_FUNCTION pytorch_q8conv_ukernel_4x8__aarch32_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
