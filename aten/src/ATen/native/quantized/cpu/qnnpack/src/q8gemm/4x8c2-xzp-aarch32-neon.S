/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>

.syntax unified

# void pytorch_q8gemm_xzp_ukernel_4x8c2__neon(
#     size_t mr,
#     size_t nr,
#     size_t k,
#     const uint8_t* restrict a,
#     size_t a_stride,
#     const int32_t* restrict a_sum,
#     const void* restrict w,
#     uint8_t* restrict c,
#     size_t c_stride,
#     const union pytorch_qnnp_q31_requantization_params requantization_params[restrict static 1])
BEGIN_FUNCTION pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon
    .arm
#ifndef __APPLE__
    .arch armv7-a
    .fpu neon
#endif

    # Load w
    # - ip = w
    LDR ip, [sp, 8]

    # Load bias0123(q8), bias4567(q9)
    # q8 := vacc0x0123
    # q9 := vacc0x4567
    VLD1.8 {d16-d19}, [ip]!

    # q10 := vacc1x0123
    VMOV.I32 q10, q8
    # q11 := vacc1x4567
    VMOV.I32 q11, q9
    # q12 := vacc2x0123
    VMOV.I32 q12, q8
    # q13 := vacc2x4567
    VMOV.I32 q13, q9
    # q14 := vacc3x0123
    VMOV.I32 q14, q8
    # q15 := vacc3x4567
    VMOV.I32 q15, q9

    PUSH {r4, r5, r6, r7, r8, r9, r10, r11}
    VPUSH {d8-d15}

    # r3 := a0
    # r4 := a1
    # r5 := a2
    # r6 := a3

    # r7 := a_sum0
    # r8 := a_sum1
    # r9 := a_sum2
    # r10 := a_sum3

    # a_sum0 := a_sum
    LDR r7, [sp, 100]

    # Load a_stride
    # - ip = a_stride
    LDR r10, [sp, 96]

    # compare mr to 2
    CMP r0, 2

    # a1 += a_stride
    ADD r4, r3, r10

    # mr < 2, a1 := a0
    MOVLO r4, r3

    # r8 := a_sum1
    ADD r8, r7, 4

    # mr < 2, a_sum1 := a_sum0
    MOVLO r8, r7

    # r5 := a2
    ADD r5, r4, r10
    # mr <= 2, a2 := a1
    MOVLS r5, r4

    # r9 := a_sum2
    ADD r9, r8, 4
    # mr <= 2, a_sum2 := a_sum1
    MOVLS r9, r8

    # compare mr to 4
    CMP r0, 4

    # r6 := a3
    ADD r6, r5, r10
    # mr != 4, a3 := a2
    MOVNE r6, r5

    # a_sum3 := a_sum2 + 1
    # r10 := a_sum3
    ADD r10, r9, 4
    # mr != 4, a_sum3 := a_sum2
    MOVNE r10, r9

    # load a_sum
    # q0: va_sum0
    VLD1.32 {d0[], d1[]}, [r7]
    # q1: va_sum1
    VLD1.32 {d2[], d3[]}, [r8]
    # q2: va_sum2
    VLD1.32 {d4[], d5[]}, [r9]
    # q3: va_sum3
    VLD1.32 {d6[], d7[]}, [r10]

    # accumulate a_sum into vacc
    # vacc0x0123 = vaddq_s32(vacc0x0123, va_sum0)
    VADD.I32 q8, q8, q0
    # vacc0x4567 = vaddq_s32(vacc0x4567, va_sum0)
    VADD.I32 q9, q9, q0
    # vacc1x0123 = vaddq_s32(vacc1x0123, va_sum1)
    VADD.I32 q10, q10, q1
    # vacc1x4567 = vaddq_s32(vacc1x4567, va_sum1)
    VADD.I32 q11, q11, q1
    # vacc2x0123 = vaddq_s32(vacc2x0123, va_sum2)
    VADD.I32 q12, q12, q2
    # vacc2x4567 = vaddq_s32(vacc2x4567, va_sum2)
    VADD.I32 q13, q13, q2
    # vacc3x0123 = vaddq_s32(vacc3x0123, va_sum3)
    VADD.I32 q14, q14, q3
    # vacc3x4567 = vaddq_s32(vacc3x4567, va_sum3)
    VADD.I32 q15, q15, q3

    # k -= 8
    SUBS r2, r2, 8

    BLO 1f

.p2align 5
0:
    # load a
    # d0 := va0x01234567
    VLD1.8 {d0}, [r3]!

    # d1 := va1x01234567
    VLD1.8 {d1}, [r4]!

    # d2 := va1x01234567
    VLD1.8 {d2}, [r5]!

    # d3 := va2x01234567
    VLD1.8 {d3}, [r6]!

    ##### k = 0, 1 #####
    # load b
    # q2 := vb01234567x01
    VLD1.8 {d4, d5}, [ip]!

    VMULL.U8 q4, d0, d4
    VPADAL.U16 q8, q4

    VMULL.U8 q5, d0, d5
    VPADAL.U16 q9, q5

    VMULL.U8 q6, d1, d4
    VPADAL.U16 q10, q6

    VMULL.U8 q7, d1, d5
    VPADAL.U16 q11, q7

    VMULL.U8 q4, d2, d4
    VPADAL.U16 q12, q4

    VMULL.U8 q5, d2, d5
    VPADAL.U16 q13, q5

    VMULL.U8 q6, d3, d4
    VPADAL.U16 q14, q6

    VMULL.U8 q7, d3, d5
    VPADAL.U16 q15, q7

    ##### k = 2, 3 #####
    # load b
    # q2 := vb01234567x01
    VLD1.8 {d4, d5}, [ip]!

    # rotate a
    VEXT.8 d0, d0, d0, 2
    VEXT.8 d1, d1, d1, 2
    VEXT.8 d2, d2, d2, 2
    VEXT.8 d3, d3, d3, 2

    VMULL.U8 q4, d0, d4
    VPADAL.U16 q8, q4

    VMULL.U8 q5, d0, d5
    VPADAL.U16 q9, q5

    VMULL.U8 q6, d1, d4
    VPADAL.U16 q10, q6

    VMULL.U8 q7, d1, d5
    VPADAL.U16 q11, q7

    VMULL.U8 q4, d2, d4
    VPADAL.U16 q12, q4

    VMULL.U8 q5, d2, d5
    VPADAL.U16 q13, q5

    VMULL.U8 q6, d3, d4
    VPADAL.U16 q14, q6

    VMULL.U8 q7, d3, d5
    VPADAL.U16 q15, q7

    ##### k = 4, 5 #####
    # load b
    # q2 := vb01234567x01
    VLD1.8 {d4, d5}, [ip]!

    # rotate a
    VEXT.8 d0, d0, d0, 2
    VEXT.8 d1, d1, d1, 2
    VEXT.8 d2, d2, d2, 2
    VEXT.8 d3, d3, d3, 2

    VMULL.U8 q4, d0, d4
    VPADAL.U16 q8, q4

    VMULL.U8 q5, d0, d5
    VPADAL.U16 q9, q5

    VMULL.U8 q6, d1, d4
    VPADAL.U16 q10, q6

    VMULL.U8 q7, d1, d5
    VPADAL.U16 q11, q7

    VMULL.U8 q4, d2, d4
    VPADAL.U16 q12, q4

    VMULL.U8 q5, d2, d5
    VPADAL.U16 q13, q5

    VMULL.U8 q6, d3, d4
    VPADAL.U16 q14, q6

    VMULL.U8 q7, d3, d5
    VPADAL.U16 q15, q7

    ##### k = 6, 7 #####
    # load b
    # q2 := vb01234567x01
    VLD1.8 {d4, d5}, [ip]!

    # rotate a
    VEXT.8 d0, d0, d0, 2
    VEXT.8 d1, d1, d1, 2
    VEXT.8 d2, d2, d2, 2
    VEXT.8 d3, d3, d3, 2

    VMULL.U8 q4, d0, d4
    VPADAL.U16 q8, q4

    VMULL.U8 q5, d0, d5
    VPADAL.U16 q9, q5

    VMULL.U8 q6, d1, d4
    VPADAL.U16 q10, q6

    VMULL.U8 q7, d1, d5
    VPADAL.U16 q11, q7

    VMULL.U8 q4, d2, d4
    VPADAL.U16 q12, q4

    VMULL.U8 q5, d2, d5
    VPADAL.U16 q13, q5

    VMULL.U8 q6, d3, d4
    VPADAL.U16 q14, q6

    VMULL.U8 q7, d3, d5
    VPADAL.U16 q15, q7

    # k -= 8
    SUBS r2, r2, 8

    # k >= 0, loop
    BHS 0b

1:
    # k >= 4
    ADDS r2, 8
    CMP r2, 4

    # branch to 2f when k < 4
    BLO 2f
    SUB r2, r2, 4

    ##### k = 0, 1 #####
    # d0 := va0x01010101
    VLD1.16 {d0[]}, [r3]!
    # d1 := va1x01010101
    VLD1.16 {d1[]}, [r4]!
    # d2 := va2x01010101
    VLD1.16 {d2[]}, [r5]!
    # d3 := va3x01010101
    VLD1.16 {d3[]}, [r6]!

    # q7 := vb01234567x01
    VLD1.8 {d14, d15}, [ip]!

    # row 0
    VMULL.U8 q2, d0, d14
    VPADAL.U16 q8, q2
    VMULL.U8 q3, d0, d15
    VPADAL.U16 q9, q3
    # row 1
    VMULL.U8 q4, d1, d14
    VPADAL.U16 q10, q4
    VMULL.U8 q5, d1, d15
    VPADAL.U16 q11, q5
    # row 2
    VMULL.U8 q2, d2, d14
    VPADAL.U16 q12, q2
    VMULL.U8 q3, d2, d15
    VPADAL.U16 q13, q3
    # row 3
    VMULL.U8 q4, d3, d14
    VPADAL.U16 q14, q4
    VMULL.U8 q5, d3, d15
    VPADAL.U16 q15, q5

    ##### k = 2, 3 #####
    # d0 := va0x01010101
    VLD1.16 {d0[]}, [r3]!
    # d1 := va1x01010101
    VLD1.16 {d1[]}, [r4]!
    # d2 := va2x01010101
    VLD1.16 {d2[]}, [r5]!
    # d3 := va3x01010101
    VLD1.16 {d3[]}, [r6]!

    # q7 := vb01234567x01
    VLD1.8 {d14, d15}, [ip]!

    # row 0
    VMULL.U8 q2, d0, d14
    VPADAL.U16 q8, q2
    VMULL.U8 q3, d0, d15
    VPADAL.U16 q9, q3
    # row 1
    VMULL.U8 q4, d1, d14
    VPADAL.U16 q10, q4
    VMULL.U8 q5, d1, d15
    VPADAL.U16 q11, q5
    # row 2
    VMULL.U8 q2, d2, d14
    VPADAL.U16 q12, q2
    VMULL.U8 q3, d2, d15
    VPADAL.U16 q13, q3
    # row 3
    VMULL.U8 q4, d3, d14
    VPADAL.U16 q14, q4
    VMULL.U8 q5, d3, d15
    VPADAL.U16 q15, q5

2:
    # k >= 2
    CMP r2, 2
    BLO 3f
    SUB r2, r2, 2

    ##### k = 0, 1 #####
    # d0 := va0x01010101
    VLD1.16 {d0[]}, [r3]!
    # d1 := va1x01010101
    VLD1.16 {d1[]}, [r4]!
    # d2 := va2x01010101
    VLD1.16 {d2[]}, [r5]!
    # d3 := va3x01010101
    VLD1.16 {d3[]}, [r6]!

    # q7 := vb01234567x01
    VLD1.8 {d14, d15}, [ip]!

    # row 0
    VMULL.U8 q2, d0, d14
    VPADAL.U16 q8, q2
    VMULL.U8 q3, d0, d15
    VPADAL.U16 q9, q3
    # row 1
    VMULL.U8 q4, d1, d14
    VPADAL.U16 q10, q4
    VMULL.U8 q5, d1, d15
    VPADAL.U16 q11, q5
    # row 2
    VMULL.U8 q2, d2, d14
    VPADAL.U16 q12, q2
    VMULL.U8 q3, d2, d15
    VPADAL.U16 q13, q3
    # row 3
    VMULL.U8 q4, d3, d14
    VPADAL.U16 q14, q4
    VMULL.U8 q5, d3, d15
    VPADAL.U16 q15, q5

3:
    # k == 1
    CMP r2, 1
    BLO 4f

    # d0 := va0x01010101
    VLD1.8 {d0[]}, [r3]
    # d1 := va1x01010101
    VLD1.8 {d1[]}, [r4]
    # d2 := va2x01010101
    VLD1.8 {d2[]}, [r5]
    # d3 := va3x01010101
    VLD1.8 {d3[]}, [r6]

    # q7 := vb01234567x01
    VLD1.8 {d14, d15}, [ip]

    # row 0
    VMULL.U8 q2, d0, d14
    VPADAL.U16 q8, q2
    VMULL.U8 q3, d0, d15
    VPADAL.U16 q9, q3
    # row 1
    VMULL.U8 q4, d1, d14
    VPADAL.U16 q10, q4
    VMULL.U8 q5, d1, d15
    VPADAL.U16 q11, q5
    # row 2
    VMULL.U8 q2, d2, d14
    VPADAL.U16 q12, q2
    VMULL.U8 q3, d2, d15
    VPADAL.U16 q13, q3
    # row 3
    VMULL.U8 q4, d3, d14
    VPADAL.U16 q14, q4
    VMULL.U8 q5, d3, d15
    VPADAL.U16 q15, q5

    .p2align 4
4:
    # Load params:
    # - ip = params
    LDR ip, [sp, 116]

    # Load multiplier:
    # - d12 = vmultiplier
    VLD1.32 {d12[]}, [ip]!

    # Load right_shift
    # - q4 = d8:d9 = vright_shift
    VLD1.32 {d8[], d9[]}, [ip]!

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

    # Load zero_point
    # - q7 = d14:d15 = vzero_point
    VLD1.16 {d14[], d15[]}, [ip]!

    VBIC q0, q12, q5
    VBIC q1, q13, q5
    VBIC q2, q14, q5
    VBIC q3, q15, q5

    VSRA.S32 q12, q0, 31
    VSRA.S32 q13, q1, 31
    VSRA.S32 q14, q2, 31
    VSRA.S32 q15, q3, 31

    # Load max:
    # - q5 = d10:d11 = vmax
    VLD1.8 {d10[], d11[]}, [ip]!

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
    LDRD r2, r3, [sp, 108]

    VQMOVN.S32 d16,  q8
    VQMOVN.S32 d17,  q9
    VQMOVN.S32 d18, q10
    VQMOVN.S32 d19, q11
    VQMOVN.S32 d20, q12
    VQMOVN.S32 d21, q13
    VQMOVN.S32 d22, q14
    VQMOVN.S32 d23, q15

    # Load min:
    # - q4 = q8:q9 = vmin
    VLD1.8 {d8[], d9[]}, [ip]!
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

END_FUNCTION pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
