/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>

.syntax unified

# void pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith(
#     size_t mr,
#     size_t nr,
#     size_t k,
#     const __fp16*restrict a,
#     size_t a_stride,
#     const __fp16*restrict w,
#     __fp16*restrict c,
#     size_t c_stride,
#     const struct pytorch_qnnp_fp16_clamping_params clamping_params[restrict static 1])
BEGIN_FUNCTION pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith
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

    # Initialize vacc0x01234567
    # - q8 = d16:d17 := vacc0x01234567 = bias01234567
    VLD1.16 {d16-d17}, [ip:64]!

    # Load a_stride
    # - r10 = a_stride
    LDR r10, [sp, 96]

    # Initialize vacc1x01234567
    # - q9 := vacc1x01234567 = vacc0x01234567
    VMOV.I16 q9, q8

    # Initialize vacc2x01234567
    # - q10 := vacc2x01234567 = vacc0x01234567
    VMOV.I16 q10, q8

    # Initialize vacc3x01234567
    # - q11 := vacc3x01234567 = vacc0x01234567
    VMOV.I16 q11, q8

    # Initialize vacc4x01234567
    # - q12 := vacc4x01234567 = vacc0x01234567
    VMOV.I16 q12, q8

    # Initialize vacc5x01234567
    # - q13 := vacc5x01234567 = vacc0x01234567
    VMOV.I16 q13, q8

    # Initialize vacc6x01234567
    # - q14 := vacc6x01234567 = vacc0x01234567
    VMOV.I16 q14, q8

    # Initialize vacc7x01234567
    # - q15 := vacc7x01234567 = vacc0x01234567
    VMOV.I16 q15, q8

    CMP r0, 2
    ADD r4, r3, r10
    MOVLO r4, r3
    ADD r5, r4, r10
    MOVLS r5, r4

    CMP r0, 4
    ADD r6, r5, r10
    MOVLO r6, r5
    ADD r7, r6, r10
    MOVLS r7, r6

    CMP r0, 6
    ADD r8, r7, r10
    MOVLO r8, r7
    ADD r9, r8, r10
    MOVLS r9, r8

    CMP r0, 8
    ADD r10, r9, r10
    MOVNE r10, r9

    SUBS r2, r2, 4
    BLO 1f

    .p2align 5
0:
    # Load a0
    # - d0 = a0
    VLD1.16 {d0}, [r3]!

    # Load a1
    # - d1 = a1
    VLD1.16 {d1}, [r4]!

    # Load a2
    # - d2 = a2
    VLD1.16 {d2}, [r5]!

    # Load a3
    # - d3 = a3
    VLD1.16 {d3}, [r6]!

    # Load a4
    # - d4 = a4
    VLD1.16 {d4}, [r7]!

    # Load a5
    # - d5 = a5
    VLD1.16 {d5}, [r8]!

    # Load a6
    # - d6 = a6
    VLD1.16 {d6}, [r9]!

    # Load a7
    # - d7 = a7
    VLD1.16 {d7}, [r10]!

    ### Channel 0 ###

    # Load b0-b15 (channel 0)
    # - q4 = d8:d9 = b0-b15
    VLD1.8 {d8-d9}, [ip:64]!

    # vacc0x01234567 += vb01234567 * va0[0];
    .word 0xF3D80140 @ VMLA.F16  q8, q4, d0[0]
    # vacc1x01234567 += vb01234567 * va1[0];
    .word 0xF3D82141 @ VMLA.F16  q9, q4, d1[0]
    # vacc2x01234567 += vb01234567 * va2[0];
    .word 0xF3D84142 @ VMLA.F16 q10, q4, d2[0]
    # vacc3x01234567 += vb01234567 * va3[0];
    .word 0xF3D86143 @ VMLA.F16 q11, q4, d3[0]
    # vacc4x01234567 += vb01234567 * va4[0];
    .word 0xF3D88144 @ VMLA.F16 q12, q4, d4[0]
    # vacc5x01234567 += vb01234567 * va5[0];
    .word 0xF3D8A145 @ VMLA.F16 q13, q4, d5[0]
    # vacc6x01234567 += vb01234567 * va6[0];
    .word 0xF3D8C146 @ VMLA.F16 q14, q4, d6[0]
    # vacc7x01234567 += vb01234567 * va7[0];
    .word 0xF3D8E147 @ VMLA.F16 q15, q4, d7[0]

    ### Channel 1 ###

    # Load b0-b15 (channel 1)
    # - q5 = d10:d11 = b0-b15
    VLD1.8 {d10-d11}, [ip:64]!

    # vacc0x01234567 += vb01234567 * va0[1];
    .word 0xF3DA0148 @ VMLA.F16  q8, q5, d0[1]
    # vacc1x01234567 += vb01234567 * va1[1];
    .word 0xF3DA2149 @ VMLA.F16  q9, q5, d1[1]
    # vacc2x01234567 += vb01234567 * va2[1];
    .word 0xF3DA414A @ VMLA.F16 q10, q5, d2[1]
    # vacc3x01234567 += vb01234567 * va3[1];
    .word 0xF3DA614B @ VMLA.F16 q11, q5, d3[1]
    # vacc4x01234567 += vb01234567 * va4[1];
    .word 0xF3DA814C @ VMLA.F16 q12, q5, d4[1]
    # vacc5x01234567 += vb01234567 * va5[1];
    .word 0xF3DAA14D @ VMLA.F16 q13, q5, d5[1]
    # vacc6x01234567 += vb01234567 * va6[1];
    .word 0xF3DAC14E @ VMLA.F16 q14, q5, d6[1]
    # vacc7x01234567 += vb01234567 * va7[1];
    .word 0xF3DAE14F @ VMLA.F16 q15, q5, d7[1]

    ### Channel 2 ###

    # Load b0-b15 (channel 2)
    # - q6 = d12:d13 = b0-b15
    VLD1.8 {d12-d13}, [ip:64]!

    # vacc0x01234567 += vb01234567 * va0[2];
    .word 0xF3DC0160 @ VMLA.F16  q8, q6, d0[2]
    # vacc1x01234567 += vb01234567 * va1[2];
    .word 0xF3DC2161 @ VMLA.F16  q9, q6, d1[2]
    # vacc2x01234567 += vb01234567 * va2[2];
    .word 0xF3DC4162 @ VMLA.F16 q10, q6, d2[2]
    # vacc3x01234567 += vb01234567 * va3[2];
    .word 0xF3DC6163 @ VMLA.F16 q11, q6, d3[2]
    # vacc4x01234567 += vb01234567 * va4[2];
    .word 0xF3DC8164 @ VMLA.F16 q12, q6, d4[2]
    # vacc5x01234567 += vb01234567 * va5[2];
    .word 0xF3DCA165 @ VMLA.F16 q13, q6, d5[2]
    # vacc6x01234567 += vb01234567 * va6[2];
    .word 0xF3DCC166 @ VMLA.F16 q14, q6, d6[2]
    # vacc7x01234567 += vb01234567 * va7[2];
    .word 0xF3DCE167 @ VMLA.F16 q15, q6, d7[2]

    ### Channel 3 ###

    # Load b0-b15 (channel 3)
    # - q7 = d14:d15 = b0-b15
    VLD1.8 {d14-d15}, [ip:64]!

    # vacc0x01234567 += vb01234567 * va0[3];
    .word 0xF3DE0168 @ VMLA.F16  q8, q7, d0[3]
    # vacc1x01234567 += vb01234567 * va1[3];
    .word 0xF3DE2169 @ VMLA.F16  q9, q7, d1[3]
    # vacc2x01234567 += vb01234567 * va2[3];
    .word 0xF3DE416A @ VMLA.F16 q10, q7, d2[3]
    # vacc3x01234567 += vb01234567 * va3[3];
    .word 0xF3DE616B @ VMLA.F16 q11, q7, d3[3]
    # vacc4x01234567 += vb01234567 * va4[3];
    .word 0xF3DE816C @ VMLA.F16 q12, q7, d4[3]
    # vacc5x01234567 += vb01234567 * va5[3];
    .word 0xF3DEA16D @ VMLA.F16 q13, q7, d5[3]
    # vacc6x01234567 += vb01234567 * va6[3];
    .word 0xF3DEC16E @ VMLA.F16 q14, q7, d6[3]
    # vacc7x01234567 += vb01234567 * va7[3];
    .word 0xF3DEE16F @ VMLA.F16 q15, q7, d7[3]

    SUBS r2, r2, 4
    BHS 0b

1:
    CMP r2, -4
    BEQ 2f

    ADD r3, r3, r2, LSL #1
    ADD r4, r4, r2, LSL #1
    ADD r5, r5, r2, LSL #1
    ADD r6, r6, r2, LSL #1
    ADD r7, r7, r2, LSL #1
    ADD r8, r8, r2, LSL #1
    ADD r9, r9, r2, LSL #1
    ADD r10, r10, r2, LSL #1

    LSL r2, r2, 4
    VDUP.32 d14, r2

    # Load a0
    # - d0 = a0
    VLD1.16 {d0}, [r3]!
    VSHL.U64 d0, d0, d14

    # Load a1
    # - d1 = a1
    VLD1.16 {d1}, [r4]!
    VSHL.U64 d1, d1, d14

    # Load a2
    # - d2 = a2
    VLD1.16 {d2}, [r5]!
    VSHL.U64 d2, d2, d14

    # Load a3
    # - d3 = a3
    VLD1.16 {d3}, [r6]!
    VSHL.U64 d3, d3, d14

    # Load a4
    # - d4 = a4
    VLD1.16 {d4}, [r7]!
    VSHL.U64 d4, d4, d14

    # Load a5
    # - d5 = a5
    VLD1.16 {d5}, [r8]!
    VSHL.U64 d5, d5, d14

    # Load a6
    # - d6 = a6
    VLD1.16 {d6}, [r9]!
    VSHL.U64 d6, d6, d14

    # Load a7
    # - d7 = a7
    VLD1.16 {d7}, [r10]!
    VSHL.U64 d7, d7, d14

    ### Channel 0 ###

    # Load b0-b15 (channel 0)
    # - q4 = d8:d9 = b0-b15
    VLD1.8 {d8-d9}, [ip:64]!

    # vacc0x01234567 += vb01234567 * va0[0];
    .word 0xF3D80140 @ VMLA.F16  q8, q4, d0[0]
    # vacc1x01234567 += vb01234567 * va1[0];
    .word 0xF3D82141 @ VMLA.F16  q9, q4, d1[0]
    # vacc2x01234567 += vb01234567 * va2[0];
    .word 0xF3D84142 @ VMLA.F16 q10, q4, d2[0]
    # vacc3x01234567 += vb01234567 * va3[0];
    .word 0xF3D86143 @ VMLA.F16 q11, q4, d3[0]
    # vacc4x01234567 += vb01234567 * va4[0];
    .word 0xF3D88144 @ VMLA.F16 q12, q4, d4[0]
    # vacc5x01234567 += vb01234567 * va5[0];
    .word 0xF3D8A145 @ VMLA.F16 q13, q4, d5[0]
    # vacc6x01234567 += vb01234567 * va6[0];
    .word 0xF3D8C146 @ VMLA.F16 q14, q4, d6[0]
    # vacc7x01234567 += vb01234567 * va7[0];
    .word 0xF3D8E147 @ VMLA.F16 q15, q4, d7[0]

    CMP r2, -32
    BLO 2f

    ### Channel 1 ###

    # Load b0-b15 (channel 1)
    # - q5 = d10:d11 = b0-b15
    VLD1.8 {d10-d11}, [ip:64]!

    # vacc0x01234567 += vb01234567 * va0[1];
    .word 0xF3DA0148 @ VMLA.F16  q8, q5, d0[1]
    # vacc1x01234567 += vb01234567 * va1[1];
    .word 0xF3DA2149 @ VMLA.F16  q9, q5, d1[1]
    # vacc2x01234567 += vb01234567 * va2[1];
    .word 0xF3DA414A @ VMLA.F16 q10, q5, d2[1]
    # vacc3x01234567 += vb01234567 * va3[1];
    .word 0xF3DA614B @ VMLA.F16 q11, q5, d3[1]
    # vacc4x01234567 += vb01234567 * va4[1];
    .word 0xF3DA814C @ VMLA.F16 q12, q5, d4[1]
    # vacc5x01234567 += vb01234567 * va5[1];
    .word 0xF3DAA14D @ VMLA.F16 q13, q5, d5[1]
    # vacc6x01234567 += vb01234567 * va6[1];
    .word 0xF3DAC14E @ VMLA.F16 q14, q5, d6[1]
    # vacc7x01234567 += vb01234567 * va7[1];
    .word 0xF3DAE14F @ VMLA.F16 q15, q5, d7[1]

    BLS 2f

    ### Channel 2 ###

    # Load b0-b15 (channel 2)
    # - q6 = d12:d13 = b0-b15
    VLD1.8 {d12-d13}, [ip:64]!

    # vacc0x01234567 += vb01234567 * va0[2];
    .word 0xF3DC0160 @ VMLA.F16  q8, q6, d0[2]
    # vacc1x01234567 += vb01234567 * va1[2];
    .word 0xF3DC2161 @ VMLA.F16  q9, q6, d1[2]
    # vacc2x01234567 += vb01234567 * va2[2];
    .word 0xF3DC4162 @ VMLA.F16 q10, q6, d2[2]
    # vacc3x01234567 += vb01234567 * va3[2];
    .word 0xF3DC6163 @ VMLA.F16 q11, q6, d3[2]
    # vacc4x01234567 += vb01234567 * va4[2];
    .word 0xF3DC8164 @ VMLA.F16 q12, q6, d4[2]
    # vacc5x01234567 += vb01234567 * va5[2];
    .word 0xF3DCA165 @ VMLA.F16 q13, q6, d5[2]
    # vacc6x01234567 += vb01234567 * va6[2];
    .word 0xF3DCC166 @ VMLA.F16 q14, q6, d6[2]
    # vacc7x01234567 += vb01234567 * va7[2];
    .word 0xF3DCE167 @ VMLA.F16 q15, q6, d7[2]

    .p2align 4
2:
    # Load params:
    # - ip = params
    LDR ip, [sp, 112]

    # Load scale:
    # - q0 = d0:d1 = vscale
    VLD1.16 {d0[], d1[]}, [ip]!

    .word 0xF3500DD0 @ VMUL.F16  q8,  q8, q0
    .word 0xF3522DD0 @ VMUL.F16  q9,  q9, q0
    .word 0xF3544DD0 @ VMUL.F16 q10, q10, q0
    .word 0xF3566DD0 @ VMUL.F16 q11, q11, q0
    .word 0xF3588DD0 @ VMUL.F16 q12, q12, q0
    .word 0xF35AADD0 @ VMUL.F16 q13, q13, q0
    .word 0xF35CCDD0 @ VMUL.F16 q14, q14, q0
    .word 0xF35EEDD0 @ VMUL.F16 q15, q15, q0

    # Load max:
    # - q1 = d2:d3 = vmax
    VLD1.16 {d2[], d3[]}, [ip]!

    .word 0xF2700FC2 @ VMIN.F16  q8,  q8, q1
    .word 0xF2722FC2 @ VMIN.F16  q9,  q9, q1
    .word 0xF2744FC2 @ VMIN.F16 q10, q10, q1
    .word 0xF2766FC2 @ VMIN.F16 q11, q11, q1
    .word 0xF2788FC2 @ VMIN.F16 q12, q12, q1
    .word 0xF27AAFC2 @ VMIN.F16 q13, q13, q1
    .word 0xF27CCFC2 @ VMIN.F16 q14, q14, q1
    .word 0xF27EEFC2 @ VMIN.F16 q15, q15, q1

    # Load min:
    # - q2 = d4:d5 = vmin
    VLD1.16 {d4[], d5[]}, [ip]

    .word 0xF2500FC4 @ VMAX.F16  q8,  q8, q2
    .word 0xF2522FC4 @ VMAX.F16  q9,  q9, q2
    .word 0xF2544FC4 @ VMAX.F16 q10, q10, q2
    .word 0xF2566FC4 @ VMAX.F16 q11, q11, q2
    .word 0xF2588FC4 @ VMAX.F16 q12, q12, q2
    .word 0xF25AAFC4 @ VMAX.F16 q13, q13, q2
    .word 0xF25CCFC4 @ VMAX.F16 q14, q14, q2
    .word 0xF25EEFC4 @ VMAX.F16 q15, q15, q2

    # Load c, c_stride:
    # - r2 = c
    # - r3 = c_stride
    LDRD r2, r3, [sp, 104]

    CMP r0, 2
    ADD r4, r2, r3
    MOVLO r4, r2
    ADD r5, r4, r3
    MOVLS r5, r4

    CMP r0, 4
    ADD r6, r5, r3
    MOVLO r6, r5
    ADD r7, r6, r3
    MOVLS r7, r6

    CMP r0, 6
    ADD r8, r7, r3
    MOVLO r8, r7
    ADD r9, r8, r3
    MOVLS r9, r8

    CMP r0, 8
    ADD r3, r9, r3
    MOVNE r3, r9

    CMP r1, 8
    BNE 4f

    VST1.16 {d16-d17}, [r2]
    VST1.16 {d18-d19}, [r4]
    VST1.16 {d20-d21}, [r5]
    VST1.16 {d22-d23}, [r6]
    VST1.16 {d24-d25}, [r7]
    VST1.16 {d26-d27}, [r8]
    VST1.16 {d28-d29}, [r9]
    VST1.16 {d30-d31}, [r3]

    VPOP {d8-d15}
    POP {r4, r5, r6, r7, r8, r9, r10, r11}
    BX lr

    .p2align 3
4:
    CMP r1, 4
    BLO 5f

    VST1.16 {d16}, [r2]!
    VST1.16 {d18}, [r4]!
    VST1.16 {d20}, [r5]!
    VST1.16 {d22}, [r6]!
    VST1.16 {d24}, [r7]!
    VST1.16 {d26}, [r8]!
    VST1.16 {d28}, [r9]!
    VST1.16 {d30}, [r3]!

    SUB r1, 4
    VMOV.I16 d16, d17
    VMOV.I16 d18, d19
    VMOV.I16 d20, d21
    VMOV.I16 d22, d23
    VMOV.I16 d24, d25
    VMOV.I16 d26, d27
    VMOV.I16 d28, d29
    VMOV.I16 d30, d31

5:
    CMP r1, 2
    BLO 6f

    VST1.32 {d16[0]}, [r2]!
    VST1.32 {d18[0]}, [r4]!
    VST1.32 {d20[0]}, [r5]!
    VST1.32 {d22[0]}, [r6]!
    VST1.32 {d24[0]}, [r7]!
    VST1.32 {d26[0]}, [r8]!
    VST1.32 {d28[0]}, [r9]!
    VST1.32 {d30[0]}, [r3]!

    SUB r1, 2
    VEXT.8 d16, d16, d16, 4
    VEXT.8 d18, d18, d18, 4
    VEXT.8 d20, d20, d20, 4
    VEXT.8 d22, d22, d22, 4
    VEXT.8 d24, d24, d24, 4
    VEXT.8 d26, d26, d26, 4
    VEXT.8 d28, d28, d28, 4
    VEXT.8 d30, d30, d30, 4

6:
    TEQ r1, 0
    BEQ 7f

    VST1.16 {d16[0]}, [r2]
    VST1.16 {d18[0]}, [r4]
    VST1.16 {d20[0]}, [r5]
    VST1.16 {d22[0]}, [r6]
    VST1.16 {d24[0]}, [r7]
    VST1.16 {d26[0]}, [r8]
    VST1.16 {d28[0]}, [r9]
    VST1.16 {d30[0]}, [r3]

7:
    VPOP {d8-d15}
    POP {r4, r5, r6, r7, r8, r9, r10, r11}
    BX lr
END_FUNCTION pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
