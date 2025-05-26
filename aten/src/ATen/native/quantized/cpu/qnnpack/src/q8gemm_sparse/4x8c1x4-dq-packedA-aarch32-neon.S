/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>
#include <requantization/runtime-assembly.h>

#ifndef __APPLE__
#define NDEF_APPLE_SYMBOLS .arch armv7-a; .fpu neon
#else
#define NDEF_APPLE_SYMBOLS
#endif

# r0 mr
# r1 nr
# r2 packed_a
# r3 packed_w

# d14 a_zero_point
# d15 b_zero_point

## Stack
# 4     a_stride
# 4     packed_w
# 4     w_row_ptr
# 4     w_block_ids_ptr
# 4     b
# 4     c
# 4     c_stride
# 4     output channel index
# 4     quantization_params
# --

.syntax unified

#  Args passed via stack.
#  TOS
#  |----------------|
#  |packed_w        | 0
#  |w_row_ptr       | 4
#  |w_block_ids_ptr | 8
#  |b               | 12
#  |c               | 16
#  |c_stride        | 20
#  |out ch indx     | 24
#  |params          | 28
#  |----------------|
#

#  After loading w pointer in ip reg.
#  And after pushing r4-r9 and d8-d15 on stack
#  |----------------|
#  |d8 - d15        | 0
#  |r4 - r11,lr     | 64
#  |w_row_ptr       | 100
#  |w_block_ids_ptr | 104
#  |b               | 108
#  |c               | 112
#  |c_stride        | 116
#  |out ch indx     | 120
#  |params          | 124
#  |----------------|
#

# void pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch32_neon(
#     size_t mr,
#     size_t nr,
#     const uint8_t* a_packed,
#     const uint8_t* packed_w,
#     const uint##W_INDEX_DTYPE_NUM_BITS##_t* w_row_ptr,
#     const uint##W_INDEX_DTYPE_NUM_BITS##_t* w_block_ids_ptr,
#     const float* b,
#     uint8_t* restrict c,
#     size_t c_stride,
#     size_t output_channel_index,
#     const union pytorch_qnnp_conv_dynamic_quantization_params quantization_params[restrict static 1])
#define MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_1X4_UKERNEL_4X8_PACKEDA__AARCH32_NEON(W_INDEX_DTYPE_NUM_BITS, W_INDEX_DTYPE_NUM_BYTES_ARG, W_INDEX_DTYPE_LOG_NUM_BYTES_ARG, LOAD_INDEX_INSTRUCTION) ;\
    BEGIN_FUNCTION pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch32_neon ;\
        .arm                                                                  ;\
        NDEF_APPLE_SYMBOLS                                                    ;\
                                                                              ;\
        PUSH {r4, r5, r6, r7, r8, r9, r10, r11, lr}                           ;\
        VPUSH {d8-d15}                                                        ;\
                                                                              ;\
        /* Store nr in r11 as well for late user. */                          ;\
        MOV r11, r1                                                           ;\
        /* Load output channel index */                                       ;\
        LDR r5, [sp, 120]                                                     ;\
        /* Load quantization params */                                        ;\
        /* - r7 = quantization_params */                                      ;\
        LDR r7, [sp, 124]                                                     ;\
        /* Load input_zero_point */                                           ;\
        VLD1.8 {d16[]}, [r7]                                                  ;\
        ADD r7, r7, 4                                                         ;\
        /* Load pointer to per channel zero points array */                   ;\
        LDR r4, [r7]                                                          ;\
        /* Add output_channel_index to the b_zero_point pointer */            ;\
        ADD r4, r4, r5                                                        ;\
                                                                              ;\
        /* We enter the loop if r1 is atleast 1. */                           ;\
        /* r1 = r1 - 1 will happen in the epilogue */                         ;\
        /* of the loop */                                                     ;\
        CMP r1, 1                                                             ;\
        BLO _7_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
                                                                              ;\
        /* Load w_row_ptr + n */                                              ;\
        LDR r5, [sp, 100]                                                     ;\
        /* r7 = blocks_id_ptr */                                              ;\
        LDR r7, [sp, 104]                                                     ;\
                                                                              ;\
        .p2align 5                                                            ;\
    _0_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        VEOR q10, q10, q10                                                    ;\
        VLD1.8 {d17[]}, [r4]!                                                 ;\
        /* ip = w_row_ptr[n], lr = w_row_ptr[n+1] */                          ;\
        /* r5 = r5 + W_INDEX_DTYPE_NUM_BYTES_ARG to point to next n */        ;\
        LOAD_INDEX_INSTRUCTION ip, [r5], W_INDEX_DTYPE_NUM_BYTES_ARG          ;\
        LOAD_INDEX_INSTRUCTION lr, [r5]                                       ;\
        /* r6 = temp_packed_w = packed_w + w_row_ptr[n] * 4 */                ;\
        /* This points to the first block of nonzero value */                 ;\
        /* for the nth row. */                                                ;\
        ADD r6, r3, ip, LSL #2                                                ;\
        /* r9 = temp_w_block_ids_ptr = w_block_ids_ptr (r7) + w_row_ptr[n] */ ;\
        /* LSL for when elements are >1 byte */                               ;\
        /* (4 bytes: LSL #2, 2 bytes: LSL #1, 1 byte: LSL #0) */              ;\
        /* This points to the block id of the first block */                  ;\
        /* It should contain lr - ip number of block ids */                   ;\
        ADD r9, r7, ip, LSL W_INDEX_DTYPE_LOG_NUM_BYTES_ARG                   ;\
        /* r8 = num_blocks that needs to be processed */                      ;\
        SUB r8, lr, ip                                                        ;\
        SUBS r8, r8, 2                                                        ;\
        BLO _1_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
                                                                              ;\
    k_loop_w##W_INDEX_DTYPE_NUM_BITS##:                                       ;\
        /* Load 2 non zero blocks of weights. Each block = 1x4. */            ;\
        VLD1.8 {d0}, [r6]!                                                    ;\
                                                                              ;\
        /* ip = block_id_ptr[0] */                                            ;\
        /* lr = block_id_ptr[1] */                                            ;\
        LOAD_INDEX_INSTRUCTION ip, [r9], W_INDEX_DTYPE_NUM_BYTES_ARG          ;\
        LOAD_INDEX_INSTRUCTION lr, [r9], W_INDEX_DTYPE_NUM_BYTES_ARG          ;\
                                                                              ;\
        /* Add offset to r2 */                                                ;\
        /* Shift by 4 because each packed block is a block of 4x4 */          ;\
        /* which 16 bytes */                                                  ;\
        ADD r10, r2, ip, LSL #4                                               ;\
        /* q9 = vxb */                                                        ;\
        VSUBL.U8 q0, d0, d17                                                  ;\
                                                                              ;\
        /* d2, d3 = 4x4 transposed */                                         ;\
        VLD1.8 {d2}, [r10]!                                                   ;\
        VLD1.8 {d3}, [r10]                                                    ;\
                                                                              ;\
        ADD r10, r2, lr, LSL #4                                               ;\
                                                                              ;\
        VSUBL.U8 q4, d2, d16  /* vxa0_t */                                    ;\
                                                                              ;\
        /* d4, d5 = next 4x4 transposed */                                    ;\
        VLD1.8 {d4}, [r10]!                                                   ;\
        VLD1.8 {d5}, [r10]                                                    ;\
                                                                              ;\
        VSUBL.U8 q5, d3, d16  /* vxa1_t */                                    ;\
        VSUBL.U8 q6, d4, d16  /* vxa4_t */                                    ;\
        VSUBL.U8 q7, d5, d16  /* vxa5_t */                                    ;\
                                                                              ;\
        /* q4, q5 = 4x4 block (16 values each of 16 bits) */                  ;\
        /* q6, q7 = 4x4 block (16 values each of 16 bits) */                  ;\
                                                                              ;\
        VMLAL.S16 q10, d8, d0[0]                                              ;\
        VMLAL.S16 q10, d9, d0[1]                                              ;\
        VMLAL.S16 q10, d10, d0[2]                                             ;\
        VMLAL.S16 q10, d11, d0[3]                                             ;\
        VMLAL.S16 q10, d12, d1[0]                                             ;\
        VMLAL.S16 q10, d13, d1[1]                                             ;\
        VMLAL.S16 q10, d14, d1[2]                                             ;\
        VMLAL.S16 q10, d15, d1[3]                                             ;\
                                                                              ;\
        SUBS r8, r8, 2                                                        ;\
                                                                              ;\
        BHS k_loop_w##W_INDEX_DTYPE_NUM_BITS                                  ;\
    _1_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        CMP r8, -2                                                            ;\
        BEQ _2_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
                                                                              ;\
        /* Load last nonzero block */                                         ;\
        /* For this we will load 4 8 bit values as one 32 bit value */        ;\
        VLD1.32 {d0[]}, [r6]!                                                 ;\
        /* q9 = vxb */                                                        ;\
        VSUBL.U8 q0, d0, d17                                                  ;\
                                                                              ;\
        /* ip = block_id_ptr[0] */                                            ;\
        LOAD_INDEX_INSTRUCTION ip, [r9]                                       ;\
                                                                              ;\
        /* Add offset to r2 */                                                ;\
        /* Shift by 4 because each packed block is a block of 4x4 */          ;\
        /* which 16 bytes */                                                  ;\
        ADD r10, r2, ip, LSL #4                                               ;\
                                                                              ;\
        VLD1.8 {d2}, [r10]!                                                   ;\
        VLD1.8 {d3}, [r10]                                                    ;\
                                                                              ;\
        VSUBL.U8 q4, d2, d16  /* vxa0_t */                                    ;\
        VSUBL.U8 q5, d3, d16  /* vxa1_t */                                    ;\
                                                                              ;\
        VMLAL.S16 q10, d8, d0[0]                                              ;\
        VMLAL.S16 q10, d9, d0[1]                                              ;\
        VMLAL.S16 q10, d10, d0[2]                                             ;\
        VMLAL.S16 q10, d11, d0[3]                                             ;\
                                                                              ;\
        .p2align 4                                                            ;\
    _2_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        /* Store result on stack */                                           ;\
                                                                              ;\
        /* -12 because TOS - 4, TOS - 8, and TOS - 12, store mr, nr and pointer to weight zp */ ;\
        /* + 128 bytes of buffer when nr = 1 */                               ;\
        /* This is needed because after processing all nrs we will */         ;\
        /* load 128 bytes from stack. This is for q10, q11 for max nr of 4 */ ;\
        /* Thus we will load accumulators back in q0, q1, q2, q3, q4, q5, q6, q7 */ ;\
        /* When nr < 4, extra q values will be fetched from stack which may overlap */ ;\
        /* with other parts of stack storing local variables. To avoid that we just */ ;\
        /* create a buffer of 128 bytes inbetween to make sure pointer increment */ ;\
        /* never produces address that is beyond the stack frame of this function. */ ;\
        SUB r9, sp, 140                                                       ;\
        /* Each iteration produce 4 values each of 4 bytes */                 ;\
        /* Thus 4 x 4 = 16 bytes 2^4 */                                       ;\
        /* In this implementation, first value will be stored at */           ;\
        /* 1st value: sp - 12 - r1 * 16 */                                    ;\
        /* 2nd value: sp - 12 - (r1 - 1) * 16 */                              ;\
        /* and so on. */                                                      ;\
        SUB r9, r9, r1, LSL #4                                                ;\
        VST1.32 {q10}, [r9]                                                   ;\
                                                                              ;\
        /* Check if nr >=1 */                                                 ;\
        SUBS r1, r1, 1                                                        ;\
        BHI _0_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
    _3_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        /* First load all the accumulators from stack */                      ;\
        /* Load nr */                                                         ;\
        SUB r9, sp, 140                                                       ;\
        SUB r9, r9, r11, LSL #4                                               ;\
        /* Now load q8-q15 */                                                 ;\
        /* This is 8x4 block (nrxmr) */                                       ;\
        /* We will transpose this to 4x8 (mrxnr) */                           ;\
        /* q8, q12  : x00, x10, x20, x30; x04, x14, x24, x34 */               ;\
        /* q9, q13  : x01, x11, x21, x31; x05, x15, x25, x35 */               ;\
        /* q10, q14 : x02, x12, x22, x32; x06, x16, x26, x36 */               ;\
        /* q11, q15 : x03, x13, x23, x33; x07, x17, x27, x37 */               ;\
        VLD1.32 {q8}, [r9]!                                                   ;\
        VLD1.32 {q9}, [r9]!                                                   ;\
        VLD1.32 {q10}, [r9]!                                                  ;\
        VLD1.32 {q11}, [r9]!                                                  ;\
        VLD1.32 {q12}, [r9]!                                                  ;\
        VLD1.32 {q13}, [r9]!                                                  ;\
        VLD1.32 {q14}, [r9]!                                                  ;\
        VLD1.32 {q15}, [r9]                                                   ;\
                                                                              ;\
        /*# Now transpose q8-11 */                                            ;\
        /* VTRN.32 q8, q9 */                                                  ;\
        /* VTRN.32 q10, q11 */                                                ;\
        /* q8 : X00, x01, x20, x21 */                                         ;\
        /* q9 : X10, x11, x30, x31 */                                         ;\
        /* q10: X02, x03, x22, x23 */                                         ;\
        /* q11: X12, x13, x32, x33 */                                         ;\
        /* VSWP d16, d17 */                                                   ;\
        /* q8 : x20, x21, x00, x01 */                                         ;\
        /* VEXT.32 q6, q8, q10, 2 */                                          ;\
        /* q6 : x00, x01, x02, x03 */                                         ;\
        /* VEXT.32 q10, q10, q8, 2 */                                         ;\
        /* q10: x22, x23, x20, x21 */                                         ;\
        /* VSWP d20, d21 */                                                   ;\
        /* VMOV q8, q6 */                                                     ;\
        /* q8 : X00, x01, x02, x03 */                                         ;\
        /* q10: x20, x21, x22, x23 */                                         ;\
        /* VSWP d18, d19 */                                                   ;\
        /* q9 : x30, x31, x10, x11 */                                         ;\
        /* VEXT.32 q6, q9, q11, 2 */                                          ;\
        /* q6 : x10, x11, x12, x13 */                                         ;\
        /* VEXT.32 q11, q11, q9, 2 */                                         ;\
        /* q11: x32, x33, x30, x31 */                                         ;\
        /* VSWP d22, d23 */                                                   ;\
        /* VMOV q9, q6 */                                                     ;\
        /* q9 : x10, x11, x12, x13 */                                         ;\
        /* q11: x30, x31, x32, x33 */                                         ;\
        /* Thus we have */                                                    ;\
        /* q8 : X00, x01, x02, x03 */                                         ;\
        /* q9 : X10, x11, x12, x13 */                                         ;\
        /* q10: X20, x21, x22, x23 */                                         ;\
        /* q11: X30, x31, x32, x33 */                                         ;\
        /* Now we can do the same for q4-q7 */                                ;\
        /* q12: X04, X05, X06, X07 */                                         ;\
        /* q13: X14, X15, X16, X17 */                                         ;\
        /* q14: X24, X25, X26, X27 */                                         ;\
        /* q15: X34, X35, X36, X37 */                                         ;\
                                                                              ;\
        VTRN.32 q8, q9                                                        ;\
        VTRN.32 q10, q11                                                      ;\
        VSWP d16, d17                                                         ;\
        VEXT.32 q6, q8, q10, 2                                                ;\
        VEXT.32 q10, q10, q8, 2                                               ;\
        VSWP d20, d21                                                         ;\
        VMOV q8, q6                                                           ;\
        VSWP d18, d19                                                         ;\
        VEXT.32 q6, q9, q11, 2                                                ;\
        VEXT.32 q11, q11, q9, 2                                               ;\
        VSWP d22, d23                                                         ;\
        VMOV q9, q6                                                           ;\
                                                                              ;\
        VTRN.32 q12, q13                                                      ;\
        VTRN.32 q14, q15                                                      ;\
        VSWP d24, d25                                                         ;\
        VEXT.32 q6, q12, q14, 2                                               ;\
        VEXT.32 q14, q14, q12, 2                                              ;\
        VSWP d28, d29                                                         ;\
        VMOV q12, q6                                                          ;\
        VSWP d26, d27                                                         ;\
        VEXT.32 q6, q13, q15, 2                                               ;\
        VEXT.32 q15, q15, q13, 2                                              ;\
        VSWP d30, d31                                                         ;\
        VMOV q13, q6                                                          ;\
                                                                              ;\
        /* Load output channel index */                                       ;\
        LDR r5, [sp, 120]                                                     ;\
        /* Load quantization params */                                        ;\
        /* - r7 = quantization_params */                                      ;\
        LDR r7, [sp, 124]                                                     ;\
        ADD r7, r7, 8                                                         ;\
        /* Load pointer to per channel requant scale */                       ;\
        LDR r7, [r7]                                                          ;\
        /* Now r7 has the base_addr + offset for multipliers */               ;\
        ADD r7, r7, r5, LSL #2                                                ;\
                                                                              ;\
        LDR r6, [sp, 108]                                                     ;\
        /* Load q6: vmultiplier_c0123 */                                      ;\
        VLD1.32 {d12, d13}, [r7]!                                             ;\
        /* Load q7: vmultiplier_c4567 */                                      ;\
        VLD1.32 {d14, d15}, [r7]                                              ;\
        VCVT.F32.S32 q8, q8                                                   ;\
        VCVT.F32.S32 q9, q9                                                   ;\
        VCVT.F32.S32 q10, q10                                                 ;\
        VLD1.32 {q0}, [r6]!                                                   ;\
        VLD1.32 {q1}, [r6]                                                    ;\
                                                                              ;\
        VCVT.F32.S32 q11, q11                                                 ;\
        VCVT.F32.S32 q12, q12                                                 ;\
        VCVT.F32.S32 q13, q13                                                 ;\
        VCVT.F32.S32 q14, q14                                                 ;\
        VCVT.F32.S32 q15, q15                                                 ;\
                                                                              ;\
        VMUL.F32 q8, q8, q6                                                   ;\
        VMUL.F32 q9, q9, q6                                                   ;\
        VMUL.F32 q10, q10, q6                                                 ;\
        VMUL.F32 q11, q11, q6                                                 ;\
        VMUL.F32 q12, q12, q7                                                 ;\
        VMUL.F32 q13, q13, q7                                                 ;\
        VMUL.F32 q14, q14, q7                                                 ;\
        VMUL.F32 q15, q15, q7                                                 ;\
                                                                              ;\
        VADD.F32 q8, q8, q0                                                   ;\
        VADD.F32 q9, q9, q0                                                   ;\
        VADD.F32 q10, q10, q0                                                 ;\
        VADD.F32 q11, q11, q0                                                 ;\
        VADD.F32 q12, q12, q1                                                 ;\
        VADD.F32 q13, q13, q1                                                 ;\
        VADD.F32 q14, q14, q1                                                 ;\
        VADD.F32 q15, q15, q1                                                 ;\
                                                                              ;\
        /* Load c, c_stride: */                                               ;\
        /* - r1 = c */                                                        ;\
        /* - r9 = c_stride */                                                 ;\
        LDR r1, [sp, 112]                                                     ;\
        LDR r9, [sp, 116]                                                     ;\
        LSL r9, r9, 2                                                         ;\
                                                                              ;\
        /* r1 = c0 = c pointer */                                             ;\
                                                                              ;\
        CMP r0, 2                                                             ;\
        /* r2 = c1 */                                                         ;\
        ADD r2, r1, r9                                                        ;\
        MOVLO r2, r1                                                          ;\
                                                                              ;\
        /* r3 = c2 */                                                         ;\
        ADD r3, r2, r9                                                        ;\
        MOVLS r3, r2                                                          ;\
                                                                              ;\
        CMP r0, 4                                                             ;\
        /* r4 = c3 */                                                         ;\
        ADD r4, r3, r9                                                        ;\
        MOVNE r4, r3                                                          ;\
                                                                              ;\
        CMP r11, 8                                                            ;\
        BNE _4_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
                                                                              ;\
        VST1.32 {q8}, [r1]!                                                   ;\
        VST1.32 {q9}, [r2]!                                                   ;\
        VST1.32 {q10}, [r3]!                                                  ;\
        VST1.32 {q11}, [r4]!                                                  ;\
        VST1.32 {q12}, [r1]                                                   ;\
        VST1.32 {q13}, [r2]                                                   ;\
        VST1.32 {q14}, [r3]                                                   ;\
        VST1.32 {q15}, [r4]                                                   ;\
                                                                              ;\
        VPOP {d8-d15}                                                         ;\
        POP {r4, r5, r6, r7, r8, r9, r10, r11, lr}                            ;\
        BX lr                                                                 ;\
                                                                              ;\
        .p2align 3                                                            ;\
    _4_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        CMP r11, 4                                                            ;\
        BLO _5_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
                                                                              ;\
        VST1.32 {q8}, [r1]!                                                   ;\
        VST1.32 {q9}, [r2]!                                                   ;\
        VST1.32 {q10}, [r3]!                                                  ;\
        VST1.32 {q11}, [r4]!                                                  ;\
                                                                              ;\
        SUB r11, 4                                                            ;\
                                                                              ;\
        VMOV.32 q8, q12                                                       ;\
        VMOV.32 q9, q13                                                       ;\
        VMOV.32 q10, q14                                                      ;\
        VMOV.32 q11, q15                                                      ;\
                                                                              ;\
    _5_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        CMP r11, 2                                                            ;\
        BLO _6_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
                                                                              ;\
        VST1.32 {d16}, [r1]!                                                  ;\
        VST1.32 {d18}, [r2]!                                                  ;\
        VST1.32 {d20}, [r3]!                                                  ;\
        VST1.32 {d22}, [r4]!                                                  ;\
                                                                              ;\
        SUB r11, 2                                                            ;\
                                                                              ;\
        VEXT.32 q8, q8, 2                                                     ;\
        VEXT.32 q9, q9, 2                                                     ;\
        VEXT.32 q10, q10, 2                                                   ;\
        VEXT.32 q11, q11, 2                                                   ;\
                                                                              ;\
    _6_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        TEQ r11, 0                                                            ;\
        BEQ _7_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
                                                                              ;\
        VST1.32 {d16[0]}, [r1]                                                ;\
        VST1.32 {d18[0]}, [r2]                                                ;\
        VST1.32 {d20[0]}, [r3]                                                ;\
        VST1.32 {d22[0]}, [r4]                                                ;\
                                                                              ;\
    _7_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        VPOP {d8-d15}                                                         ;\
        POP {r4, r5, r6, r7, r8, r9, r10, r11, lr}                            ;\
        BX lr                                                                 ;\
                                                                              ;\
    END_FUNCTION pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch32_neon

# void pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w32__aarch32_neon(
#     size_t mr,
#     size_t nr,
#     const uint8_t* a_packed,
#     const uint8_t* packed_w,
#     const uint32_t* w_row_ptr,
#     const uint32_t* w_block_ids_ptr,
#     const float* b,
#     uint8_t* restrict c,
#     size_t c_stride,
#     size_t output_channel_index,
#     const union pytorch_qnnp_conv_dynamic_quantization_params quantization_params[restrict static 1])
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_1X4_UKERNEL_4X8_PACKEDA__AARCH32_NEON(32, #4, #2, LDR)

# void pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w16__aarch32_neon(
#     size_t mr,
#     size_t nr,
#     const uint8_t* a_packed,
#     const uint8_t* packed_w,
#     const uint16_t* w_row_ptr,
#     const uint16_t* w_block_ids_ptr,
#     const float* b,
#     uint8_t* restrict c,
#     size_t c_stride,
#     size_t output_channel_index,
#     const union pytorch_qnnp_conv_dynamic_quantization_params quantization_params[restrict static 1])
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_1X4_UKERNEL_4X8_PACKEDA__AARCH32_NEON(16, #2, #1, LDRH)

# void pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w8__aarch32_neon(
#     size_t mr,
#     size_t nr,
#     const uint8_t* a_packed,
#     const uint8_t* packed_w,
#     const uint8_t* w_row_ptr,
#     const uint8_t* w_block_ids_ptr,
#     const float* b,
#     uint8_t* restrict c,
#     size_t c_stride,
#     size_t output_channel_index,
#     const union pytorch_qnnp_conv_dynamic_quantization_params quantization_params[restrict static 1])
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_1X4_UKERNEL_4X8_PACKEDA__AARCH32_NEON(8, #1, #0, LDRB)

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

#undef NDEF_APPLE_SYMBOLS
#undef MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_1X4_UKERNEL_4X8_PACKEDA__AARCH32_NEON
