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

# void pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch32_neon(
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
#define MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_8X1_UKERNEL_4X8_PACKEDA__AARCH32_NEON(W_INDEX_DTYPE_NUM_BITS, W_INDEX_DTYPE_NUM_BYTES_ARG, W_INDEX_DTYPE_LOG_NUM_BYTES_ARG, LOAD_INDEX_INSTRUCTION) ;\
    BEGIN_FUNCTION pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch32_neon ;\
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
        VLD1.8 {d14[]}, [r7]                                                  ;\
        ADD r7, r7, 4                                                         ;\
        /* Load pointer to per channel zero points array */                   ;\
        LDR r4, [r7]                                                          ;\
        /* Add output_channel_index to the b_zero_point pointer */            ;\
        ADD r4, r4, r5                                                        ;\
                                                                              ;\
        /* Load w_row_ptr + n */                                              ;\
        LDR r5, [sp, 100]                                                     ;\
        /* r7 = blocks_id_ptr */                                              ;\
        LDR r7, [sp, 104]                                                     ;\
                                                                              ;\
        VEOR q8, q8, q8                                                       ;\
        VEOR q9, q9, q9                                                       ;\
        VEOR q10, q10, q10                                                    ;\
        VEOR q11, q11, q11                                                    ;\
        VEOR q12, q12, q12                                                    ;\
        VEOR q13, q13, q13                                                    ;\
        VEOR q14, q14, q14                                                    ;\
        VEOR q15, q15, q15                                                    ;\
        VLD1.8 {d15}, [r4]                                                    ;\
        /* ip = w_row_ptr[n], lr = w_row_ptr[n+1] */                          ;\
        /* r5 = r5 + W_INDEX_DTYPE_NUM_BYTES_ARG to point to next n */        ;\
        LOAD_INDEX_INSTRUCTION ip, [r5], W_INDEX_DTYPE_NUM_BYTES_ARG          ;\
        LOAD_INDEX_INSTRUCTION lr, [r5]                                       ;\
        /* r6 = temp_packed_w = packed_w + w_row_ptr[n] * 8 */                ;\
        /* * 8 because each block contains 8 values */                        ;\
        /* This points to the first block of nonzero value */                 ;\
        /* for the nth row. */                                                ;\
        ADD r6, r3, ip, LSL #3                                                ;\
        /* r9 = temp_w_block_ids_ptr = w_block_ids_ptr (r7) + w_row_ptr[n] */ ;\
        /* LSL for when elements are >1 byte */                               ;\
        /* (4 bytes: LSL #2, 2 bytes: LSL #1, 1 byte: LSL #0) */              ;\
        /* This points to the col block id of the first block */              ;\
        /* It should contain lr - ip number of block ids */                   ;\
        /* Note that in this kernel sparsity pattern is 8x1. */               ;\
        /* Thus each block contains only 1 k as opposed to */                 ;\
        /* 1x4 where each block contains 4 k. */                              ;\
        ADD r9, r7, ip, LSL W_INDEX_DTYPE_LOG_NUM_BYTES_ARG                   ;\
        /* r8 = num_blocks that needs to be processed */                      ;\
        SUB r8, lr, ip                                                        ;\
        SUBS r8, r8, 2                                                        ;\
        BLO _1_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
                                                                              ;\
        .p2align 5                                                            ;\
    k_loop_w##W_INDEX_DTYPE_NUM_BITS##:                                       ;\
        /* Load 2 non zero blocks of weights. Each block = 8x1. */            ;\
        VLD1.8 {d0}, [r6]!                                                    ;\
        VLD1.8 {d2}, [r6]!                                                    ;\
                                                                              ;\
        /* ip = block_id_ptr[0] */                                            ;\
        /* lr = block_id_ptr[1] */                                            ;\
        LOAD_INDEX_INSTRUCTION ip, [r9], W_INDEX_DTYPE_NUM_BYTES_ARG          ;\
        LOAD_INDEX_INSTRUCTION lr, [r9], W_INDEX_DTYPE_NUM_BYTES_ARG          ;\
                                                                              ;\
        /* Add offset to r2 */                                                ;\
        /* Shift by 4 because each packed block is a block of 4x1 */          ;\
        /* which 4 bytes */                                                   ;\
        ADD r10, r2, ip, LSL #2                                               ;\
        /* q9 = vxb */                                                        ;\
        VSUBL.U8 q0, d0, d15                                                  ;\
        VSUBL.U8 q1, d2, d15                                                  ;\
                                                                              ;\
        /* d4 = 4x1 transposed */                                             ;\
        VLD1.32 {d4[]}, [r10]                                                 ;\
                                                                              ;\
        ADD r10, r2, lr, LSL #2                                               ;\
                                                                              ;\
        VSUBL.U8 q2, d4, d14  /* vxa0_t */                                    ;\
                                                                              ;\
        /* d5 = next 4x1 transposed */                                        ;\
        VLD1.32 {d6[]}, [r10]                                                 ;\
                                                                              ;\
        VSUBL.U8 q3, d6, d14  /* vxa1_t */                                    ;\
                                                                              ;\
        /* q0 = d0, d1 = 8x1 block of weight for k */                         ;\
        /* q1 = d2, d3 = 8x1 block of weight for k + 1 */                     ;\
        /* q2's d4 = 4x1 block of activation for k */                         ;\
        /* q3's d6 = 4x1 block of activation for k + 1 */                     ;\
                                                                              ;\
        /* Generate 4x8 block as two 4x4 blocks */                            ;\
                                                                              ;\
        VMLAL.S16 q8, d0, d4[0]                                               ;\
        VMLAL.S16 q9, d1, d4[0]                                               ;\
        VMLAL.S16 q10, d0, d4[1]                                              ;\
        VMLAL.S16 q11, d1, d4[1]                                              ;\
        VMLAL.S16 q12, d0, d4[2]                                              ;\
        VMLAL.S16 q13, d1, d4[2]                                              ;\
        VMLAL.S16 q14, d0, d4[3]                                              ;\
        VMLAL.S16 q15, d1, d4[3]                                              ;\
                                                                              ;\
        VMLAL.S16 q8, d2, d6[0]                                               ;\
        VMLAL.S16 q9, d3, d6[0]                                               ;\
        VMLAL.S16 q10, d2, d6[1]                                              ;\
        VMLAL.S16 q11, d3, d6[1]                                              ;\
        VMLAL.S16 q12, d2, d6[2]                                              ;\
        VMLAL.S16 q13, d3, d6[2]                                              ;\
        VMLAL.S16 q14, d2, d6[3]                                              ;\
        VMLAL.S16 q15, d3, d6[3]                                              ;\
                                                                              ;\
        SUBS r8, r8, 2                                                        ;\
                                                                              ;\
        BHS k_loop_w##W_INDEX_DTYPE_NUM_BITS                                  ;\
    _1_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        CMP r8, -2                                                            ;\
        BEQ _3_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
                                                                              ;\
        /* Load last nonzero block */                                         ;\
        /* For this we will load 4 8 bit values as one 32 bit value */        ;\
        VLD1.8 {d0}, [r6]                                                     ;\
        /* q9 = vxb */                                                        ;\
        VSUBL.U8 q0, d0, d15                                                  ;\
                                                                              ;\
        /* ip = block_id_ptr[0] */                                            ;\
        LOAD_INDEX_INSTRUCTION ip, [r9]                                       ;\
                                                                              ;\
        /* Add offset to r2 */                                                ;\
        /* Shift by 4 because each packed block is a block of 4x1 */          ;\
        /* which 4 bytes */                                                   ;\
        ADD r10, r2, ip, LSL #2                                               ;\
                                                                              ;\
        VLD1.32 {d4[]}, [r10]!                                                ;\
                                                                              ;\
        VSUBL.U8 q2, d4, d14  /* vxa0_t */                                    ;\
                                                                              ;\
        VMLAL.S16 q8, d0, d4[0]                                               ;\
        VMLAL.S16 q9, d1, d4[0]                                               ;\
        VMLAL.S16 q10, d0, d4[1]                                              ;\
        VMLAL.S16 q11, d1, d4[1]                                              ;\
        VMLAL.S16 q12, d0, d4[2]                                              ;\
        VMLAL.S16 q13, d1, d4[2]                                              ;\
        VMLAL.S16 q14, d0, d4[3]                                              ;\
        VMLAL.S16 q15, d1, d4[3]                                              ;\
                                                                              ;\
                                                                              ;\
        .p2align 4                                                            ;\
    _3_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
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
        VMUL.F32 q9, q9, q7                                                   ;\
        VMUL.F32 q10, q10, q6                                                 ;\
        VMUL.F32 q11, q11, q7                                                 ;\
        VMUL.F32 q12, q12, q6                                                 ;\
        VMUL.F32 q13, q13, q7                                                 ;\
        VMUL.F32 q14, q14, q6                                                 ;\
        VMUL.F32 q15, q15, q7                                                 ;\
                                                                              ;\
        VADD.F32 q8, q8, q0                                                   ;\
        VADD.F32 q9, q9, q1                                                   ;\
        VADD.F32 q10, q10, q0                                                 ;\
        VADD.F32 q11, q11, q1                                                 ;\
        VADD.F32 q12, q12, q0                                                 ;\
        VADD.F32 q13, q13, q1                                                 ;\
        VADD.F32 q14, q14, q0                                                 ;\
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
        VST1.32 {q10}, [r2]!                                                  ;\
        VST1.32 {q12}, [r3]!                                                  ;\
        VST1.32 {q14}, [r4]!                                                  ;\
        VST1.32 {q9}, [r1]                                                    ;\
        VST1.32 {q11}, [r2]                                                   ;\
        VST1.32 {q13}, [r3]                                                   ;\
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
        VST1.32 {q10}, [r2]!                                                  ;\
        VST1.32 {q12}, [r3]!                                                  ;\
        VST1.32 {q14}, [r4]!                                                  ;\
                                                                              ;\
        SUB r11, 4                                                            ;\
                                                                              ;\
        VMOV.32 q8, q9                                                        ;\
        VMOV.32 q10, q11                                                      ;\
        VMOV.32 q12, q13                                                      ;\
        VMOV.32 q14, q15                                                      ;\
                                                                              ;\
    _5_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        CMP r11, 2                                                            ;\
        BLO _6_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
                                                                              ;\
        VST1.32 {d16}, [r1]!                                                  ;\
        VST1.32 {d20}, [r2]!                                                  ;\
        VST1.32 {d24}, [r3]!                                                  ;\
        VST1.32 {d28}, [r4]!                                                  ;\
                                                                              ;\
        SUB r11, 2                                                            ;\
                                                                              ;\
        VEXT.32 q8, q8, 2                                                     ;\
        VEXT.32 q10, q10, 2                                                   ;\
        VEXT.32 q12, q12, 2                                                   ;\
        VEXT.32 q14, q14, 2                                                   ;\
                                                                              ;\
    _6_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        TEQ r11, 0                                                            ;\
        BEQ _7_w##W_INDEX_DTYPE_NUM_BITS                                      ;\
                                                                              ;\
        VST1.32 {d16[0]}, [r1]                                                ;\
        VST1.32 {d20[0]}, [r2]                                                ;\
        VST1.32 {d24[0]}, [r3]                                                ;\
        VST1.32 {d28[0]}, [r4]                                                ;\
                                                                              ;\
    _7_w##W_INDEX_DTYPE_NUM_BITS##:                                           ;\
        VPOP {d8-d15}                                                         ;\
        POP {r4, r5, r6, r7, r8, r9, r10, r11, lr}                            ;\
        BX lr                                                                 ;\
                                                                              ;\
    END_FUNCTION pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch32_neon

# void pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w32__aarch32_neon(
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
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_8X1_UKERNEL_4X8_PACKEDA__AARCH32_NEON(32, #4, #2, LDR)

# void pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w16__aarch32_neon(
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
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_8X1_UKERNEL_4X8_PACKEDA__AARCH32_NEON(16, #2, #1, LDRH)

# void pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w8__aarch32_neon(
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
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_8X1_UKERNEL_4X8_PACKEDA__AARCH32_NEON(8, #1, #0, LDRB)

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

#undef NDEF_APPLE_SYMBOLS
#undef MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_8X1_UKERNEL_4X8_PACKEDA__AARCH32_NEON
