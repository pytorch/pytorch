/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
#define NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_5 .p2align 5
#define NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_4 .p2align 4
#define NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_3 .p2align 3
#else
#define NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_5
#define NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_4
#define NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_3
#endif

# Macro for separating instructions. For most builds, ; can be used, but for
# ARM64 + Mach, ; begins a comment, and %% is used to separate instructions
#if defined(__MACH__)
#define XX %%
#else
#define XX ;
#endif

.macro TRANSPOSE_4X4_S32 vin0, vin1, vin2, vin3, temp0, temp1, temp2, temp3
    TRN1 \temp0\().4s, \vin0\().4s, \vin1\().4s
    TRN2 \temp1\().4s, \vin0\().4s, \vin1\().4s
    TRN1 \temp2\().4s, \vin2\().4s, \vin3\().4s
    TRN2 \temp3\().4s, \vin2\().4s, \vin3\().4s
    TRN1 \vin0\().2d, \temp0\().2d, \temp2\().2d
    TRN1 \vin1\().2d, \temp1\().2d, \temp3\().2d
    TRN2 \vin2\().2d, \temp0\().2d, \temp2\().2d
    TRN2 \vin3\().2d, \temp1\().2d, \temp3\().2d
.endm

# params
# c_stride

#  Args passed via stack.
#  TOS
#  |-----------|
#  |c_stride   | 0
#  |out ch indx| 8
#  |params     | 16
#  |-----------|

# void pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch64_neon(
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
#define MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_1X4_UKERNEL_8X8_PACKEDA__AARCH64_NEON(W_INDEX_DTYPE_NUM_BITS, W_INDEX_DTYPE_NUM_BYTES_ARG, W_INDEX_DTYPE_LOG_NUM_BYTES_ARG, LOAD_INDEX_INSTRUCTION) XX\
    BEGIN_FUNCTION pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch64_neon XX\
                                                                             XX\
        STP d15, d14, [sp, -16]                                              XX\
        STP d13, d12, [sp, -32]                                              XX\
        STP d11, d10, [sp, -48]                                              XX\
        STP d9, d8, [sp, -64]                                                XX\
                                                                             XX\
        MOV x11, x1                                                          XX\
        /* Load output channel index */                                      XX\
        LDR x10, [sp, 8]                                                     XX\
        /* Load params */                                                    XX\
        LDR x8, [sp, 16]                                                     XX\
                                                                             XX\
        /* Load a_zero_point */                                              XX\
        LD1R {v24.8b}, [x8]                                                  XX\
        ADD x8, x8, 8                                                        XX\
                                                                             XX\
        /* Load pointer to per channel zero points array */                  XX\
        LDR x17, [x8], 8                                                     XX\
                                                                             XX\
        /* Load pointer to per channel multiplier */                         XX\
        LDR x13, [x8]                                                        XX\
                                                                             XX\
        /* Add offset to the base pointer */                                 XX\
        ADD x17, x17, x10                                                    XX\
        /* Mul by 4 to get byte offset for multiplier */                     XX\
        LSL x10, x10, 2                                                      XX\
        /* Add offset to the base pointer for multiplier */                  XX\
        ADD x13, x13, x10                                                    XX\
                                                                             XX\
        /* Load b_zero_point */                                              XX\
        LD1 {v25.8b}, [x17]                                                  XX\
        /* Load multiplier c0123 */                                          XX\
        LD1 {v26.4s}, [x13], 16                                              XX\
        /* Load multiplier c4567 */                                          XX\
        LD1 {v30.4s}, [x13]                                                  XX\
                                                                             XX\
        EOR x12, x12, x12                                                    XX\
        EOR x13, x13, x13                                                    XX\
                                                                             XX\
        CMP x1, 1                                                            XX\
        B.LO _7_w##W_INDEX_DTYPE_NUM_BITS                                    XX\
                                                                             XX\
        NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_5                          XX\
    _0_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        /* v8 := zero */                                                     XX\
        EOR v8.16b, v8.16b, v8.16b                                           XX\
        /* v9 := zero */                                                     XX\
        EOR v9.16b, v9.16b, v9.16b                                           XX\
                                                                             XX\
        DUP v29.8b, v25.b[0]                                                 XX\
        /* w12 = w_row_ptr[n], x13 = w_row_ptr[n+1] */                       XX\
        /* x4 = x4 + W_INDEX_DTYPE_NUM_BYTES_ARG to point to next n */       XX\
        LOAD_INDEX_INSTRUCTION w12, [x4], W_INDEX_DTYPE_NUM_BYTES_ARG        XX\
        LOAD_INDEX_INSTRUCTION w13, [x4]                                     XX\
        /* x10 = temp_packed_w = packed_w + w_row_ptr[n] * 4 */              XX\
        /* This points to the first block of nonzero value */                XX\
        /* for the nth row. */                                               XX\
        ADD x10, x3, x12, LSL #2                                             XX\
        /* x9 = temp_w_block_ids_ptr = w_block_ids_ptr (x5) + w_row_ptr[n] */ XX\
        /* LSL for when elements are >1 byte */                              XX\
        /* (4 bytes: LSL #2, 2 bytes: LSL #1, 1 byte: LSL #0) */             XX\
        /* This points to the block id of the first block */                 XX\
        /* It should contain x13 - x12 number of block ids */                XX\
        ADD x9, x5, x12, LSL W_INDEX_DTYPE_LOG_NUM_BYTES_ARG                 XX\
        /* x8 = num_blocks that needs to be processed */                     XX\
        SUB x8, x13, x12                                                     XX\
        SUBS x8, x8, 2                                                       XX\
        B.LO _1_w##W_INDEX_DTYPE_NUM_BITS                                    XX\
                                                                             XX\
    k_loop_w##W_INDEX_DTYPE_NUM_BITS##:                                      XX\
        /* b0-7 (channel 0) */                                               XX\
        LD1 {v10.8b}, [x10], 8                                               XX\
        USUBL v10.8h, v10.8b, v29.8b                                         XX\
                                                                             XX\
        /* x12 = block_id_ptr[0] */                                          XX\
        /* x13 = block_id_ptr[1] */                                          XX\
        LOAD_INDEX_INSTRUCTION w12, [x9], W_INDEX_DTYPE_NUM_BYTES_ARG        XX\
        LOAD_INDEX_INSTRUCTION w13, [x9], W_INDEX_DTYPE_NUM_BYTES_ARG        XX\
        /* Add offset to x2 */                                               XX\
        /* Shift by 5 because each packed block is a block of 8x4 */         XX\
        /* which 32 bytes */                                                 XX\
        ADD x16, x2, x12, LSL #5                                             XX\
        ADD x17, x2, x13, LSL #5                                             XX\
                                                                             XX\
        LD1 {v0.8b}, [x16], 8                                                XX\
        LD1 {v1.8b}, [x16], 8                                                XX\
        LD1 {v2.8b}, [x16], 8                                                XX\
        LD1 {v3.8b}, [x16]                                                   XX\
        LD1 {v4.8b}, [x17], 8                                                XX\
        LD1 {v5.8b}, [x17], 8                                                XX\
        LD1 {v6.8b}, [x17], 8                                                XX\
        LD1 {v7.8b}, [x17]                                                   XX\
                                                                             XX\
        USUBL v0.8h, v0.8b, v24.8b                                           XX\
        USUBL v1.8h, v1.8b, v24.8b                                           XX\
        USUBL v2.8h, v2.8b, v24.8b                                           XX\
        USUBL v3.8h, v3.8b, v24.8b                                           XX\
        USUBL v4.8h, v4.8b, v24.8b                                           XX\
        USUBL v5.8h, v5.8b, v24.8b                                           XX\
        USUBL v6.8h, v6.8b, v24.8b                                           XX\
        USUBL v7.8h, v7.8b, v24.8b                                           XX\
                                                                             XX\
        SMLAL v8.4s, v0.4h, v10.h[0]                                         XX\
        SMLAL2 v9.4s, v0.8h, v10.h[0]                                        XX\
        SMLAL v8.4s, v1.4h, v10.h[1]                                         XX\
        SMLAL2 v9.4s, v1.8h, v10.h[1]                                        XX\
        SMLAL v8.4s, v2.4h, v10.h[2]                                         XX\
        SMLAL2 v9.4s, v2.8h, v10.h[2]                                        XX\
        SMLAL v8.4s, v3.4h, v10.h[3]                                         XX\
        SMLAL2 v9.4s, v3.8h, v10.h[3]                                        XX\
        SMLAL v8.4s, v4.4h, v10.h[4]                                         XX\
        SMLAL2 v9.4s, v4.8h, v10.h[4]                                        XX\
        SMLAL v8.4s, v5.4h, v10.h[5]                                         XX\
        SMLAL2 v9.4s, v5.8h, v10.h[5]                                        XX\
        SMLAL v8.4s, v6.4h, v10.h[6]                                         XX\
        SMLAL2 v9.4s, v6.8h, v10.h[6]                                        XX\
        SUBS x8, x8, 2                                                       XX\
        SMLAL v8.4s, v7.4h, v10.h[7]                                         XX\
        SMLAL2 v9.4s, v7.8h, v10.h[7]                                        XX\
                                                                             XX\
                                                                             XX\
        B.HS k_loop_w##W_INDEX_DTYPE_NUM_BITS                                XX\
                                                                             XX\
    _1_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        CMP x8, -2                                                           XX\
        B.EQ _2_w##W_INDEX_DTYPE_NUM_BITS                                    XX\
                                                                             XX\
        /* b0-7 (channel 0) */                                               XX\
        LD1R {v10.4s}, [x10]                                                 XX\
        USUBL v10.8h, v10.8b, v29.8b                                         XX\
                                                                             XX\
        /* x12 = block_id_ptr[0] */                                          XX\
        LOAD_INDEX_INSTRUCTION w12, [x9]                                     XX\
        /* Add offset to x2 */                                               XX\
        /* Shift by 5 because each packed block is a block of 8x4 */         XX\
        /* which 32 bytes */                                                 XX\
        ADD x16, x2, x12, LSL #5                                             XX\
                                                                             XX\
        LD1 {v0.8b}, [x16], 8                                                XX\
        LD1 {v1.8b}, [x16], 8                                                XX\
        LD1 {v2.8b}, [x16], 8                                                XX\
        LD1 {v3.8b}, [x16]                                                   XX\
                                                                             XX\
        USUBL v0.8h, v0.8b, v24.8b                                           XX\
        USUBL v1.8h, v1.8b, v24.8b                                           XX\
        USUBL v2.8h, v2.8b, v24.8b                                           XX\
        USUBL v3.8h, v3.8b, v24.8b                                           XX\
                                                                             XX\
        SMLAL v8.4s, v0.4h, v10.h[0]                                         XX\
        SMLAL2 v9.4s, v0.8h, v10.h[0]                                        XX\
        SMLAL v8.4s, v1.4h, v10.h[1]                                         XX\
        SMLAL2 v9.4s, v1.8h, v10.h[1]                                        XX\
        SMLAL v8.4s, v2.4h, v10.h[2]                                         XX\
        SMLAL2 v9.4s, v2.8h, v10.h[2]                                        XX\
        SMLAL v8.4s, v3.4h, v10.h[3]                                         XX\
        SMLAL2 v9.4s, v3.8h, v10.h[3]                                        XX\
                                                                             XX\
        NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_4                          XX\
    _2_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        /* Store result on stack */                                          XX\
                                                                             XX\
        /* -64 because all d8-d15 are on stack */                            XX\
        /* + 256 bytes of buffer when nr = 1 */                              XX\
        /* 256 because we are doing 8x8 block with each value being 4 bytes */ XX\
        /* Thus 64 * 4 = 256 */                                              XX\
        /* 256 + 64 = 320 */                                                 XX\
        /* This is needed because after processing all nrs we will */        XX\
        /* load 256  bytes from stack. */                                    XX\
        /* Thus we will load accumulators back in v8, v9, v10, v11, v12, v13, v14, v15 */ XX\
        /* v16, v17, v18, v19, v20, v21, v22, v23 */                         XX\
        /* When nr < 8, say nr = 1, extra v values will be fetched from stack which may overlap */ XX\
        /* with other parts of stack storing local variables. To avoid that we just */ XX\
        /* create a buffer of 256 bytes inbetween to make sure pointer increment */ XX\
        /* never produces address that is beyond the stack frame of this function. */ XX\
        SUB x9, sp, 320                                                      XX\
        /* Each iteration produce 8 values each of 4 bytes */                XX\
        /* Thus 8 x 4 = 32 bytes 2^5 */                                      XX\
        /* In this implementation, first value will be stored at */          XX\
        /* 1st value: sp - 64 - r1 * 32 */                                   XX\
        /* 2nd value: sp - 12 - (r1 - 1) * 32 */                             XX\
        /* and so on. */                                                     XX\
        SUB x9, x9, x1, LSL #5                                               XX\
        ST1 {v8.4s}, [x9], 16                                                XX\
        ST1 {v9.4s}, [x9]                                                    XX\
                                                                             XX\
        /* Shift zero point vector by 8 to load */                           XX\
        /* zero point of the next channel */                                 XX\
        SRI v25.2d, v25.2d, #8                                               XX\
        /* Check if nr >=1 */                                                XX\
        SUBS x1, x1, 1                                                       XX\
        BHI _0_w##W_INDEX_DTYPE_NUM_BITS                                     XX\
    _3_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        /* First load all the accumulators from stack */                     XX\
        /* Load nr */                                                        XX\
        SUB x9, sp, 320                                                      XX\
        SUB x9, x9, x11, LSL #5                                              XX\
        /* Now load v8-v15 */                                                XX\
        /* This is 8x4 block (nrxmr) */                                      XX\
        /* We will transpose this to 4x8 (mrxnr) */                          XX\
        /* v8, v9   : x00, x10, x20, x30; x40, x50, x60, x70 */              XX\
        /* v10, v11 : x01, x11, x21, x31; x41, x51, x61, x71 */              XX\
        /* v12, v13 : x02, x12, x22, x32; x42, x52, x62, x72 */              XX\
        /* v14, v15 : x03, x13, x23, x33; x43, x53, x63, x73 */              XX\
        /* */                                                                XX\
        /* v16, v17 : x04, x14, x24, x34; x44, x54, x64, x74 */              XX\
        /* v18, v19 : x05, x15, x25, x35; x45, x55, x65, x75 */              XX\
        /* v20, v21 : x06, x16, x26, x36; x46, x56, x66, x76 */              XX\
        /* v22, v23 : x07, x17, x27, x37; x47, x57, x67, x77 */              XX\
        LD1 {v8.4s}, [x9], 16                                                XX\
        LD1 {v9.4s}, [x9], 16                                                XX\
        LD1 {v10.4s}, [x9], 16                                               XX\
        LD1 {v11.4s}, [x9], 16                                               XX\
        LD1 {v12.4s}, [x9], 16                                               XX\
        LD1 {v13.4s}, [x9], 16                                               XX\
        LD1 {v14.4s}, [x9], 16                                               XX\
        LD1 {v15.4s}, [x9], 16                                               XX\
        LD1 {v16.4s}, [x9], 16                                               XX\
        LD1 {v17.4s}, [x9], 16                                               XX\
        LD1 {v18.4s}, [x9], 16                                               XX\
        LD1 {v19.4s}, [x9], 16                                               XX\
        LD1 {v20.4s}, [x9], 16                                               XX\
        LD1 {v21.4s}, [x9], 16                                               XX\
        LD1 {v22.4s}, [x9], 16                                               XX\
        LD1 {v23.4s}, [x9]                                                   XX\
                                                                             XX\
        /* We can tranpose one 4x4 block using macro */                      XX\
        /* TRANSPOSE_4X4_S32 v8, v10, v12, v14, v0, v1, v2, v3 */            XX\
        /* After this we have */                                             XX\
        /* v8  : x00, x01, x02, x03 */                                       XX\
        /* v10 : x10, x11, x12, x13 */                                       XX\
        /* v12 : x20, x21, x22, x23 */                                       XX\
        /* v14 : x30, x31, x32, x33 */                                       XX\
        /* Then using */                                                     XX\
        /* TRANSPOSE_4X4_S32 v16, v18, v20, v22, v4, v5, v6, v7 */           XX\
        /* We get */                                                         XX\
        /* v16 : x04, x05, x06, x07 */                                       XX\
        /* v18 : x14, x15, x16, x17 */                                       XX\
        /* v20 : x24, x25, x26, x27 */                                       XX\
        /* v22 : x34, x35, x36, x37 */                                       XX\
        /* Similarly we can transpose other two 4x4 blocks and we get */     XX\
        /* tranposed 8x8 */                                                  XX\
                                                                             XX\
        TRANSPOSE_4X4_S32 v8, v10, v12, v14, v0, v1, v2, v3                  XX\
        TRANSPOSE_4X4_S32 v16, v18, v20, v22, v4, v5, v6, v7                 XX\
        TRANSPOSE_4X4_S32 v9, v11, v13, v15, v0, v1, v2, v3                  XX\
        TRANSPOSE_4X4_S32 v17, v19, v21, v23, v4, v5, v6, v7                 XX\
                                                                             XX\
        /* row 0: v8, v16 */                                                 XX\
        /* row 1: v10, v18 */                                                XX\
        /* row 2: v12, v20 */                                                XX\
        /* row 3: v14, v22 */                                                XX\
        /* row 4: v9, v17 */                                                 XX\
        /* row 5: v11, v19 */                                                XX\
        /* row 6: v13, v21 */                                                XX\
        /* row 7: v15, v23 */                                                XX\
                                                                             XX\
        /* Load c_stride & params */                                         XX\
        LDR x16, [sp]                                                        XX\
        LSL x16, x16, 2                                                      XX\
        LD1 {v24.4s}, [x6], 16                                               XX\
        LD1 {v25.4s}, [x6]                                                   XX\
                                                                             XX\
        SCVTF v8.4s, v8.4s                                                   XX\
        SCVTF v9.4s, v9.4s                                                   XX\
        SCVTF v10.4s, v10.4s                                                 XX\
        SCVTF v11.4s, v11.4s                                                 XX\
        SCVTF v12.4s, v12.4s                                                 XX\
        SCVTF v13.4s, v13.4s                                                 XX\
        SCVTF v14.4s, v14.4s                                                 XX\
        SCVTF v15.4s, v15.4s                                                 XX\
        SCVTF v16.4s, v16.4s                                                 XX\
        SCVTF v17.4s, v17.4s                                                 XX\
        SCVTF v18.4s, v18.4s                                                 XX\
        SCVTF v19.4s, v19.4s                                                 XX\
        SCVTF v20.4s, v20.4s                                                 XX\
        SCVTF v21.4s, v21.4s                                                 XX\
        SCVTF v22.4s, v22.4s                                                 XX\
        SCVTF v23.4s, v23.4s                                                 XX\
                                                                             XX\
        FMUL v8.4s, v8.4s, v26.4s                                            XX\
        FMUL v16.4s, v16.4s, v30.4s                                          XX\
        FMUL v10.4s, v10.4s, v26.4s                                          XX\
        FMUL v18.4s, v18.4s, v30.4s                                          XX\
        FMUL v12.4s, v12.4s, v26.4s                                          XX\
        FMUL v20.4s, v20.4s, v30.4s                                          XX\
        FMUL v14.4s, v14.4s, v26.4s                                          XX\
        FMUL v22.4s, v22.4s, v30.4s                                          XX\
        FMUL v9.4s, v9.4s, v26.4s                                            XX\
        FMUL v17.4s, v17.4s, v30.4s                                          XX\
        FMUL v11.4s, v11.4s, v26.4s                                          XX\
        FMUL v19.4s, v19.4s, v30.4s                                          XX\
        FMUL v13.4s, v13.4s, v26.4s                                          XX\
        FMUL v21.4s, v21.4s, v30.4s                                          XX\
        FMUL v15.4s, v15.4s, v26.4s                                          XX\
        FMUL v23.4s, v23.4s, v30.4s                                          XX\
                                                                             XX\
        FADD v8.4s, v8.4s, v24.4s                                            XX\
        FADD v16.4s, v16.4s, v25.4s                                          XX\
        FADD v10.4s, v10.4s, v24.4s                                          XX\
        FADD v18.4s, v18.4s, v25.4s                                          XX\
        FADD v12.4s, v12.4s, v24.4s                                          XX\
        FADD v20.4s, v20.4s, v25.4s                                          XX\
        FADD v14.4s, v14.4s, v24.4s                                          XX\
        FADD v22.4s, v22.4s, v25.4s                                          XX\
        FADD v9.4s, v9.4s, v24.4s                                            XX\
        FADD v17.4s, v17.4s, v25.4s                                          XX\
        FADD v11.4s, v11.4s, v24.4s                                          XX\
        FADD v19.4s, v19.4s, v25.4s                                          XX\
        FADD v13.4s, v13.4s, v24.4s                                          XX\
        FADD v21.4s, v21.4s, v25.4s                                          XX\
        FADD v15.4s, v15.4s, v24.4s                                          XX\
        FADD v23.4s, v23.4s, v25.4s                                          XX\
                                                                             XX\
        /* Compute c0-c7 */                                                  XX\
                                                                             XX\
        ADD  x9, x7, x16                                                     XX\
        CMP x0, 2                                                            XX\
        CSEL x9, x7, x9, LO                                                  XX\
                                                                             XX\
        ADD x10, x9,  x16                                                    XX\
        CSEL x10, x9, x10, LS                                                XX\
                                                                             XX\
        ADD x8, x10, x16                                                     XX\
        CMP x0, 4                                                            XX\
        CSEL x8, x10, x8, LO                                                 XX\
                                                                             XX\
        ADD x12, x8, x16                                                     XX\
        CSEL x12, x8, x12, LS                                                XX\
                                                                             XX\
        ADD x13, x12, x16                                                    XX\
        CMP x0, 6                                                            XX\
        CSEL x13, x12, x13, LO                                               XX\
                                                                             XX\
        ADD x14, x13, x16                                                    XX\
        CSEL x14, x13, x14, LS                                               XX\
                                                                             XX\
        ADD x15, x14, x16                                                    XX\
        CMP x0, 8                                                            XX\
        CSEL x15, x14, x15, NE                                               XX\
                                                                             XX\
        CMP x11, 8                                                           XX\
        B.NE _4_w##W_INDEX_DTYPE_NUM_BITS                                    XX\
                                                                             XX\
        ST1 {v8.4s}, [x7], 16                                                XX\
        ST1 {v16.4s}, [x7]                                                   XX\
        ST1 {v10.4s}, [x9], 16                                               XX\
        ST1 {v18.4s}, [x9]                                                   XX\
        ST1 {v12.4s}, [x10], 16                                              XX\
        ST1 {v20.4s}, [x10]                                                  XX\
        ST1 {v14.4s}, [x8], 16                                               XX\
        ST1 {v22.4s}, [x8]                                                   XX\
        ST1 {v9.4s}, [x12], 16                                               XX\
        ST1 {v17.4s}, [x12]                                                  XX\
        ST1 {v11.4s}, [x13], 16                                              XX\
        ST1 {v19.4s}, [x13]                                                  XX\
        ST1 {v13.4s}, [x14], 16                                              XX\
        ST1 {v21.4s}, [x14]                                                  XX\
        ST1 {v15.4s}, [x15], 16                                              XX\
        ST1 {v23.4s}, [x15]                                                  XX\
                                                                             XX\
        LDP d9, d8, [sp, -64]                                                XX\
        LDP d11, d10, [sp, -48]                                              XX\
        LDP d13, d12, [sp, -32]                                              XX\
        LDP d15, d14, [sp, -16]                                              XX\
                                                                             XX\
        RET                                                                  XX\
                                                                             XX\
        NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_3                          XX\
    _4_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        CMP x11, 4                                                           XX\
        B.LO _5_w##W_INDEX_DTYPE_NUM_BITS                                    XX\
                                                                             XX\
        ST1 {v8.4s}, [x7], 16                                                XX\
        ST1 {v10.4s}, [x9], 16                                               XX\
        ST1 {v12.4s}, [x10], 16                                              XX\
        ST1 {v14.4s}, [x8], 16                                               XX\
        ST1 {v9.4s}, [x12], 16                                               XX\
        ST1 {v11.4s}, [x13], 16                                              XX\
        ST1 {v13.4s}, [x14], 16                                              XX\
        ST1 {v15.4s}, [x15], 16                                              XX\
                                                                             XX\
        SUB x11, x11, 4                                                      XX\
                                                                             XX\
        MOV v8.16b, v16.16b                                                  XX\
        MOV v10.16b, v18.16b                                                 XX\
        MOV v12.16b, v20.16b                                                 XX\
        MOV v14.16b, v22.16b                                                 XX\
        MOV v9.16b, v17.16b                                                  XX\
        MOV v11.16b, v19.16b                                                 XX\
        MOV v13.16b, v21.16b                                                 XX\
        MOV v15.16b, v23.16b                                                 XX\
                                                                             XX\
    _5_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        CMP x11, 2                                                           XX\
        B.LO _6_w##W_INDEX_DTYPE_NUM_BITS                                    XX\
                                                                             XX\
        ST1 {v8.2s}, [x7], 8                                                 XX\
        ST1 {v10.2s}, [x9], 8                                                XX\
        ST1 {v12.2s}, [x10], 8                                               XX\
        ST1 {v14.2s}, [x8], 8                                                XX\
        ST1 {v9.2s}, [x12], 8                                                XX\
        ST1 {v11.2s}, [x13], 8                                               XX\
        ST1 {v13.2s}, [x14], 8                                               XX\
        ST1 {v15.2s}, [x15], 8                                               XX\
                                                                             XX\
        SUB x11, x11, 2                                                      XX\
                                                                             XX\
        EXT v8.16b, v8.16b, v8.16b, 8                                        XX\
        EXT v10.16b, v10.16b, v10.16b, 8                                     XX\
        EXT v12.16b, v12.16b, v12.16b, 8                                     XX\
        EXT v14.16b, v14.16b, v14.16b, 8                                     XX\
        EXT v9.16b, v9.16b, v9.16b, 8                                        XX\
        EXT v11.16b, v11.16b, v11.16b, 8                                     XX\
        EXT v13.16b, v13.16b, v13.16b, 8                                     XX\
        EXT v15.16b, v15.16b, v15.16b, 8                                     XX\
                                                                             XX\
    _6_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        CMP x11, 1                                                           XX\
        B.LO _7_w##W_INDEX_DTYPE_NUM_BITS                                    XX\
                                                                             XX\
        ST1 {v8.s}[0], [x7]                                                  XX\
        ST1 {v10.s}[0], [x9]                                                 XX\
        ST1 {v12.s}[0], [x10]                                                XX\
        ST1 {v14.s}[0], [x8]                                                 XX\
        ST1 {v9.s}[0], [x12]                                                 XX\
        ST1 {v11.s}[0], [x13]                                                XX\
        ST1 {v13.s}[0], [x14]                                                XX\
        ST1 {v15.s}[0], [x15]                                                XX\
                                                                             XX\
    _7_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        LDP d9, d8, [sp, -64]                                                XX\
        LDP d11, d10, [sp, -48]                                              XX\
        LDP d13, d12, [sp, -32]                                              XX\
        LDP d15, d14, [sp, -16]                                              XX\
                                                                             XX\
        RET                                                                  XX\
                                                                             XX\
    END_FUNCTION pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch64_neon

# void pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w32__aarch64_neon(
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
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_1X4_UKERNEL_8X8_PACKEDA__AARCH64_NEON(32, #4, #2, LDR)

# void pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w16__aarch64_neon(
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
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_1X4_UKERNEL_8X8_PACKEDA__AARCH64_NEON(16, #2, #1, LDRH)

# void pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w8__aarch64_neon(
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
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_1X4_UKERNEL_8X8_PACKEDA__AARCH64_NEON(8, #1, #0, LDRB)

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

#undef NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_5
#undef NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_4
#undef NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_3
#undef MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_1X4_UKERNEL_8X8_PACKEDA__AARCH64_NEON
#undef XX
