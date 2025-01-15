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

# params
# c_stride

#  Args passed via stack.
#  TOS
#  |-----------|
#  |c_stride   | 0
#  |out ch indx| 8
#  |params     | 16
#  |-----------|

# void pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch64_neon(
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
#define MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_8X1_UKERNEL_8X8_PACKEDA__AARCH64_NEON(W_INDEX_DTYPE_NUM_BITS, W_INDEX_DTYPE_NUM_BYTES_ARG, W_INDEX_DTYPE_LOG_NUM_BYTES_ARG, LOAD_INDEX_INSTRUCTION) XX\
    BEGIN_FUNCTION pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch64_neon XX\
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
        EOR v8.16b, v8.16b, v8.16b                                           XX\
        EOR v9.16b, v9.16b, v9.16b                                           XX\
        EOR v10.16b, v10.16b, v10.16b                                        XX\
        EOR v11.16b, v11.16b, v11.16b                                        XX\
        EOR v12.16b, v12.16b, v12.16b                                        XX\
        EOR v13.16b, v13.16b, v13.16b                                        XX\
        EOR v14.16b, v14.16b, v14.16b                                        XX\
        EOR v15.16b, v15.16b, v15.16b                                        XX\
        EOR v16.16b, v16.16b, v16.16b                                        XX\
        EOR v17.16b, v17.16b, v17.16b                                        XX\
        EOR v18.16b, v18.16b, v18.16b                                        XX\
        EOR v19.16b, v19.16b, v19.16b                                        XX\
        EOR v20.16b, v20.16b, v20.16b                                        XX\
        EOR v21.16b, v21.16b, v21.16b                                        XX\
        EOR v22.16b, v22.16b, v22.16b                                        XX\
        EOR v23.16b, v23.16b, v23.16b                                        XX\
                                                                             XX\
        /* w12 = w_row_ptr[n], x13 = w_row_ptr[n+1] */                       XX\
        /* x4 = x4 + W_INDEX_DTYPE_NUM_BYTES_ARG to point to next n */       XX\
        LOAD_INDEX_INSTRUCTION w12, [x4], W_INDEX_DTYPE_NUM_BYTES_ARG        XX\
        LOAD_INDEX_INSTRUCTION w13, [x4]                                     XX\
        /* x10 = temp_packed_w = packed_w + w_row_ptr[n] * 8 */              XX\
        /* This points to the first block of nonzero value */                XX\
        /* for the nth row. */                                               XX\
        ADD x10, x3, x12, LSL #3                                             XX\
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
        NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_5                          XX\
    k_loop_w##W_INDEX_DTYPE_NUM_BITS##:                                      XX\
        /* k_loop processes two k values */                                  XX\
        /* Load two 8x1 blocks */                                            XX\
        LD1 {v0.8b}, [x10], 8                                                XX\
        LD1 {v1.8b}, [x10], 8                                                XX\
        USUBL v0.8h, v0.8b, v25.8b                                           XX\
        USUBL v1.8h, v1.8b, v25.8b                                           XX\
                                                                             XX\
        /* x12 = block_id_ptr[0] */                                          XX\
        /* x13 = block_id_ptr[1] */                                          XX\
        LOAD_INDEX_INSTRUCTION w12, [x9], W_INDEX_DTYPE_NUM_BYTES_ARG        XX\
        LOAD_INDEX_INSTRUCTION w13, [x9], W_INDEX_DTYPE_NUM_BYTES_ARG        XX\
        /* Add offset to x2 */                                               XX\
        /* Shift by 3 because each packed block is a block of 8x1 */         XX\
        /* which 8 bytes */                                                  XX\
        ADD x16, x2, x12, LSL #3                                             XX\
        ADD x17, x2, x13, LSL #3                                             XX\
                                                                             XX\
        /* Load two 8x1 blocks of activation */                              XX\
        /* First 8x1 for first channel */                                    XX\
        /* second 8x1 for next channel */                                    XX\
        LD1 {v2.8b}, [x16]                                                   XX\
        LD1 {v3.8b}, [x17]                                                   XX\
                                                                             XX\
        USUBL v2.8h, v2.8b, v24.8b                                           XX\
        USUBL v3.8h, v3.8b, v24.8b                                           XX\
                                                                             XX\
        /* First channel */                                                  XX\
        SMLAL v8.4s, v0.4h, v2.h[0]                                          XX\
        SMLAL2 v9.4s, v0.8h, v2.h[0]                                         XX\
        SMLAL v10.4s, v0.4h, v2.h[1]                                         XX\
        SMLAL2 v11.4s, v0.8h, v2.h[1]                                        XX\
        SMLAL v12.4s, v0.4h, v2.h[2]                                         XX\
        SMLAL2 v13.4s, v0.8h, v2.h[2]                                        XX\
        SMLAL v14.4s, v0.4h, v2.h[3]                                         XX\
        SMLAL2 v15.4s, v0.8h, v2.h[3]                                        XX\
        SMLAL v16.4s, v0.4h, v2.h[4]                                         XX\
        SMLAL2 v17.4s, v0.8h, v2.h[4]                                        XX\
        SMLAL v18.4s, v0.4h, v2.h[5]                                         XX\
        SMLAL2 v19.4s, v0.8h, v2.h[5]                                        XX\
        SMLAL v20.4s, v0.4h, v2.h[6]                                         XX\
        SMLAL2 v21.4s, v0.8h, v2.h[6]                                        XX\
        SMLAL v22.4s, v0.4h, v2.h[7]                                         XX\
        SMLAL2 v23.4s, v0.8h, v2.h[7]                                        XX\
                                                                             XX\
        SUBS x8, x8, 2                                                       XX\
        /* Second channel */                                                 XX\
        SMLAL v8.4s, v1.4h, v3.h[0]                                          XX\
        SMLAL2 v9.4s, v1.8h, v3.h[0]                                         XX\
        SMLAL v10.4s, v1.4h, v3.h[1]                                         XX\
        SMLAL2 v11.4s, v1.8h, v3.h[1]                                        XX\
        SMLAL v12.4s, v1.4h, v3.h[2]                                         XX\
        SMLAL2 v13.4s, v1.8h, v3.h[2]                                        XX\
        SMLAL v14.4s, v1.4h, v3.h[3]                                         XX\
        SMLAL2 v15.4s, v1.8h, v3.h[3]                                        XX\
        SMLAL v16.4s, v1.4h, v3.h[4]                                         XX\
        SMLAL2 v17.4s, v1.8h, v3.h[4]                                        XX\
        SMLAL v18.4s, v1.4h, v3.h[5]                                         XX\
        SMLAL2 v19.4s, v1.8h, v3.h[5]                                        XX\
        SMLAL v20.4s, v1.4h, v3.h[6]                                         XX\
        SMLAL2 v21.4s, v1.8h, v3.h[6]                                        XX\
        SMLAL v22.4s, v1.4h, v3.h[7]                                         XX\
        SMLAL2 v23.4s, v1.8h, v3.h[7]                                        XX\
                                                                             XX\
        B.HS k_loop_w##W_INDEX_DTYPE_NUM_BITS                                XX\
                                                                             XX\
    _1_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        CMP x8, -2                                                           XX\
        B.EQ _3_w##W_INDEX_DTYPE_NUM_BITS                                    XX\
                                                                             XX\
        LD1 {v0.8b}, [x10]                                                   XX\
        USUBL v0.8h, v0.8b, v25.8b                                           XX\
                                                                             XX\
        /* x12 = block_id_ptr[0] */                                          XX\
        LOAD_INDEX_INSTRUCTION w12, [x9]                                     XX\
        /* Add offset to x2 */                                               XX\
        ADD x16, x2, x12, LSL #3                                             XX\
                                                                             XX\
        LD1 {v2.8b}, [x16]                                                   XX\
        USUBL v2.8h, v2.8b, v24.8b                                           XX\
                                                                             XX\
        SMLAL v8.4s, v0.4h, v2.h[0]                                          XX\
        SMLAL2 v9.4s, v0.8h, v2.h[0]                                         XX\
        SMLAL v10.4s, v0.4h, v2.h[1]                                         XX\
        SMLAL2 v11.4s, v0.8h, v2.h[1]                                        XX\
        SMLAL v12.4s, v0.4h, v2.h[2]                                         XX\
        SMLAL2 v13.4s, v0.8h, v2.h[2]                                        XX\
        SMLAL v14.4s, v0.4h, v2.h[3]                                         XX\
        SMLAL2 v15.4s, v0.8h, v2.h[3]                                        XX\
        SMLAL v16.4s, v0.4h, v2.h[4]                                         XX\
        SMLAL2 v17.4s, v0.8h, v2.h[4]                                        XX\
        SMLAL v18.4s, v0.4h, v2.h[5]                                         XX\
        SMLAL2 v19.4s, v0.8h, v2.h[5]                                        XX\
        SMLAL v20.4s, v0.4h, v2.h[6]                                         XX\
        SMLAL2 v21.4s, v0.8h, v2.h[6]                                        XX\
        SMLAL v22.4s, v0.4h, v2.h[7]                                         XX\
        SMLAL2 v23.4s, v0.8h, v2.h[7]                                        XX\
                                                                             XX\
        NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_4                          XX\
    _3_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        /* row 0: v8, v9 */                                                  XX\
        /* row 1: v10, v11 */                                                XX\
        /* row 2: v12, v13 */                                                XX\
        /* row 3: v14, v15 */                                                XX\
        /* row 4: v16, v17 */                                                XX\
        /* row 5: v18, v19 */                                                XX\
        /* row 6: v20, v21 */                                                XX\
        /* row 7: v22, v23 */                                                XX\
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
        FMUL v9.4s, v9.4s, v30.4s                                            XX\
        FMUL v10.4s, v10.4s, v26.4s                                          XX\
        FMUL v11.4s, v11.4s, v30.4s                                          XX\
        FMUL v12.4s, v12.4s, v26.4s                                          XX\
        FMUL v13.4s, v13.4s, v30.4s                                          XX\
        FMUL v14.4s, v14.4s, v26.4s                                          XX\
        FMUL v15.4s, v15.4s, v30.4s                                          XX\
        FMUL v16.4s, v16.4s, v26.4s                                          XX\
        FMUL v17.4s, v17.4s, v30.4s                                          XX\
        FMUL v18.4s, v18.4s, v26.4s                                          XX\
        FMUL v19.4s, v19.4s, v30.4s                                          XX\
        FMUL v20.4s, v20.4s, v26.4s                                          XX\
        FMUL v21.4s, v21.4s, v30.4s                                          XX\
        FMUL v22.4s, v22.4s, v26.4s                                          XX\
        FMUL v23.4s, v23.4s, v30.4s                                          XX\
                                                                             XX\
        FADD v8.4s, v8.4s, v24.4s                                            XX\
        FADD v9.4s, v9.4s, v25.4s                                            XX\
        FADD v10.4s, v10.4s, v24.4s                                          XX\
        FADD v11.4s, v11.4s, v25.4s                                          XX\
        FADD v12.4s, v12.4s, v24.4s                                          XX\
        FADD v13.4s, v13.4s, v25.4s                                          XX\
        FADD v14.4s, v14.4s, v24.4s                                          XX\
        FADD v15.4s, v15.4s, v25.4s                                          XX\
        FADD v16.4s, v16.4s, v24.4s                                          XX\
        FADD v17.4s, v17.4s, v25.4s                                          XX\
        FADD v18.4s, v18.4s, v24.4s                                          XX\
        FADD v19.4s, v19.4s, v25.4s                                          XX\
        FADD v20.4s, v20.4s, v24.4s                                          XX\
        FADD v21.4s, v21.4s, v25.4s                                          XX\
        FADD v22.4s, v22.4s, v24.4s                                          XX\
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
        ST1 {v9.4s}, [x7]                                                    XX\
        ST1 {v10.4s}, [x9], 16                                               XX\
        ST1 {v11.4s}, [x9]                                                   XX\
        ST1 {v12.4s}, [x10], 16                                              XX\
        ST1 {v13.4s}, [x10]                                                  XX\
        ST1 {v14.4s}, [x8], 16                                               XX\
        ST1 {v15.4s}, [x8]                                                   XX\
        ST1 {v16.4s}, [x12], 16                                              XX\
        ST1 {v17.4s}, [x12]                                                  XX\
        ST1 {v18.4s}, [x13], 16                                              XX\
        ST1 {v19.4s}, [x13]                                                  XX\
        ST1 {v20.4s}, [x14], 16                                              XX\
        ST1 {v21.4s}, [x14]                                                  XX\
        ST1 {v22.4s}, [x15], 16                                              XX\
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
        ST1 {v16.4s}, [x12], 16                                              XX\
        ST1 {v18.4s}, [x13], 16                                              XX\
        ST1 {v20.4s}, [x14], 16                                              XX\
        ST1 {v22.4s}, [x15], 16                                              XX\
                                                                             XX\
        SUB x11, x11, 4                                                      XX\
                                                                             XX\
        MOV v8.16b, v9.16b                                                   XX\
        MOV v10.16b, v11.16b                                                 XX\
        MOV v12.16b, v13.16b                                                 XX\
        MOV v14.16b, v15.16b                                                 XX\
        MOV v16.16b, v17.16b                                                 XX\
        MOV v18.16b, v19.16b                                                 XX\
        MOV v20.16b, v21.16b                                                 XX\
        MOV v22.16b, v23.16b                                                 XX\
                                                                             XX\
    _5_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        CMP x11, 2                                                           XX\
        B.LO _6_w##W_INDEX_DTYPE_NUM_BITS                                    XX\
                                                                             XX\
        ST1 {v8.2s}, [x7], 8                                                 XX\
        ST1 {v10.2s}, [x9], 8                                                XX\
        ST1 {v12.2s}, [x10], 8                                               XX\
        ST1 {v14.2s}, [x8], 8                                                XX\
        ST1 {v16.2s}, [x12], 8                                               XX\
        ST1 {v18.2s}, [x13], 8                                               XX\
        ST1 {v20.2s}, [x14], 8                                               XX\
        ST1 {v22.2s}, [x15], 8                                               XX\
                                                                             XX\
        SUB x11, x11, 2                                                      XX\
                                                                             XX\
        EXT v8.16b, v8.16b, v8.16b, 8                                        XX\
        EXT v10.16b, v10.16b, v10.16b, 8                                     XX\
        EXT v12.16b, v12.16b, v12.16b, 8                                     XX\
        EXT v14.16b, v14.16b, v14.16b, 8                                     XX\
        EXT v16.16b, v16.16b, v16.16b, 8                                     XX\
        EXT v18.16b, v18.16b, v18.16b, 8                                     XX\
        EXT v20.16b, v20.16b, v20.16b, 8                                     XX\
        EXT v22.16b, v22.16b, v22.16b, 8                                     XX\
                                                                             XX\
    _6_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        CMP x11, 1                                                           XX\
        B.LO _7_w##W_INDEX_DTYPE_NUM_BITS                                    XX\
                                                                             XX\
        ST1 {v8.s}[0], [x7]                                                  XX\
        ST1 {v10.s}[0], [x9]                                                 XX\
        ST1 {v12.s}[0], [x10]                                                XX\
        ST1 {v14.s}[0], [x8]                                                 XX\
        ST1 {v16.s}[0], [x12]                                                XX\
        ST1 {v18.s}[0], [x13]                                                XX\
        ST1 {v20.s}[0], [x14]                                                XX\
        ST1 {v22.s}[0], [x15]                                                XX\
                                                                             XX\
    _7_w##W_INDEX_DTYPE_NUM_BITS##:                                          XX\
        LDP d9, d8, [sp, -64]                                                XX\
        LDP d11, d10, [sp, -48]                                              XX\
        LDP d13, d12, [sp, -32]                                              XX\
        LDP d15, d14, [sp, -16]                                              XX\
                                                                             XX\
        RET                                                                  XX\
                                                                             XX\
    END_FUNCTION pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w##W_INDEX_DTYPE_NUM_BITS##__aarch64_neon

# void pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w32__aarch64_neon(
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
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_8X1_UKERNEL_8X8_PACKEDA__AARCH64_NEON(32, #4, #2, LDR)

# void pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w16__aarch64_neon(
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
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_8X1_UKERNEL_8X8_PACKEDA__AARCH64_NEON(16, #2, #1, LDRH)

# void pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w8__aarch64_neon(
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
MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_8X1_UKERNEL_8X8_PACKEDA__AARCH64_NEON(8, #1, #0, LDRB)

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

#undef NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_5
#undef NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_4
#undef NDEF_IGNORE_CODE_ALIGN_DIRECTIVES_P2ALIGN_3
#undef MAKE_PYTORCH_Q8GEMM_DQ_SPARSE_8X1_UKERNEL_8X8_PACKEDA__AARCH64_NEON
#undef XX
