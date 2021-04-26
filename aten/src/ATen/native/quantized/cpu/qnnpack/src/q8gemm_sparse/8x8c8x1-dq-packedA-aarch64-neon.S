/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>

# params
# c_stride

#  Args passed via stack.
#  TOS
#  |-----------|
#  |c_stride   | 0
#  |out ch indx| 8
#  |params     | 16
#  |-----------|

# void pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA__aarch32_neon(
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
BEGIN_FUNCTION pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA__aarch64_neon

    STP d15, d14, [sp, -16]
    STP d13, d12, [sp, -32]
    STP d11, d10, [sp, -48]
    STP d9, d8, [sp, -64]

    MOV x11, x1
    # Load output channel index
    LDR x10, [sp, 8]
    # Load params
    LDR x8, [sp, 16]

    # Load a_zero_point
    LD1R {v24.8b}, [x8]
    ADD x8, x8, 8

    # Load pointer to per channel zero points array
    LDR x17, [x8], 8

    # Load pointer to per channel multiplier
    LDR x13, [x8]

    # Add offset to the base pointer
    ADD x17, x17, x10
    # Mul by 4 to get byte offset for multiplier
    LSL x10, x10, 2
    # Add offset to the base pointer for multiplier
    ADD x13, x13, x10

    # Load b_zero_point
    LD1 {v25.8b}, [x17]
    # Load multiplier c0123
    LD1 {v26.4s}, [x13], 16
    # Load multiplier c4567
    LD1 {v30.4s}, [x13]

    EOR x12, x12, x12
    EOR x13, x13, x13

    EOR v8.16b, v8.16b, v8.16b
    EOR v9.16b, v9.16b, v9.16b
    EOR v10.16b, v10.16b, v10.16b
    EOR v11.16b, v11.16b, v11.16b
    EOR v12.16b, v12.16b, v12.16b
    EOR v13.16b, v13.16b, v13.16b
    EOR v14.16b, v14.16b, v14.16b
    EOR v15.16b, v15.16b, v15.16b
    EOR v16.16b, v16.16b, v16.16b
    EOR v17.16b, v17.16b, v17.16b
    EOR v18.16b, v18.16b, v18.16b
    EOR v19.16b, v19.16b, v19.16b
    EOR v20.16b, v20.16b, v20.16b
    EOR v21.16b, v21.16b, v21.16b
    EOR v22.16b, v22.16b, v22.16b
    EOR v23.16b, v23.16b, v23.16b

    # w12 = w_row_ptr[n], x13 = w_row_ptr[n+1]
    # x4 = x4 + 4 to point to next n
    LDR w12, [x4], #4
    LDR w13, [x4]
    # x10 = temp_packed_w = packed_w + w_row_ptr[n] * 8
    # This points to the first block of nonzero value
    # for the nth row.
    ADD x10, x3, x12, LSL #3
    # x9 = temp_w_block_ids_ptr = w_block_ids_ptr (x5) + w_row_ptr[n]
    # LSL2 because each element is 4 bytes
    # This points to the block id of the first block
    # It should contain x13 - x12 number of block ids
    ADD x9, x5, x12, LSL #2
    # x8 = num_blocks that needs to be processed
    SUB x8, x13, x12
    SUBS x8, x8, 2
    B.LO 1f

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 5
#endif
k_loop:
    # k_loop processes two k values
    # Load two 8x1 blocks
    LD1 {v0.8b}, [x10], 8
    LD1 {v1.8b}, [x10], 8
    USUBL v0.8h, v0.8b, v25.8b
    USUBL v1.8h, v1.8b, v25.8b

    #x12 = block_id_ptr[0]
    #x13 = block_id_ptr[1]
    LDR w12, [x9], #4
    LDR w13, [x9], #4
    # Add offset to x2
    # Shift by 3 because each packed block is a block of 8x1
    # which 8 bytes
    ADD x16, x2, x12, LSL #3
    ADD x17, x2, x13, LSL #3

    # Load two 8x1 blocks of activation
    # First 8x1 for first channel
    # second 8x1 for next channel
    LD1 {v2.8b}, [x16]
    LD1 {v3.8b}, [x17]

    USUBL v2.8h, v2.8b, v24.8b
    USUBL v3.8h, v3.8b, v24.8b

    # First channel
    SMLAL v8.4s, v0.4h, v2.h[0]
    SMLAL2 v9.4s, v0.8h, v2.h[0]
    SMLAL v10.4s, v0.4h, v2.h[1]
    SMLAL2 v11.4s, v0.8h, v2.h[1]
    SMLAL v12.4s, v0.4h, v2.h[2]
    SMLAL2 v13.4s, v0.8h, v2.h[2]
    SMLAL v14.4s, v0.4h, v2.h[3]
    SMLAL2 v15.4s, v0.8h, v2.h[3]
    SMLAL v16.4s, v0.4h, v2.h[4]
    SMLAL2 v17.4s, v0.8h, v2.h[4]
    SMLAL v18.4s, v0.4h, v2.h[5]
    SMLAL2 v19.4s, v0.8h, v2.h[5]
    SMLAL v20.4s, v0.4h, v2.h[6]
    SMLAL2 v21.4s, v0.8h, v2.h[6]
    SMLAL v22.4s, v0.4h, v2.h[7]
    SMLAL2 v23.4s, v0.8h, v2.h[7]

    SUBS x8, x8, 2
    # Second channel
    SMLAL v8.4s, v1.4h, v3.h[0]
    SMLAL2 v9.4s, v1.8h, v3.h[0]
    SMLAL v10.4s, v1.4h, v3.h[1]
    SMLAL2 v11.4s, v1.8h, v3.h[1]
    SMLAL v12.4s, v1.4h, v3.h[2]
    SMLAL2 v13.4s, v1.8h, v3.h[2]
    SMLAL v14.4s, v1.4h, v3.h[3]
    SMLAL2 v15.4s, v1.8h, v3.h[3]
    SMLAL v16.4s, v1.4h, v3.h[4]
    SMLAL2 v17.4s, v1.8h, v3.h[4]
    SMLAL v18.4s, v1.4h, v3.h[5]
    SMLAL2 v19.4s, v1.8h, v3.h[5]
    SMLAL v20.4s, v1.4h, v3.h[6]
    SMLAL2 v21.4s, v1.8h, v3.h[6]
    SMLAL v22.4s, v1.4h, v3.h[7]
    SMLAL2 v23.4s, v1.8h, v3.h[7]

    B.HS k_loop

1:
    CMP x8, -2
    B.EQ 3f

    LD1 {v0.8b}, [x10]
    USUBL v0.8h, v0.8b, v25.8b

    #x12 = block_id_ptr[0]
    LDR w12, [x9]
    # Add offset to x2
    ADD x16, x2, x12, LSL #3

    LD1 {v2.8b}, [x16]
    USUBL v2.8h, v2.8b, v24.8b

    SMLAL v8.4s, v0.4h, v2.h[0]
    SMLAL2 v9.4s, v0.8h, v2.h[0]
    SMLAL v10.4s, v0.4h, v2.h[1]
    SMLAL2 v11.4s, v0.8h, v2.h[1]
    SMLAL v12.4s, v0.4h, v2.h[2]
    SMLAL2 v13.4s, v0.8h, v2.h[2]
    SMLAL v14.4s, v0.4h, v2.h[3]
    SMLAL2 v15.4s, v0.8h, v2.h[3]
    SMLAL v16.4s, v0.4h, v2.h[4]
    SMLAL2 v17.4s, v0.8h, v2.h[4]
    SMLAL v18.4s, v0.4h, v2.h[5]
    SMLAL2 v19.4s, v0.8h, v2.h[5]
    SMLAL v20.4s, v0.4h, v2.h[6]
    SMLAL2 v21.4s, v0.8h, v2.h[6]
    SMLAL v22.4s, v0.4h, v2.h[7]
    SMLAL2 v23.4s, v0.8h, v2.h[7]

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 4
#endif
3:
    # row 0: v8, v9
    # row 1: v10, v11
    # row 2: v12, v13
    # row 3: v14, v15
    # row 4: v16, v17
    # row 5: v18, v19
    # row 6: v20, v21
    # row 7: v22, v23

    # Load c_stride & params
    LDR x16, [sp]
    LSL x16, x16, 2
    LD1 {v24.4s}, [x6], 16
    LD1 {v25.4s}, [x6]

    SCVTF v8.4s, v8.4s
    SCVTF v9.4s, v9.4s
    SCVTF v10.4s, v10.4s
    SCVTF v11.4s, v11.4s
    SCVTF v12.4s, v12.4s
    SCVTF v13.4s, v13.4s
    SCVTF v14.4s, v14.4s
    SCVTF v15.4s, v15.4s
    SCVTF v16.4s, v16.4s
    SCVTF v17.4s, v17.4s
    SCVTF v18.4s, v18.4s
    SCVTF v19.4s, v19.4s
    SCVTF v20.4s, v20.4s
    SCVTF v21.4s, v21.4s
    SCVTF v22.4s, v22.4s
    SCVTF v23.4s, v23.4s

    FMUL v8.4s, v8.4s, v26.4s
    FMUL v9.4s, v9.4s, v30.4s
    FMUL v10.4s, v10.4s, v26.4s
    FMUL v11.4s, v11.4s, v30.4s
    FMUL v12.4s, v12.4s, v26.4s
    FMUL v13.4s, v13.4s, v30.4s
    FMUL v14.4s, v14.4s, v26.4s
    FMUL v15.4s, v15.4s, v30.4s
    FMUL v16.4s, v16.4s, v26.4s
    FMUL v17.4s, v17.4s, v30.4s
    FMUL v18.4s, v18.4s, v26.4s
    FMUL v19.4s, v19.4s, v30.4s
    FMUL v20.4s, v20.4s, v26.4s
    FMUL v21.4s, v21.4s, v30.4s
    FMUL v22.4s, v22.4s, v26.4s
    FMUL v23.4s, v23.4s, v30.4s

    FADD v8.4s, v8.4s, v24.4s
    FADD v9.4s, v9.4s, v25.4s
    FADD v10.4s, v10.4s, v24.4s
    FADD v11.4s, v11.4s, v25.4s
    FADD v12.4s, v12.4s, v24.4s
    FADD v13.4s, v13.4s, v25.4s
    FADD v14.4s, v14.4s, v24.4s
    FADD v15.4s, v15.4s, v25.4s
    FADD v16.4s, v16.4s, v24.4s
    FADD v17.4s, v17.4s, v25.4s
    FADD v18.4s, v18.4s, v24.4s
    FADD v19.4s, v19.4s, v25.4s
    FADD v20.4s, v20.4s, v24.4s
    FADD v21.4s, v21.4s, v25.4s
    FADD v22.4s, v22.4s, v24.4s
    FADD v23.4s, v23.4s, v25.4s

    // Compute c0-c7

    ADD  x9, x7, x16
    CMP x0, 2
    CSEL x9, x7, x9, LO

    ADD x10, x9,  x16
    CSEL x10, x9, x10, LS

    ADD x8, x10, x16
    CMP x0, 4
    CSEL x8, x10, x8, LO

    ADD x12, x8, x16
    CSEL x12, x8, x12, LS

    ADD x13, x12, x16
    CMP x0, 6
    CSEL x13, x12, x13, LO

    ADD x14, x13, x16
    CSEL x14, x13, x14, LS

    ADD x15, x14, x16
    CMP x0, 8
    CSEL x15, x14, x15, NE

    CMP x11, 8
    B.NE 4f

    ST1 {v8.4s}, [x7], 16
    ST1 {v9.4s}, [x7]
    ST1 {v10.4s}, [x9], 16
    ST1 {v11.4s}, [x9]
    ST1 {v12.4s}, [x10], 16
    ST1 {v13.4s}, [x10]
    ST1 {v14.4s}, [x8], 16
    ST1 {v15.4s}, [x8]
    ST1 {v16.4s}, [x12], 16
    ST1 {v17.4s}, [x12]
    ST1 {v18.4s}, [x13], 16
    ST1 {v19.4s}, [x13]
    ST1 {v20.4s}, [x14], 16
    ST1 {v21.4s}, [x14]
    ST1 {v22.4s}, [x15], 16
    ST1 {v23.4s}, [x15]

    LDP d9, d8, [sp, -64]
    LDP d11, d10, [sp, -48]
    LDP d13, d12, [sp, -32]
    LDP d15, d14, [sp, -16]

    RET

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 3
#endif
4:
    CMP x11, 4
    B.LO 5f

    ST1 {v8.4s}, [x7], 16
    ST1 {v10.4s}, [x9], 16
    ST1 {v12.4s}, [x10], 16
    ST1 {v14.4s}, [x8], 16
    ST1 {v16.4s}, [x12], 16
    ST1 {v18.4s}, [x13], 16
    ST1 {v20.4s}, [x14], 16
    ST1 {v22.4s}, [x15], 16

    SUB x11, x11, 4

    MOV v8.16b, v9.16b
    MOV v10.16b, v11.16b
    MOV v12.16b, v13.16b
    MOV v14.16b, v15.16b
    MOV v16.16b, v17.16b
    MOV v18.16b, v19.16b
    MOV v20.16b, v21.16b
    MOV v22.16b, v23.16b

5:
    CMP x11, 2
    B.LO 6f

    ST1 {v8.2s}, [x7], 8
    ST1 {v10.2s}, [x9], 8
    ST1 {v12.2s}, [x10], 8
    ST1 {v14.2s}, [x8], 8
    ST1 {v16.2s}, [x12], 8
    ST1 {v18.2s}, [x13], 8
    ST1 {v20.2s}, [x14], 8
    ST1 {v22.2s}, [x15], 8

    SUB x11, x11, 2

    EXT v8.16b, v8.16b, v8.16b, 8
    EXT v10.16b, v10.16b, v10.16b, 8
    EXT v12.16b, v12.16b, v12.16b, 8
    EXT v14.16b, v14.16b, v14.16b, 8
    EXT v16.16b, v16.16b, v16.16b, 8
    EXT v18.16b, v18.16b, v18.16b, 8
    EXT v20.16b, v20.16b, v20.16b, 8
    EXT v22.16b, v22.16b, v22.16b, 8

6:
    CMP x11, 1
    B.LO 7f

    ST1 {v8.s}[0], [x7]
    ST1 {v10.s}[0], [x9]
    ST1 {v12.s}[0], [x10]
    ST1 {v14.s}[0], [x8]
    ST1 {v16.s}[0], [x12]
    ST1 {v18.s}[0], [x13]
    ST1 {v20.s}[0], [x14]
    ST1 {v22.s}[0], [x15]

7:
    LDP d9, d8, [sp, -64]
    LDP d11, d10, [sp, -48]
    LDP d13, d12, [sp, -32]
    LDP d15, d14, [sp, -16]

    RET

END_FUNCTION pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA__aarch64_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
