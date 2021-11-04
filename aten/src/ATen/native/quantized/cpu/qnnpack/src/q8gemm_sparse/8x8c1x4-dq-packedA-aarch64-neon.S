/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>

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

# void pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon(
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
BEGIN_FUNCTION pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA__aarch64_neon

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

    CMP x1, 1
    B.LO 7f

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 5
#endif
0:
    # v8 := zero
    EOR v8.16b, v8.16b, v8.16b
    # v9 := zero
    EOR v9.16b, v9.16b, v9.16b

    DUP v29.8b, v25.b[0]
    # w12 = w_row_ptr[n], x13 = w_row_ptr[n+1]
    # x4 = x4 + 4 to point to next n
    LDR w12, [x4], #4
    LDR w13, [x4]
    # x10 = temp_packed_w = packed_w + w_row_ptr[n] * 4
    # This points to the first block of nonzero value
    # for the nth row.
    ADD x10, x3, x12, LSL #2
    # x9 = temp_w_block_ids_ptr = w_block_ids_ptr (x5) + w_row_ptr[n]
    # LSL2 because each element is 4 bytes
    # This points to the block id of the first block
    # It should contain x13 - x12 number of block ids
    ADD x9, x5, x12, LSL #2
    # x8 = num_blocks that needs to be processed
    SUB x8, x13, x12
    SUBS x8, x8, 2
    B.LO 1f

k_loop:
    // b0-7 (channel 0)
    LD1 {v10.8b}, [x10], 8
    USUBL v10.8h, v10.8b, v29.8b

    #x12 = block_id_ptr[0]
    #x13 = block_id_ptr[1]
    LDR w12, [x9], #4
    LDR w13, [x9], #4
    # Add offset to x2
    # Shift by 5 because each packed block is a block of 8x4
    # which 32 bytes
    ADD x16, x2, x12, LSL #5
    ADD x17, x2, x13, LSL #5

    LD1 {v0.8b}, [x16], 8
    LD1 {v1.8b}, [x16], 8
    LD1 {v2.8b}, [x16], 8
    LD1 {v3.8b}, [x16]
    LD1 {v4.8b}, [x17], 8
    LD1 {v5.8b}, [x17], 8
    LD1 {v6.8b}, [x17], 8
    LD1 {v7.8b}, [x17]

    USUBL v0.8h, v0.8b, v24.8b
    USUBL v1.8h, v1.8b, v24.8b
    USUBL v2.8h, v2.8b, v24.8b
    USUBL v3.8h, v3.8b, v24.8b
    USUBL v4.8h, v4.8b, v24.8b
    USUBL v5.8h, v5.8b, v24.8b
    USUBL v6.8h, v6.8b, v24.8b
    USUBL v7.8h, v7.8b, v24.8b

    SMLAL v8.4s, v0.4h, v10.h[0]
    SMLAL2 v9.4s, v0.8h, v10.h[0]
    SMLAL v8.4s, v1.4h, v10.h[1]
    SMLAL2 v9.4s, v1.8h, v10.h[1]
    SMLAL v8.4s, v2.4h, v10.h[2]
    SMLAL2 v9.4s, v2.8h, v10.h[2]
    SMLAL v8.4s, v3.4h, v10.h[3]
    SMLAL2 v9.4s, v3.8h, v10.h[3]
    SMLAL v8.4s, v4.4h, v10.h[4]
    SMLAL2 v9.4s, v4.8h, v10.h[4]
    SMLAL v8.4s, v5.4h, v10.h[5]
    SMLAL2 v9.4s, v5.8h, v10.h[5]
    SMLAL v8.4s, v6.4h, v10.h[6]
    SMLAL2 v9.4s, v6.8h, v10.h[6]
    SUBS x8, x8, 2
    SMLAL v8.4s, v7.4h, v10.h[7]
    SMLAL2 v9.4s, v7.8h, v10.h[7]


    B.HS k_loop

1:
    CMP x8, -2
    B.EQ 2f

    // b0-7 (channel 0)
    LD1R {v10.4s}, [x10]
    USUBL v10.8h, v10.8b, v29.8b

    #x12 = block_id_ptr[0]
    LDR w12, [x9]
    # Add offset to x2
    # Shift by 5 because each packed block is a block of 8x4
    # which 32 bytes
    ADD x16, x2, x12, LSL #5

    LD1 {v0.8b}, [x16], 8
    LD1 {v1.8b}, [x16], 8
    LD1 {v2.8b}, [x16], 8
    LD1 {v3.8b}, [x16]

    USUBL v0.8h, v0.8b, v24.8b
    USUBL v1.8h, v1.8b, v24.8b
    USUBL v2.8h, v2.8b, v24.8b
    USUBL v3.8h, v3.8b, v24.8b

    SMLAL v8.4s, v0.4h, v10.h[0]
    SMLAL2 v9.4s, v0.8h, v10.h[0]
    SMLAL v8.4s, v1.4h, v10.h[1]
    SMLAL2 v9.4s, v1.8h, v10.h[1]
    SMLAL v8.4s, v2.4h, v10.h[2]
    SMLAL2 v9.4s, v2.8h, v10.h[2]
    SMLAL v8.4s, v3.4h, v10.h[3]
    SMLAL2 v9.4s, v3.8h, v10.h[3]

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 4
#endif
2:
    # Store result on stack

    # -64 because all d8-d15 are on stack
    # + 256 bytes of buffer when nr = 1
    # 256 because we are doing 8x8 block with each value being 4 bytes
    # Thus 64 * 4 = 256
    # 256 + 64 = 320
    # This is needed because after processing all nrs we will
    # load 256  bytes from stack.
    # Thus we will load accumulators back in v8, v9, v10, v11, v12, v13, v14, v15
    # v16, v17, v18, v19, v20, v21, v22, v23
    # When nr < 8, say nr = 1, extra v values will be fetched from stack which may overlap
    # with other parts of stack storing local variables. To avoid that we just
    # create a buffer of 256 bytes inbetween to make sure pointer increment
    # never produces address that is beyond the stack frame of this function.
    SUB x9, sp, 320
    # Each iteration produce 8 values each of 4 bytes
    # Thus 8 x 4 = 32 bytes 2^5
    # In this implementation, first value will be stored at
    # 1st value: sp - 64 - r1 * 32
    # 2nd value: sp - 12 - (r1 - 1) * 32
    # and so on.
    SUB x9, x9, x1, LSL #5
    ST1 {v8.4s}, [x9], 16
    ST1 {v9.4s}, [x9]

    # Shift zero point vector by 8 to load
    # zero point of the next channel
    SRI v25.2d, v25.2d, #8
    # Check if nr >=1
    SUBS x1, x1, 1
    BHI 0b
3:
    # First load all the accumulators from stack
    # Load nr
    SUB x9, sp, 320
    SUB x9, x9, x11, LSL #5
    # Now load v8-v15
    # This is 8x4 block (nrxmr)
    # We will transpose this to 4x8 (mrxnr)
    # v8, v9   : x00, x10, x20, x30; x40, x50, x60, x70
    # v10, v11 : x01, x11, x21, x31; x41, x51, x61, x71
    # v12, v13 : x02, x12, x22, x32; x42, x52, x62, x72
    # v14, v15 : x03, x13, x23, x33; x43, x53, x63, x73
    #
    # v16, v17 : x04, x14, x24, x34; x44, x54, x64, x74
    # v18, v19 : x05, x15, x25, x35; x45, x55, x65, x75
    # v20, v21 : x06, x16, x26, x36; x46, x56, x66, x76
    # v22, v23 : x07, x17, x27, x37; x47, x57, x67, x77
    LD1 {v8.4s}, [x9], 16
    LD1 {v9.4s}, [x9], 16
    LD1 {v10.4s}, [x9], 16
    LD1 {v11.4s}, [x9], 16
    LD1 {v12.4s}, [x9], 16
    LD1 {v13.4s}, [x9], 16
    LD1 {v14.4s}, [x9], 16
    LD1 {v15.4s}, [x9], 16
    LD1 {v16.4s}, [x9], 16
    LD1 {v17.4s}, [x9], 16
    LD1 {v18.4s}, [x9], 16
    LD1 {v19.4s}, [x9], 16
    LD1 {v20.4s}, [x9], 16
    LD1 {v21.4s}, [x9], 16
    LD1 {v22.4s}, [x9], 16
    LD1 {v23.4s}, [x9]

    # We can tranpose one 4x4 block using macro
    # TRANSPOSE_4X4_S32 v8, v10, v12, v14, v0, v1, v2, v3
    # After this we have
    # v8  : x00, x01, x02, x03
    # v10 : x10, x11, x12, x13
    # v12 : x20, x21, x22, x23
    # v14 : x30, x31, x32, x33
    # Then using
    # TRANSPOSE_4X4_S32 v16, v18, v20, v22, v4, v5, v6, v7
    # We get
    # v16 : x04, x05, x06, x07
    # v18 : x14, x15, x16, x17
    # v20 : x24, x25, x26, x27
    # v22 : x34, x35, x36, x37
    # Similarly we can transpose other two 4x4 blocks and we get
    # tranposed 8x8

    TRANSPOSE_4X4_S32 v8, v10, v12, v14, v0, v1, v2, v3
    TRANSPOSE_4X4_S32 v16, v18, v20, v22, v4, v5, v6, v7
    TRANSPOSE_4X4_S32 v9, v11, v13, v15, v0, v1, v2, v3
    TRANSPOSE_4X4_S32 v17, v19, v21, v23, v4, v5, v6, v7

    # row 0: v8, v16
    # row 1: v10, v18
    # row 2: v12, v20
    # row 3: v14, v22
    # row 4: v9, v17
    # row 5: v11, v19
    # row 6: v13, v21
    # row 7: v15, v23

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
    FMUL v16.4s, v16.4s, v30.4s
    FMUL v10.4s, v10.4s, v26.4s
    FMUL v18.4s, v18.4s, v30.4s
    FMUL v12.4s, v12.4s, v26.4s
    FMUL v20.4s, v20.4s, v30.4s
    FMUL v14.4s, v14.4s, v26.4s
    FMUL v22.4s, v22.4s, v30.4s
    FMUL v9.4s, v9.4s, v26.4s
    FMUL v17.4s, v17.4s, v30.4s
    FMUL v11.4s, v11.4s, v26.4s
    FMUL v19.4s, v19.4s, v30.4s
    FMUL v13.4s, v13.4s, v26.4s
    FMUL v21.4s, v21.4s, v30.4s
    FMUL v15.4s, v15.4s, v26.4s
    FMUL v23.4s, v23.4s, v30.4s

    FADD v8.4s, v8.4s, v24.4s
    FADD v16.4s, v16.4s, v25.4s
    FADD v10.4s, v10.4s, v24.4s
    FADD v18.4s, v18.4s, v25.4s
    FADD v12.4s, v12.4s, v24.4s
    FADD v20.4s, v20.4s, v25.4s
    FADD v14.4s, v14.4s, v24.4s
    FADD v22.4s, v22.4s, v25.4s
    FADD v9.4s, v9.4s, v24.4s
    FADD v17.4s, v17.4s, v25.4s
    FADD v11.4s, v11.4s, v24.4s
    FADD v19.4s, v19.4s, v25.4s
    FADD v13.4s, v13.4s, v24.4s
    FADD v21.4s, v21.4s, v25.4s
    FADD v15.4s, v15.4s, v24.4s
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
    ST1 {v16.4s}, [x7]
    ST1 {v10.4s}, [x9], 16
    ST1 {v18.4s}, [x9]
    ST1 {v12.4s}, [x10], 16
    ST1 {v20.4s}, [x10]
    ST1 {v14.4s}, [x8], 16
    ST1 {v22.4s}, [x8]
    ST1 {v9.4s}, [x12], 16
    ST1 {v17.4s}, [x12]
    ST1 {v11.4s}, [x13], 16
    ST1 {v19.4s}, [x13]
    ST1 {v13.4s}, [x14], 16
    ST1 {v21.4s}, [x14]
    ST1 {v15.4s}, [x15], 16
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
    ST1 {v9.4s}, [x12], 16
    ST1 {v11.4s}, [x13], 16
    ST1 {v13.4s}, [x14], 16
    ST1 {v15.4s}, [x15], 16

    SUB x11, x11, 4

    MOV v8.16b, v16.16b
    MOV v10.16b, v18.16b
    MOV v12.16b, v20.16b
    MOV v14.16b, v22.16b
    MOV v9.16b, v17.16b
    MOV v11.16b, v19.16b
    MOV v13.16b, v21.16b
    MOV v15.16b, v23.16b

5:
    CMP x11, 2
    B.LO 6f

    ST1 {v8.2s}, [x7], 8
    ST1 {v10.2s}, [x9], 8
    ST1 {v12.2s}, [x10], 8
    ST1 {v14.2s}, [x8], 8
    ST1 {v9.2s}, [x12], 8
    ST1 {v11.2s}, [x13], 8
    ST1 {v13.2s}, [x14], 8
    ST1 {v15.2s}, [x15], 8

    SUB x11, x11, 2

    EXT v8.16b, v8.16b, v8.16b, 8
    EXT v10.16b, v10.16b, v10.16b, 8
    EXT v12.16b, v12.16b, v12.16b, 8
    EXT v14.16b, v14.16b, v14.16b, 8
    EXT v9.16b, v9.16b, v9.16b, 8
    EXT v11.16b, v11.16b, v11.16b, 8
    EXT v13.16b, v13.16b, v13.16b, 8
    EXT v15.16b, v15.16b, v15.16b, 8

6:
    CMP x11, 1
    B.LO 7f

    ST1 {v8.s}[0], [x7]
    ST1 {v10.s}[0], [x9]
    ST1 {v12.s}[0], [x10]
    ST1 {v14.s}[0], [x8]
    ST1 {v9.s}[0], [x12]
    ST1 {v11.s}[0], [x13]
    ST1 {v13.s}[0], [x14]
    ST1 {v15.s}[0], [x15]

7:
    LDP d9, d8, [sp, -64]
    LDP d11, d10, [sp, -48]
    LDP d13, d12, [sp, -32]
    LDP d15, d14, [sp, -16]

    RET

END_FUNCTION pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA__aarch64_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
