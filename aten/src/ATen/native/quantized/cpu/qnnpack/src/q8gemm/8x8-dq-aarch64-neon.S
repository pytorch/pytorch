/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>
#include <requantization/runtime-assembly.h>

# params
# c_stride

#  Args passed via stack.
#  TOS
#  |-----------|
#  |c_stride   | 0
#  |out ch indx| 8
#  |params     | 16
#  |-----------|

# void pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon(
#     size_t mr,
#     size_t nr,
#     size_t k,
#     const uint8_t*restrict a,
#     size_t a_stride,
#     const void*restrict w,
#     const float*restrict b,
#     uint8_t*restrict c,
#     size_t c_stride,
#     size_t output_channel_index,
#     const union pytorch_qnnp_conv_quantization_params quantization_params[restrict static 1])
BEGIN_FUNCTION pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon

    STP d15, d14, [sp, -16]
    STP d13, d12, [sp, -32]
    STP d11, d10, [sp, -48]
    STP d9, d8, [sp, -64]

    # Skip over bias0123, bias4567
    ADD x5, x5, 32

    # Load c_stride & params
    LDR x16, [sp]
    # Load output channel index
    LDR x10, [sp, 8]
    # Load params
    LDR x8, [sp, 16]

    # Load a_zero_point
    LD1R {v24.8b}, [x8]
    ADD x8, x8, 8

    # Load pointer to per channel zero points array
    LDR x17, [x8], 8

    # v8 := zero
    EOR v8.16b, v8.16b, v8.16b
    # v9 := zero
    EOR v9.16b, v9.16b, v9.16b

    # v10 := zero
    EOR v10.16b, v10.16b, v10.16b
    # v11 := zero
    EOR v11.16b, v11.16b, v11.16b

    # Load pointer to per channel multiplier
    LDR x13, [x8]

    # v12 := zero
    EOR v12.16b, v12.16b, v12.16b
    # v13 := zero
    EOR v13.16b, v13.16b, v13.16b

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

    # v14 := zero
    EOR v14.16b, v14.16b, v14.16b
    # v15 := zero
    EOR v15.16b, v15.16b, v15.16b

    # v16 := zero
    EOR v16.16b, v16.16b, v16.16b
    # v17 := zero
    EOR v17.16b, v17.16b, v17.16b

    # v18 := zero
    EOR v18.16b, v18.16b, v18.16b
    # v19 := zero
    EOR v19.16b, v19.16b, v19.16b

    # v20 := zero
    EOR v20.16b, v20.16b, v20.16b
    # v21 := zero
    EOR v21.16b, v21.16b, v21.16b

    # v22 := zero
    EOR v22.16b, v22.16b, v22.16b
    # v23 := zero
    EOR v23.16b, v23.16b, v23.16b

    # a1
    CMP x0, 2
    ADD x9, x3, x4
    CSEL x9, x3, x9, LO

    # a2
    ADD x10, x9, x4
    CSEL x10, x9, x10, LS

    # a3
    CMP x0, 4
    ADD x11, x10, x4
    CSEL x11, x10, x11, LO

    # a4
    ADD x12, x11, x4
    CSEL x12, x11, x12, LS

    # a5
    CMP x0, 6
    ADD x13, x12, x4
    CSEL x13, x12, x13, LO

    # a6
    ADD x14, x13, x4
    CSEL x14, x13, x14, LS

    # a7
    CMP x0, 8
    ADD x15, x14, x4
    CSEL x15, x14, x15, NE

    SUBS x2, x2, 8
    B.LO 1f

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 5
#endif
0:
    // b0-7 (channel 0)
    LD1 {v27.8b}, [x5], 8
    USUBL v27.8h, v27.8b, v25.8b

    # va0 - va7 := va - va_zero_point
    LD1 {v0.8b}, [x3], 8
    SUB_ZERO_POINT v0.8h, v0.8b, v24.8b
    LD1 {v1.8b}, [x9], 8
    SUB_ZERO_POINT v1.8h, v1.8b, v24.8b
    LD1 {v2.8b}, [x10], 8
    SUB_ZERO_POINT v2.8h, v2.8b, v24.8b
    LD1 {v3.8b}, [x11], 8
    SUB_ZERO_POINT v3.8h, v3.8b, v24.8b
    LD1 {v4.8b}, [x12], 8
    SUB_ZERO_POINT v4.8h, v4.8b, v24.8b
    LD1 {v5.8b}, [x13], 8
    SUB_ZERO_POINT v5.8h, v5.8b, v24.8b
    LD1 {v6.8b}, [x14], 8
    SUB_ZERO_POINT v6.8h, v6.8b, v24.8b
    LD1 {v7.8b}, [x15], 8
    SUB_ZERO_POINT v7.8h, v7.8b, v24.8b

    // b0-7 (channel 1)
    LD1 {v28.8b}, [x5], 8

    SMLAL v8.4s, v27.4h, v0.h[0]    // vacc0x0123 += vb0123 * va0[0]
    SMLAL2 v9.4s, v27.8h, v0.h[0]   // vacc0x4567 += vb4567 * va0[0]
    SMLAL v10.4s, v27.4h, v1.h[0]   // vacc1x0123 += vb0123 * va1[0]
    SMLAL2 v11.4s, v27.8h, v1.h[0]  // vacc1x4567 += vb4567 * va1[0]
    SMLAL v12.4s, v27.4h, v2.h[0]   // vacc2x0123 += vb0123 * va2[0]
    SMLAL2 v13.4s, v27.8h, v2.h[0]  // vacc2x4567 += vb4567 * va2[0]
    SMLAL v14.4s, v27.4h, v3.h[0]   // vacc3x0123 += vb0123 * va3[0]
    SMLAL2 v15.4s, v27.8h, v3.h[0]  // vacc3x4567 += vb4567 * va3[0]
    USUBL v28.8h, v28.8b, v25.8b
    SMLAL v16.4s, v27.4h, v4.h[0]   // vacc4x0123 += vb0123 * va4[0]
    SMLAL2 v17.4s, v27.8h, v4.h[0]  // vacc4x4567 += vb4567 * va4[0]
    SMLAL v18.4s, v27.4h, v5.h[0]   // vacc5x0123 += vb0123 * va5[0]
    SMLAL2 v19.4s, v27.8h, v5.h[0]  // vacc5x4567 += vb4567 * va5[0]
    SMLAL v20.4s, v27.4h, v6.h[0]   // vacc6x0123 += vb0123 * va6[0]
    SMLAL2 v21.4s, v27.8h, v6.h[0]  // vacc6x4567 += vb4567 * va6[0]
    SMLAL v22.4s, v27.4h, v7.h[0]   // vacc7x0123 += vb0123 * va7[0]
    SMLAL2 v23.4s, v27.8h, v7.h[0]  // vacc7x4567 += vb4567 * va7[0]

    // b0-7 (channel 2)
    LD1 {v27.8b}, [x5], 8

    SMLAL v8.4s, v28.4h, v0.h[1]    // vacc0x0123 += vb0123 * va0[1]
    SMLAL2 v9.4s, v28.8h, v0.h[1]   // vacc0x4567 += vb4567 * va0[1]
    SMLAL v10.4s, v28.4h, v1.h[1]   // vacc1x0123 += vb0123 * va1[1]
    SMLAL2 v11.4s, v28.8h, v1.h[1]  // vacc1x4567 += vb4567 * va1[1]
    SMLAL v12.4s, v28.4h, v2.h[1]   // vacc2x0123 += vb0123 * va2[1]
    SMLAL2 v13.4s, v28.8h, v2.h[1]  // vacc2x4567 += vb4567 * va2[1]
    SMLAL v14.4s, v28.4h, v3.h[1]   // vacc3x0123 += vb0123 * va3[1]
    SMLAL2 v15.4s, v28.8h, v3.h[1]  // vacc3x4567 += vb4567 * va3[1]
    USUBL v27.8h, v27.8b, v25.8b
    SMLAL v16.4s, v28.4h, v4.h[1]   // vacc4x0123 += vb0123 * va4[1]
    SMLAL2 v17.4s, v28.8h, v4.h[1]  // vacc4x4567 += vb4567 * va4[1]
    SMLAL v18.4s, v28.4h, v5.h[1]   // vacc5x0123 += vb0123 * va5[1]
    SMLAL2 v19.4s, v28.8h, v5.h[1]  // vacc5x4567 += vb4567 * va5[1]
    SMLAL v20.4s, v28.4h, v6.h[1]   // vacc6x0123 += vb0123 * va6[1]
    SMLAL2 v21.4s, v28.8h, v6.h[1]  // vacc6x4567 += vb4567 * va6[1]
    SMLAL v22.4s, v28.4h, v7.h[1]   // vacc7x0123 += vb0123 * va7[1]
    SMLAL2 v23.4s, v28.8h, v7.h[1]  // vacc7x4567 += vb4567 * va7[1]

    // b0-7 (channel 3)
    LD1 {v28.8b}, [x5], 8

    SMLAL v8.4s, v27.4h, v0.h[2]    // vacc0x0123 += vb0123 * va0[2]
    SMLAL2 v9.4s, v27.8h, v0.h[2]   // vacc0x4567 += vb4567 * va0[2]
    SMLAL v10.4s, v27.4h, v1.h[2]   // vacc1x0123 += vb0123 * va1[2]
    SMLAL2 v11.4s, v27.8h, v1.h[2]  // vacc1x4567 += vb4567 * va1[2]
    SMLAL v12.4s, v27.4h, v2.h[2]   // vacc2x0123 += vb0123 * va2[2]
    SMLAL2 v13.4s, v27.8h, v2.h[2]  // vacc2x4567 += vb4567 * va2[2]
    SMLAL v14.4s, v27.4h, v3.h[2]   // vacc3x0123 += vb0123 * va3[2]
    SMLAL2 v15.4s, v27.8h, v3.h[2]  // vacc3x4567 += vb4567 * va3[2]
    USUBL v28.8h, v28.8b, v25.8b
    SMLAL v16.4s, v27.4h, v4.h[2]   // vacc4x0123 += vb0123 * va4[2]
    SMLAL2 v17.4s, v27.8h, v4.h[2]  // vacc4x4567 += vb4567 * va4[2]
    SMLAL v18.4s, v27.4h, v5.h[2]   // vacc5x0123 += vb0123 * va5[2]
    SMLAL2 v19.4s, v27.8h, v5.h[2]  // vacc5x4567 += vb4567 * va5[2]
    SMLAL v20.4s, v27.4h, v6.h[2]   // vacc6x0123 += vb0123 * va6[2]
    SMLAL2 v21.4s, v27.8h, v6.h[2]  // vacc6x4567 += vb4567 * va6[2]
    SMLAL v22.4s, v27.4h, v7.h[2]   // vacc7x0123 += vb0123 * va7[2]
    SMLAL2 v23.4s, v27.8h, v7.h[2]  // vacc7x4567 += vb4567 * va7[2]

    // b0-7 (channel 4)
    LD1 {v27.8b}, [x5], 8

    SMLAL v8.4s, v28.4h, v0.h[3]    // vacc0x0123 += vb0123 * va0[3]
    SMLAL2 v9.4s, v28.8h, v0.h[3]   // vacc0x4567 += vb4567 * va0[3]
    SMLAL v10.4s, v28.4h, v1.h[3]   // vacc1x0123 += vb0123 * va1[3]
    SMLAL2 v11.4s, v28.8h, v1.h[3]  // vacc1x4567 += vb4567 * va1[3]
    SMLAL v12.4s, v28.4h, v2.h[3]   // vacc2x0123 += vb0123 * va2[3]
    SMLAL2 v13.4s, v28.8h, v2.h[3]  // vacc2x4567 += vb4567 * va2[3]
    SMLAL v14.4s, v28.4h, v3.h[3]   // vacc3x0123 += vb0123 * va3[3]
    SMLAL2 v15.4s, v28.8h, v3.h[3]  // vacc3x4567 += vb4567 * va3[3]
    USUBL v27.8h, v27.8b, v25.8b
    SMLAL v16.4s, v28.4h, v4.h[3]   // vacc4x0123 += vb0123 * va4[3]
    SMLAL2 v17.4s, v28.8h, v4.h[3]  // vacc4x4567 += vb4567 * va4[3]
    SMLAL v18.4s, v28.4h, v5.h[3]   // vacc5x0123 += vb0123 * va5[3]
    SMLAL2 v19.4s, v28.8h, v5.h[3]  // vacc5x4567 += vb4567 * va5[3]
    SMLAL v20.4s, v28.4h, v6.h[3]   // vacc6x0123 += vb0123 * va6[3]
    SMLAL2 v21.4s, v28.8h, v6.h[3]  // vacc6x4567 += vb4567 * va6[3]
    SMLAL v22.4s, v28.4h, v7.h[3]   // vacc7x0123 += vb0123 * va7[3]
    SMLAL2 v23.4s, v28.8h, v7.h[3]  // vacc7x4567 += vb4567 * va7[3]

    // b0-7 (channel 5)
    LD1 {v28.8b}, [x5], 8

    SMLAL v8.4s, v27.4h, v0.h[4]    // vacc0x0123 += vb0123 * va0[4]
    SMLAL2 v9.4s, v27.8h, v0.h[4]   // vacc0x4567 += vb4567 * va0[4]
    SMLAL v10.4s, v27.4h, v1.h[4]   // vacc1x0123 += vb0123 * va1[4]
    SMLAL2 v11.4s, v27.8h, v1.h[4]  // vacc1x4567 += vb4567 * va1[4]
    SMLAL v12.4s, v27.4h, v2.h[4]   // vacc2x0123 += vb0123 * va2[4]
    SMLAL2 v13.4s, v27.8h, v2.h[4]  // vacc2x4567 += vb4567 * va2[4]
    SMLAL v14.4s, v27.4h, v3.h[4]   // vacc3x0123 += vb0123 * va3[4]
    SMLAL2 v15.4s, v27.8h, v3.h[4]  // vacc3x4567 += vb4567 * va3[4]
    USUBL v28.8h, v28.8b, v25.8b
    SMLAL v16.4s, v27.4h, v4.h[4]   // vacc4x0123 += vb0123 * va4[4]
    SMLAL2 v17.4s, v27.8h, v4.h[4]  // vacc4x4567 += vb4567 * va4[4]
    SMLAL v18.4s, v27.4h, v5.h[4]   // vacc5x0123 += vb0123 * va5[4]
    SMLAL2 v19.4s, v27.8h, v5.h[4]  // vacc5x4567 += vb4567 * va5[4]
    SMLAL v20.4s, v27.4h, v6.h[4]   // vacc6x0123 += vb0123 * va6[4]
    SMLAL2 v21.4s, v27.8h, v6.h[4]  // vacc6x4567 += vb4567 * va6[4]
    SMLAL v22.4s, v27.4h, v7.h[4]   // vacc7x0123 += vb0123 * va7[4]
    SMLAL2 v23.4s, v27.8h, v7.h[4]  // vacc7x4567 += vb4567 * va7[4]

    // b0-7 (channel 6)
    LD1 {v27.8b}, [x5], 8

    SMLAL v8.4s, v28.4h, v0.h[5]    // vacc0x0123 += vb0123 * va0[5]
    SMLAL2 v9.4s, v28.8h, v0.h[5]   // vacc0x4567 += vb4567 * va0[5]
    SMLAL v10.4s, v28.4h, v1.h[5]   // vacc1x0123 += vb0123 * va1[5]
    SMLAL2 v11.4s, v28.8h, v1.h[5]  // vacc1x4567 += vb4567 * va1[5]
    SMLAL v12.4s, v28.4h, v2.h[5]   // vacc2x0123 += vb0123 * va2[5]
    SMLAL2 v13.4s, v28.8h, v2.h[5]  // vacc2x4567 += vb4567 * va2[5]
    SMLAL v14.4s, v28.4h, v3.h[5]   // vacc3x0123 += vb0123 * va3[5]
    SMLAL2 v15.4s, v28.8h, v3.h[5]  // vacc3x4567 += vb4567 * va3[5]
    USUBL v27.8h, v27.8b, v25.8b
    SMLAL v16.4s, v28.4h, v4.h[5]   // vacc4x0123 += vb0123 * va4[5]
    SMLAL2 v17.4s, v28.8h, v4.h[5]  // vacc4x4567 += vb4567 * va4[5]
    SMLAL v18.4s, v28.4h, v5.h[5]   // vacc5x0123 += vb0123 * va5[5]
    SMLAL2 v19.4s, v28.8h, v5.h[5]  // vacc5x4567 += vb4567 * va5[5]
    SMLAL v20.4s, v28.4h, v6.h[5]   // vacc6x0123 += vb0123 * va6[5]
    SMLAL2 v21.4s, v28.8h, v6.h[5]  // vacc6x4567 += vb4567 * va6[5]
    SMLAL v22.4s, v28.4h, v7.h[5]   // vacc7x0123 += vb0123 * va7[5]
    SMLAL2 v23.4s, v28.8h, v7.h[5]  // vacc7x4567 += vb4567 * va7[5]

    // b0-7 (channel 7)
    LD1 {v28.8b}, [x5], 8

    SMLAL v8.4s, v27.4h, v0.h[6]    // vacc0x0123 += vb0123 * va0[6]
    SMLAL2 v9.4s, v27.8h, v0.h[6]   // vacc0x4567 += vb4567 * va0[6]
    SMLAL v10.4s, v27.4h, v1.h[6]   // vacc1x0123 += vb0123 * va1[6]
    SMLAL2 v11.4s, v27.8h, v1.h[6]  // vacc1x4567 += vb4567 * va1[6]
    SMLAL v12.4s, v27.4h, v2.h[6]   // vacc2x0123 += vb0123 * va2[6]
    SMLAL2 v13.4s, v27.8h, v2.h[6]  // vacc2x4567 += vb4567 * va2[6]
    SMLAL v14.4s, v27.4h, v3.h[6]   // vacc3x0123 += vb0123 * va3[6]
    SMLAL2 v15.4s, v27.8h, v3.h[6]  // vacc3x4567 += vb4567 * va3[6]
    USUBL v28.8h, v28.8b, v25.8b
    SMLAL v16.4s, v27.4h, v4.h[6]   // vacc4x0123 += vb0123 * va4[6]
    SMLAL2 v17.4s, v27.8h, v4.h[6]  // vacc4x4567 += vb4567 * va4[6]
    SMLAL v18.4s, v27.4h, v5.h[6]   // vacc5x0123 += vb0123 * va5[6]
    SMLAL2 v19.4s, v27.8h, v5.h[6]  // vacc5x4567 += vb4567 * va5[6]
    SMLAL v20.4s, v27.4h, v6.h[6]   // vacc6x0123 += vb0123 * va6[6]
    SMLAL2 v21.4s, v27.8h, v6.h[6]  // vacc6x4567 += vb4567 * va6[6]
    SMLAL v22.4s, v27.4h, v7.h[6]   // vacc7x0123 += vb0123 * va7[6]
    SMLAL2 v23.4s, v27.8h, v7.h[6]  // vacc7x4567 += vb4567 * va7[6]

    SUBS x2, x2, 8

    SMLAL v8.4s, v28.4h, v0.h[7]    // vacc0x0123 += vb0123 * va0[7]
    SMLAL2 v9.4s, v28.8h, v0.h[7]   // vacc0x4567 += vb4567 * va0[7]
    SMLAL v10.4s, v28.4h, v1.h[7]   // vacc1x0123 += vb0123 * va1[7]
    SMLAL2 v11.4s, v28.8h, v1.h[7]  // vacc1x4567 += vb4567 * va1[7]
    SMLAL v12.4s, v28.4h, v2.h[7]   // vacc2x0123 += vb0123 * va2[7]
    SMLAL2 v13.4s, v28.8h, v2.h[7]  // vacc2x4567 += vb4567 * va2[7]
    SMLAL v14.4s, v28.4h, v3.h[7]   // vacc3x0123 += vb0123 * va3[7]
    SMLAL2 v15.4s, v28.8h, v3.h[7]  // vacc3x4567 += vb4567 * va3[7]
    SMLAL v16.4s, v28.4h, v4.h[7]   // vacc4x0123 += vb0123 * va4[7]
    SMLAL2 v17.4s, v28.8h, v4.h[7]  // vacc4x4567 += vb4567 * va4[7]
    SMLAL v18.4s, v28.4h, v5.h[7]   // vacc5x0123 += vb0123 * va5[7]
    SMLAL2 v19.4s, v28.8h, v5.h[7]  // vacc5x4567 += vb4567 * va5[7]
    SMLAL v20.4s, v28.4h, v6.h[7]   // vacc6x0123 += vb0123 * va6[7]
    SMLAL2 v21.4s, v28.8h, v6.h[7]  // vacc6x4567 += vb4567 * va6[7]
    SMLAL v22.4s, v28.4h, v7.h[7]   // vacc7x0123 += vb0123 * va7[7]
    SMLAL2 v23.4s, v28.8h, v7.h[7]  // vacc7x4567 += vb4567 * va7[7]

    B.HS 0b

1:
    CMP x2, -8
    B.EQ 2f

    // Adjust a0-a7
    ADD x3, x3, x2
    ADD x9, x9, x2
    ADD x10, x10, x2
    ADD x11, x11, x2
    ADD x12, x12, x2
    ADD x13, x13, x2
    ADD x14, x14, x2
    ADD x15, x15, x2

    // a_shift = 8 * k - 64
    LSL x2, x2, 3
    FMOV d29, x2
    USHL d24, d24, d29

    // Load x0-a7
    LD1 {v0.8b}, [x3], 8
    USHL d0, d0, d29
    SUB_ZERO_POINT v0.8h, v0.8b, v24.8b

    LD1 {v1.8b}, [x9], 8
    USHL d1, d1, d29
    SUB_ZERO_POINT v1.8h, v1.8b, v24.8b

    LD1 {v2.8b}, [x10], 8
    USHL d2, d2, d29
    SUB_ZERO_POINT v2.8h, v2.8b, v24.8b

    LD1 {v3.8b}, [x11], 8
    USHL d3, d3, d29
    SUB_ZERO_POINT v3.8h, v3.8b, v24.8b

    LD1 {v4.8b}, [x12], 8
    USHL d4, d4, d29
    SUB_ZERO_POINT v4.8h, v4.8b, v24.8b

    LD1 {v5.8b}, [x13], 8
    USHL d5, d5, d29
    SUB_ZERO_POINT v5.8h, v5.8b, v24.8b

    LD1 {v6.8b}, [x14], 8
    USHL d6, d6, d29
    SUB_ZERO_POINT v6.8h, v6.8b, v24.8b

    LD1 {v7.8b}, [x15], 8
    USHL d7, d7, d29
    SUB_ZERO_POINT v7.8h, v7.8b, v24.8b

    // Channel 0
    LD1 {v27.8b}, [x5], 8
    USUBL v27.8h, v27.8b, v25.8b

    SMLAL v8.4s, v27.4h, v0.h[0]    // vacc0x0123 += vb0123 * va0[0]
    SMLAL2 v9.4s, v27.8h, v0.h[0]   // vacc0x4567 += vb4567 * va0[0]
    SMLAL v10.4s, v27.4h, v1.h[0]   // vacc1x0123 += vb0123 * va1[0]
    SMLAL2 v11.4s, v27.8h, v1.h[0]  // vacc1x4567 += vb4567 * va1[0]
    SMLAL v12.4s, v27.4h, v2.h[0]   // vacc2x0123 += vb0123 * va2[0]
    SMLAL2 v13.4s, v27.8h, v2.h[0]  // vacc2x4567 += vb4567 * va2[0]
    SMLAL v14.4s, v27.4h, v3.h[0]   // vacc3x0123 += vb0123 * va3[0]
    SMLAL2 v15.4s, v27.8h, v3.h[0]  // vacc3x4567 += vb4567 * va3[0]
    SMLAL v16.4s, v27.4h, v4.h[0]   // vacc4x0123 += vb0123 * va4[0]
    SMLAL2 v17.4s, v27.8h, v4.h[0]  // vacc4x4567 += vb4567 * va4[0]
    SMLAL v18.4s, v27.4h, v5.h[0]   // vacc5x0123 += vb0123 * va5[0]
    SMLAL2 v19.4s, v27.8h, v5.h[0]  // vacc5x4567 += vb4567 * va5[0]
    SMLAL v20.4s, v27.4h, v6.h[0]   // vacc6x0123 += vb0123 * va6[0]
    SMLAL2 v21.4s, v27.8h, v6.h[0]  // vacc6x4567 += vb4567 * va6[0]
    SMLAL v22.4s, v27.4h, v7.h[0]   // vacc7x0123 += vb0123 * va7[0]
    SMLAL2 v23.4s, v27.8h, v7.h[0]  // vacc7x4567 += vb4567 * va7[0]

    CMP x2, -48
    B.LO 2f

    // Channel 1
    LD1 {v28.8b}, [x5], 8
    USUBL v28.8h, v28.8b, v25.8b

    SMLAL v8.4s, v28.4h, v0.h[1]    // vacc0x0123 += vb0123 * va0[1]
    SMLAL2 v9.4s, v28.8h, v0.h[1]   // vacc0x4567 += vb4567 * va0[1]
    SMLAL v10.4s, v28.4h, v1.h[1]   // vacc1x0123 += vb0123 * va1[1]
    SMLAL2 v11.4s, v28.8h, v1.h[1]  // vacc1x4567 += vb4567 * va1[1]
    SMLAL v12.4s, v28.4h, v2.h[1]   // vacc2x0123 += vb0123 * va2[1]
    SMLAL2 v13.4s, v28.8h, v2.h[1]  // vacc2x4567 += vb4567 * va2[1]
    SMLAL v14.4s, v28.4h, v3.h[1]   // vacc3x0123 += vb0123 * va3[1]
    SMLAL2 v15.4s, v28.8h, v3.h[1]  // vacc3x4567 += vb4567 * va3[1]
    SMLAL v16.4s, v28.4h, v4.h[1]   // vacc4x0123 += vb0123 * va4[1]
    SMLAL2 v17.4s, v28.8h, v4.h[1]  // vacc4x4567 += vb4567 * va4[1]
    SMLAL v18.4s, v28.4h, v5.h[1]   // vacc5x0123 += vb0123 * va5[1]
    SMLAL2 v19.4s, v28.8h, v5.h[1]  // vacc5x4567 += vb4567 * va5[1]
    SMLAL v20.4s, v28.4h, v6.h[1]   // vacc6x0123 += vb0123 * va6[1]
    SMLAL2 v21.4s, v28.8h, v6.h[1]  // vacc6x4567 += vb4567 * va6[1]
    SMLAL v22.4s, v28.4h, v7.h[1]   // vacc7x0123 += vb0123 * va7[1]
    SMLAL2 v23.4s, v28.8h, v7.h[1]  // vacc7x4567 += vb4567 * va7[1]

    B.LS 2f

    // Channel 2
    LD1 {v27.8b}, [x5], 8
    USUBL v27.8h, v27.8b, v25.8b

    SMLAL v8.4s, v27.4h, v0.h[2]    // vacc0x0123 += vb0123 * va0[2]
    SMLAL2 v9.4s, v27.8h, v0.h[2]   // vacc0x4567 += vb4567 * va0[2]
    SMLAL v10.4s, v27.4h, v1.h[2]   // vacc1x0123 += vb0123 * va1[2]
    SMLAL2 v11.4s, v27.8h, v1.h[2]  // vacc1x4567 += vb4567 * va1[2]
    SMLAL v12.4s, v27.4h, v2.h[2]   // vacc2x0123 += vb0123 * va2[2]
    SMLAL2 v13.4s, v27.8h, v2.h[2]  // vacc2x4567 += vb4567 * va2[2]
    SMLAL v14.4s, v27.4h, v3.h[2]   // vacc3x0123 += vb0123 * va3[2]
    SMLAL2 v15.4s, v27.8h, v3.h[2]  // vacc3x4567 += vb4567 * va3[2]
    SMLAL v16.4s, v27.4h, v4.h[2]   // vacc4x0123 += vb0123 * va4[2]
    SMLAL2 v17.4s, v27.8h, v4.h[2]  // vacc4x4567 += vb4567 * va4[2]
    SMLAL v18.4s, v27.4h, v5.h[2]   // vacc5x0123 += vb0123 * va5[2]
    SMLAL2 v19.4s, v27.8h, v5.h[2]  // vacc5x4567 += vb4567 * va5[2]
    SMLAL v20.4s, v27.4h, v6.h[2]   // vacc6x0123 += vb0123 * va6[2]
    SMLAL2 v21.4s, v27.8h, v6.h[2]  // vacc6x4567 += vb4567 * va6[2]
    SMLAL v22.4s, v27.4h, v7.h[2]   // vacc7x0123 += vb0123 * va7[2]
    SMLAL2 v23.4s, v27.8h, v7.h[2]  // vacc7x4567 += vb4567 * va7[2]

    CMP x2, -32
    B.LO 2f

    // Channel 3
    LD1 {v28.8b}, [x5], 8
    USUBL v28.8h, v28.8b, v25.8b

    SMLAL v8.4s, v28.4h, v0.h[3]    // vacc0x0123 += vb0123 * va0[3]
    SMLAL2 v9.4s, v28.8h, v0.h[3]   // vacc0x4567 += vb4567 * va0[3]
    SMLAL v10.4s, v28.4h, v1.h[3]   // vacc1x0123 += vb0123 * va1[3]
    SMLAL2 v11.4s, v28.8h, v1.h[3]  // vacc1x4567 += vb4567 * va1[3]
    SMLAL v12.4s, v28.4h, v2.h[3]   // vacc2x0123 += vb0123 * va2[3]
    SMLAL2 v13.4s, v28.8h, v2.h[3]  // vacc2x4567 += vb4567 * va2[3]
    SMLAL v14.4s, v28.4h, v3.h[3]   // vacc3x0123 += vb0123 * va3[3]
    SMLAL2 v15.4s, v28.8h, v3.h[3]  // vacc3x4567 += vb4567 * va3[3]
    SMLAL v16.4s, v28.4h, v4.h[3]   // vacc4x0123 += vb0123 * va4[3]
    SMLAL2 v17.4s, v28.8h, v4.h[3]  // vacc4x4567 += vb4567 * va4[3]
    SMLAL v18.4s, v28.4h, v5.h[3]   // vacc5x0123 += vb0123 * va5[3]
    SMLAL2 v19.4s, v28.8h, v5.h[3]  // vacc5x4567 += vb4567 * va5[3]
    SMLAL v20.4s, v28.4h, v6.h[3]   // vacc6x0123 += vb0123 * va6[3]
    SMLAL2 v21.4s, v28.8h, v6.h[3]  // vacc6x4567 += vb4567 * va6[3]
    SMLAL v22.4s, v28.4h, v7.h[3]   // vacc7x0123 += vb0123 * va7[3]
    SMLAL2 v23.4s, v28.8h, v7.h[3]  // vacc7x4567 += vb4567 * va7[3]

    B.LS 2f

    // Channel 4
    LD1 {v27.8b}, [x5], 8
    USUBL v27.8h, v27.8b, v25.8b

    SMLAL v8.4s, v27.4h, v0.h[4]    // vacc0x0123 += vb0123 * va0[4]
    SMLAL2 v9.4s, v27.8h, v0.h[4]   // vacc0x4567 += vb4567 * va0[4]
    SMLAL v10.4s, v27.4h, v1.h[4]   // vacc1x0123 += vb0123 * va1[4]
    SMLAL2 v11.4s, v27.8h, v1.h[4]  // vacc1x4567 += vb4567 * va1[4]
    SMLAL v12.4s, v27.4h, v2.h[4]   // vacc2x0123 += vb0123 * va2[4]
    SMLAL2 v13.4s, v27.8h, v2.h[4]  // vacc2x4567 += vb4567 * va2[4]
    SMLAL v14.4s, v27.4h, v3.h[4]   // vacc3x0123 += vb0123 * va3[4]
    SMLAL2 v15.4s, v27.8h, v3.h[4]  // vacc3x4567 += vb4567 * va3[4]
    SMLAL v16.4s, v27.4h, v4.h[4]   // vacc4x0123 += vb0123 * va4[4]
    SMLAL2 v17.4s, v27.8h, v4.h[4]  // vacc4x4567 += vb4567 * va4[4]
    SMLAL v18.4s, v27.4h, v5.h[4]   // vacc5x0123 += vb0123 * va5[4]
    SMLAL2 v19.4s, v27.8h, v5.h[4]  // vacc5x4567 += vb4567 * va5[4]
    SMLAL v20.4s, v27.4h, v6.h[4]   // vacc6x0123 += vb0123 * va6[4]
    SMLAL2 v21.4s, v27.8h, v6.h[4]  // vacc6x4567 += vb4567 * va6[4]
    SMLAL v22.4s, v27.4h, v7.h[4]   // vacc7x0123 += vb0123 * va7[4]
    SMLAL2 v23.4s, v27.8h, v7.h[4]  // vacc7x4567 += vb4567 * va7[4]

    CMP x2, -16
    B.LO 2f

    // Channel 5
    LD1 {v28.8b}, [x5], 8
    USUBL v28.8h, v28.8b, v25.8b

    SMLAL v8.4s, v28.4h, v0.h[5]    // vacc0x0123 += vb0123 * va0[5]
    SMLAL2 v9.4s, v28.8h, v0.h[5]   // vacc0x4567 += vb4567 * va0[5]
    SMLAL v10.4s, v28.4h, v1.h[5]   // vacc1x0123 += vb0123 * va1[5]
    SMLAL2 v11.4s, v28.8h, v1.h[5]  // vacc1x4567 += vb4567 * va1[5]
    SMLAL v12.4s, v28.4h, v2.h[5]   // vacc2x0123 += vb0123 * va2[5]
    SMLAL2 v13.4s, v28.8h, v2.h[5]  // vacc2x4567 += vb4567 * va2[5]
    SMLAL v14.4s, v28.4h, v3.h[5]   // vacc3x0123 += vb0123 * va3[5]
    SMLAL2 v15.4s, v28.8h, v3.h[5]  // vacc3x4567 += vb4567 * va3[5]
    SMLAL v16.4s, v28.4h, v4.h[5]   // vacc4x0123 += vb0123 * va4[5]
    SMLAL2 v17.4s, v28.8h, v4.h[5]  // vacc4x4567 += vb4567 * va4[5]
    SMLAL v18.4s, v28.4h, v5.h[5]   // vacc5x0123 += vb0123 * va5[5]
    SMLAL2 v19.4s, v28.8h, v5.h[5]  // vacc5x4567 += vb4567 * va5[5]
    SMLAL v20.4s, v28.4h, v6.h[5]   // vacc6x0123 += vb0123 * va6[5]
    SMLAL2 v21.4s, v28.8h, v6.h[5]  // vacc6x4567 += vb4567 * va6[5]
    SMLAL v22.4s, v28.4h, v7.h[5]   // vacc7x0123 += vb0123 * va7[5]
    SMLAL2 v23.4s, v28.8h, v7.h[5]  // vacc7x4567 += vb4567 * va7[5]

    B.LS 2f

    // Channel 6
    LD1 {v27.8b}, [x5], 8
    USUBL v27.8h, v27.8b, v25.8b

    SMLAL v8.4s, v27.4h, v0.h[6]    // vacc0x0123 += vb0123 * va0[6]
    SMLAL2 v9.4s, v27.8h, v0.h[6]   // vacc0x4567 += vb4567 * va0[6]
    SMLAL v10.4s, v27.4h, v1.h[6]   // vacc1x0123 += vb0123 * va1[6]
    SMLAL2 v11.4s, v27.8h, v1.h[6]  // vacc1x4567 += vb4567 * va1[6]
    SMLAL v12.4s, v27.4h, v2.h[6]   // vacc2x0123 += vb0123 * va2[6]
    SMLAL2 v13.4s, v27.8h, v2.h[6]  // vacc2x4567 += vb4567 * va2[6]
    SMLAL v14.4s, v27.4h, v3.h[6]   // vacc3x0123 += vb0123 * va3[6]
    SMLAL2 v15.4s, v27.8h, v3.h[6]  // vacc3x4567 += vb4567 * va3[6]
    SMLAL v16.4s, v27.4h, v4.h[6]   // vacc4x0123 += vb0123 * va4[6]
    SMLAL2 v17.4s, v27.8h, v4.h[6]  // vacc4x4567 += vb4567 * va4[6]
    SMLAL v18.4s, v27.4h, v5.h[6]   // vacc5x0123 += vb0123 * va5[6]
    SMLAL2 v19.4s, v27.8h, v5.h[6]  // vacc5x4567 += vb4567 * va5[6]
    SMLAL v20.4s, v27.4h, v6.h[6]   // vacc6x0123 += vb0123 * va6[6]
    SMLAL2 v21.4s, v27.8h, v6.h[6]  // vacc6x4567 += vb4567 * va6[6]
    SMLAL v22.4s, v27.4h, v7.h[6]   // vacc7x0123 += vb0123 * va7[6]
    SMLAL2 v23.4s, v27.8h, v7.h[6]  // vacc7x4567 += vb4567 * va7[6]

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 4
#endif
2:
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

    ADD  x9, x7,  x16
    CMP x0, 2
    CSEL x9, x7, x9, LO

    ADD x10, x9,  x16
    CSEL x10, x9, x10, LS

    ADD x11, x10, x16
    CMP x0, 4
    CSEL x11, x10, x11, LO

    ADD x12, x11, x16
    CSEL x12, x11, x12, LS

    ADD x13, x12, x16
    CMP x0, 6
    CSEL x13, x12, x13, LO

    ADD x14, x13, x16
    CSEL x14, x13, x14, LS

    ADD x15, x14, x16
    CMP x0, 8
    CSEL x15, x14, x15, NE

    CMP x1, 8
    B.NE 4f

    ST1 {v8.4s}, [x7], 16
    ST1 {v9.4s}, [x7]
    ST1 {v10.4s}, [x9], 16
    ST1 {v11.4s}, [x9]
    ST1 {v12.4s}, [x10], 16
    ST1 {v13.4s}, [x10]
    ST1 {v14.4s}, [x11], 16
    ST1 {v15.4s}, [x11]
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
    CMP x1, 4
    B.LO 5f

    ST1 {v8.4s}, [x7], 16
    ST1 {v10.4s}, [x9], 16
    ST1 {v12.4s}, [x10], 16
    ST1 {v14.4s}, [x11], 16
    ST1 {v16.4s}, [x12], 16
    ST1 {v18.4s}, [x13], 16
    ST1 {v20.4s}, [x14], 16
    ST1 {v22.4s}, [x15], 16

    SUB x1, x1, 4

    MOV V8.16b, V9.16b
    MOV v10.16b, v11.16b
    MOV v12.16b, V13.16b
    MOV V14.16b, V15.16b
    MOV V16.16b, V17.16b
    MOV V18.16b, V19.16b
    MOV V20.16b, V21.16b
    MOV V22.16b, V23.16b

5:
    CMP x1, 2
    B.LO 6f

    ST1 {v8.2s}, [x7], 8
    ST1 {v10.2s}, [x9], 8
    ST1 {v12.2s}, [x10], 8
    ST1 {v14.2s}, [x11], 8
    ST1 {v16.2s}, [x12], 8
    ST1 {v18.2s}, [x13], 8
    ST1 {v20.2s}, [x14], 8
    ST1 {v22.2s}, [x15], 8

    SUB x1, x1, 2

    EXT v8.16b, v8.16b, v8.16b, 8
    EXT v10.16b, v10.16b, v10.16b, 8
    EXT v12.16b, v12.16b, v12.16b, 8
    EXT V14.16b, V14.16b, V14.16b, 8
    EXT V16.16b, V16.16b, V16.16b, 8
    EXT V18.16b, V18.16b, V18.16b, 8
    EXT V20.16b, V20.16b, V20.16b, 8
    EXT V22.16b, V22.16b, V22.16b, 8

6:
    CMP x1, 1
    B.LO 7f

    ST1 {v8.s}[0], [x7]
    ST1 {v10.s}[0], [x9]
    ST1 {v12.s}[0], [x10]
    ST1 {v14.s}[0], [x11]
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

END_FUNCTION pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
