/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>
#include <requantization/runtime-assembly.h>


#  Args passed via 8 registers (64 bytes)
#  x0: mr
#  x1: nr
#  x2: kc
#  x3: ks
#  x4: a
#  x5: w
#  x6: c
#  x7: c_stride
#

#  Args passed via stack.
#  TOS
#  |-----------|
#  |out ch indx| 0
#  |params     | 8
#  |-----------|

# void pytorch_q8conv_ukernel_8x8__aarch64_neon(
#    size_t mr,
#    size_t nr,
#    size_t kc,
#    size_t ks,
#    const uint8_t** restrict a,
#    const void* restrict w,
#    uint8_t* restrict c,
#    size_t c_stride,
#    size_t output_channel_index,
#    const union pytorch_qnnp_q31_requantization_params quantization_params[restrict static 1])
BEGIN_FUNCTION pytorch_q8conv_ukernel_8x8__aarch64_neon
    # Load params: x8
    # Load output channel index: x9
    # Note since this is an offset into a byte pointer
    # We do not need to multiply with size of pointer type
    LDP x9, x8, [sp]

    STP d15, d14, [sp, -16]
    STP d13, d12, [sp, -32]
    STP d11, d10, [sp, -48]
    STP d9, d8, [sp, -64]

    # Load bias0123, bias4567
    LD1 {v8.4s, v9.4s}, [x5], 32

    # Load pointer to per channel zero points array
    # And go to a_zero_point with post-index
    LDR x10, [x8], 8
    # Add offset to the base pointer
    ADD x10, x10, x9

    # v10 := vacc1x0123
    MOV v10.16b, v8.16b

    # v11 := vacc1x4567
    MOV v11.16b, v9.16b
    # Load b_zero_point
    LD1 {v25.8b}, [x10]
    # Load a_zero_point
    LD1R {v24.8b}, [x8]
    # Load pointer to per channel requant scale
    LDR x10, [x8, 8]!
    ADD x8, x8, 8

    # v12 := vacc2x0123
    MOV v12.16b, v8.16b
    # v13 := vacc2x4567
    MOV v13.16b, v9.16b

    # v14 := vacc3x0123
    MOV v14.16b, v8.16b
    # v15 := vacc3x4567
    MOV v15.16b, v9.16b

    # v16 := vacc4x0123
    MOV v16.16b, v8.16b
    # v17 := vacc4x4567
    MOV v17.16b, v9.16b

    # v18 := vacc5x0123
    MOV v18.16b, v8.16b
    # v19 := vacc5x4567
    MOV v19.16b, v9.16b

    # v20 := vacc6x0123
    MOV v20.16b, v8.16b
    # v21 := vacc6x4567
    MOV v21.16b, v9.16b

    # v22 := vacc7x0123
    MOV v22.16b, v8.16b
    # v23 := vacc7x4567
    MOV v23.16b, v9.16b

    # Fold mul by 4 to get byte offset for requant scale.
    # Add offset to the base pointer
    ADD x10, x10, x9, lsl#2
    // Load requantization_scale
    // - v26 = requantization_scale channels 0-3
    // - v31 = requantization_scale channels 4-7
    LD1 {v26.4s}, [x10], 16
    LD1 {v30.4s}, [x10]

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 4
#endif
3:
    MOV x17, x2

    LDR x16, [x4], 8  // a0
    LDR x9,  [x4], 8  // a1
    LDR x10, [x4], 8  // a2
    LDR x11, [x4], 8  // a3
    LDR x12, [x4], 8  // a4
    LDR x13, [x4], 8  // a5
    LDR x14, [x4], 8  // a6
    LDR x15, [x4], 8  // a7

    SUBS x17, x17, 8
    B.LO 1f

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 5
#endif
0:
    # b0-7 (channel 0)
    LD1 {v27.8b}, [x5], 8
    USUBL v27.8h, v27.8b, v25.8b

    # va0 - va7 := va - va_offset
    LD1 {v0.8b}, [x16], 8
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

    SUBS x17, x17, 8

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
    CMP x17, -8
    B.EQ 2f

    // Adjust a0-a7
    ADD x16, x16, x17
    ADD x9, x9, x17
    ADD x10, x10, x17
    ADD x11, x11, x17
    ADD x12, x12, x17
    ADD x13, x13, x17
    ADD x14, x14, x17
    ADD x15, x15, x17

    // a_shift = 8 * k - 64
    LSL x17, x17, 3
    FMOV d29, x17
    USHL d31, d24, d29

    // Load x0-a7
    LD1 {v0.8b}, [x16], 8
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

    CMP x17, -48
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

    CMP x17, -32
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

    CMP x17, -16
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

    SUB x3, x3, 1
    CBNZ x3, 3b

    // Load zero_point:
    // - v29 = vzero_point
    LD1R {v29.8h}, [x8], 2

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

    // Load max:
    // - v30 = vmax
    LD1R {v30.16b}, [x8], 1

    // Load min:
    // - v31 = vmin
    LD1R {v31.16b}, [x8]

    FCVTNS v8.4s, v8.4s
    FCVTNS v9.4s, v9.4s
    FCVTNS v10.4s, v10.4s
    FCVTNS v11.4s, v11.4s
    FCVTNS v12.4s, v12.4s
    FCVTNS v13.4s, v13.4s
    FCVTNS v14.4s, v14.4s
    FCVTNS v15.4s, v15.4s

    FCVTNS v16.4s, v16.4s
    FCVTNS v17.4s, v17.4s
    FCVTNS v18.4s, v18.4s
    FCVTNS v19.4s, v19.4s
    FCVTNS v20.4s, v20.4s
    FCVTNS v21.4s, v21.4s
    FCVTNS v22.4s, v22.4s
    FCVTNS v23.4s, v23.4s

    SQXTN   v8.4h,  v8.4s
    SQXTN  v10.4h, v10.4s
    SQXTN  v12.4h, v12.4s
    SQXTN  v14.4h, v14.4s
    SQXTN  v16.4h, v16.4s
    SQXTN  v18.4h, v18.4s
    SQXTN  v20.4h, v20.4s
    SQXTN  v22.4h, v22.4s

    SQXTN2  v8.8h,  v9.4s
    SQXTN2 v10.8h, v11.4s
    SQXTN2 v12.8h, v13.4s
    SQXTN2 v14.8h, v15.4s
    SQXTN2 v16.8h, v17.4s
    SQXTN2 v18.8h, v19.4s
    SQXTN2 v20.8h, v21.4s
    SQXTN2 v22.8h, v23.4s

    SQADD  v8.8h,  v8.8h, v29.8h
    SQADD v10.8h, v10.8h, v29.8h
    SQADD v12.8h, v12.8h, v29.8h
    SQADD v14.8h, v14.8h, v29.8h
    SQADD v16.8h, v16.8h, v29.8h
    SQADD v18.8h, v18.8h, v29.8h
    SQADD v20.8h, v20.8h, v29.8h
    SQADD v22.8h, v22.8h, v29.8h

    SQXTUN    v8.8b,  v8.8h
    SQXTUN   v12.8b, v12.8h
    SQXTUN   v16.8b, v16.8h
    SQXTUN   v20.8b, v20.8h

    SQXTUN2  v8.16b, v10.8h
    SQXTUN2 v12.16b, v14.8h
    SQXTUN2 v16.16b, v18.8h
    SQXTUN2 v20.16b, v22.8h

    UMIN  v8.16b,  v8.16b, v30.16b
    UMIN v12.16b, v12.16b, v30.16b
    UMIN v16.16b, v16.16b, v30.16b
    UMIN v20.16b, v20.16b, v30.16b

    UMAX  v8.16b,  v8.16b, v31.16b
    UMAX v12.16b, v12.16b, v31.16b
    UMAX v16.16b, v16.16b, v31.16b
    UMAX v20.16b, v20.16b, v31.16b

    // Compute c0-c7

    ADD  x9, x6,  x7
    CMP x0, 2
    CSEL x9, x6, x9, LO

    ADD x10, x9,  x7
    CSEL x10, x9, x10, LS

    ADD x11, x10, x7
    CMP x0, 4
    CSEL x11, x10, x11, LO

    ADD x12, x11, x7
    CSEL x12, x11, x12, LS

    ADD x13, x12, x7
    CMP x0, 6
    CSEL x13, x12, x13, LO

    ADD x14, x13, x7
    CSEL x14, x13, x14, LS

    ADD x15, x14, x7
    CMP x0, 8
    CSEL x15, x14, x15, NE

    CMP x1, 8
    B.NE 4f

    // Store results
    ST1  {v8.d}[0],  [x6]
    ST1  {v8.d}[1],  [x9]
    ST1 {v12.d}[0], [x10]
    ST1 {v12.d}[1], [x11]
    ST1 {v16.d}[0], [x12]
    ST1 {v16.d}[1], [x13]
    ST1 {v20.d}[0], [x14]
    ST1 {v20.d}[1], [x15]

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

    ST1  {v8.s}[0],  [x6], 4
    ST1  {v8.s}[2],  [x9], 4
    ST1 {v12.s}[0], [x10], 4
    ST1 {v12.s}[2], [x11], 4
    ST1 {v16.s}[0], [x12], 4
    ST1 {v16.s}[2], [x13], 4
    ST1 {v20.s}[0], [x14], 4
    ST1 {v20.s}[2], [x15], 4

    SUB x1, x1, 4
    EXT  v8.16b,  v8.16b,  v8.16b, 4
    EXT v12.16b, v12.16b, v12.16b, 4
    EXT v16.16b, v16.16b, v16.16b, 4
    EXT v20.16b, v20.16b, v20.16b, 4

5:
    CMP x1, 2
    B.LO 6f

    ST1  {v8.h}[0],  [x6], 2
    ST1  {v8.h}[4],  [x9], 2
    ST1 {v12.h}[0], [x10], 2
    ST1 {v12.h}[4], [x11], 2
    ST1 {v16.h}[0], [x12], 2
    ST1 {v16.h}[4], [x13], 2
    ST1 {v20.h}[0], [x14], 2
    ST1 {v20.h}[4], [x15], 2

    SUB x1, x1, 2
    EXT  v8.16b,  v8.16b,  v8.16b, 2
    EXT v12.16b, v12.16b, v12.16b, 2
    EXT v16.16b, v16.16b, v16.16b, 2
    EXT v20.16b, v20.16b, v20.16b, 2

6:
    CMP x1, 1
    B.LO 7f

    ST1  {v8.b}[0],  [x6]
    ST1  {v8.b}[8],  [x9]
    ST1 {v12.b}[0], [x10]
    ST1 {v12.b}[8], [x11]
    ST1 {v16.b}[0], [x12]
    ST1 {v16.b}[8], [x13]
    ST1 {v20.b}[0], [x14]
    ST1 {v20.b}[8], [x15]

7:
    LDP d9, d8, [sp, -64]
    LDP d11, d10, [sp, -48]
    LDP d13, d12, [sp, -32]
    LDP d15, d14, [sp, -16]

    RET

END_FUNCTION pytorch_q8conv_ukernel_8x8__aarch64_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
