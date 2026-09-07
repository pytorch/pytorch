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
#  |------------|
#  |c_stride    | 0
#  |out ch index| 8
#  |params      | 16
#  |------------|

# This is an mr=1 variant of 8x8-dq-aarch64-neon.S. It computes a single
# activation row (v8-v9) for the dynamic-quantization path, for a batch/beam
# dimension of exactly 1. Callers dispatch on that exact size, so the kernel
# needs no row bounds checks. The weight packing (nr=8, kr=1) is identical to
# the 8x8 kernel, so no repacking is required.

# void pytorch_q8gemm_dq_ukernel_1x8__aarch64_neon(
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
BEGIN_FUNCTION pytorch_q8gemm_dq_ukernel_1x8__aarch64_neon

    STP d9, d8, [sp, -16]

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

    SUBS x2, x2, 8
    B.LO 1f

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 5
#endif
0:
    // b0-7 (channel 0)
    LD1 {v27.8b}, [x5], 8
    USUBL v27.8h, v27.8b, v25.8b

    # va0 := va - va_zero_point
    LD1 {v0.8b}, [x3], 8
    SUB_ZERO_POINT v0.8h, v0.8b, v24.8b

    // b0-7 (channel 1)
    LD1 {v28.8b}, [x5], 8

    SMLAL v8.4s, v27.4h, v0.h[0]    // vacc0x0123 += vb0123 * va0[0]
    SMLAL2 v9.4s, v27.8h, v0.h[0]   // vacc0x4567 += vb4567 * va0[0]
    USUBL v28.8h, v28.8b, v25.8b

    // b0-7 (channel 2)
    LD1 {v27.8b}, [x5], 8

    SMLAL v8.4s, v28.4h, v0.h[1]    // vacc0x0123 += vb0123 * va0[1]
    SMLAL2 v9.4s, v28.8h, v0.h[1]   // vacc0x4567 += vb4567 * va0[1]
    USUBL v27.8h, v27.8b, v25.8b

    // b0-7 (channel 3)
    LD1 {v28.8b}, [x5], 8

    SMLAL v8.4s, v27.4h, v0.h[2]    // vacc0x0123 += vb0123 * va0[2]
    SMLAL2 v9.4s, v27.8h, v0.h[2]   // vacc0x4567 += vb4567 * va0[2]
    USUBL v28.8h, v28.8b, v25.8b

    // b0-7 (channel 4)
    LD1 {v27.8b}, [x5], 8

    SMLAL v8.4s, v28.4h, v0.h[3]    // vacc0x0123 += vb0123 * va0[3]
    SMLAL2 v9.4s, v28.8h, v0.h[3]   // vacc0x4567 += vb4567 * va0[3]
    USUBL v27.8h, v27.8b, v25.8b

    // b0-7 (channel 5)
    LD1 {v28.8b}, [x5], 8

    SMLAL v8.4s, v27.4h, v0.h[4]    // vacc0x0123 += vb0123 * va0[4]
    SMLAL2 v9.4s, v27.8h, v0.h[4]   // vacc0x4567 += vb4567 * va0[4]
    USUBL v28.8h, v28.8b, v25.8b

    // b0-7 (channel 6)
    LD1 {v27.8b}, [x5], 8

    SMLAL v8.4s, v28.4h, v0.h[5]    // vacc0x0123 += vb0123 * va0[5]
    SMLAL2 v9.4s, v28.8h, v0.h[5]   // vacc0x4567 += vb4567 * va0[5]
    USUBL v27.8h, v27.8b, v25.8b

    // b0-7 (channel 7)
    LD1 {v28.8b}, [x5], 8

    SMLAL v8.4s, v27.4h, v0.h[6]    // vacc0x0123 += vb0123 * va0[6]
    SMLAL2 v9.4s, v27.8h, v0.h[6]   // vacc0x4567 += vb4567 * va0[6]
    USUBL v28.8h, v28.8b, v25.8b

    SUBS x2, x2, 8

    SMLAL v8.4s, v28.4h, v0.h[7]    // vacc0x0123 += vb0123 * va0[7]
    SMLAL2 v9.4s, v28.8h, v0.h[7]   // vacc0x4567 += vb4567 * va0[7]

    B.HS 0b

1:
    CMP x2, -8
    B.EQ 2f

    // Adjust a0
    ADD x3, x3, x2

    // a_shift = 8 * k - 64
    LSL x2, x2, 3
    FMOV d29, x2
    USHL d24, d24, d29

    // Load a0
    LD1 {v0.8b}, [x3], 8
    USHL d0, d0, d29
    SUB_ZERO_POINT v0.8h, v0.8b, v24.8b

    // Channel 0
    LD1 {v27.8b}, [x5], 8
    USUBL v27.8h, v27.8b, v25.8b

    SMLAL v8.4s, v27.4h, v0.h[0]    // vacc0x0123 += vb0123 * va0[0]
    SMLAL2 v9.4s, v27.8h, v0.h[0]   // vacc0x4567 += vb4567 * va0[0]

    CMP x2, -48
    B.LO 2f

    // Channel 1
    LD1 {v28.8b}, [x5], 8
    USUBL v28.8h, v28.8b, v25.8b

    SMLAL v8.4s, v28.4h, v0.h[1]    // vacc0x0123 += vb0123 * va0[1]
    SMLAL2 v9.4s, v28.8h, v0.h[1]   // vacc0x4567 += vb4567 * va0[1]

    B.LS 2f

    // Channel 2
    LD1 {v27.8b}, [x5], 8
    USUBL v27.8h, v27.8b, v25.8b

    SMLAL v8.4s, v27.4h, v0.h[2]    // vacc0x0123 += vb0123 * va0[2]
    SMLAL2 v9.4s, v27.8h, v0.h[2]   // vacc0x4567 += vb4567 * va0[2]

    CMP x2, -32
    B.LO 2f

    // Channel 3
    LD1 {v28.8b}, [x5], 8
    USUBL v28.8h, v28.8b, v25.8b

    SMLAL v8.4s, v28.4h, v0.h[3]    // vacc0x0123 += vb0123 * va0[3]
    SMLAL2 v9.4s, v28.8h, v0.h[3]   // vacc0x4567 += vb4567 * va0[3]

    B.LS 2f

    // Channel 4
    LD1 {v27.8b}, [x5], 8
    USUBL v27.8h, v27.8b, v25.8b

    SMLAL v8.4s, v27.4h, v0.h[4]    // vacc0x0123 += vb0123 * va0[4]
    SMLAL2 v9.4s, v27.8h, v0.h[4]   // vacc0x4567 += vb4567 * va0[4]

    CMP x2, -16
    B.LO 2f

    // Channel 5
    LD1 {v28.8b}, [x5], 8
    USUBL v28.8h, v28.8b, v25.8b

    SMLAL v8.4s, v28.4h, v0.h[5]    // vacc0x0123 += vb0123 * va0[5]
    SMLAL2 v9.4s, v28.8h, v0.h[5]   // vacc0x4567 += vb4567 * va0[5]

    B.LS 2f

    // Channel 6
    LD1 {v27.8b}, [x5], 8
    USUBL v27.8h, v27.8b, v25.8b

    SMLAL v8.4s, v27.4h, v0.h[6]    // vacc0x0123 += vb0123 * va0[6]
    SMLAL2 v9.4s, v27.8h, v0.h[6]   // vacc0x4567 += vb4567 * va0[6]

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 4
#endif
2:
    LSL x16, x16, 2
    LD1 {v24.4s}, [x6], 16
    LD1 {v25.4s}, [x6]

    SCVTF v8.4s, v8.4s
    SCVTF v9.4s, v9.4s

    FMUL v8.4s, v8.4s, v26.4s
    FMUL v9.4s, v9.4s, v30.4s

    FADD v8.4s, v8.4s, v24.4s
    FADD v9.4s, v9.4s, v25.4s

    // Compute c0

    CMP x1, 8
    B.NE 4f

    ST1 {v8.4s}, [x7], 16
    ST1 {v9.4s}, [x7]

    LDP d9, d8, [sp, -16]

    RET

#ifndef IGNORE_CODE_ALIGN_DIRECTIVES
    .p2align 3
#endif
4:
    CMP x1, 4
    B.LO 5f

    ST1 {v8.4s}, [x7], 16

    SUB x1, x1, 4

    MOV V8.16b, V9.16b

5:
    CMP x1, 2
    B.LO 6f

    ST1 {v8.2s}, [x7], 8

    SUB x1, x1, 2

    EXT v8.16b, v8.16b, v8.16b, 8

6:
    CMP x1, 1
    B.LO 7f

    ST1 {v8.s}[0], [x7]

7:
    LDP d9, d8, [sp, -16]

    RET

END_FUNCTION pytorch_q8gemm_dq_ukernel_1x8__aarch64_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
