/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>
#include <requantization/runtime-assembly.h>

# r0 mr
# r1 nr
# r2 a
# r3 a_stride

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
# 36 bytes of parameters on stack
# 16    r4-r7
# 64    d8-d15

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
#  |packed_w        | 100
#  |w_row_ptr       | 104
#  |w_block_ids_ptr | 108
#  |b               | 112
#  |c               | 116
#  |c_stride        | 120
#  |out ch indx     | 124
#  |params          | 128
#  |----------------|
#  

# void pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon(
#     size_t mr,
#     size_t nr,
#     const uint8_t* a,
#     size_t a_stride,
#     const uint8_t* packed_w,
#     const uint32_t* w_row_ptr,
#     const uint32_t* w_block_ids_ptr,
#     const float* b,
#     uint8_t* restrict c,
#     size_t c_stride,
#     size_t output_channel_index,
#     const union pytorch_qnnp_conv_dynamic_quantization_params quantization_params[restrict static 1])
BEGIN_FUNCTION pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon
    .arm
#ifndef __APPLE__
    .arch armv7-a
    .fpu neon
#endif

    PUSH {r4, r5, r6, r7, r8, r9, r10, r11, lr}
    VPUSH {d8-d15}

    # Store mr and nr on stack to later load from since
    # we will not need it once all 8 a pointers are
    # initialized.
    # Furthermore, we need all the register we can get.
    STR r0, [sp, #-4]
    STR r1, [sp, #-8]
    # Load output channel index
    LDR r5, [sp, 124]
    # Load quantization params
    # - r7 = quantization_params
    LDR r7, [sp, 128]
    # Load input_zero_point
    VLD1.8 {d16[]}, [r7]
    ADD r7, r7, 4
    # Load pointer to per channel zero points array
    # Post-index: After load increment r7 by 4
    LDR r4, [r7]
    # Add output_channel_index to the b_zero_point pointer
    ADD r4, r4, r5
    # Store pointer to zero point on stack
    STR r4, [sp, #-12]

    # Copy mr to r9
    MOV r9, r0
    # Stride to r10
    MOV r10, r3
    # r0 = a0 = a pointer
    MOV r0, r2
    # Use r8 to store nr
    # and r1 to store a1
    MOV r8, r1

    CMP r9, 2
    # r1 = a1
    ADD r1, r0, r10
    MOVLO r1, r0

    # r2 = a2
    ADD r2, r1, r10
    MOVLS r2, r1

    CMP r9, 4
    # r3 = a3
    ADD r3, r2, r10
    MOVLO r3, r2

    # r4 = a4
    ADD r4, r3, r10
    MOVLS r4, r3

    CMP r9, 6
    # r5 = a5
    ADD r5, r4, r10
    MOVLO r5, r4

    # r6 = a6
    ADD r6, r5, r10
    MOVLS r6, r5

    CMP r9, 8
    # r7 = a7
    ADD r7, r6, r10
    MOVNE r7, r6

    # We enter the loop if r8 is atleast 1.
    # r8 = r8 - 1 will happen in the epilogue
    # of the loop
    CMP r8, 1
    BLO 7f

    .p2align 5
0:
    # Load pointer to weight zero point
    LDR r9, [sp, #-12]
    # Load w_row_ptr + n
    LDR r11, [sp, 104]
    VEOR q10, q10, q10
    VLD1.8 {d17[]}, [r9]!
    # Store updated pointer to weight zero point on stack
    STR r9, [sp, #-12]
    # ip = w_row_ptr[n], lr = w_row_ptr[n+1]
    # r11 = r11 + 4 to point to next n
    LDR ip, [r11], #4
    LDR lr, [r11]
    VEOR q11, q11, q11
    STR r11, [sp, 104]
    # r9 = packed_w
    LDR r9, [sp, 100]
    # r10 = blocks_id_ptr
    LDR r10, [sp, 108]
    # r9 = temp_packed_w = packed_w + w_row_ptr[n] * 4
    # This points to the first block of nonzero value
    # for the nth row.
    ADD r9, r9, ip, LSL #2
    # r10 = temp_w_block_ids_ptr = w_block_ids_ptr + w_row_ptr[n]
    # LSL2 because each element is 4 bytes
    # This points to the block id of the first block
    # It should contain r9 - r8 number of block ids
    ADD r10, r10, ip, LSL #2
    # r11 = num_blocks that needs to be processed
    SUB r11, lr, ip
    SUBS r11, r11, 2
    BLO 1f

k_loop:
    # Load 2 non zero blocks of weights. Each block = 1x4.
    VLD1.8 {d19}, [r9]!
    # q9 = vxb
    VSUBL.U8 q9, d19, d17

    #ip = block_id_ptr[0]
    #lr = block_id_ptr[1]
    LDR ip, [r10], #4
    LDR lr, [r10], #4

    # Add offset to r0
    # Shift by 2 because each block id corresponds to 4 uint8_t elements
    ADD r0, r0, ip, LSL #2
    ADD r1, r1, ip, LSL #2
    ADD r2, r2, ip, LSL #2
    ADD r3, r3, ip, LSL #2
    ADD r4, r4, ip, LSL #2
    ADD r5, r5, ip, LSL #2
    ADD r6, r6, ip, LSL #2
    ADD r7, r7, ip, LSL #2

    VLD1.32 {d0[]}, [r0]
    VLD1.32 {d2[]}, [r1]
    VLD1.32 {d4[]}, [r2]
    VLD1.32 {d6[]}, [r3]
    VLD1.32 {d8[]}, [r4]
    VLD1.32 {d10[]}, [r5]
    VLD1.32 {d12[]}, [r6]
    VLD1.32 {d14[]}, [r7]

    SUB ip, lr, ip
    ADD r0, r0, ip, LSL #2
    ADD r1, r1, ip, LSL #2
    ADD r2, r2, ip, LSL #2
    ADD r3, r3, ip, LSL #2
    ADD r4, r4, ip, LSL #2
    ADD r5, r5, ip, LSL #2
    ADD r6, r6, ip, LSL #2
    ADD r7, r7, ip, LSL #2

    VLD1.32 {d1[]}, [r0]
    VLD1.32 {d3[]}, [r1]
    VLD1.32 {d5[]}, [r2]
    VLD1.32 {d7[]}, [r3]
    VLD1.32 {d9[]}, [r4]
    VLD1.32 {d11[]}, [r5]
    VLD1.32 {d13[]}, [r6]
    VLD1.32 {d15[]}, [r7]

    SUB r0, r0, lr, LSL #2
    SUB r1, r1, lr, LSL #2
    SUB r2, r2, lr, LSL #2
    SUB r3, r3, lr, LSL #2
    SUB r4, r4, lr, LSL #2
    SUB r5, r5, lr, LSL #2
    SUB r6, r6, lr, LSL #2
    SUB r7, r7, lr, LSL #2

    VEXT.8 d0, d0, d1, #4
    VEXT.8 d1, d2, d3, #4
    VEXT.8 d2, d4, d5, #4
    VTRN.8 d0, d1
    VEXT.8 d3, d6, d7, #4
    VEXT.8 d4, d8, d9, #4
    VTRN.8 d2, d3
    VEXT.8 d5, d10, d11, #4
    VTRN.16 q0, q1
    VTRN.8 d4, d5
    VEXT.8 d6, d12, d13, #4
    VEXT.8 d7, d14, d15, #4

    VTRN.8 d6, d7
    VTRN.16 q2, q3

    VTRN.32 q0, q2
    VTRN.32 q1, q3
    # Sub offset from r0
    # This we have to do because we dont have a vector load
    # instruction with pre-indexed register offset

    # Add offset to r1
    # Sub offset from r1
    # This we have to do because we dont have a vector load
    # instruction with pre-indexed register offset

    # Add offset to r2
    # Sub offset from r2
    # This we have to do because we dont have a vector load
    # instruction with pre-indexed register offset

    # Add offset to r3
    # Sub offset from r3
    # This we have to do because we dont have a vector load
    # instruction with pre-indexed register offset

    # Add offset to r4
    # Sub offset from r4
    # This we have to do because we dont have a vector load
    # instruction with pre-indexed register offset

    # Add offset to r5
    # Sub offset from r5
    # This we have to do because we dont have a vector load
    # instruction with pre-indexed register offset

    # Add offset to r6
    # Sub offset from r6
    # This we have to do because we dont have a vector load
    # instruction with pre-indexed register offset

    # Add offset to r7
    # Sub offset from r7
    # This we have to do because we dont have a vector load
    # instruction with pre-indexed register offset

    #
    # d0, d1 = q0 = va0, va1
    # d2, d3 = q1 = va2, va3
    # d4, d5 = q2 = va4, va5
    # d6, d7 = q3 = va6, va7
    #Now transpose


    VSUBL.U8 q4, d0, d16  // vxa0_t
    VSUBL.U8 q5, d1, d16  // vxa1_t
    VSUBL.U8 q6, d2, d16  // vxa2_t
    VSUBL.U8 q7, d3, d16  // vxa3_t
    VSUBL.U8 q12, d4, d16  // vxa4_t
    VSUBL.U8 q13, d5, d16  // vxa5_t
    VSUBL.U8 q14, d6, d16  // vxa6_t
    VSUBL.U8 q15, d7, d16  // vxa7_t
    # This setup without the VMOVs is a perfect
    # setup for double buffering + tranpose
    # Tranposed result is stored in q0, q1, q2, q3.
    # vxa* ares in q4-q7, q12-q13.
    # Now q0-q3 are free to store next iterations of
    # activations + tranpose
    # We will do this as a later optimization.

    VMOV q0, q9

    VMLAL.S16 q10, d8, d0[0]
    VMLAL.S16 q11, d9, d0[0]
    VMLAL.S16 q10, d10, d0[1]
    VMLAL.S16 q11, d11, d0[1]
    VMLAL.S16 q10, d12, d0[2]
    VMLAL.S16 q11, d13, d0[2]
    VMLAL.S16 q10, d14, d0[3]
    VMLAL.S16 q11, d15, d0[3]
    VMLAL.S16 q10, d24, d1[0]
    VMLAL.S16 q11, d25, d1[0]
    VMLAL.S16 q10, d26, d1[1]
    VMLAL.S16 q11, d27, d1[1]
    VMLAL.S16 q10, d28, d1[2]
    VMLAL.S16 q11, d29, d1[2]
    VMLAL.S16 q10, d30, d1[3]
    VMLAL.S16 q11, d31, d1[3]

    SUBS r11, r11, 2

    BHS k_loop
1:
    CMP r11, -2
    BEQ 2f

    # Load last nonzero block
    # For this we will load 4 8 bit values as one 32 bit value
    VLD1.32 {d19[]}, [r9]!
    # q9 = vxb
    VSUBL.U8 q9, d19, d17

    #ip = block_id_ptr[0]
    LDR ip, [r10]

    # Add offset to r0, r4
    # Shift by 2 because each block id corresponds to 4 uint8_t elements
    ADD r0, r0, ip, LSL #2
    ADD r4, r4, ip, LSL #2
    ADD r1, r1, ip, LSL #2
    ADD r5, r5, ip, LSL #2
    ADD r2, r2, ip, LSL #2
    ADD r6, r6, ip, LSL #2
    ADD r3, r3, ip, LSL #2
    ADD r7, r7, ip, LSL #2

    VLD1.32 {d0[]}, [r0]
    VLD1.32 {d1[]}, [r4]
    VLD1.32 {d2[]}, [r1]
    VLD1.32 {d3[]}, [r5]
    VLD1.32 {d4[]}, [r2]
    VLD1.32 {d5[]}, [r6]
    VLD1.32 {d6[]}, [r3]
    VLD1.32 {d7[]}, [r7]

    SUB r0, r0, ip, LSL #2
    SUB r1, r1, ip, LSL #2
    SUB r5, r5, ip, LSL #2
    SUB r4, r4, ip, LSL #2
    SUB r2, r2, ip, LSL #2
    SUB r6, r6, ip, LSL #2
    SUB r3, r3, ip, LSL #2
    SUB r7, r7, ip, LSL #2

    VEXT.8 d0, d0, d1, #4
    # Add offset to r1, r5
    # d1 = va1, va5
    VEXT.8 d1, d2, d3, #4
    # Add offset to r2, r6
    # d2 = va2, va6
    VEXT.8 d2, d4, d5, #4
    # Add offset to r3, r7
    # d3 = va3, va7
    VEXT.8 d3, d6, d7, #4

    #
    # d0 = va0, va4
    # d1 = va1, va5
    # d2 = va2, va6
    # d3 = va3, va7
    #Now transpose
    VTRN.8 d0, d1
    VTRN.8 d2, d3
    VTRN.16 d0, d2
    VTRN.16 d1, d3


    VSUBL.U8 q4, d0, d16  // vxa04_t
    VSUBL.U8 q5, d1, d16  // vxa15_t
    VSUBL.U8 q6, d2, d16  // vxa26_t
    VSUBL.U8 q7, d3, d16  // vxa37_t

    VMOV q0, q9

    VMLAL.S16 q10, d8, d0[0]
    VMLAL.S16 q11, d9, d0[0]
    VMLAL.S16 q10, d10, d0[1]
    VMLAL.S16 q11, d11, d0[1]
    VMLAL.S16 q10, d12, d0[2]
    VMLAL.S16 q11, d13, d0[2]
    VMLAL.S16 q10, d14, d0[3]
    VMLAL.S16 q11, d15, d0[3]

    .p2align 4
2:
    # Store result on stack

    # -12 because TOS - 4, TOS - 8, and TOS - 12, store mr, nr and pointer to weight zp
    # + 128 bytes of buffer when nr = 1
    # This is needed because after processing all nrs we will
    # load 128 bytes from stack. This is for q10, q11 for max nr of 4
    # Thus we will load accumulators back in q0, q1, q2, q3, q4, q5, q6, q7
    # When nr < 4, extra q values will be fetched from stack which may overlap
    # with other parts of stack storing local variables. To avoid that we just
    # create a buffer of 128 bytes inbetween to make sure pointer increment
    # never produces address that is beyond the stack frame of this function.
    SUB r9, sp, 140
    # Each iteration produce 8 values each of 4 bytes
    # Thus 8 x 4 = 32 bytes 2^5
    # In this implementation, first value will be stored at
    # 1st value: sp - 12 - r8 * 32
    # 2nd value: sp - 12 - (r8 - 1) * 32
    # and so on.
    SUB r9, r9, r8, LSL #5
    VST1.32 {q10}, [r9]!
    VST1.32 {q11}, [r9]

    # Check if nr >=1
    SUBS r8, r8, 1
    BHI 0b
3:
    # First load all the accumulators from stack
    # Load nr
    LDR r8, [sp, #-8]
    SUB r9, sp, 140
    SUB r9, r9, r8, LSL #5
    # Now load q8-q15
    # This is 4x8 block (nrxmr)
    # We will transpose this to 8x4 (mrxnr)
    # q8, q12  : x00, x10, x20, x30; x40, x50, x60, x70
    # q9, q13  : x01, x11, x21, x31; x41, x51, x61, x71
    # q10, q14 : x02, x12, x22, x32; x42, x52, x62, x72
    # q11, q15 : x03, x13, x23, x33; x43, x53, x63, x73
    VLD1.32 {q8}, [r9]!
    VLD1.32 {q12}, [r9]!
    VLD1.32 {q9}, [r9]!
    VLD1.32 {q13}, [r9]!
    VLD1.32 {q10}, [r9]!
    VLD1.32 {q14}, [r9]!
    VLD1.32 {q11}, [r9]!
    VLD1.32 {q15}, [r9]

    ## Now transpose q8-11
    # VTRN.32 q8, q9
    # VTRN.32 q10, q11
    # q8 : X00, x01, x20, x21
    # q9 : X10, x11, x30, x31
    # q10: X02, x03, x22, x23
    # q11: X12, x13, x32, x33
    # VSWP d16, d17
    # q8 : x20, x21, x00, x01
    # VEXT.32 q6, q8, q10, 2
    # q6 : x00, x01, x02, x03
    # VEXT.32 q10, q10, q8, 2
    # q10: x22, x23, x20, x21
    # VSWP d20, d21
    # VMOV q8, q6
    # q8 : X00, x01, x02, x03
    # q10: x20, x21, x22, x23
    # VSWP d18, d19
    # q9 : x30, x31, x10, x11
    # VEXT.32 q6, q9, q11, 2
    # q6 : x10, x11, x12, x13
    # VEXT.32 q11, q11, q9, 2
    # q11: x32, x33, x30, x31
    # VSWP d22, d23
    # VMOV q9, q6
    # q9 : x10, x11, x12, x13
    # q11: x30, x31, x32, x33
    # Thus we have
    # q8 : X00, x01, x02, x03
    # q9 : X10, x11, x12, x13
    # q10: X20, x21, x22, x23
    # q11: X30, x31, x32, x33
    # Now we can do the same for q4-q7
    # q12: X40, X41, X42, X43
    # q13: X50, X51, X52, X53
    # q14: X60, X61, X62, X63
    # q15: X70, X71, X72, X73
    # NEED TO VALIDATE THIS
    VTRN.32 q8, q9
    VTRN.32 q10, q11
    VSWP d16, d17
    VEXT.32 q6, q8, q10, 2
    VEXT.32 q10, q10, q8, 2
    VSWP d20, d21
    VMOV q8, q6
    VSWP d18, d19
    VEXT.32 q6, q9, q11, 2
    VEXT.32 q11, q11, q9, 2
    VSWP d22, d23
    VMOV q9, q6

    VTRN.32 q12, q13
    VTRN.32 q14, q15
    VSWP d24, d25
    VEXT.32 q6, q12, q14, 2
    VEXT.32 q14, q14, q12, 2
    VSWP d28, d29
    VMOV q12, q6
    VSWP d26, d27
    VEXT.32 q6, q13, q15, 2
    VEXT.32 q15, q15, q13, 2
    VSWP d30, d31
    VMOV q13, q6

    # Load output channel index
    LDR r5, [sp, 124]
    # Load quantization params
    # - r7 = quantization_params
    LDR r7, [sp, 128]
    ADD r7, r7, 8
    # Load pointer to per channel requant scale
    LDR r7, [r7]
    # Now r7 has the base_addr + offset for multipliers
    ADD r7, r7, r5, LSL #2

    LDR r6, [sp, 112]
    # Load q6: vmultiplier_c0123
    VLD1.32 {d12, d13}, [r7]!
    VCVT.F32.S32 q8, q8
    VCVT.F32.S32 q9, q9
    VCVT.F32.S32 q10, q10
    VLD1.32 {q0}, [r6]

    VCVT.F32.S32 q11, q11
    VCVT.F32.S32 q12, q12
    VCVT.F32.S32 q13, q13
    VCVT.F32.S32 q14, q14
    VCVT.F32.S32 q15, q15

    VMUL.F32 q8, q8, q6
    VMUL.F32 q9, q9, q6
    VMUL.F32 q10, q10, q6
    VMUL.F32 q11, q11, q6
    VMUL.F32 q12, q12, q6
    VMUL.F32 q13, q13, q6
    VMUL.F32 q14, q14, q6
    VMUL.F32 q15, q15, q6

    VADD.F32 q8, q8, q0
    VADD.F32 q9, q9, q0
    VADD.F32 q10, q10, q0
    VADD.F32 q11, q11, q0
    VADD.F32 q12, q12, q0
    VADD.F32 q13, q13, q0
    VADD.F32 q14, q14, q0
    VADD.F32 q15, q15, q0

    # Load c, c_stride:
    # - r0 = c
    # - r10 = c_stride
    LDR r0, [sp, 116]
    LDR r10, [sp, 120]
    LSL r10, r10, 2

    # load mr from stack
    LDR r9, [sp, #-4]
    # load nr from stack
    LDR r8, [sp, #-8]
    # r0 = c0 = c pointer

    CMP r9, 2
    # r1 = c1
    ADD r1, r0, r10
    MOVLO r1, r0

    # r2 = c2
    ADD r2, r1, r10
    MOVLS r2, r1

    CMP r9, 4
    # r3 = c3
    ADD r3, r2, r10
    MOVLO r3, r2

    # r4 = c4
    ADD r4, r3, r10
    MOVLS r4, r3

    CMP r9, 6
    # r5 = c5
    ADD r5, r4, r10
    MOVLO r5, r4

    # r6 = c6
    ADD r6, r5, r10
    MOVLS r6, r5

    CMP r9, 8
    # r7 = c7
    ADD r7, r6, r10
    MOVNE r7, r6

    CMP r8, 4
    BNE 4f

    VST1.32 {q8}, [r0]
    VST1.32 {q9}, [r1]
    VST1.32 {q10}, [r2]
    VST1.32 {q11}, [r3]
    VST1.32 {q12}, [r4]
    VST1.32 {q13}, [r5]
    VST1.32 {q14}, [r6]
    VST1.32 {q15}, [r7]

    VPOP {d8-d15}
    POP {r4, r5, r6, r7, r8, r9, r10, r11, lr}
    BX lr

    .p2align 3
4:
    CMP r8, 2
    BLO 5f

    VST1.32 {d16}, [r0]!
    VST1.32 {d18}, [r1]!
    VST1.32 {d20}, [r2]!
    VST1.32 {d22}, [r3]!
    VST1.32 {d24}, [r4]!
    VST1.32 {d26}, [r5]!
    VST1.32 {d28}, [r6]!
    VST1.32 {d30}, [r7]!

    SUB r8, 2

    VMOV.32 d16, d17
    VMOV.32 d18, d19
    VMOV.32 d20, d21
    VMOV.32 d22, d23
    VMOV.32 d24, d25
    VMOV.32 d26, d27
    VMOV.32 d28, d29
    VMOV.32 d30, d31

5:
    CMP r8, 0
    BEQ 7f

    VST1.32 {d16[0]}, [r0]
    VST1.32 {d18[0]}, [r1]
    VST1.32 {d20[0]}, [r2]
    VST1.32 {d22[0]}, [r3]
    VST1.32 {d24[0]}, [r4]
    VST1.32 {d26[0]}, [r5]
    VST1.32 {d28[0]}, [r6]
    VST1.32 {d30[0]}, [r7]

7:
    VPOP {d8-d15}
    POP {r4, r5, r6, r7, r8, r9, r10, r11, lr}
    BX lr

END_FUNCTION pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
