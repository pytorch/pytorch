/*
 * Copyright (c) 2020 - Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>
#include <requantization/runtime-assembly.h>

.syntax unified

	.fnstart
BEGIN_FUNCTION pytorch_q8vadd_ukernel__aarch32_neon
    .arm
#ifndef __APPLE__
    .arch armv7-a
    .fpu neon
#endif
@ %bb.0:
	.save	{r4, r5, r6, r10, r11, lr}
	push	{r4, r5, r6, r10, r11, lr}
	.setfp	r11, sp, #16
	add	r11, sp, #16
	.vsave	{d8, d9, d10, d11, d12, d13, d14, d15}
	vpush	{d8, d9, d10, d11, d12, d13, d14, d15}
	.pad	#24
	sub	sp, sp, #24
	bfc	sp, #0, #4
	ldr	r6, [r11, #8]
	mov	r4, #28
	cmp	r0, #7
	mov	r5, r6
	ldrb	lr, [r6, #15]
	vld1.8	{d20[]}, [r5], r4
	ldrb	r4, [r6, #14]
	ldr	r12, [r5]
	add	r5, r6, #1
	vmov.8	d18[0], lr
	vld1.8	{d21[]}, [r5]
	vmov.8	d6[0], r4
	vldr	s0, [r6, #4]
	vldr	s4, [r6, #8]
	vldr	s8, [r6, #24]
	bls	.LBB0_16
@ %bb.1:
	vdup.32	q11, d4[0]
	cmp	r0, #32
	vdup.32	q13, d2[0]
	vdup.32	q14, d0[0]
	vdup.32	q12, r12
	blo	.LBB0_5
@ %bb.2:
	sub	r6, r0, #32
	vdup.8	q15, lr
	vdup.8	q0, r4
	bic	r12, r6, #15
	vld1.8	{d4, d5}, [r2]
	add	r2, r2, #16
	mov	lr, r0
	vld1.8	{d2, d3}, [r1]
	add	r1, r1, #16
	vst1.64	{d6, d7}, [sp:128]      @ 16-byte Spill
	vsubl.u8	q3, d5, d21
	vsubl.u8	q2, d4, d21
	vsubl.u8	q5, d3, d20
	vsubl.u8	q1, d2, d20
.LBB0_3:                                @ =>This Inner Loop Header: Depth=1
	sub	lr, lr, #16
	vmovl.s16	q8, d7
	cmp	lr, #31
	vmovl.s16	q6, d5
	vmovl.s16	q7, d6
	vmovl.s16	q4, d4
	vcvt.f32.s32	q8, q8
	vcvt.f32.s32	q7, q7
	vcvt.f32.s32	q6, q6
	vmovl.s16	q3, d10
	vmovl.s16	q5, d11
	vcvt.f32.s32	q4, q4
	vmovl.s16	q2, d3
	vmovl.s16	q1, d2
	vcvt.f32.s32	q5, q5
	vmul.f32	q8, q13, q8
	vcvt.f32.s32	q2, q2
	vmul.f32	q7, q13, q7
	vmul.f32	q6, q13, q6
	vcvt.f32.s32	q3, q3
	vcvt.f32.s32	q1, q1
	vmul.f32	q4, q13, q4
	vmla.f32	q8, q14, q5
	vmla.f32	q6, q14, q2
	vmla.f32	q7, q14, q3
	vmla.f32	q4, q14, q1
	vld1.8	{d4, d5}, [r2]
	add	r2, r2, #16
	vld1.8	{d2, d3}, [r1]
	add	r1, r1, #16
	vadd.f32	q8, q11, q8
	vadd.f32	q7, q11, q7
	vadd.f32	q6, q11, q6
	vadd.f32	q4, q11, q4
	vsub.i32	q8, q8, q12
	vsub.i32	q7, q7, q12
	vsub.i32	q6, q6, q12
	vsub.i32	q4, q4, q12
	vqmovn.s32	d17, q8
	vqmovn.s32	d16, q7
	vqmovn.s32	d15, q6
	vqmovn.s32	d14, q4
	vsubl.u8	q3, d5, d21
	vqmovun.s16	d17, q8
	vqmovun.s16	d16, q7
	vsubl.u8	q2, d4, d21
	vsubl.u8	q5, d3, d20
	vsubl.u8	q1, d2, d20
	vmax.u8	q8, q8, q15
	vmin.u8	q8, q8, q0
	vst1.8	{d16, d17}, [r3]!
	bhi	.LBB0_3
@ %bb.4:
	sub	r0, r0, r12
	sub	r2, r2, #16
	sub	r1, r1, #16
	vld1.64	{d6, d7}, [sp:128]      @ 16-byte Reload
	sub	r0, r0, #16
	cmp	r0, #8
	blo	.LBB0_8
.LBB0_5:
	vdup.8	d30, d6[0]
	mov	r12, r0
	vdup.8	d31, d18[0]
.LBB0_6:                                @ =>This Inner Loop Header: Depth=1
	vld1.8	{d16}, [r1]!
	sub	r12, r12, #8
	cmp	r12, #7
	vsubl.u8	q8, d16, d20
	vld1.8	{d0}, [r2]!
	vsubl.u8	q0, d0, d21
	vmovl.s16	q1, d17
	vmovl.s16	q8, d16
	vmovl.s16	q2, d1
	vcvt.f32.s32	q1, q1
	vcvt.f32.s32	q8, q8
	vmovl.s16	q0, d0
	vcvt.f32.s32	q2, q2
	vmul.f32	q1, q14, q1
	vcvt.f32.s32	q0, q0
	vmul.f32	q8, q14, q8
	vmla.f32	q1, q13, q2
	vmla.f32	q8, q13, q0
	vadd.f32	q0, q11, q1
	vadd.f32	q8, q11, q8
	vsub.i32	q0, q0, q12
	vsub.i32	q8, q8, q12
	vqmovn.s32	d1, q0
	vqmovn.s32	d0, q8
	vqmovun.s16	d16, q0
	vmax.u8	d16, d16, d31
	vmin.u8	d16, d16, d30
	vst1.8	{d16}, [r3]!
	bhi	.LBB0_6
@ %bb.7:
	and	r0, r0, #7
.LBB0_8:
	cmp	r0, #0
	beq	.LBB0_15
@ %bb.9:
	sub	r6, r0, #8
	vldr	s1, .LCPI0_0
	add	r1, r1, r6
	tst	r0, #4
	lsl	r5, r6, #3
	vld1.8	{d16}, [r1]
	vmov	s0, r5
	add	r1, r2, r6
	vshl.u64	d16, d16, d0
	vld1.8	{d17}, [r1]
	vsubl.u8	q15, d16, d20
	vshl.u64	d17, d17, d0
	vmovl.s16	q0, d31
	vmovl.s16	q15, d30
	vsubl.u8	q8, d17, d21
	vcvt.f32.s32	q10, q0
	vcvt.f32.s32	q15, q15
	vmovl.s16	q0, d17
	vmovl.s16	q8, d16
	vmul.f32	q10, q14, q10
	vcvt.f32.s32	q0, q0
	vcvt.f32.s32	q8, q8
	vmul.f32	q14, q14, q15
	vmla.f32	q10, q13, q0
	vmla.f32	q14, q13, q8
	vadd.f32	q8, q11, q10
	vadd.f32	q10, q11, q14
	vsub.i32	q8, q8, q12
	vsub.i32	q10, q10, q12
	vqmovn.s32	d17, q8
	vqmovn.s32	d16, q10
	vqmovun.s16	d16, q8
	vdup.8	d17, d18[0]
	vmax.u8	d16, d16, d17
	vdup.8	d17, d6[0]
	vmin.u8	d17, d16, d17
	bne	.LBB0_12
@ %bb.10:
	vorr	d16, d17, d17
	tst	r0, #2
	beq	.LBB0_13
.LBB0_11:
	vext.8	d17, d16, d16, #2
	vst1.16	{d16[0]}, [r3]!
	tst	r0, #1
	bne	.LBB0_14
	b	.LBB0_15
.LBB0_12:
	vext.8	d16, d17, d17, #4
	vst1.32	{d17[0]}, [r3]!
	tst	r0, #2
	bne	.LBB0_11
.LBB0_13:
	vorr	d17, d16, d16
	tst	r0, #1
	beq	.LBB0_15
.LBB0_14:
	vst1.8	{d17[0]}, [r3]
.LBB0_15:
	sub	sp, r11, #80
	vpop	{d8, d9, d10, d11, d12, d13, d14, d15}
	pop	{r4, r5, r6, r10, r11, pc}
.LBB0_16:
	cmp	r0, #0
	beq	.LBB0_15
@ %bb.17:
	vmov.32	d22[0], r12
	vmovl.u8	q12, d20
	vmovl.u8	q10, d21
	vdup.8	d17, d18[0]
	vdup.32	d18, d22[0]
	vdup.8	d16, d6[0]
	vdup.32	d19, d4[0]
	vdup.32	d22, d2[0]
	vdup.32	d23, d0[0]
.LBB0_18:                               @ =>This Inner Loop Header: Depth=1
	vld1.8	{d26[]}, [r1]!
	subs	r0, r0, #1
	vmovl.u8	q13, d26
	vld1.8	{d28[]}, [r2]!
	vmovl.u8	q14, d28
	vsub.i16	d26, d26, d24
	vsub.i16	d28, d28, d20
	vmovl.s16	q13, d26
	vmovl.s16	q14, d28
	vcvt.f32.s32	d26, d26
	vcvt.f32.s32	d27, d28
	vmul.f32	d26, d23, d26
	vmla.f32	d26, d22, d27
	vadd.f32	d26, d19, d26
	vsub.i32	d26, d26, d18
	vorr	d27, d26, d26
	vqmovn.s32	d26, q13
	vorr	d27, d26, d26
	vqmovun.s16	d26, q13
	vmax.u8	d26, d26, d17
	vmin.u8	d26, d26, d16
	vst1.8	{d26[0]}, [r3]!
	bne	.LBB0_18
	b	.LBB0_15
	.p2align	2
@ %bb.19:
.LCPI0_0:
	.long	0                       @ float 0
	.fnend
END_FUNCTION pytorch_q8vadd_ukernel__aarch32_neon
