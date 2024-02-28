/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/q8gemm.h>

void pytorch_q8gemm_xzp_ukernel_4x8c2__neon(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const int32_t* restrict a_sum,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    const union pytorch_qnnp_q31_requantization_params
        requantization_params[restrict static 1]) {
  int32x4_t vacc0x0123 = vld1q_s32(w);
  w = (const void*)((uintptr_t)w + 16);
  int32x4_t vacc0x4567 = vld1q_s32(w);
  w = (const void*)((uintptr_t)w + 16);
  int32x4_t vacc1x0123 = vacc0x0123;
  int32x4_t vacc1x4567 = vacc0x4567;
  int32x4_t vacc2x0123 = vacc0x0123;
  int32x4_t vacc2x4567 = vacc0x4567;
  int32x4_t vacc3x0123 = vacc0x0123;
  int32x4_t vacc3x4567 = vacc0x4567;

  const uint8_t* a0 = a;
  const uint8_t* a1 = a0;
  const int32_t* a_sum0 = a_sum;
  const int32_t* a_sum1 = a_sum0;
  if (mr >= 2) {
    a1 += a_stride;
    a_sum1 += 1;
  }
  const uint8_t* a2 = a1;
  const int32_t* a_sum2 = a_sum1;
  if (mr > 2) {
    a2 += a_stride;
    a_sum2 += 1;
  }
  const uint8_t* a3 = a2;
  const int32_t* a_sum3 = a_sum2;
  if (mr == 4) {
    a3 += a_stride;
    a_sum3 += 1;
  }

  const int32x4_t va_sum0 = vld1q_dup_s32(a_sum0);
  const int32x4_t va_sum1 = vld1q_dup_s32(a_sum1);
  const int32x4_t va_sum2 = vld1q_dup_s32(a_sum2);
  const int32x4_t va_sum3 = vld1q_dup_s32(a_sum3);
  vacc0x0123 = vaddq_s32(vacc0x0123, va_sum0);
  vacc0x4567 = vaddq_s32(vacc0x4567, va_sum0);
  vacc1x0123 = vaddq_s32(vacc1x0123, va_sum1);
  vacc1x4567 = vaddq_s32(vacc1x4567, va_sum1);
  vacc2x0123 = vaddq_s32(vacc2x0123, va_sum2);
  vacc2x4567 = vaddq_s32(vacc2x4567, va_sum2);
  vacc3x0123 = vaddq_s32(vacc3x0123, va_sum3);
  vacc3x4567 = vaddq_s32(vacc3x4567, va_sum3);

  for (; k >= 8; k -= 8) {
    uint8x8_t va0x01234567 = vld1_u8(a0);
    a0 += 8;
    uint8x8_t va1x01234567 = vld1_u8(a1);
    a1 += 8;
    uint8x8_t va2x01234567 = vld1_u8(a2);
    a2 += 8;
    uint8x8_t va3x01234567 = vld1_u8(a3);
    a3 += 8;

    /* k = 0, 1 */
    const uint8x16_t vb01234567x01 = vld1q_u8(w);
    w += 16;

    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x01234567, vget_low_u8(vb01234567x01))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x01234567, vget_high_u8(vb01234567x01))));

    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01234567, vget_low_u8(vb01234567x01))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01234567, vget_high_u8(vb01234567x01))));

    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01234567, vget_low_u8(vb01234567x01))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01234567, vget_high_u8(vb01234567x01))));

    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01234567, vget_low_u8(vb01234567x01))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01234567, vget_high_u8(vb01234567x01))));

    /* k = 2, 3 */
    va0x01234567 = vext_u8(va0x01234567, va0x01234567, 2);
    va1x01234567 = vext_u8(va1x01234567, va1x01234567, 2);
    va2x01234567 = vext_u8(va2x01234567, va2x01234567, 2);
    va3x01234567 = vext_u8(va3x01234567, va3x01234567, 2);

    const uint8x16_t vb01234567x23 = vld1q_u8(w);
    w += 16;

    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x01234567, vget_low_u8(vb01234567x23))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x01234567, vget_high_u8(vb01234567x23))));

    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01234567, vget_low_u8(vb01234567x23))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01234567, vget_high_u8(vb01234567x23))));

    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01234567, vget_low_u8(vb01234567x23))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01234567, vget_high_u8(vb01234567x23))));

    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01234567, vget_low_u8(vb01234567x23))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01234567, vget_high_u8(vb01234567x23))));

    /* k = 4, 5 */
    va0x01234567 = vext_u8(va0x01234567, va0x01234567, 2);
    va1x01234567 = vext_u8(va1x01234567, va1x01234567, 2);
    va2x01234567 = vext_u8(va2x01234567, va2x01234567, 2);
    va3x01234567 = vext_u8(va3x01234567, va3x01234567, 2);

    const uint8x16_t vb01234567x45 = vld1q_u8(w);
    w += 16;

    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x01234567, vget_low_u8(vb01234567x45))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x01234567, vget_high_u8(vb01234567x45))));

    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01234567, vget_low_u8(vb01234567x45))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01234567, vget_high_u8(vb01234567x45))));

    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01234567, vget_low_u8(vb01234567x45))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01234567, vget_high_u8(vb01234567x45))));

    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01234567, vget_low_u8(vb01234567x45))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01234567, vget_high_u8(vb01234567x45))));

    /* k = 6, 7 */
    va0x01234567 = vext_u8(va0x01234567, va0x01234567, 2);
    va1x01234567 = vext_u8(va1x01234567, va1x01234567, 2);
    va2x01234567 = vext_u8(va2x01234567, va2x01234567, 2);
    va3x01234567 = vext_u8(va3x01234567, va3x01234567, 2);

    const uint8x16_t vb01234567x67 = vld1q_u8(w);
    w += 16;

    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x01234567, vget_low_u8(vb01234567x67))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x01234567, vget_high_u8(vb01234567x67))));

    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01234567, vget_low_u8(vb01234567x67))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01234567, vget_high_u8(vb01234567x67))));

    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01234567, vget_low_u8(vb01234567x67))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01234567, vget_high_u8(vb01234567x67))));

    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01234567, vget_low_u8(vb01234567x67))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01234567, vget_high_u8(vb01234567x67))));
  }

  /* for k < 8, reuse the packing scheme for the original xzp ukernel */
  if (k & 4) {
    /* k = 0, 1 */
    const uint8x8_t va0x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a0, 1)));
    a0 += 2;
    const uint8x8_t va1x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a1, 1)));
    a1 += 2;
    const uint8x8_t va2x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a2, 1)));
    a2 += 2;
    const uint8x8_t va3x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a3, 1)));
    a3 += 2;
    const uint8x16_t vb01234567x01 = vld1q_u8(w);
    w += 16;
    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x01010101, vget_low_u8(vb01234567x01))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x01010101, vget_high_u8(vb01234567x01))));
    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01010101, vget_low_u8(vb01234567x01))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01010101, vget_high_u8(vb01234567x01))));
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01010101, vget_low_u8(vb01234567x01))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01010101, vget_high_u8(vb01234567x01))));
    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01010101, vget_low_u8(vb01234567x01))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01010101, vget_high_u8(vb01234567x01))));

    /* k = 2, 3 */
    const uint8x8_t va0x23232323 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a0, 1)));
    a0 += 2;
    const uint8x8_t va1x23232323 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a1, 1)));
    a1 += 2;
    const uint8x8_t va2x23232323 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a2, 1)));
    a2 += 2;
    const uint8x8_t va3x23232323 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a3, 1)));
    a3 += 2;
    const uint8x16_t vb01234567x23 = vld1q_u8(w);
    w += 16;
    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x23232323, vget_low_u8(vb01234567x23))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x23232323, vget_high_u8(vb01234567x23))));
    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x23232323, vget_low_u8(vb01234567x23))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x23232323, vget_high_u8(vb01234567x23))));
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x23232323, vget_low_u8(vb01234567x23))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x23232323, vget_high_u8(vb01234567x23))));
    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x23232323, vget_low_u8(vb01234567x23))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x23232323, vget_high_u8(vb01234567x23))));
  }
  if (k & 2) {
    /* k = 0, 1 */
    const uint8x8_t va0x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a0, 1)));
    a0 += 2;
    const uint8x8_t va1x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a1, 1)));
    a1 += 2;
    const uint8x8_t va2x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a2, 1)));
    a2 += 2;
    const uint8x8_t va3x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a3, 1)));
    a3 += 2;
    const uint8x16_t vb01234567x01 = vld1q_u8(w);
    w += 16;
    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x01010101, vget_low_u8(vb01234567x01))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x01010101, vget_high_u8(vb01234567x01))));
    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01010101, vget_low_u8(vb01234567x01))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01010101, vget_high_u8(vb01234567x01))));
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01010101, vget_low_u8(vb01234567x01))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01010101, vget_high_u8(vb01234567x01))));
    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01010101, vget_low_u8(vb01234567x01))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01010101, vget_high_u8(vb01234567x01))));
  }
  if (k & 1) {
    const uint8x8_t va0x00000000 = vld1_dup_u8(a0);
    const uint8x8_t va1x00000000 = vld1_dup_u8(a1);
    const uint8x8_t va2x00000000 = vld1_dup_u8(a2);
    const uint8x8_t va3x00000000 = vld1_dup_u8(a3);
    const uint8x16_t vb01234567x0 = vld1q_u8(w);
    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x00000000, vget_low_u8(vb01234567x0))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x00000000, vget_high_u8(vb01234567x0))));
    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x00000000, vget_low_u8(vb01234567x0))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x00000000, vget_high_u8(vb01234567x0))));
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x00000000, vget_low_u8(vb01234567x0))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x00000000, vget_high_u8(vb01234567x0))));
    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x00000000, vget_low_u8(vb01234567x0))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x00000000, vget_high_u8(vb01234567x0))));
  }

  const int32x4_t vmultiplier =
      vld1q_dup_s32(&requantization_params->neon.multiplier);
  vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
  vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);
  vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier);
  vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier);
  vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier);
  vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier);
  vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier);
  vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier);

  const int32x4_t vright_shift =
      vld1q_dup_s32(&requantization_params->neon.right_shift);
  const int32x4_t vzero_shift_mask =
      vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
  vacc0x0123 =
      vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);
  vacc0x4567 =
      vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), 31);
  vacc1x0123 =
      vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), 31);
  vacc1x4567 =
      vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask), 31);
  vacc2x0123 =
      vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask), 31);
  vacc2x4567 =
      vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask), 31);
  vacc3x0123 =
      vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask), 31);
  vacc3x4567 =
      vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask), 31);

  vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
  vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
  vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift);
  vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift);
  vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift);
  vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift);
  vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift);
  vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift);

  const int16x8_t vzero_point =
      vld1q_dup_s16(&requantization_params->neon.zero_point);
#ifdef __aarch64__
  const int16x8_t vacc0x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), vzero_point);
  const int16x8_t vacc1x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), vzero_point);
  const int16x8_t vacc2x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), vzero_point);
  const int16x8_t vacc3x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), vzero_point);

  uint8x16_t vout0x01234567_1x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
  uint8x16_t vout2x01234567_3x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);
#else
  const int16x8_t vacc0x01234567 = vqaddq_s16(
      vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)),
      vzero_point);
  const int16x8_t vacc1x01234567 = vqaddq_s16(
      vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)),
      vzero_point);
  const int16x8_t vacc2x01234567 = vqaddq_s16(
      vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)),
      vzero_point);
  const int16x8_t vacc3x01234567 = vqaddq_s16(
      vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)),
      vzero_point);

  uint8x16_t vout0x01234567_1x01234567 =
      vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc1x01234567));
  uint8x16_t vout2x01234567_3x01234567 =
      vcombine_u8(vqmovun_s16(vacc2x01234567), vqmovun_s16(vacc3x01234567));
#endif
  const uint8x16_t vmin = vld1q_dup_u8(&requantization_params->neon.min);
  const uint8x16_t vmax = vld1q_dup_u8(&requantization_params->neon.max);

  vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, vmin);
  vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, vmin);
  vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, vmax);
  vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, vmax);

  uint8_t* c0 = c;
  uint8_t* c1 = c0;
  if (mr >= 2) {
    c1 += c_stride;
  }
  uint8_t* c2 = c1;
  if (mr > 2) {
    c2 += c_stride;
  }
  uint8_t* c3 = c2;
  if (mr == 4) {
    c3 += c_stride;
  }
  if (nr == 8) {
    vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
    vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
    vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
    vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
  } else {
    if (nr >= 4) {
      vst1q_lane_u32(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u32_u8(vout0x01234567_1x01234567),
          0);
      c0 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u32_u8(vout0x01234567_1x01234567),
          2);
      c1 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u32_u8(vout2x01234567_3x01234567),
          0);
      c2 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u32_u8(vout2x01234567_3x01234567),
          2);
      c3 += 4;
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      nr -= 4;
    }
    if (nr >= 2) {
      vst1q_lane_u16(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          0);
      c0 += 2;
      vst1q_lane_u16(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          4);
      c1 += 2;
      vst1q_lane_u16(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          0);
      c2 += 2;
      vst1q_lane_u16(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          4);
      c3 += 2;
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      nr -= 2;
    }
    if (nr != 0) {
      vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
      vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
      vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
      vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
    }
  }
}
