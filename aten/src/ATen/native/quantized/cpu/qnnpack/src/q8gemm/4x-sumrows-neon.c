/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/q8gemm.h>

void pytorch_q8sumrows_ukernel_4x__neon(
    const uint8_t* restrict a,
    size_t m,
    size_t k,
    size_t stride,
    const int32_t multiplier,
    int32_t* restrict a_sum) {
  const uint8_t* a0 = a;
  const uint8_t* a1 = a0;
  if (m >= 2) {
    a1 += stride;
  }
  const uint8_t* a2 = a1;
  if (m > 2) {
    a2 += stride;
  }
  const uint8_t* a3 = a2;
  if (m == 4) {
    a3 += stride;
  }

  uint32x4_t vacc0x0123 = vmovq_n_u32(0); // row 0
  uint32x4_t vacc1x0123 = vmovq_n_u32(0); // row 1
  uint32x4_t vacc2x0123 = vmovq_n_u32(0); // row 2
  uint32x4_t vacc3x0123 = vmovq_n_u32(0); // row 3
  for (; k >= 16; k -= 16) {
    // row 0
    const uint8x16_t va0x0_15 = vld1q_u8(a0);
    a0 += 16;
    vacc0x0123 = vpadalq_u16(
        vacc0x0123, vaddl_u8(vget_low_u8(va0x0_15), vget_high_u8(va0x0_15)));

    // row 1
    const uint8x16_t va1x0_15 = vld1q_u8(a1);
    a1 += 16;
    vacc1x0123 = vpadalq_u16(
        vacc1x0123, vaddl_u8(vget_low_u8(va1x0_15), vget_high_u8(va1x0_15)));

    // row 2
    const uint8x16_t va2x0_15 = vld1q_u8(a2);
    a2 += 16;
    vacc2x0123 = vpadalq_u16(
        vacc2x0123, vaddl_u8(vget_low_u8(va2x0_15), vget_high_u8(va2x0_15)));

    // row 3
    const uint8x16_t va3x0_15 = vld1q_u8(a3);
    a3 += 16;
    vacc3x0123 = vpadalq_u16(
        vacc3x0123, vaddl_u8(vget_low_u8(va3x0_15), vget_high_u8(va3x0_15)));
  }

  if (k >= 8) {
    vacc0x0123 = vaddw_u16(vacc0x0123, vpaddl_u8(vld1_u8(a0)));
    a0 += 8;
    vacc1x0123 = vaddw_u16(vacc1x0123, vpaddl_u8(vld1_u8(a1)));
    a1 += 8;
    vacc2x0123 = vaddw_u16(vacc2x0123, vpaddl_u8(vld1_u8(a2)));
    a2 += 8;
    vacc3x0123 = vaddw_u16(vacc3x0123, vpaddl_u8(vld1_u8(a3)));
    a3 += 8;
    k -= 8;
  }

  if (k >= 4) {
    vacc0x0123 = vaddw_u16(
        vacc0x0123,
        vget_low_u16(vmovl_u8(vreinterpret_u8_u32(
            vld1_dup_u32(__builtin_assume_aligned((const uint32_t*)a0, 1))))));
    a0 += 4;
    vacc1x0123 = vaddw_u16(
        vacc1x0123,
        vget_low_u16(vmovl_u8(vreinterpret_u8_u32(
            vld1_dup_u32(__builtin_assume_aligned((const uint32_t*)a1, 1))))));
    a1 += 4;
    vacc2x0123 = vaddw_u16(
        vacc2x0123,
        vget_low_u16(vmovl_u8(vreinterpret_u8_u32(
            vld1_dup_u32(__builtin_assume_aligned((const uint32_t*)a2, 1))))));
    a2 += 4;
    vacc3x0123 = vaddw_u16(
        vacc3x0123,
        vget_low_u16(vmovl_u8(vreinterpret_u8_u32(
            vld1_dup_u32(__builtin_assume_aligned((const uint32_t*)a3, 1))))));
    a3 += 4;
    k -= 4;
  }

  const uint32x2_t vsum0x01 =
      vpadd_u32(vget_low_u32(vacc0x0123), vget_high_u32(vacc0x0123));
  const uint32x2_t vsum1x01 =
      vpadd_u32(vget_low_u32(vacc1x0123), vget_high_u32(vacc1x0123));
  const uint32x2_t vsum2x01 =
      vpadd_u32(vget_low_u32(vacc2x0123), vget_high_u32(vacc2x0123));
  const uint32x2_t vsum3x01 =
      vpadd_u32(vget_low_u32(vacc3x0123), vget_high_u32(vacc3x0123));
  uint32x4_t vacc0123 = vcombine_u32(
      vpadd_u32(vsum0x01, vsum1x01), vpadd_u32(vsum2x01, vsum3x01));

  if (k >= 2) {
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
    const uint8x8_t va0x01_1x010101 = vext_u8(va0x01010101, va1x01010101, 2);
    const uint8x8_t va2x01_3x010101 = vext_u8(va2x01010101, va3x01010101, 6);
    const uint8x8_t va0123x01 = vext_u8(va0x01_1x010101, va2x01_3x010101, 4);
    vacc0123 = vaddw_u16(vacc0123, vpaddl_u8(va0123x01));
    k -= 2;
  }

  if (k > 0) {
    uint8x8_t vax0x1x2x3 = vmov_n_u8(0);
    vax0x1x2x3 = vld1_lane_u8(a0, vax0x1x2x3, 0);
    vax0x1x2x3 = vld1_lane_u8(a1, vax0x1x2x3, 2);
    vax0x1x2x3 = vld1_lane_u8(a2, vax0x1x2x3, 4);
    vax0x1x2x3 = vld1_lane_u8(a3, vax0x1x2x3, 6);
    vacc0123 = vaddw_u16(vacc0123, vpaddl_u8(vax0x1x2x3));
  }

  int32x4_t vsum0123 = vmulq_n_s32(vreinterpretq_s32_u32(vacc0123), multiplier);
  if (m == 4) {
    vst1q_s32(a_sum, vsum0123);
  } else {
    if (m >= 2) {
      vst1_s32(a_sum, vget_low_s32(vsum0123));
      a_sum += 2;
      vsum0123 = vextq_s32(vsum0123, vsum0123, 2);
      m -= 2;
    }
    if (m != 0) {
      vst1q_lane_s32(a_sum, vsum0123, 0);
    }
  }
}
