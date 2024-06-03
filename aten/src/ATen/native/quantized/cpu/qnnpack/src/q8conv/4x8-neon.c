/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/q8conv.h>
#include <requantization/runtime-neon.h>

void pytorch_q8conv_ukernel_4x8__neon(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[restrict static 1]) {
  const uint8x8_t va_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);
  // Assumes that kernel_zero_points is an array padded with necessary elements
  // in order to make it multiple of 8.
  const uint8x8_t vb_zero_point =
      vld1_u8((const uint8_t*)&quantization_params->neon.kernel_zero_points
          [output_channel_index]);

  int32x4_t vacc0x0123 = vld1q_s32(w);
  w = (void*)((uintptr_t)w + sizeof(int32x4_t));
  int32x4_t vacc0x4567 = vld1q_s32(w);
  w = (void*)((uintptr_t)w + sizeof(int32x4_t));
  int32x4_t vacc1x0123 = vacc0x0123;
  int32x4_t vacc1x4567 = vacc0x4567;
  int32x4_t vacc2x0123 = vacc0x0123;
  int32x4_t vacc2x4567 = vacc0x4567;
  int32x4_t vacc3x0123 = vacc0x0123;
  int32x4_t vacc3x4567 = vacc0x4567;

  do {
    const uint8_t* restrict a0 = *a++;
    const uint8_t* restrict a1 = *a++;
    const uint8_t* restrict a2 = *a++;
    const uint8_t* restrict a3 = *a++;

    size_t k = kc;
    for (; k >= 8; k -= 8) {
      const uint8x8_t va0 = vld1_u8(a0);
      a0 += 8;
      const uint8x8_t va1 = vld1_u8(a1);
      a1 += 8;
      const uint8x8_t va2 = vld1_u8(a2);
      a2 += 8;
      const uint8x8_t va3 = vld1_u8(a3);
      a3 += 8;
      const int16x8_t vxa0 =
          vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
      const int16x8_t vxa1 =
          vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));
      const int16x8_t vxa2 =
          vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));
      const int16x8_t vxa3 =
          vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));

      {
        const uint8x8_t vb01234567 = vld1_u8(w);
        w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
        const int16x8_t vxb01234567 =
            vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(
            vacc0x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa0), 0);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa1), 0);
        vacc1x4567 = vmlal_lane_s16(
            vacc1x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa1), 0);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa2), 0);
        vacc2x4567 = vmlal_lane_s16(
            vacc2x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa2), 0);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa3), 0);
        vacc3x4567 = vmlal_lane_s16(
            vacc3x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa3), 0);
      }

      {
        const uint8x8_t vb01234567 = vld1_u8(w);
        w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
        const int16x8_t vxb01234567 =
            vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(
            vacc0x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa0), 1);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa1), 1);
        vacc1x4567 = vmlal_lane_s16(
            vacc1x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa1), 1);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa2), 1);
        vacc2x4567 = vmlal_lane_s16(
            vacc2x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa2), 1);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa3), 1);
        vacc3x4567 = vmlal_lane_s16(
            vacc3x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa3), 1);
      }

      {
        const uint8x8_t vb01234567 = vld1_u8(w);
        w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
        const int16x8_t vxb01234567 =
            vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(
            vacc0x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa0), 2);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa1), 2);
        vacc1x4567 = vmlal_lane_s16(
            vacc1x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa1), 2);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa2), 2);
        vacc2x4567 = vmlal_lane_s16(
            vacc2x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa2), 2);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa3), 2);
        vacc3x4567 = vmlal_lane_s16(
            vacc3x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa3), 2);
      }

      {
        const uint8x8_t vb01234567 = vld1_u8(w);
        w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
        const int16x8_t vxb01234567 =
            vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(
            vacc0x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa0), 3);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa1), 3);
        vacc1x4567 = vmlal_lane_s16(
            vacc1x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa1), 3);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa2), 3);
        vacc2x4567 = vmlal_lane_s16(
            vacc2x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa2), 3);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa3), 3);
        vacc3x4567 = vmlal_lane_s16(
            vacc3x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa3), 3);
      }

      {
        const uint8x8_t vb01234567 = vld1_u8(w);
        w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
        const int16x8_t vxb01234567 =
            vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(
            vacc0x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa0), 0);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa1), 0);
        vacc1x4567 = vmlal_lane_s16(
            vacc1x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa1), 0);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa2), 0);
        vacc2x4567 = vmlal_lane_s16(
            vacc2x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa2), 0);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa3), 0);
        vacc3x4567 = vmlal_lane_s16(
            vacc3x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa3), 0);
      }

      {
        const uint8x8_t vb01234567 = vld1_u8(w);
        w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
        const int16x8_t vxb01234567 =
            vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(
            vacc0x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa0), 1);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa1), 1);
        vacc1x4567 = vmlal_lane_s16(
            vacc1x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa1), 1);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa2), 1);
        vacc2x4567 = vmlal_lane_s16(
            vacc2x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa2), 1);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa3), 1);
        vacc3x4567 = vmlal_lane_s16(
            vacc3x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa3), 1);
      }

      {
        const uint8x8_t vb01234567 = vld1_u8(w);
        w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
        const int16x8_t vxb01234567 =
            vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(
            vacc0x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa0), 2);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa1), 2);
        vacc1x4567 = vmlal_lane_s16(
            vacc1x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa1), 2);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa2), 2);
        vacc2x4567 = vmlal_lane_s16(
            vacc2x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa2), 2);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa3), 2);
        vacc3x4567 = vmlal_lane_s16(
            vacc3x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa3), 2);
      }

      {
        const uint8x8_t vb01234567 = vld1_u8(w);
        w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
        const int16x8_t vxb01234567 =
            vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(
            vacc0x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa0), 3);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa1), 3);
        vacc1x4567 = vmlal_lane_s16(
            vacc1x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa1), 3);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa2), 3);
        vacc2x4567 = vmlal_lane_s16(
            vacc2x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa2), 3);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa3), 3);
        vacc3x4567 = vmlal_lane_s16(
            vacc3x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa3), 3);
      }
    }
    if (k != 0) {
      const size_t a_predecrement = 8 - k;
      const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
      const uint8x8_t va0 = vreinterpret_u8_u64(vshl_u64(
          vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
      const uint8x8_t va1 = vreinterpret_u8_u64(vshl_u64(
          vreinterpret_u64_u8(vld1_u8(a1 - a_predecrement)), va_shift));
      const uint8x8_t va2 = vreinterpret_u8_u64(vshl_u64(
          vreinterpret_u64_u8(vld1_u8(a2 - a_predecrement)), va_shift));
      const uint8x8_t va3 = vreinterpret_u8_u64(vshl_u64(
          vreinterpret_u64_u8(vld1_u8(a3 - a_predecrement)), va_shift));
      const int16x8_t vxa0 =
          vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
      const int16x8_t vxa1 =
          vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));
      const int16x8_t vxa2 =
          vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));
      const int16x8_t vxa3 =
          vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));

      {
        const uint8x8_t vb01234567 = vld1_u8(w);
        w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
        const int16x8_t vxb01234567 =
            vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(
            vacc0x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa0), 0);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa1), 0);
        vacc1x4567 = vmlal_lane_s16(
            vacc1x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa1), 0);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa2), 0);
        vacc2x4567 = vmlal_lane_s16(
            vacc2x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa2), 0);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa3), 0);
        vacc3x4567 = vmlal_lane_s16(
            vacc3x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa3), 0);
      }

      if (k >= 2) {
        const uint8x8_t vb01234567 = vld1_u8(w);
        w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
        const int16x8_t vxb01234567 =
            vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(
            vacc0x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa0), 1);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa1), 1);
        vacc1x4567 = vmlal_lane_s16(
            vacc1x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa1), 1);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa2), 1);
        vacc2x4567 = vmlal_lane_s16(
            vacc2x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa2), 1);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa3), 1);
        vacc3x4567 = vmlal_lane_s16(
            vacc3x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa3), 1);

        if (k > 2) {
          const uint8x8_t vb01234567 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const int16x8_t vxb01234567 =
              vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

          vacc0x0123 = vmlal_lane_s16(
              vacc0x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa0), 2);
          vacc0x4567 = vmlal_lane_s16(
              vacc0x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa0), 2);
          vacc1x0123 = vmlal_lane_s16(
              vacc1x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa1), 2);
          vacc1x4567 = vmlal_lane_s16(
              vacc1x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa1), 2);
          vacc2x0123 = vmlal_lane_s16(
              vacc2x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa2), 2);
          vacc2x4567 = vmlal_lane_s16(
              vacc2x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa2), 2);
          vacc3x0123 = vmlal_lane_s16(
              vacc3x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa3), 2);
          vacc3x4567 = vmlal_lane_s16(
              vacc3x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa3), 2);

          if (k >= 4) {
            const uint8x8_t vb01234567 = vld1_u8(w);
            w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
            const int16x8_t vxb01234567 =
                vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

            vacc0x0123 = vmlal_lane_s16(
                vacc0x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa0), 3);
            vacc0x4567 = vmlal_lane_s16(
                vacc0x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa0), 3);
            vacc1x0123 = vmlal_lane_s16(
                vacc1x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa1), 3);
            vacc1x4567 = vmlal_lane_s16(
                vacc1x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa1), 3);
            vacc2x0123 = vmlal_lane_s16(
                vacc2x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa2), 3);
            vacc2x4567 = vmlal_lane_s16(
                vacc2x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa2), 3);
            vacc3x0123 = vmlal_lane_s16(
                vacc3x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa3), 3);
            vacc3x4567 = vmlal_lane_s16(
                vacc3x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa3), 3);

            if (k > 4) {
              const uint8x8_t vb01234567 = vld1_u8(w);
              w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
              const int16x8_t vxb01234567 =
                  vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

              vacc0x0123 = vmlal_lane_s16(
                  vacc0x0123,
                  vget_low_s16(vxb01234567),
                  vget_high_s16(vxa0),
                  0);
              vacc0x4567 = vmlal_lane_s16(
                  vacc0x4567,
                  vget_high_s16(vxb01234567),
                  vget_high_s16(vxa0),
                  0);
              vacc1x0123 = vmlal_lane_s16(
                  vacc1x0123,
                  vget_low_s16(vxb01234567),
                  vget_high_s16(vxa1),
                  0);
              vacc1x4567 = vmlal_lane_s16(
                  vacc1x4567,
                  vget_high_s16(vxb01234567),
                  vget_high_s16(vxa1),
                  0);
              vacc2x0123 = vmlal_lane_s16(
                  vacc2x0123,
                  vget_low_s16(vxb01234567),
                  vget_high_s16(vxa2),
                  0);
              vacc2x4567 = vmlal_lane_s16(
                  vacc2x4567,
                  vget_high_s16(vxb01234567),
                  vget_high_s16(vxa2),
                  0);
              vacc3x0123 = vmlal_lane_s16(
                  vacc3x0123,
                  vget_low_s16(vxb01234567),
                  vget_high_s16(vxa3),
                  0);
              vacc3x4567 = vmlal_lane_s16(
                  vacc3x4567,
                  vget_high_s16(vxb01234567),
                  vget_high_s16(vxa3),
                  0);

              if (k >= 6) {
                const uint8x8_t vb01234567 = vld1_u8(w);
                w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
                const int16x8_t vxb01234567 =
                    vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

                vacc0x0123 = vmlal_lane_s16(
                    vacc0x0123,
                    vget_low_s16(vxb01234567),
                    vget_high_s16(vxa0),
                    1);
                vacc0x4567 = vmlal_lane_s16(
                    vacc0x4567,
                    vget_high_s16(vxb01234567),
                    vget_high_s16(vxa0),
                    1);
                vacc1x0123 = vmlal_lane_s16(
                    vacc1x0123,
                    vget_low_s16(vxb01234567),
                    vget_high_s16(vxa1),
                    1);
                vacc1x4567 = vmlal_lane_s16(
                    vacc1x4567,
                    vget_high_s16(vxb01234567),
                    vget_high_s16(vxa1),
                    1);
                vacc2x0123 = vmlal_lane_s16(
                    vacc2x0123,
                    vget_low_s16(vxb01234567),
                    vget_high_s16(vxa2),
                    1);
                vacc2x4567 = vmlal_lane_s16(
                    vacc2x4567,
                    vget_high_s16(vxb01234567),
                    vget_high_s16(vxa2),
                    1);
                vacc3x0123 = vmlal_lane_s16(
                    vacc3x0123,
                    vget_low_s16(vxb01234567),
                    vget_high_s16(vxa3),
                    1);
                vacc3x4567 = vmlal_lane_s16(
                    vacc3x4567,
                    vget_high_s16(vxb01234567),
                    vget_high_s16(vxa3),
                    1);

                if (k > 6) {
                  const uint8x8_t vb01234567 = vld1_u8(w);
                  w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
                  const int16x8_t vxb01234567 = vreinterpretq_s16_u16(
                      vsubl_u8(vb01234567, vb_zero_point));

                  vacc0x0123 = vmlal_lane_s16(
                      vacc0x0123,
                      vget_low_s16(vxb01234567),
                      vget_high_s16(vxa0),
                      2);
                  vacc0x4567 = vmlal_lane_s16(
                      vacc0x4567,
                      vget_high_s16(vxb01234567),
                      vget_high_s16(vxa0),
                      2);
                  vacc1x0123 = vmlal_lane_s16(
                      vacc1x0123,
                      vget_low_s16(vxb01234567),
                      vget_high_s16(vxa1),
                      2);
                  vacc1x4567 = vmlal_lane_s16(
                      vacc1x4567,
                      vget_high_s16(vxb01234567),
                      vget_high_s16(vxa1),
                      2);
                  vacc2x0123 = vmlal_lane_s16(
                      vacc2x0123,
                      vget_low_s16(vxb01234567),
                      vget_high_s16(vxa2),
                      2);
                  vacc2x4567 = vmlal_lane_s16(
                      vacc2x4567,
                      vget_high_s16(vxb01234567),
                      vget_high_s16(vxa2),
                      2);
                  vacc3x0123 = vmlal_lane_s16(
                      vacc3x0123,
                      vget_low_s16(vxb01234567),
                      vget_high_s16(vxa3),
                      2);
                  vacc3x4567 = vmlal_lane_s16(
                      vacc3x4567,
                      vget_high_s16(vxb01234567),
                      vget_high_s16(vxa3),
                      2);
                }
              }
            }
          }
        }
      }
    }
  } while (--ks != 0);

  // Doing 2 VLD1 instead of 1 VLD2 because A75 has higher latency
  // 8 vs. 5 for VLD2 with both VLD1 and VLD2 having throughput of
  // 2 per cycle. So probably this is better.
  const float32x4_t requantization_scale_c0123 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[output_channel_index]
          );
  const float32x4_t requantization_scale_c4567 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[
              output_channel_index + 4]);

  /*
   * Convert int32_t input to FP32 and multiply by FP32 scale.
   * Both operations involve statistically unbiased roundings:
   * - Large int32_t values can't be exactly represented as FP32. The
   * conversion instruction in ARM NEON would round it to nearest FP32 value
   * with ties to even.
   * - Product of two FP32 values is generally not exactly representation as
   * an FP32 value, and will be rounded to nearest FP32 value with ties to
   * even.
   */
  const float32x4_t vacc0x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc0x0123), requantization_scale_c0123);
  const float32x4_t vacc1x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc1x0123), requantization_scale_c0123);
  const float32x4_t vacc2x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc2x0123), requantization_scale_c0123);
  const float32x4_t vacc3x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc3x0123), requantization_scale_c0123);
  const float32x4_t vacc0x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc0x4567), requantization_scale_c4567);
  const float32x4_t vacc1x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc1x4567), requantization_scale_c4567);
  const float32x4_t vacc2x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc2x4567), requantization_scale_c4567);
  const float32x4_t vacc3x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc3x4567), requantization_scale_c4567);

#ifdef __aarch64__
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  /*
   * Leverage "Floating-point Convert to Signed integer, rounding to nearest
   * with ties to even" instruction. This is an ARMv8 instruction (always
   * available in AArch64), which saturates result on overflow. We don't need
   * to specifically consider saturated results, they will be clamped at the
   * last stage.
   */
  vacc0x0123 = vcvtnq_s32_f32(vacc0x0123_f);
  vacc1x0123 = vcvtnq_s32_f32(vacc1x0123_f);
  vacc2x0123 = vcvtnq_s32_f32(vacc2x0123_f);
  vacc3x0123 = vcvtnq_s32_f32(vacc3x0123_f);
  vacc0x4567 = vcvtnq_s32_f32(vacc0x4567_f);
  vacc1x4567 = vcvtnq_s32_f32(vacc1x4567_f);
  vacc2x4567 = vcvtnq_s32_f32(vacc2x4567_f);
  vacc3x4567 = vcvtnq_s32_f32(vacc3x4567_f);

  const int16x8_t vacc0x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
  const int16x8_t vacc1x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
  const int16x8_t vacc2x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
  const int16x8_t vacc3x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);

  uint8x16_t vout0x01234567_1x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
  uint8x16_t vout2x01234567_3x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);

  const uint8x16_t voutput_min =
      vld1q_dup_u8(&quantization_params->neon.output_min);
  const uint8x16_t voutput_max =
      vld1q_dup_u8(&quantization_params->neon.output_max);

  vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
  vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
  vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
  vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);
#else
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);
  /*
   * ARMv7 NEON offers only a floating-point to integer conversion instruction
   * with rounding towards zero. In lieu of conversion instruction with
   * rounding-to-nearest-even, we use a magic trick of adding a large number
   * (1.5 * 2**23) to scaled value to cause rounding to integer, and then
   * substracing this magic number as integer. This trick works only in a
   * limited range (absolute value of input must be less than 2**22), so
   * generally we have to clamp input to this range before using the magic.
   * However, clamping to any smaller range works just as well, and thus we
   * clamp to [qmin - zero point, qmax - zero point] range so that after we
   * add zero point to the result, it gets into target [qmin, qmax] range.
   */
  const float32x4_t vacc0x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc0x0123_f, vfmin), vfmax);
  const float32x4_t vacc1x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc1x0123_f, vfmin), vfmax);
  const float32x4_t vacc2x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc2x0123_f, vfmin), vfmax);
  const float32x4_t vacc3x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc3x0123_f, vfmin), vfmax);
  const float32x4_t vacc0x4567_f_clamped =
      vminq_f32(vmaxq_f32(vacc0x4567_f, vfmin), vfmax);
  const float32x4_t vacc1x4567_f_clamped =
      vminq_f32(vmaxq_f32(vacc1x4567_f, vfmin), vfmax);
  const float32x4_t vacc2x4567_f_clamped =
      vminq_f32(vmaxq_f32(vacc2x4567_f, vfmin), vfmax);
  const float32x4_t vacc3x4567_f_clamped =
      vminq_f32(vmaxq_f32(vacc3x4567_f, vfmin), vfmax);

  /*
   * Conversion to integer using the "magic trick". Rounding is performed in
   * the output of addition operation, and result is rounded to nearest even
   * integer with ties to even.
   */
  vacc0x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc0x0123_f_clamped, vfmagic)), vimagic);
  vacc1x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc1x0123_f_clamped, vfmagic)), vimagic);
  vacc2x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc2x0123_f_clamped, vfmagic)), vimagic);
  vacc3x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc3x0123_f_clamped, vfmagic)), vimagic);
  vacc0x4567 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc0x4567_f_clamped, vfmagic)), vimagic);
  vacc1x4567 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc1x4567_f_clamped, vfmagic)), vimagic);
  vacc2x4567 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc2x4567_f_clamped, vfmagic)), vimagic);
  vacc3x4567 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc3x4567_f_clamped, vfmagic)), vimagic);

  const int16x8_t vacc0x01234567 =
      vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
  const int16x8_t vacc1x01234567 =
      vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567));
  const int16x8_t vacc2x01234567 =
      vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567));
  const int16x8_t vacc3x01234567 =
      vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567));

  uint8x16_t vout0x01234567_1x01234567 =
      vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc1x01234567));
  uint8x16_t vout2x01234567_3x01234567 =
      vcombine_u8(vqmovun_s16(vacc2x01234567), vqmovun_s16(vacc3x01234567));
#endif

  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*)((uintptr_t)c0 + c_stride);
  if (mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*)((uintptr_t)c1 + c_stride);
  if (mr <= 2) {
    c2 = c1;
  }
  uint8_t* c3 = (uint8_t*)((uintptr_t)c2 + c_stride);
  if (mr != 4) {
    c3 = c2;
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
