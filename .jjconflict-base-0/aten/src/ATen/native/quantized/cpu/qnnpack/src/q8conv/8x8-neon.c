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

void pytorch_q8conv_ukernel_8x8__neon(
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
  int32x4_t vacc4x0123 = vacc0x0123;
  int32x4_t vacc4x4567 = vacc0x4567;
  int32x4_t vacc5x0123 = vacc0x0123;
  int32x4_t vacc5x4567 = vacc0x4567;
  int32x4_t vacc6x0123 = vacc0x0123;
  int32x4_t vacc6x4567 = vacc0x4567;
  int32x4_t vacc7x0123 = vacc0x0123;
  int32x4_t vacc7x4567 = vacc0x4567;

  do {
    const uint8_t* restrict a0 = *a++;
    const uint8_t* restrict a1 = *a++;
    const uint8_t* restrict a2 = *a++;
    const uint8_t* restrict a3 = *a++;
    const uint8_t* restrict a4 = *a++;
    const uint8_t* restrict a5 = *a++;
    const uint8_t* restrict a6 = *a++;
    const uint8_t* restrict a7 = *a++;

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
      const uint8x8_t va4 = vld1_u8(a4);
      a4 += 8;
      const uint8x8_t va5 = vld1_u8(a5);
      a5 += 8;
      const uint8x8_t va6 = vld1_u8(a6);
      a6 += 8;
      const uint8x8_t va7 = vld1_u8(a7);
      a7 += 8;
      const int16x8_t vxa0 =
          vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
      const int16x8_t vxa1 =
          vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));
      const int16x8_t vxa2 =
          vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));
      const int16x8_t vxa3 =
          vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));
      const int16x8_t vxa4 =
          vreinterpretq_s16_u16(sub_zero_point(va4, va_zero_point));
      const int16x8_t vxa5 =
          vreinterpretq_s16_u16(sub_zero_point(va5, va_zero_point));
      const int16x8_t vxa6 =
          vreinterpretq_s16_u16(sub_zero_point(va6, va_zero_point));
      const int16x8_t vxa7 =
          vreinterpretq_s16_u16(sub_zero_point(va7, va_zero_point));

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
        vacc4x0123 = vmlal_lane_s16(
            vacc4x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa4), 0);
        vacc4x4567 = vmlal_lane_s16(
            vacc4x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa4), 0);
        vacc5x0123 = vmlal_lane_s16(
            vacc5x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa5), 0);
        vacc5x4567 = vmlal_lane_s16(
            vacc5x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa5), 0);
        vacc6x0123 = vmlal_lane_s16(
            vacc6x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa6), 0);
        vacc6x4567 = vmlal_lane_s16(
            vacc6x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa6), 0);
        vacc7x0123 = vmlal_lane_s16(
            vacc7x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa7), 0);
        vacc7x4567 = vmlal_lane_s16(
            vacc7x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa7), 0);
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
        vacc4x0123 = vmlal_lane_s16(
            vacc4x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa4), 1);
        vacc4x4567 = vmlal_lane_s16(
            vacc4x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa4), 1);
        vacc5x0123 = vmlal_lane_s16(
            vacc5x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa5), 1);
        vacc5x4567 = vmlal_lane_s16(
            vacc5x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa5), 1);
        vacc6x0123 = vmlal_lane_s16(
            vacc6x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa6), 1);
        vacc6x4567 = vmlal_lane_s16(
            vacc6x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa6), 1);
        vacc7x0123 = vmlal_lane_s16(
            vacc7x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa7), 1);
        vacc7x4567 = vmlal_lane_s16(
            vacc7x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa7), 1);
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
        vacc4x0123 = vmlal_lane_s16(
            vacc4x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa4), 2);
        vacc4x4567 = vmlal_lane_s16(
            vacc4x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa4), 2);
        vacc5x0123 = vmlal_lane_s16(
            vacc5x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa5), 2);
        vacc5x4567 = vmlal_lane_s16(
            vacc5x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa5), 2);
        vacc6x0123 = vmlal_lane_s16(
            vacc6x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa6), 2);
        vacc6x4567 = vmlal_lane_s16(
            vacc6x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa6), 2);
        vacc7x0123 = vmlal_lane_s16(
            vacc7x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa7), 2);
        vacc7x4567 = vmlal_lane_s16(
            vacc7x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa7), 2);
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
        vacc4x0123 = vmlal_lane_s16(
            vacc4x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa4), 3);
        vacc4x4567 = vmlal_lane_s16(
            vacc4x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa4), 3);
        vacc5x0123 = vmlal_lane_s16(
            vacc5x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa5), 3);
        vacc5x4567 = vmlal_lane_s16(
            vacc5x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa5), 3);
        vacc6x0123 = vmlal_lane_s16(
            vacc6x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa6), 3);
        vacc6x4567 = vmlal_lane_s16(
            vacc6x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa6), 3);
        vacc7x0123 = vmlal_lane_s16(
            vacc7x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa7), 3);
        vacc7x4567 = vmlal_lane_s16(
            vacc7x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa7), 3);
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
        vacc4x0123 = vmlal_lane_s16(
            vacc4x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa4), 0);
        vacc4x4567 = vmlal_lane_s16(
            vacc4x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa4), 0);
        vacc5x0123 = vmlal_lane_s16(
            vacc5x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa5), 0);
        vacc5x4567 = vmlal_lane_s16(
            vacc5x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa5), 0);
        vacc6x0123 = vmlal_lane_s16(
            vacc6x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa6), 0);
        vacc6x4567 = vmlal_lane_s16(
            vacc6x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa6), 0);
        vacc7x0123 = vmlal_lane_s16(
            vacc7x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa7), 0);
        vacc7x4567 = vmlal_lane_s16(
            vacc7x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa7), 0);
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
        vacc4x0123 = vmlal_lane_s16(
            vacc4x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa4), 1);
        vacc4x4567 = vmlal_lane_s16(
            vacc4x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa4), 1);
        vacc5x0123 = vmlal_lane_s16(
            vacc5x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa5), 1);
        vacc5x4567 = vmlal_lane_s16(
            vacc5x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa5), 1);
        vacc6x0123 = vmlal_lane_s16(
            vacc6x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa6), 1);
        vacc6x4567 = vmlal_lane_s16(
            vacc6x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa6), 1);
        vacc7x0123 = vmlal_lane_s16(
            vacc7x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa7), 1);
        vacc7x4567 = vmlal_lane_s16(
            vacc7x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa7), 1);
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
        vacc4x0123 = vmlal_lane_s16(
            vacc4x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa4), 2);
        vacc4x4567 = vmlal_lane_s16(
            vacc4x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa4), 2);
        vacc5x0123 = vmlal_lane_s16(
            vacc5x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa5), 2);
        vacc5x4567 = vmlal_lane_s16(
            vacc5x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa5), 2);
        vacc6x0123 = vmlal_lane_s16(
            vacc6x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa6), 2);
        vacc6x4567 = vmlal_lane_s16(
            vacc6x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa6), 2);
        vacc7x0123 = vmlal_lane_s16(
            vacc7x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa7), 2);
        vacc7x4567 = vmlal_lane_s16(
            vacc7x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa7), 2);
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
        vacc4x0123 = vmlal_lane_s16(
            vacc4x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa4), 3);
        vacc4x4567 = vmlal_lane_s16(
            vacc4x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa4), 3);
        vacc5x0123 = vmlal_lane_s16(
            vacc5x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa5), 3);
        vacc5x4567 = vmlal_lane_s16(
            vacc5x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa5), 3);
        vacc6x0123 = vmlal_lane_s16(
            vacc6x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa6), 3);
        vacc6x4567 = vmlal_lane_s16(
            vacc6x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa6), 3);
        vacc7x0123 = vmlal_lane_s16(
            vacc7x0123, vget_low_s16(vxb01234567), vget_high_s16(vxa7), 3);
        vacc7x4567 = vmlal_lane_s16(
            vacc7x4567, vget_high_s16(vxb01234567), vget_high_s16(vxa7), 3);
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
      const uint8x8_t va4 = vreinterpret_u8_u64(vshl_u64(
          vreinterpret_u64_u8(vld1_u8(a4 - a_predecrement)), va_shift));
      const uint8x8_t va5 = vreinterpret_u8_u64(vshl_u64(
          vreinterpret_u64_u8(vld1_u8(a5 - a_predecrement)), va_shift));
      const uint8x8_t va6 = vreinterpret_u8_u64(vshl_u64(
          vreinterpret_u64_u8(vld1_u8(a6 - a_predecrement)), va_shift));
      const uint8x8_t va7 = vreinterpret_u8_u64(vshl_u64(
          vreinterpret_u64_u8(vld1_u8(a7 - a_predecrement)), va_shift));
      const int16x8_t vxa0 =
          vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
      const int16x8_t vxa1 =
          vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));
      const int16x8_t vxa2 =
          vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));
      const int16x8_t vxa3 =
          vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));
      const int16x8_t vxa4 =
          vreinterpretq_s16_u16(sub_zero_point(va4, va_zero_point));
      const int16x8_t vxa5 =
          vreinterpretq_s16_u16(sub_zero_point(va5, va_zero_point));
      const int16x8_t vxa6 =
          vreinterpretq_s16_u16(sub_zero_point(va6, va_zero_point));
      const int16x8_t vxa7 =
          vreinterpretq_s16_u16(sub_zero_point(va7, va_zero_point));

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
        vacc4x0123 = vmlal_lane_s16(
            vacc4x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa4), 0);
        vacc4x4567 = vmlal_lane_s16(
            vacc4x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa4), 0);
        vacc5x0123 = vmlal_lane_s16(
            vacc5x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa5), 0);
        vacc5x4567 = vmlal_lane_s16(
            vacc5x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa5), 0);
        vacc6x0123 = vmlal_lane_s16(
            vacc6x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa6), 0);
        vacc6x4567 = vmlal_lane_s16(
            vacc6x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa6), 0);
        vacc7x0123 = vmlal_lane_s16(
            vacc7x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa7), 0);
        vacc7x4567 = vmlal_lane_s16(
            vacc7x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa7), 0);
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
        vacc4x0123 = vmlal_lane_s16(
            vacc4x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa4), 1);
        vacc4x4567 = vmlal_lane_s16(
            vacc4x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa4), 1);
        vacc5x0123 = vmlal_lane_s16(
            vacc5x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa5), 1);
        vacc5x4567 = vmlal_lane_s16(
            vacc5x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa5), 1);
        vacc6x0123 = vmlal_lane_s16(
            vacc6x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa6), 1);
        vacc6x4567 = vmlal_lane_s16(
            vacc6x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa6), 1);
        vacc7x0123 = vmlal_lane_s16(
            vacc7x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa7), 1);
        vacc7x4567 = vmlal_lane_s16(
            vacc7x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa7), 1);

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
          vacc4x0123 = vmlal_lane_s16(
              vacc4x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa4), 2);
          vacc4x4567 = vmlal_lane_s16(
              vacc4x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa4), 2);
          vacc5x0123 = vmlal_lane_s16(
              vacc5x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa5), 2);
          vacc5x4567 = vmlal_lane_s16(
              vacc5x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa5), 2);
          vacc6x0123 = vmlal_lane_s16(
              vacc6x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa6), 2);
          vacc6x4567 = vmlal_lane_s16(
              vacc6x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa6), 2);
          vacc7x0123 = vmlal_lane_s16(
              vacc7x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa7), 2);
          vacc7x4567 = vmlal_lane_s16(
              vacc7x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa7), 2);

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
            vacc4x0123 = vmlal_lane_s16(
                vacc4x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa4), 3);
            vacc4x4567 = vmlal_lane_s16(
                vacc4x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa4), 3);
            vacc5x0123 = vmlal_lane_s16(
                vacc5x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa5), 3);
            vacc5x4567 = vmlal_lane_s16(
                vacc5x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa5), 3);
            vacc6x0123 = vmlal_lane_s16(
                vacc6x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa6), 3);
            vacc6x4567 = vmlal_lane_s16(
                vacc6x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa6), 3);
            vacc7x0123 = vmlal_lane_s16(
                vacc7x0123, vget_low_s16(vxb01234567), vget_low_s16(vxa7), 3);
            vacc7x4567 = vmlal_lane_s16(
                vacc7x4567, vget_high_s16(vxb01234567), vget_low_s16(vxa7), 3);

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
              vacc4x0123 = vmlal_lane_s16(
                  vacc4x0123,
                  vget_low_s16(vxb01234567),
                  vget_high_s16(vxa4),
                  0);
              vacc4x4567 = vmlal_lane_s16(
                  vacc4x4567,
                  vget_high_s16(vxb01234567),
                  vget_high_s16(vxa4),
                  0);
              vacc5x0123 = vmlal_lane_s16(
                  vacc5x0123,
                  vget_low_s16(vxb01234567),
                  vget_high_s16(vxa5),
                  0);
              vacc5x4567 = vmlal_lane_s16(
                  vacc5x4567,
                  vget_high_s16(vxb01234567),
                  vget_high_s16(vxa5),
                  0);
              vacc6x0123 = vmlal_lane_s16(
                  vacc6x0123,
                  vget_low_s16(vxb01234567),
                  vget_high_s16(vxa6),
                  0);
              vacc6x4567 = vmlal_lane_s16(
                  vacc6x4567,
                  vget_high_s16(vxb01234567),
                  vget_high_s16(vxa6),
                  0);
              vacc7x0123 = vmlal_lane_s16(
                  vacc7x0123,
                  vget_low_s16(vxb01234567),
                  vget_high_s16(vxa7),
                  0);
              vacc7x4567 = vmlal_lane_s16(
                  vacc7x4567,
                  vget_high_s16(vxb01234567),
                  vget_high_s16(vxa7),
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
                vacc4x0123 = vmlal_lane_s16(
                    vacc4x0123,
                    vget_low_s16(vxb01234567),
                    vget_high_s16(vxa4),
                    1);
                vacc4x4567 = vmlal_lane_s16(
                    vacc4x4567,
                    vget_high_s16(vxb01234567),
                    vget_high_s16(vxa4),
                    1);
                vacc5x0123 = vmlal_lane_s16(
                    vacc5x0123,
                    vget_low_s16(vxb01234567),
                    vget_high_s16(vxa5),
                    1);
                vacc5x4567 = vmlal_lane_s16(
                    vacc5x4567,
                    vget_high_s16(vxb01234567),
                    vget_high_s16(vxa5),
                    1);
                vacc6x0123 = vmlal_lane_s16(
                    vacc6x0123,
                    vget_low_s16(vxb01234567),
                    vget_high_s16(vxa6),
                    1);
                vacc6x4567 = vmlal_lane_s16(
                    vacc6x4567,
                    vget_high_s16(vxb01234567),
                    vget_high_s16(vxa6),
                    1);
                vacc7x0123 = vmlal_lane_s16(
                    vacc7x0123,
                    vget_low_s16(vxb01234567),
                    vget_high_s16(vxa7),
                    1);
                vacc7x4567 = vmlal_lane_s16(
                    vacc7x4567,
                    vget_high_s16(vxb01234567),
                    vget_high_s16(vxa7),
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
                  vacc4x0123 = vmlal_lane_s16(
                      vacc4x0123,
                      vget_low_s16(vxb01234567),
                      vget_high_s16(vxa4),
                      2);
                  vacc4x4567 = vmlal_lane_s16(
                      vacc4x4567,
                      vget_high_s16(vxb01234567),
                      vget_high_s16(vxa4),
                      2);
                  vacc5x0123 = vmlal_lane_s16(
                      vacc5x0123,
                      vget_low_s16(vxb01234567),
                      vget_high_s16(vxa5),
                      2);
                  vacc5x4567 = vmlal_lane_s16(
                      vacc5x4567,
                      vget_high_s16(vxb01234567),
                      vget_high_s16(vxa5),
                      2);
                  vacc6x0123 = vmlal_lane_s16(
                      vacc6x0123,
                      vget_low_s16(vxb01234567),
                      vget_high_s16(vxa6),
                      2);
                  vacc6x4567 = vmlal_lane_s16(
                      vacc6x4567,
                      vget_high_s16(vxb01234567),
                      vget_high_s16(vxa6),
                      2);
                  vacc7x0123 = vmlal_lane_s16(
                      vacc7x0123,
                      vget_low_s16(vxb01234567),
                      vget_high_s16(vxa7),
                      2);
                  vacc7x4567 = vmlal_lane_s16(
                      vacc7x4567,
                      vget_high_s16(vxb01234567),
                      vget_high_s16(vxa7),
                      2);
                }
              }
            }
          }
        }
      }
    }
  } while (--ks != 0);

  const float32x4_t requantization_scale_c0123 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[output_channel_index]
          );
  const float32x4_t requantization_scale_c4567 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[
              output_channel_index + 4]);

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
  const float32x4_t vacc4x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc4x0123), requantization_scale_c0123);
  const float32x4_t vacc5x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc5x0123), requantization_scale_c0123);
  const float32x4_t vacc6x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc6x0123), requantization_scale_c0123);
  const float32x4_t vacc7x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc7x0123), requantization_scale_c0123);
  const float32x4_t vacc4x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc4x4567), requantization_scale_c4567);
  const float32x4_t vacc5x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc5x4567), requantization_scale_c4567);
  const float32x4_t vacc6x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc6x4567), requantization_scale_c4567);
  const float32x4_t vacc7x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc7x4567), requantization_scale_c4567);

#ifdef __aarch64__
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);

  vacc0x0123 = vcvtnq_s32_f32(vacc0x0123_f);
  vacc1x0123 = vcvtnq_s32_f32(vacc1x0123_f);
  vacc2x0123 = vcvtnq_s32_f32(vacc2x0123_f);
  vacc3x0123 = vcvtnq_s32_f32(vacc3x0123_f);
  vacc0x4567 = vcvtnq_s32_f32(vacc0x4567_f);
  vacc1x4567 = vcvtnq_s32_f32(vacc1x4567_f);
  vacc2x4567 = vcvtnq_s32_f32(vacc2x4567_f);
  vacc3x4567 = vcvtnq_s32_f32(vacc3x4567_f);
  vacc4x0123 = vcvtnq_s32_f32(vacc4x0123_f);
  vacc5x0123 = vcvtnq_s32_f32(vacc5x0123_f);
  vacc6x0123 = vcvtnq_s32_f32(vacc6x0123_f);
  vacc7x0123 = vcvtnq_s32_f32(vacc7x0123_f);
  vacc4x4567 = vcvtnq_s32_f32(vacc4x4567_f);
  vacc5x4567 = vcvtnq_s32_f32(vacc5x4567_f);
  vacc6x4567 = vcvtnq_s32_f32(vacc6x4567_f);
  vacc7x4567 = vcvtnq_s32_f32(vacc7x4567_f);

  const int16x8_t vacc0x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
  const int16x8_t vacc1x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
  const int16x8_t vacc2x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
  const int16x8_t vacc3x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
  const int16x8_t vacc4x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc4x0123), vacc4x4567), voutput_zero_point);
  const int16x8_t vacc5x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc5x0123), vacc5x4567), voutput_zero_point);
  const int16x8_t vacc6x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc6x0123), vacc6x4567), voutput_zero_point);
  const int16x8_t vacc7x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc7x0123), vacc7x4567), voutput_zero_point);

  uint8x16_t vout0x01234567_1x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
  uint8x16_t vout2x01234567_3x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);
  uint8x16_t vout4x01234567_5x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc4x01234567), vacc5x01234567);
  uint8x16_t vout6x01234567_7x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc6x01234567), vacc7x01234567);

  const uint8x16_t voutput_min =
      vld1q_dup_u8(&quantization_params->neon.output_min);
  const uint8x16_t voutput_max =
      vld1q_dup_u8(&quantization_params->neon.output_max);

  vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
  vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
  vout4x01234567_5x01234567 = vmaxq_u8(vout4x01234567_5x01234567, voutput_min);
  vout6x01234567_7x01234567 = vmaxq_u8(vout6x01234567_7x01234567, voutput_min);
  vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
  vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);
  vout4x01234567_5x01234567 = vminq_u8(vout4x01234567_5x01234567, voutput_max);
  vout6x01234567_7x01234567 = vminq_u8(vout6x01234567_7x01234567, voutput_max);
#else
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);

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
  const float32x4_t vacc4x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc4x0123_f, vfmin), vfmax);
  const float32x4_t vacc5x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc5x0123_f, vfmin), vfmax);
  const float32x4_t vacc6x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc6x0123_f, vfmin), vfmax);
  const float32x4_t vacc7x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc7x0123_f, vfmin), vfmax);
  const float32x4_t vacc4x4567_f_clamped =
      vminq_f32(vmaxq_f32(vacc4x4567_f, vfmin), vfmax);
  const float32x4_t vacc5x4567_f_clamped =
      vminq_f32(vmaxq_f32(vacc5x4567_f, vfmin), vfmax);
  const float32x4_t vacc6x4567_f_clamped =
      vminq_f32(vmaxq_f32(vacc6x4567_f, vfmin), vfmax);
  const float32x4_t vacc7x4567_f_clamped =
      vminq_f32(vmaxq_f32(vacc7x4567_f, vfmin), vfmax);

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
  vacc4x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc4x0123_f_clamped, vfmagic)), vimagic);
  vacc5x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc5x0123_f_clamped, vfmagic)), vimagic);
  vacc6x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc6x0123_f_clamped, vfmagic)), vimagic);
  vacc7x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc7x0123_f_clamped, vfmagic)), vimagic);
  vacc4x4567 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc4x4567_f_clamped, vfmagic)), vimagic);
  vacc5x4567 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc5x4567_f_clamped, vfmagic)), vimagic);
  vacc6x4567 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc6x4567_f_clamped, vfmagic)), vimagic);
  vacc7x4567 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc7x4567_f_clamped, vfmagic)), vimagic);

  const int16x8_t vacc0x01234567 =
      vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
  const int16x8_t vacc1x01234567 =
      vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567));
  const int16x8_t vacc2x01234567 =
      vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567));
  const int16x8_t vacc3x01234567 =
      vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567));
  const int16x8_t vacc4x01234567 =
      vcombine_s16(vqmovn_s32(vacc4x0123), vqmovn_s32(vacc4x4567));
  const int16x8_t vacc5x01234567 =
      vcombine_s16(vqmovn_s32(vacc5x0123), vqmovn_s32(vacc5x4567));
  const int16x8_t vacc6x01234567 =
      vcombine_s16(vqmovn_s32(vacc6x0123), vqmovn_s32(vacc6x4567));
  const int16x8_t vacc7x01234567 =
      vcombine_s16(vqmovn_s32(vacc7x0123), vqmovn_s32(vacc7x4567));

  uint8x16_t vout0x01234567_1x01234567 =
      vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc1x01234567));
  uint8x16_t vout2x01234567_3x01234567 =
      vcombine_u8(vqmovun_s16(vacc2x01234567), vqmovun_s16(vacc3x01234567));
  uint8x16_t vout4x01234567_5x01234567 =
      vcombine_u8(vqmovun_s16(vacc4x01234567), vqmovun_s16(vacc5x01234567));
  uint8x16_t vout6x01234567_7x01234567 =
      vcombine_u8(vqmovun_s16(vacc6x01234567), vqmovun_s16(vacc7x01234567));
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
  if (mr < 4) {
    c3 = c2;
  }
  uint8_t* c4 = (uint8_t*)((uintptr_t)c3 + c_stride);
  if (mr <= 4) {
    c4 = c3;
  }
  uint8_t* c5 = (uint8_t*)((uintptr_t)c4 + c_stride);
  if (mr < 6) {
    c5 = c4;
  }
  uint8_t* c6 = (uint8_t*)((uintptr_t)c5 + c_stride);
  if (mr <= 6) {
    c6 = c5;
  }
  uint8_t* c7 = (uint8_t*)((uintptr_t)c6 + c_stride);
  if (mr != 8) {
    c7 = c6;
  }
  if (nr == 8) {
    vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
    vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
    vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
    vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
    vst1_u8(c4, vget_low_u8(vout4x01234567_5x01234567));
    vst1_u8(c5, vget_high_u8(vout4x01234567_5x01234567));
    vst1_u8(c6, vget_low_u8(vout6x01234567_7x01234567));
    vst1_u8(c7, vget_high_u8(vout6x01234567_7x01234567));
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
      vst1q_lane_u32(
          __builtin_assume_aligned(c4, 1),
          vreinterpretq_u32_u8(vout4x01234567_5x01234567),
          0);
      c4 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c5, 1),
          vreinterpretq_u32_u8(vout4x01234567_5x01234567),
          2);
      c5 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c6, 1),
          vreinterpretq_u32_u8(vout6x01234567_7x01234567),
          0);
      c6 += 4;
      vst1q_lane_u32(
          __builtin_assume_aligned(c7, 1),
          vreinterpretq_u32_u8(vout6x01234567_7x01234567),
          2);
      c7 += 4;
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      vout4x01234567_5x01234567 =
          vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 4);
      vout6x01234567_7x01234567 =
          vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 4);
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
      vst1q_lane_u16(
          __builtin_assume_aligned(c4, 1),
          vreinterpretq_u16_u8(vout4x01234567_5x01234567),
          0);
      c4 += 2;
      vst1q_lane_u16(
          __builtin_assume_aligned(c5, 1),
          vreinterpretq_u16_u8(vout4x01234567_5x01234567),
          4);
      c5 += 2;
      vst1q_lane_u16(
          __builtin_assume_aligned(c6, 1),
          vreinterpretq_u16_u8(vout6x01234567_7x01234567),
          0);
      c6 += 2;
      vst1q_lane_u16(
          __builtin_assume_aligned(c7, 1),
          vreinterpretq_u16_u8(vout6x01234567_7x01234567),
          4);
      c7 += 2;
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      vout4x01234567_5x01234567 =
          vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 2);
      vout6x01234567_7x01234567 =
          vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 2);
      nr -= 2;
    }
    if (nr != 0) {
      vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
      vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
      vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
      vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
      vst1q_lane_u8(c4, vout4x01234567_5x01234567, 0);
      vst1q_lane_u8(c5, vout4x01234567_5x01234567, 8);
      vst1q_lane_u8(c6, vout6x01234567_7x01234567, 0);
      vst1q_lane_u8(c7, vout6x01234567_7x01234567, 8);
    }
  }
}
