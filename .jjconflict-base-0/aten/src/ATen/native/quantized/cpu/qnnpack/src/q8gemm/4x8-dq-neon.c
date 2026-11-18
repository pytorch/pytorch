/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/q8gemm.h>
#include <requantization/runtime-neon.h>

void pytorch_q8gemm_dq_ukernel_4x8__neon(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    const float* restrict b,
    float* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const struct pytorch_qnnp_conv_dynamic_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  int32x4_t vacc0x0123 = {};
  int32x4_t vacc0x4567 = {};
  int32x4_t vacc1x0123 = {};
  int32x4_t vacc1x4567 = {};
  int32x4_t vacc2x0123 = {};
  int32x4_t vacc2x4567 = {};
  int32x4_t vacc3x0123 = {};
  int32x4_t vacc3x4567 = {};
  w = (const void*)((uintptr_t)w + 32);

  const uint8_t* a0 = a;
  const uint8_t* a1 = (const uint8_t*)((uintptr_t)a0 + a_stride);
  if (mr < 2) {
    a1 = a0;
  }
  const uint8_t* a2 = (const uint8_t*)((uintptr_t)a1 + a_stride);
  if (mr <= 2) {
    a2 = a1;
  }
  const uint8_t* a3 = (const uint8_t*)((uintptr_t)a2 + a_stride);
  if (mr != 4) {
    a3 = a2;
  }

  const uint8x8_t va_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->input_zero_point);
  // Assumes that kernel_zero_points is an array padded with necessary elements
  // in order to make it multiple of 8.
  const uint8x8_t vb_zero_point =
      vld1_u8((const uint8_t*)&quantization_params->kernel_zero_points
          [output_channel_index]);

  const float32x4_t vmultiplier_c0123 =
      vld1q_f32(&quantization_params->multipliers[output_channel_index]);
  const float32x4_t vmultiplier_c4567 =
      vld1q_f32(&quantization_params->multipliers[output_channel_index + 4]);
  const float32x4_t vbias[] = {
    vld1q_f32(b),
    vld1q_f32(b + 4),
  };

  for (; k >= 8; k -= 8) {
    const uint8x8_t va0 = vld1_u8(a0);
    a0 += 8;
    const int16x8_t vxa0 =
        vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
    const uint8x8_t va1 = vld1_u8(a1);
    a1 += 8;
    const int16x8_t vxa1 =
        vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));
    const uint8x8_t va2 = vld1_u8(a2);
    a2 += 8;
    const int16x8_t vxa2 =
        vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));
    const uint8x8_t va3 = vld1_u8(a3);
    a3 += 8;
    const int16x8_t vxa3 =
        vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));

    const uint8x8_t vb01234567c0 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb01234567c0 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);

    const uint8x8_t vb01234567c1 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb01234567c1 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);

    const uint8x8_t vb01234567c2 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb01234567c2 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);

    const uint8x8_t vb01234567c3 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb01234567c3 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);

    const uint8x8_t vb01234567c4 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb01234567c4 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);

    const uint8x8_t vb01234567c5 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb01234567c5 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);

    const uint8x8_t vb01234567c6 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb01234567c6 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);

    const uint8x8_t vb01234567c7 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb01234567c7 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
  }
  if (k != 0) {
    const size_t a_predecrement = 8 - k;
    const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
    const uint8x8_t va0 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
    const int16x8_t vxa0 =
        vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
    const uint8x8_t va1 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a1 - a_predecrement)), va_shift));
    const int16x8_t vxa1 =
        vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));
    const uint8x8_t va2 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a2 - a_predecrement)), va_shift));
    const int16x8_t vxa2 =
        vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));
    const uint8x8_t va3 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a3 - a_predecrement)), va_shift));
    const int16x8_t vxa3 =
        vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));

    const uint8x8_t vb01234567c0 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb01234567c0 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);

    if (k >= 2) {
      const uint8x8_t vb01234567c1 = vld1_u8(w);
      w = (const void*)((uintptr_t)w + 8);
      const int16x8_t vxb01234567c1 =
          vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(
          vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(
          vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc1x0123 = vmlal_lane_s16(
          vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
      vacc1x4567 = vmlal_lane_s16(
          vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
      vacc2x0123 = vmlal_lane_s16(
          vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
      vacc2x4567 = vmlal_lane_s16(
          vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
      vacc3x0123 = vmlal_lane_s16(
          vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
      vacc3x4567 = vmlal_lane_s16(
          vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);

      if (k >= 3) {
        const uint8x8_t vb01234567c2 = vld1_u8(w);
        w = (const void*)((uintptr_t)w + 8);
        const int16x8_t vxb01234567c2 =
            vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(
            vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
        vacc1x4567 = vmlal_lane_s16(
            vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
        vacc2x4567 = vmlal_lane_s16(
            vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
        vacc3x4567 = vmlal_lane_s16(
            vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);

        if (k >= 4) {
          const uint8x8_t vb01234567c3 = vld1_u8(w);
          w = (const void*)((uintptr_t)w + 8);
          const int16x8_t vxb01234567c3 =
              vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));

          vacc0x0123 = vmlal_lane_s16(
              vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
          vacc0x4567 = vmlal_lane_s16(
              vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
          vacc1x0123 = vmlal_lane_s16(
              vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
          vacc1x4567 = vmlal_lane_s16(
              vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
          vacc2x0123 = vmlal_lane_s16(
              vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
          vacc2x4567 = vmlal_lane_s16(
              vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
          vacc3x0123 = vmlal_lane_s16(
              vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
          vacc3x4567 = vmlal_lane_s16(
              vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);

          if (k >= 5) {
            const uint8x8_t vb01234567c4 = vld1_u8(w);
            w = (const void*)((uintptr_t)w + 8);
            const int16x8_t vxb01234567c4 =
                vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));

            vacc0x0123 = vmlal_lane_s16(
                vacc0x0123,
                vget_low_s16(vxb01234567c4),
                vget_high_s16(vxa0),
                0);
            vacc0x4567 = vmlal_lane_s16(
                vacc0x4567,
                vget_high_s16(vxb01234567c4),
                vget_high_s16(vxa0),
                0);
            vacc1x0123 = vmlal_lane_s16(
                vacc1x0123,
                vget_low_s16(vxb01234567c4),
                vget_high_s16(vxa1),
                0);
            vacc1x4567 = vmlal_lane_s16(
                vacc1x4567,
                vget_high_s16(vxb01234567c4),
                vget_high_s16(vxa1),
                0);
            vacc2x0123 = vmlal_lane_s16(
                vacc2x0123,
                vget_low_s16(vxb01234567c4),
                vget_high_s16(vxa2),
                0);
            vacc2x4567 = vmlal_lane_s16(
                vacc2x4567,
                vget_high_s16(vxb01234567c4),
                vget_high_s16(vxa2),
                0);
            vacc3x0123 = vmlal_lane_s16(
                vacc3x0123,
                vget_low_s16(vxb01234567c4),
                vget_high_s16(vxa3),
                0);
            vacc3x4567 = vmlal_lane_s16(
                vacc3x4567,
                vget_high_s16(vxb01234567c4),
                vget_high_s16(vxa3),
                0);

            if (k >= 6) {
              const uint8x8_t vb01234567c5 = vld1_u8(w);
              w = (const void*)((uintptr_t)w + 8);
              const int16x8_t vxb01234567c5 =
                  vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

              vacc0x0123 = vmlal_lane_s16(
                  vacc0x0123,
                  vget_low_s16(vxb01234567c5),
                  vget_high_s16(vxa0),
                  1);
              vacc0x4567 = vmlal_lane_s16(
                  vacc0x4567,
                  vget_high_s16(vxb01234567c5),
                  vget_high_s16(vxa0),
                  1);
              vacc1x0123 = vmlal_lane_s16(
                  vacc1x0123,
                  vget_low_s16(vxb01234567c5),
                  vget_high_s16(vxa1),
                  1);
              vacc1x4567 = vmlal_lane_s16(
                  vacc1x4567,
                  vget_high_s16(vxb01234567c5),
                  vget_high_s16(vxa1),
                  1);
              vacc2x0123 = vmlal_lane_s16(
                  vacc2x0123,
                  vget_low_s16(vxb01234567c5),
                  vget_high_s16(vxa2),
                  1);
              vacc2x4567 = vmlal_lane_s16(
                  vacc2x4567,
                  vget_high_s16(vxb01234567c5),
                  vget_high_s16(vxa2),
                  1);
              vacc3x0123 = vmlal_lane_s16(
                  vacc3x0123,
                  vget_low_s16(vxb01234567c5),
                  vget_high_s16(vxa3),
                  1);
              vacc3x4567 = vmlal_lane_s16(
                  vacc3x4567,
                  vget_high_s16(vxb01234567c5),
                  vget_high_s16(vxa3),
                  1);

              if (k >= 7) {
                const uint8x8_t vb01234567c6 = vld1_u8(w);
                w = (const void*)((uintptr_t)w + 8);
                const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(
                    vsubl_u8(vb01234567c6, vb_zero_point));

                vacc0x0123 = vmlal_lane_s16(
                    vacc0x0123,
                    vget_low_s16(vxb01234567c6),
                    vget_high_s16(vxa0),
                    2);
                vacc0x4567 = vmlal_lane_s16(
                    vacc0x4567,
                    vget_high_s16(vxb01234567c6),
                    vget_high_s16(vxa0),
                    2);
                vacc1x0123 = vmlal_lane_s16(
                    vacc1x0123,
                    vget_low_s16(vxb01234567c6),
                    vget_high_s16(vxa1),
                    2);
                vacc1x4567 = vmlal_lane_s16(
                    vacc1x4567,
                    vget_high_s16(vxb01234567c6),
                    vget_high_s16(vxa1),
                    2);
                vacc2x0123 = vmlal_lane_s16(
                    vacc2x0123,
                    vget_low_s16(vxb01234567c6),
                    vget_high_s16(vxa2),
                    2);
                vacc2x4567 = vmlal_lane_s16(
                    vacc2x4567,
                    vget_high_s16(vxb01234567c6),
                    vget_high_s16(vxa2),
                    2);
                vacc3x0123 = vmlal_lane_s16(
                    vacc3x0123,
                    vget_low_s16(vxb01234567c6),
                    vget_high_s16(vxa3),
                    2);
                vacc3x4567 = vmlal_lane_s16(
                    vacc3x4567,
                    vget_high_s16(vxb01234567c6),
                    vget_high_s16(vxa3),
                    2);
              }
            }
          }
        }
      }
    }
  }

  float32x4_t vout0[] = {
    vaddq_f32(vmulq_f32(vmultiplier_c0123, vcvtq_f32_s32(vacc0x0123)), vbias[0]),
    vaddq_f32(vmulq_f32(vmultiplier_c4567, vcvtq_f32_s32(vacc0x4567)), vbias[1]),
  };
  float32x4_t vout1[] = {
    vaddq_f32(vmulq_f32(vmultiplier_c0123, vcvtq_f32_s32(vacc1x0123)), vbias[0]),
    vaddq_f32(vmulq_f32(vmultiplier_c4567, vcvtq_f32_s32(vacc1x4567)), vbias[1]),
  };
  float32x4_t vout2[] = {
    vaddq_f32(vmulq_f32(vmultiplier_c0123, vcvtq_f32_s32(vacc2x0123)), vbias[0]),
    vaddq_f32(vmulq_f32(vmultiplier_c4567, vcvtq_f32_s32(vacc2x4567)), vbias[1]),
  };
  float32x4_t vout3[] = {
    vaddq_f32(vmulq_f32(vmultiplier_c0123, vcvtq_f32_s32(vacc3x0123)), vbias[0]),
    vaddq_f32(vmulq_f32(vmultiplier_c4567, vcvtq_f32_s32(vacc3x4567)), vbias[1]),
  };

  float32x4_t * vout0_ptr = vout0;
  float32x4_t * vout1_ptr = vout1;
  float32x4_t * vout2_ptr = vout2;
  float32x4_t * vout3_ptr = vout3;

  float* c0 = c;
  float* c1 = c0 + c_stride;
  if (mr < 2) {
    c1 = c0;
  }
  float* c2 = c1 + c_stride;
  if (mr <= 2) {
    c2 = c1;
  }
  float* c3 = c2 + c_stride;
  if (mr != 4) {
    c3 = c2;
  }

  for (; nr >= 4; nr -= 4) {
    vst1q_f32(c0, *vout0_ptr++);
    vst1q_f32(c1, *vout1_ptr++);
    vst1q_f32(c2, *vout2_ptr++);
    vst1q_f32(c3, *vout3_ptr++);

    c0 += 4;
    c1 += 4;
    c2 += 4;
    c3 += 4;
  }

  if (nr >= 2) {
    vst1_f32(c0, vget_low_f32(*vout0_ptr));
    vst1_f32(c1, vget_low_f32(*vout1_ptr));
    vst1_f32(c2, vget_low_f32(*vout2_ptr));
    vst1_f32(c3, vget_low_f32(*vout3_ptr));

    c0 += 2;
    (*vout0_ptr)[0] = (*vout0_ptr)[2];
    c1 += 2;
    (*vout1_ptr)[0] = (*vout1_ptr)[2];
    c2 += 2;
    (*vout2_ptr)[0] = (*vout2_ptr)[2];
    c3 += 2;
    (*vout3_ptr)[0] = (*vout3_ptr)[2];

    nr -= 2;
  }

  if (nr != 0) {
    vst1q_lane_f32(c0, *vout0_ptr, 0);
    vst1q_lane_f32(c1, *vout1_ptr, 0);
    vst1q_lane_f32(c2, *vout2_ptr, 0);
    vst1q_lane_f32(c3, *vout3_ptr, 0);
  }
}
