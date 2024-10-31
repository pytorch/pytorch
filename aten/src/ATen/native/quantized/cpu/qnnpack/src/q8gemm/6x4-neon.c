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

void pytorch_q8gemm_ukernel_6x4__neon(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[restrict static 1]) {
  int32x4_t vacc0x0123 = vld1q_s32(w);
  w = (const void*)((uintptr_t)w + 16);
  int32x4_t vacc1x0123 = vacc0x0123;
  int32x4_t vacc2x0123 = vacc0x0123;
  int32x4_t vacc3x0123 = vacc0x0123;
  int32x4_t vacc4x0123 = vacc0x0123;
  int32x4_t vacc5x0123 = vacc0x0123;

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
  if (mr < 4) {
    a3 = a2;
  }
  const uint8_t* a4 = (const uint8_t*)((uintptr_t)a3 + a_stride);
  if (mr <= 4) {
    a4 = a3;
  };
  const uint8_t* a5 = (const uint8_t*)((uintptr_t)a4 + a_stride);
  if (mr != 6) {
    a5 = a4;
  }

  const uint8x8_t va_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);
  uint8x8_t vb_zero_point =
      vld1_u8((const uint8_t*)&quantization_params->neon.kernel_zero_points
          [output_channel_index]);
  // Since only lower 4 values are used in this kernel. We replicate lower 4
  // values in upper 4 values. Still we end up loading 8 values assuming
  // zero point array is always multiple of 8.
  vb_zero_point = vset_lane_u8(vget_lane_u8(vb_zero_point, 0), vb_zero_point, 4);
  vb_zero_point = vset_lane_u8(vget_lane_u8(vb_zero_point, 1), vb_zero_point, 5);
  vb_zero_point = vset_lane_u8(vget_lane_u8(vb_zero_point, 2), vb_zero_point, 6);
  vb_zero_point = vset_lane_u8(vget_lane_u8(vb_zero_point, 3), vb_zero_point, 7);
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
    const uint8x8_t va4 = vld1_u8(a4);
    a4 += 8;
    const int16x8_t vxa4 =
        vreinterpretq_s16_u16(sub_zero_point(va4, va_zero_point));
    const uint8x8_t va5 = vld1_u8(a5);
    a5 += 8;
    const int16x8_t vxa5 =
        vreinterpretq_s16_u16(sub_zero_point(va5, va_zero_point));

    const uint8x8_t vb0123c01 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb0123c01 =
        vreinterpretq_s16_u16(vsubl_u8(vb0123c01, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb0123c01), vget_low_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb0123c01), vget_low_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb0123c01), vget_low_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb0123c01), vget_low_s16(vxa3), 0);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb0123c01), vget_low_s16(vxa4), 0);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb0123c01), vget_low_s16(vxa5), 0);

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_high_s16(vxb0123c01), vget_low_s16(vxa0), 1);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_high_s16(vxb0123c01), vget_low_s16(vxa1), 1);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_high_s16(vxb0123c01), vget_low_s16(vxa2), 1);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_high_s16(vxb0123c01), vget_low_s16(vxa3), 1);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_high_s16(vxb0123c01), vget_low_s16(vxa4), 1);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_high_s16(vxb0123c01), vget_low_s16(vxa5), 1);

    const uint8x8_t vb0123c23 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb0123c23 =
        vreinterpretq_s16_u16(vsubl_u8(vb0123c23, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb0123c23), vget_low_s16(vxa0), 2);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb0123c23), vget_low_s16(vxa1), 2);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb0123c23), vget_low_s16(vxa2), 2);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb0123c23), vget_low_s16(vxa3), 2);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb0123c23), vget_low_s16(vxa4), 2);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb0123c23), vget_low_s16(vxa5), 2);

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_high_s16(vxb0123c23), vget_low_s16(vxa0), 3);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_high_s16(vxb0123c23), vget_low_s16(vxa1), 3);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_high_s16(vxb0123c23), vget_low_s16(vxa2), 3);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_high_s16(vxb0123c23), vget_low_s16(vxa3), 3);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_high_s16(vxb0123c23), vget_low_s16(vxa4), 3);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_high_s16(vxb0123c23), vget_low_s16(vxa5), 3);

    const uint8x8_t vb0123c45 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb0123c45 =
        vreinterpretq_s16_u16(vsubl_u8(vb0123c45, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa3), 0);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa4), 0);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb0123c45), vget_high_s16(vxa5), 0);

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_high_s16(vxb0123c45), vget_high_s16(vxa0), 1);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_high_s16(vxb0123c45), vget_high_s16(vxa1), 1);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_high_s16(vxb0123c45), vget_high_s16(vxa2), 1);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_high_s16(vxb0123c45), vget_high_s16(vxa3), 1);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_high_s16(vxb0123c45), vget_high_s16(vxa4), 1);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_high_s16(vxb0123c45), vget_high_s16(vxa5), 1);

    const uint8x8_t vb0123c67 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb0123c67 =
        vreinterpretq_s16_u16(vsubl_u8(vb0123c67, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa0), 2);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa1), 2);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa2), 2);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa3), 2);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa4), 2);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb0123c67), vget_high_s16(vxa5), 2);

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa0), 3);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa1), 3);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa2), 3);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa3), 3);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa4), 3);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_high_s16(vxb0123c67), vget_high_s16(vxa5), 3);
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
    const uint8x8_t va4 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a4 - a_predecrement)), va_shift));
    const int16x8_t vxa4 =
        vreinterpretq_s16_u16(sub_zero_point(va4, va_zero_point));
    const uint8x8_t va5 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a5 - a_predecrement)), va_shift));
    const int16x8_t vxa5 =
        vreinterpretq_s16_u16(sub_zero_point(va5, va_zero_point));

    const uint8x8_t vb0123c0 = vreinterpret_u8_u32(vld1_dup_u32(w));
    w = (const void*)((uintptr_t)w + 4);
    const int16x8_t vxb0123c0 =
        vreinterpretq_s16_u16(vsubl_u8(vb0123c0, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa3), 0);
    vacc4x0123 = vmlal_lane_s16(
        vacc4x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa4), 0);
    vacc5x0123 = vmlal_lane_s16(
        vacc5x0123, vget_low_s16(vxb0123c0), vget_low_s16(vxa5), 0);

    if (k >= 2) {
      const uint8x8_t vb0123c1 = vreinterpret_u8_u32(vld1_dup_u32(w));
      w = (const void*)((uintptr_t)w + 4);
      const int16x8_t vxb0123c1 =
          vreinterpretq_s16_u16(vsubl_u8(vb0123c1, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(
          vacc0x0123, vget_low_s16(vxb0123c1), vget_low_s16(vxa0), 1);
      vacc1x0123 = vmlal_lane_s16(
          vacc1x0123, vget_low_s16(vxb0123c1), vget_low_s16(vxa1), 1);
      vacc2x0123 = vmlal_lane_s16(
          vacc2x0123, vget_low_s16(vxb0123c1), vget_low_s16(vxa2), 1);
      vacc3x0123 = vmlal_lane_s16(
          vacc3x0123, vget_low_s16(vxb0123c1), vget_low_s16(vxa3), 1);
      vacc4x0123 = vmlal_lane_s16(
          vacc4x0123, vget_low_s16(vxb0123c1), vget_low_s16(vxa4), 1);
      vacc5x0123 = vmlal_lane_s16(
          vacc5x0123, vget_low_s16(vxb0123c1), vget_low_s16(vxa5), 1);

      if (k > 2) {
        const uint8x8_t vb0123c2 = vreinterpret_u8_u32(vld1_dup_u32(w));
        w = (const void*)((uintptr_t)w + 4);
        const int16x8_t vxb0123c2 =
            vreinterpretq_s16_u16(vsubl_u8(vb0123c2, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(
            vacc0x0123, vget_low_s16(vxb0123c2), vget_low_s16(vxa0), 2);
        vacc1x0123 = vmlal_lane_s16(
            vacc1x0123, vget_low_s16(vxb0123c2), vget_low_s16(vxa1), 2);
        vacc2x0123 = vmlal_lane_s16(
            vacc2x0123, vget_low_s16(vxb0123c2), vget_low_s16(vxa2), 2);
        vacc3x0123 = vmlal_lane_s16(
            vacc3x0123, vget_low_s16(vxb0123c2), vget_low_s16(vxa3), 2);
        vacc4x0123 = vmlal_lane_s16(
            vacc4x0123, vget_low_s16(vxb0123c2), vget_low_s16(vxa4), 2);
        vacc5x0123 = vmlal_lane_s16(
            vacc5x0123, vget_low_s16(vxb0123c2), vget_low_s16(vxa5), 2);

        if (k >= 4) {
          const uint8x8_t vb0123c3 = vreinterpret_u8_u32(vld1_dup_u32(w));
          w = (const void*)((uintptr_t)w + 4);
          const int16x8_t vxb0123c3 =
              vreinterpretq_s16_u16(vsubl_u8(vb0123c3, vb_zero_point));

          vacc0x0123 = vmlal_lane_s16(
              vacc0x0123, vget_low_s16(vxb0123c3), vget_low_s16(vxa0), 3);
          vacc1x0123 = vmlal_lane_s16(
              vacc1x0123, vget_low_s16(vxb0123c3), vget_low_s16(vxa1), 3);
          vacc2x0123 = vmlal_lane_s16(
              vacc2x0123, vget_low_s16(vxb0123c3), vget_low_s16(vxa2), 3);
          vacc3x0123 = vmlal_lane_s16(
              vacc3x0123, vget_low_s16(vxb0123c3), vget_low_s16(vxa3), 3);
          vacc4x0123 = vmlal_lane_s16(
              vacc4x0123, vget_low_s16(vxb0123c3), vget_low_s16(vxa4), 3);
          vacc5x0123 = vmlal_lane_s16(
              vacc5x0123, vget_low_s16(vxb0123c3), vget_low_s16(vxa5), 3);

          if (k > 4) {
            const uint8x8_t vb0123c4 = vreinterpret_u8_u32(vld1_dup_u32(w));
            w = (const void*)((uintptr_t)w + 4);
            const int16x8_t vxb0123c4 =
                vreinterpretq_s16_u16(vsubl_u8(vb0123c4, vb_zero_point));

            vacc0x0123 = vmlal_lane_s16(
                vacc0x0123, vget_low_s16(vxb0123c4), vget_high_s16(vxa0), 0);
            vacc1x0123 = vmlal_lane_s16(
                vacc1x0123, vget_low_s16(vxb0123c4), vget_high_s16(vxa1), 0);
            vacc2x0123 = vmlal_lane_s16(
                vacc2x0123, vget_low_s16(vxb0123c4), vget_high_s16(vxa2), 0);
            vacc3x0123 = vmlal_lane_s16(
                vacc3x0123, vget_low_s16(vxb0123c4), vget_high_s16(vxa3), 0);
            vacc4x0123 = vmlal_lane_s16(
                vacc4x0123, vget_low_s16(vxb0123c4), vget_high_s16(vxa4), 0);
            vacc5x0123 = vmlal_lane_s16(
                vacc5x0123, vget_low_s16(vxb0123c4), vget_high_s16(vxa5), 0);

            if (k >= 6) {
              const uint8x8_t vb0123c5 = vreinterpret_u8_u32(vld1_dup_u32(w));
              w = (const void*)((uintptr_t)w + 4);
              const int16x8_t vxb0123c5 =
                  vreinterpretq_s16_u16(vsubl_u8(vb0123c5, vb_zero_point));

              vacc0x0123 = vmlal_lane_s16(
                  vacc0x0123, vget_low_s16(vxb0123c5), vget_high_s16(vxa0), 1);
              vacc1x0123 = vmlal_lane_s16(
                  vacc1x0123, vget_low_s16(vxb0123c5), vget_high_s16(vxa1), 1);
              vacc2x0123 = vmlal_lane_s16(
                  vacc2x0123, vget_low_s16(vxb0123c5), vget_high_s16(vxa2), 1);
              vacc3x0123 = vmlal_lane_s16(
                  vacc3x0123, vget_low_s16(vxb0123c5), vget_high_s16(vxa3), 1);
              vacc4x0123 = vmlal_lane_s16(
                  vacc4x0123, vget_low_s16(vxb0123c5), vget_high_s16(vxa4), 1);
              vacc5x0123 = vmlal_lane_s16(
                  vacc5x0123, vget_low_s16(vxb0123c5), vget_high_s16(vxa5), 1);

              if (k > 6) {
                const uint8x8_t vb0123c6 = vreinterpret_u8_u32(vld1_dup_u32(w));
                const int16x8_t vxb0123c6 =
                    vreinterpretq_s16_u16(vsubl_u8(vb0123c6, vb_zero_point));

                vacc0x0123 = vmlal_lane_s16(
                    vacc0x0123,
                    vget_low_s16(vxb0123c6),
                    vget_high_s16(vxa0),
                    2);
                vacc1x0123 = vmlal_lane_s16(
                    vacc1x0123,
                    vget_low_s16(vxb0123c6),
                    vget_high_s16(vxa1),
                    2);
                vacc2x0123 = vmlal_lane_s16(
                    vacc2x0123,
                    vget_low_s16(vxb0123c6),
                    vget_high_s16(vxa2),
                    2);
                vacc3x0123 = vmlal_lane_s16(
                    vacc3x0123,
                    vget_low_s16(vxb0123c6),
                    vget_high_s16(vxa3),
                    2);
                vacc4x0123 = vmlal_lane_s16(
                    vacc4x0123,
                    vget_low_s16(vxb0123c6),
                    vget_high_s16(vxa4),
                    2);
                vacc5x0123 = vmlal_lane_s16(
                    vacc5x0123,
                    vget_low_s16(vxb0123c6),
                    vget_high_s16(vxa5),
                    2);
              }
            }
          }
        }
      }
    }
  }

  const float32x4_t requantization_scale_v =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[
              output_channel_index]);

  const float32x4_t vacc0x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc0x0123), requantization_scale_v);
  const float32x4_t vacc1x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc1x0123), requantization_scale_v);
  const float32x4_t vacc2x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc2x0123), requantization_scale_v);
  const float32x4_t vacc3x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc3x0123), requantization_scale_v);
  const float32x4_t vacc4x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc4x0123), requantization_scale_v);
  const float32x4_t vacc5x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc5x0123), requantization_scale_v);

#ifdef __aarch64__
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  vacc0x0123 = vcvtnq_s32_f32(vacc0x0123_f);
  vacc1x0123 = vcvtnq_s32_f32(vacc1x0123_f);
  vacc2x0123 = vcvtnq_s32_f32(vacc2x0123_f);
  vacc3x0123 = vcvtnq_s32_f32(vacc3x0123_f);
  vacc4x0123 = vcvtnq_s32_f32(vacc4x0123_f);
  vacc5x0123 = vcvtnq_s32_f32(vacc5x0123_f);

  const int16x8_t vacc01x0123 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc1x0123), voutput_zero_point);
  const int16x8_t vacc23x0123 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc3x0123), voutput_zero_point);
  const int16x8_t vacc45x0123 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc4x0123), vacc5x0123), voutput_zero_point);

  uint8x16_t vout0123x0123 =
      vqmovun_high_s16(vqmovun_s16(vacc01x0123), vacc23x0123);
  uint8x8_t vout45x0123 = vqmovun_s16(vacc45x0123);

  const uint8x16_t voutput_min =
      vld1q_dup_u8(&quantization_params->neon.output_min);
  const uint8x16_t voutput_max =
      vld1q_dup_u8(&quantization_params->neon.output_max);

  vout0123x0123 = vmaxq_u8(vout0123x0123, voutput_min);
  vout45x0123 = vmax_u8(vout45x0123, vget_low_u8(voutput_min));
  vout0123x0123 = vminq_u8(vout0123x0123, voutput_max);
  vout45x0123 = vmin_u8(vout45x0123, vget_low_u8(voutput_max));
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
  const float32x4_t vacc4x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc4x0123_f, vfmin), vfmax);
  const float32x4_t vacc5x0123_f_clamped =
      vminq_f32(vmaxq_f32(vacc5x0123_f, vfmin), vfmax);

  vacc0x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc0x0123_f_clamped, vfmagic)), vimagic);
  vacc1x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc1x0123_f_clamped, vfmagic)), vimagic);
  vacc2x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc2x0123_f_clamped, vfmagic)), vimagic);
  vacc3x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc3x0123_f_clamped, vfmagic)), vimagic);
  vacc4x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc4x0123_f_clamped, vfmagic)), vimagic);
  vacc5x0123 = vsubq_s32(
      vreinterpretq_s32_f32(vaddq_f32(vacc5x0123_f_clamped, vfmagic)), vimagic);

  const int16x8_t vacc01x0123 =
      vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc1x0123));
  const int16x8_t vacc23x0123 =
      vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc3x0123));
  const int16x8_t vacc45x0123 =
      vcombine_s16(vqmovn_s32(vacc4x0123), vqmovn_s32(vacc5x0123));

  uint8x16_t vout0123x0123 =
      vcombine_u8(vqmovun_s16(vacc01x0123), vqmovun_s16(vacc23x0123));
  uint8x8_t vout45x0123 = vqmovun_s16(vacc45x0123);
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
  if (mr != 6) {
    c5 = c4;
  }
  if (nr == 4) {
    vst1q_lane_u32(
        __builtin_assume_aligned(c0, 1),
        vreinterpretq_u32_u8(vout0123x0123),
        0);
    vst1q_lane_u32(
        __builtin_assume_aligned(c1, 1),
        vreinterpretq_u32_u8(vout0123x0123),
        1);
    vst1q_lane_u32(
        __builtin_assume_aligned(c2, 1),
        vreinterpretq_u32_u8(vout0123x0123),
        2);
    vst1q_lane_u32(
        __builtin_assume_aligned(c3, 1),
        vreinterpretq_u32_u8(vout0123x0123),
        3);
    vst1_lane_u32(
        __builtin_assume_aligned(c4, 1), vreinterpret_u32_u8(vout45x0123), 0);
    vst1_lane_u32(
        __builtin_assume_aligned(c5, 1), vreinterpret_u32_u8(vout45x0123), 1);
  } else {
    if (nr >= 2) {
      vst1q_lane_u16(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u16_u8(vout0123x0123),
          0);
      c0 += 2;
      vst1q_lane_u16(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u16_u8(vout0123x0123),
          2);
      c1 += 2;
      vst1q_lane_u16(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u16_u8(vout0123x0123),
          4);
      c2 += 2;
      vst1q_lane_u16(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u16_u8(vout0123x0123),
          6);
      c3 += 2;
      vst1_lane_u16(
          __builtin_assume_aligned(c4, 1), vreinterpret_u16_u8(vout45x0123), 0);
      c4 += 2;
      vst1_lane_u16(
          __builtin_assume_aligned(c5, 1), vreinterpret_u16_u8(vout45x0123), 2);
      c5 += 2;
      vout0123x0123 = vextq_u8(vout0123x0123, vout0123x0123, 2);
      vout45x0123 = vext_u8(vout45x0123, vout45x0123, 2);
      nr -= 2;
    }
    if (nr != 0) {
      vst1q_lane_u8(__builtin_assume_aligned(c0, 1), vout0123x0123, 0);
      vst1q_lane_u8(__builtin_assume_aligned(c1, 1), vout0123x0123, 4);
      vst1q_lane_u8(__builtin_assume_aligned(c2, 1), vout0123x0123, 8);
      vst1q_lane_u8(__builtin_assume_aligned(c3, 1), vout0123x0123, 12);
      vst1_lane_u8(__builtin_assume_aligned(c4, 1), vout45x0123, 0);
      vst1_lane_u8(__builtin_assume_aligned(c5, 1), vout45x0123, 4);
    }
  }
}
