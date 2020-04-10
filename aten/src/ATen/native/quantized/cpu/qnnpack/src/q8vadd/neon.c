/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/common.h>
#include <qnnpack/q8vadd.h>

void pytorch_q8vadd_ukernel__neon(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union pytorch_qnnp_add_quantization_params
        quantization_params[restrict static 1]) {
  const uint8x8_t va_zero_point =
      vld1_dup_u8(&quantization_params->neon.a_zero_point);
  const uint8x8_t vb_zero_point =
      vld1_dup_u8(&quantization_params->neon.b_zero_point);
  const int16x8_t vy_zero_point =
      vld1q_dup_s16(&quantization_params->neon.y_zero_point);
  const float32x4_t va_multiplier =
      vld1q_dup_f32(&quantization_params->neon.a_scale);
  const float32x4_t vb_multiplier =
      vld1q_dup_f32(&quantization_params->neon.b_scale);
  const uint8x16_t vy_max = vld1q_dup_u8(&quantization_params->neon.y_max);
  const uint8x16_t vy_min = vld1q_dup_u8(&quantization_params->neon.y_min);
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);

  uint8x16_t va01;
  uint8x16_t va23;
  uint8x16_t vb01;
  uint8x16_t vb23;
  int16x8_t vxa0;
  int16x8_t vxb0;
  int16x8_t vxa1;
  int16x8_t vxb1;
  int16x8_t vxa2;
  int16x8_t vxb2;
  int16x8_t vxa3;
  int16x8_t vxb3;
  if
    PYTORCH_QNNP_LIKELY(n >= 8) {
#ifdef __aarch64__
      for (; n >= 32; n -= 32) {
        va01 = vld1q_u8(a);
        a += 16;
        vb01 = vld1q_u8(b);
        b += 16;
        va23 = vld1q_u8(a);
        a += 16;
        vb23 = vld1q_u8(b);
        b += 16;

        /* Subtract zero point */
        vxa0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va01), va_zero_point));
        vxb0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb01), vb_zero_point));
        vxa1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va01), va_zero_point));
        vxb1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb01), vb_zero_point));
        vxa2 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va23), va_zero_point));
        vxb2 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb23), vb_zero_point));
        vxa3 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va23), va_zero_point));
        vxb3 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb23), vb_zero_point));

        /*
         * Convert vxa0/1/2/3 to float by movl/vget_low/high
         * convert from s32 to fp32
         * multiply a by a_scale.
         * to multiply add acc = scaled_a + b*b_scale
         * convert accumulator from fp32 to s32
         *   (sequence will be different for 64 vs 32bit arm)
         */
        const int32x4_t vxa0_0 = vmovl_s16(vget_low_s16(vxa0));
        const int32x4_t vxa0_1 = vmovl_high_s16(vxa0);
        const int32x4_t vxa1_0 = vmovl_s16(vget_low_s16(vxa1));
        const int32x4_t vxa1_1 = vmovl_high_s16(vxa1);
        const int32x4_t vxa2_0 = vmovl_s16(vget_low_s16(vxa2));
        const int32x4_t vxa2_1 = vmovl_high_s16(vxa2);
        const int32x4_t vxa3_0 = vmovl_s16(vget_low_s16(vxa3));
        const int32x4_t vxa3_1 = vmovl_high_s16(vxa3);
        const int32x4_t vxb0_0 = vmovl_s16(vget_low_s16(vxb0));
        const int32x4_t vxb0_1 = vmovl_high_s16(vxb0);
        const int32x4_t vxb1_0 = vmovl_s16(vget_low_s16(vxb1));
        const int32x4_t vxb1_1 = vmovl_high_s16(vxb1);
        const int32x4_t vxb2_0 = vmovl_s16(vget_low_s16(vxb2));
        const int32x4_t vxb2_1 = vmovl_high_s16(vxb2);
        const int32x4_t vxb3_0 = vmovl_s16(vget_low_s16(vxb3));
        const int32x4_t vxb3_1 = vmovl_high_s16(vxb3);

        float32x4_t vxa0_0_f = vcvtq_f32_s32(vxa0_0);
        float32x4_t vxa0_1_f = vcvtq_f32_s32(vxa0_1);
        float32x4_t vxa1_0_f = vcvtq_f32_s32(vxa1_0);
        float32x4_t vxa1_1_f = vcvtq_f32_s32(vxa1_1);
        float32x4_t vxa2_0_f = vcvtq_f32_s32(vxa2_0);
        float32x4_t vxa2_1_f = vcvtq_f32_s32(vxa2_1);
        float32x4_t vxa3_0_f = vcvtq_f32_s32(vxa3_0);
        float32x4_t vxa3_1_f = vcvtq_f32_s32(vxa3_1);
        float32x4_t vxb0_0_f = vcvtq_f32_s32(vxb0_0);
        float32x4_t vxb0_1_f = vcvtq_f32_s32(vxb0_1);
        float32x4_t vxb1_0_f = vcvtq_f32_s32(vxb1_0);
        float32x4_t vxb1_1_f = vcvtq_f32_s32(vxb1_1);
        float32x4_t vxb2_0_f = vcvtq_f32_s32(vxb2_0);
        float32x4_t vxb2_1_f = vcvtq_f32_s32(vxb2_1);
        float32x4_t vxb3_0_f = vcvtq_f32_s32(vxb3_0);
        float32x4_t vxb3_1_f = vcvtq_f32_s32(vxb3_1);

        float32x4_t vacc0_0_f = vmulq_f32(vxa0_0_f, va_multiplier);
        float32x4_t vacc0_1_f = vmulq_f32(vxa0_1_f, va_multiplier);
        float32x4_t vacc1_0_f = vmulq_f32(vxa1_0_f, va_multiplier);
        float32x4_t vacc1_1_f = vmulq_f32(vxa1_1_f, va_multiplier);
        float32x4_t vacc2_0_f = vmulq_f32(vxa2_0_f, va_multiplier);
        float32x4_t vacc2_1_f = vmulq_f32(vxa2_1_f, va_multiplier);
        float32x4_t vacc3_0_f = vmulq_f32(vxa3_0_f, va_multiplier);
        float32x4_t vacc3_1_f = vmulq_f32(vxa3_1_f, va_multiplier);

        vacc0_0_f = vfmaq_f32(vacc0_0_f, vxb0_0_f, vb_multiplier);
        vacc0_1_f = vfmaq_f32(vacc0_1_f, vxb0_1_f, vb_multiplier);
        vacc1_0_f = vfmaq_f32(vacc1_0_f, vxb1_0_f, vb_multiplier);
        vacc1_1_f = vfmaq_f32(vacc1_1_f, vxb1_1_f, vb_multiplier);
        vacc2_0_f = vfmaq_f32(vacc2_0_f, vxb2_0_f, vb_multiplier);
        vacc2_1_f = vfmaq_f32(vacc2_1_f, vxb2_1_f, vb_multiplier);
        vacc3_0_f = vfmaq_f32(vacc3_0_f, vxb3_0_f, vb_multiplier);
        vacc3_1_f = vfmaq_f32(vacc3_1_f, vxb3_1_f, vb_multiplier);

        int32x4_t vacc0_lo = vcvtnq_s32_f32(vacc0_0_f);
        int32x4_t vacc0_hi = vcvtnq_s32_f32(vacc0_1_f);
        int32x4_t vacc1_lo = vcvtnq_s32_f32(vacc1_0_f);
        int32x4_t vacc1_hi = vcvtnq_s32_f32(vacc1_1_f);
        int32x4_t vacc2_lo = vcvtnq_s32_f32(vacc2_0_f);
        int32x4_t vacc2_hi = vcvtnq_s32_f32(vacc2_1_f);
        int32x4_t vacc3_lo = vcvtnq_s32_f32(vacc3_0_f);
        int32x4_t vacc3_hi = vcvtnq_s32_f32(vacc3_1_f);


        /* Pack, saturate, and add output zero point */
        const int16x8_t vacc0 = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc0_lo), vacc0_hi), vy_zero_point);
        const int16x8_t vacc1 = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc1_lo), vacc1_hi), vy_zero_point);
        const int16x8_t vacc2 = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc2_lo), vacc2_hi), vy_zero_point);
        const int16x8_t vacc3 = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc3_lo), vacc3_hi), vy_zero_point);

        uint8x16_t vy01 = vqmovun_high_s16(vqmovun_s16(vacc0), vacc1);
        uint8x16_t vy23 = vqmovun_high_s16(vqmovun_s16(vacc2), vacc3);

        vy01 = vmaxq_u8(vy01, vy_min);
        vy23 = vmaxq_u8(vy23, vy_min);
        vy01 = vminq_u8(vy01, vy_max);
        vy23 = vminq_u8(vy23, vy_max);

        vst1q_u8(y, vy01);
        y += 16;
        vst1q_u8(y, vy23);
        y += 16;
      }
#else
      if (n >=32) {
        va01 = vld1q_u8(a);
        a += 16;
        vb01 = vld1q_u8(b);
        b += 16;
        /* Subtract zero point */
        vxa0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va01), va_zero_point));
        vxb0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb01), vb_zero_point));
        vxa1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va01), va_zero_point));
        vxb1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb01), vb_zero_point));
      for (; n >= 32; n -= 16) {
        /*
         * Convert vxa0/1 to float by movl/vget_low/high
         * convert from s32 to fp32
         * multiply a by a_scale.
         * to multiply add acc = scaled_a + b*b_scale
         * convert accumulator from fp32 to s32
         *   (sequence will be different for 64 vs 32bit arm)
         */
        const int32x4_t vxa0_0 = vmovl_s16(vget_low_s16(vxa0));
        const int32x4_t vxa0_1 = vmovl_s16(vget_high_s16(vxa0));
        const int32x4_t vxa1_0 = vmovl_s16(vget_low_s16(vxa1));
        const int32x4_t vxa1_1 = vmovl_s16(vget_high_s16(vxa1));
        const int32x4_t vxb0_0 = vmovl_s16(vget_low_s16(vxb0));
        const int32x4_t vxb0_1 = vmovl_s16(vget_high_s16(vxb0));
        const int32x4_t vxb1_0 = vmovl_s16(vget_low_s16(vxb1));
        const int32x4_t vxb1_1 = vmovl_s16(vget_high_s16(vxb1));

        float32x4_t vxa0_0_f = vcvtq_f32_s32(vxa0_0);
        float32x4_t vxa0_1_f = vcvtq_f32_s32(vxa0_1);
        float32x4_t vxa1_0_f = vcvtq_f32_s32(vxa1_0);
        float32x4_t vxa1_1_f = vcvtq_f32_s32(vxa1_1);
        float32x4_t vxb0_0_f = vcvtq_f32_s32(vxb0_0);
        float32x4_t vxb0_1_f = vcvtq_f32_s32(vxb0_1);
        float32x4_t vxb1_0_f = vcvtq_f32_s32(vxb1_0);
        float32x4_t vxb1_1_f = vcvtq_f32_s32(vxb1_1);

        float32x4_t vacc0_0_f = vmulq_f32(vxa0_0_f, va_multiplier);
        float32x4_t vacc0_1_f = vmulq_f32(vxa0_1_f, va_multiplier);
        float32x4_t vacc1_0_f = vmulq_f32(vxa1_0_f, va_multiplier);
        float32x4_t vacc1_1_f = vmulq_f32(vxa1_1_f, va_multiplier);

        vacc0_0_f = vmlaq_f32(vacc0_0_f, vxb0_0_f, vb_multiplier);
        vacc0_1_f = vmlaq_f32(vacc0_1_f, vxb0_1_f, vb_multiplier);
        vacc1_0_f = vmlaq_f32(vacc1_0_f, vxb1_0_f, vb_multiplier);
        vacc1_1_f = vmlaq_f32(vacc1_1_f, vxb1_1_f, vb_multiplier);

        va01 = vld1q_u8(a);
        a += 16;
        vb01 = vld1q_u8(b);
        b += 16;

        /* Subtract zero point */
        vxa0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va01), va_zero_point));
        vxb0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb01), vb_zero_point));
        vxa1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va01), va_zero_point));
        vxb1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb01), vb_zero_point));

        int32x4_t vacc0_lo = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc0_0_f, vfmagic)), vimagic);
        int32x4_t vacc0_hi = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc0_1_f, vfmagic)), vimagic);
        int32x4_t vacc1_lo = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc1_0_f, vfmagic)), vimagic);
        int32x4_t vacc1_hi = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc1_1_f, vfmagic)), vimagic);

        /* Pack, saturate, and add output zero point */
        const int16x8_t vacc0 =
            vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi));
        const int16x8_t vacc1 =
            vcombine_s16(vqmovn_s32(vacc1_lo), vqmovn_s32(vacc1_hi));

        uint8x16_t vy01 = vcombine_u8(vqmovun_s16(vacc0), vqmovun_s16(vacc1));

        vy01 = vmaxq_u8(vy01, vy_min);
        vy01 = vminq_u8(vy01, vy_max);

        vst1q_u8(y, vy01);
        y += 16;
      }
      a = a - 16;
      b = b - 16;
      }
#endif
      for (; n >= 8; n -= 8) {
        const uint8x8_t va = vld1_u8(a);
        a += 8;
        const uint8x8_t vb = vld1_u8(b);
        b += 8;

        /* Subtract zero point */
        const int16x8_t vxa =
            vreinterpretq_s16_u16(vsubl_u8(va, va_zero_point));
        const int16x8_t vxb =
            vreinterpretq_s16_u16(vsubl_u8(vb, vb_zero_point));

#ifdef __aarch64__
        const int32x4_t vxa_0 = vmovl_s16(vget_low_s16(vxa));
        const int32x4_t vxa_1 = vmovl_high_s16(vxa);
        const int32x4_t vxb_0 = vmovl_s16(vget_low_s16(vxb));
        const int32x4_t vxb_1 = vmovl_high_s16(vxb);

        float32x4_t vxa_0_f = vcvtq_f32_s32(vxa_0);
        float32x4_t vxa_1_f = vcvtq_f32_s32(vxa_1);
        float32x4_t vxb_0_f = vcvtq_f32_s32(vxb_0);
        float32x4_t vxb_1_f = vcvtq_f32_s32(vxb_1);

        float32x4_t vacc_0_f = vmulq_f32(vxa_0_f, va_multiplier);
        float32x4_t vacc_1_f = vmulq_f32(vxa_1_f, va_multiplier);

        vacc_0_f = vmlaq_f32(vacc_0_f, vxb_0_f, vb_multiplier);
        vacc_1_f = vmlaq_f32(vacc_1_f, vxb_1_f, vb_multiplier);

        int32x4_t vacc_lo = vcvtnq_s32_f32(vacc_0_f);
        int32x4_t vacc_hi = vcvtnq_s32_f32(vacc_1_f);

        /* Pack, saturate, and add output zero point */
        const int16x8_t vacc = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), vy_zero_point);
        uint8x8_t vy = vqmovun_s16(vacc);
        vy = vmax_u8(vy, vget_low_u8(vy_min));
        vy = vmin_u8(vy, vget_low_u8(vy_max));
#else
        const int32x4_t vxa_0 = vmovl_s16(vget_low_s16(vxa));
        const int32x4_t vxa_1 = vmovl_s16(vget_high_s16(vxa));
        const int32x4_t vxb_0 = vmovl_s16(vget_low_s16(vxb));
        const int32x4_t vxb_1 = vmovl_s16(vget_high_s16(vxb));

        float32x4_t vxa_0_f = vcvtq_f32_s32(vxa_0);
        float32x4_t vxa_1_f = vcvtq_f32_s32(vxa_1);
        float32x4_t vxb_0_f = vcvtq_f32_s32(vxb_0);
        float32x4_t vxb_1_f = vcvtq_f32_s32(vxb_1);

        float32x4_t vacc_0_f = vmulq_f32(vxa_0_f, va_multiplier);
        float32x4_t vacc_1_f = vmulq_f32(vxa_1_f, va_multiplier);

        vacc_0_f = vmlaq_f32(vacc_0_f, vxb_0_f, vb_multiplier);
        vacc_1_f = vmlaq_f32(vacc_1_f, vxb_1_f, vb_multiplier);

        int32x4_t vacc_lo = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_0_f, vfmagic)), vimagic);
        int32x4_t vacc_hi = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_1_f, vfmagic)), vimagic);
        /* Pack, saturate, and add output zero point */
        const int16x8_t vacc =
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
        uint8x8_t vy = vqmovun_s16(vacc);
        vy = vmax_u8(vy, vget_low_u8(vy_min));
        vy = vmin_u8(vy, vget_low_u8(vy_max));
#endif

        vst1_u8(y, vy);
        y += 8;
      }
      if (n != 0) {
        const size_t n_increment = n - 8;
        const int64x1_t vld_shift = vmov_n_s64(8 * n_increment);
        const uint8x8_t va = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(a + n_increment)), vld_shift));
        const uint8x8_t vb = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(b + n_increment)), vld_shift));

        /* Subtract zero point */
        const int16x8_t vxa =
            vreinterpretq_s16_u16(vsubl_u8(va, va_zero_point));
        const int16x8_t vxb =
            vreinterpretq_s16_u16(vsubl_u8(vb, vb_zero_point));

#ifdef __aarch64__
        const int32x4_t vxa_0 = vmovl_s16(vget_low_s16(vxa));
        const int32x4_t vxa_1 = vmovl_high_s16(vxa);
        const int32x4_t vxb_0 = vmovl_s16(vget_low_s16(vxb));
        const int32x4_t vxb_1 = vmovl_high_s16(vxb);

        float32x4_t vxa_0_f = vcvtq_f32_s32(vxa_0);
        float32x4_t vxa_1_f = vcvtq_f32_s32(vxa_1);
        float32x4_t vxb_0_f = vcvtq_f32_s32(vxb_0);
        float32x4_t vxb_1_f = vcvtq_f32_s32(vxb_1);

        float32x4_t vacc_0_f = vmulq_f32(vxa_0_f, va_multiplier);
        float32x4_t vacc_1_f = vmulq_f32(vxa_1_f, va_multiplier);

        vacc_0_f = vmlaq_f32(vacc_0_f, vxb_0_f, vb_multiplier);
        vacc_1_f = vmlaq_f32(vacc_1_f, vxb_1_f, vb_multiplier);

        int32x4_t vacc_lo = vcvtnq_s32_f32(vacc_0_f);
        int32x4_t vacc_hi = vcvtnq_s32_f32(vacc_1_f);

        /* Pack, saturate, and add output zero point */
        const int16x8_t vacc = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), vy_zero_point);
        uint8x8_t vy = vqmovun_s16(vacc);
        vy = vmax_u8(vy, vget_low_u8(vy_min));
        vy = vmin_u8(vy, vget_low_u8(vy_max));
#else
        const int32x4_t vxa_0 = vmovl_s16(vget_low_s16(vxa));
        const int32x4_t vxa_1 = vmovl_s16(vget_high_s16(vxa));
        const int32x4_t vxb_0 = vmovl_s16(vget_low_s16(vxb));
        const int32x4_t vxb_1 = vmovl_s16(vget_high_s16(vxb));

        float32x4_t vxa_0_f = vcvtq_f32_s32(vxa_0);
        float32x4_t vxa_1_f = vcvtq_f32_s32(vxa_1);
        float32x4_t vxb_0_f = vcvtq_f32_s32(vxb_0);
        float32x4_t vxb_1_f = vcvtq_f32_s32(vxb_1);

        float32x4_t vacc_0_f = vmulq_f32(vxa_0_f, va_multiplier);
        float32x4_t vacc_1_f = vmulq_f32(vxa_1_f, va_multiplier);

        vacc_0_f = vmlaq_f32(vacc_0_f, vxb_0_f, vb_multiplier);
        vacc_1_f = vmlaq_f32(vacc_1_f, vxb_1_f, vb_multiplier);

        int32x4_t vacc_lo = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_0_f, vfmagic)), vimagic);
        int32x4_t vacc_hi = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_1_f, vfmagic)), vimagic);
        /* Pack, saturate, and add output zero point */
        const int16x8_t vacc =
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
        uint8x8_t vy = vqmovun_s16(vacc);
        vy = vmax_u8(vy, vget_low_u8(vy_min));
        vy = vmin_u8(vy, vget_low_u8(vy_max));
#endif

        if (n & 4) {
          vst1_lane_u32(
              __builtin_assume_aligned(y, 1), vreinterpret_u32_u8(vy), 0);
          y += 4;
          vy = vext_u8(vy, vy, 4);
        }
        if (n & 2) {
          vst1_lane_u16(
              __builtin_assume_aligned(y, 1), vreinterpret_u16_u8(vy), 0);
          y += 2;
          vy = vext_u8(vy, vy, 2);
        }
        if (n & 1) {
          vst1_lane_u8(y, vy, 0);
        }
      }
    }
  else {
    for (; n != 0; n--) {
      const uint8x8_t va = vld1_dup_u8(a);
      a += 1;
      const uint8x8_t vb = vld1_dup_u8(b);
      b += 1;

      /* Subtract zero point */
      const int16x4_t vxa =
          vreinterpret_s16_u16(vget_low_u16(vsubl_u8(va, va_zero_point)));
      const int16x4_t vxb =
          vreinterpret_s16_u16(vget_low_u16(vsubl_u8(vb, vb_zero_point)));

      float32x2_t vxa_f = vcvt_f32_s32(vget_low_s32(vmovl_s16(vxa)));
      float32x2_t vxb_f = vcvt_f32_s32(vget_low_s32(vmovl_s16(vxb)));

      float32x2_t vacc_f = vmul_f32(vxa_f, vget_low_f32(va_multiplier));
      vacc_f = vmla_f32(vacc_f, vxb_f, vget_low_f32(vb_multiplier));

#ifdef __aarch64__
      int32x2_t vacc = vcvtn_s32_f32(vacc_f);

      const int16x4_t vacc16 = vqadd_s16(
          vqmovn_s32(vcombine_s32(vacc, vacc)), vget_low_s16(vy_zero_point));

      /* Pack, saturate, and add output zero point */
      uint8x8_t vy = vqmovun_s16(vcombine_s16(vacc16, vacc16));
      vy = vmin_u8(vy, vget_low_u8(vy_max));
      vy = vmax_u8(vy, vget_low_u8(vy_min));
#else
      int32x2_t vacc = vsub_s32(
          vreinterpret_s32_f32(vadd_f32(vacc_f, vget_low_f32(vfmagic))),
          vget_low_s32(vimagic));
      const int16x4_t vacc16 = vqmovn_s32(vcombine_s32(vacc, vacc));

      /* Pack, saturate, and add output zero point */
      uint8x8_t vy = vqmovun_s16(vcombine_s16(vacc16, vacc16));
      vy = vmax_u8(vy, vget_low_u8(vy_min));
      vy = vmin_u8(vy, vget_low_u8(vy_max));
#endif
      vst1_lane_u8(y, vy, 0);
      y += 1;
    }
  }
}
