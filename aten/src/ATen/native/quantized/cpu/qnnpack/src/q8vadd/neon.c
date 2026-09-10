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
  const int32x4_t va_multiplier =
      vld1q_dup_s32(&quantization_params->neon.a_multiplier);
  const int32x4_t vb_multiplier =
      vld1q_dup_s32(&quantization_params->neon.b_multiplier);
  const int32x4_t vright_shift =
      vld1q_dup_s32(&quantization_params->neon.right_shift);
  const int32x4_t vzero_shift_mask =
      vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
  const uint8x16_t vy_max = vld1q_dup_u8(&quantization_params->neon.y_max);
  const uint8x16_t vy_min = vld1q_dup_u8(&quantization_params->neon.y_min);
  if
    PYTORCH_QNNP_LIKELY(n >= 8) {
#ifdef __aarch64__
      for (; n >= 32; n -= 32) {
        const uint8x16_t va01 = vld1q_u8(a);
        a += 16;
        const uint8x16_t vb01 = vld1q_u8(b);
        b += 16;
        const uint8x16_t va23 = vld1q_u8(a);
        a += 16;
        const uint8x16_t vb23 = vld1q_u8(b);
        b += 16;

        /* Subtract zero point */
        const int16x8_t vxa0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va01), va_zero_point));
        const int16x8_t vxb0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb01), vb_zero_point));
        const int16x8_t vxa1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va01), va_zero_point));
        const int16x8_t vxb1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb01), vb_zero_point));
        const int16x8_t vxa2 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va23), va_zero_point));
        const int16x8_t vxb2 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb23), vb_zero_point));
        const int16x8_t vxa3 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va23), va_zero_point));
        const int16x8_t vxb3 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb23), vb_zero_point));

        /* Multiply by factors and accumulate products */
        int32x4_t vacc0_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa0)), va_multiplier);
        int32x4_t vacc1_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa1)), va_multiplier);
        int32x4_t vacc2_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa2)), va_multiplier);
        int32x4_t vacc3_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa3)), va_multiplier);
        int32x4_t vacc0_hi = vmulq_s32(vmovl_high_s16(vxa0), va_multiplier);
        int32x4_t vacc1_hi = vmulq_s32(vmovl_high_s16(vxa1), va_multiplier);
        int32x4_t vacc2_hi = vmulq_s32(vmovl_high_s16(vxa2), va_multiplier);
        int32x4_t vacc3_hi = vmulq_s32(vmovl_high_s16(vxa3), va_multiplier);

        vacc0_lo =
            vmlaq_s32(vacc0_lo, vmovl_s16(vget_low_s16(vxb0)), vb_multiplier);
        vacc1_lo =
            vmlaq_s32(vacc1_lo, vmovl_s16(vget_low_s16(vxb1)), vb_multiplier);
        vacc2_lo =
            vmlaq_s32(vacc2_lo, vmovl_s16(vget_low_s16(vxb2)), vb_multiplier);
        vacc3_lo =
            vmlaq_s32(vacc3_lo, vmovl_s16(vget_low_s16(vxb3)), vb_multiplier);
        vacc0_hi = vmlaq_s32(vacc0_hi, vmovl_high_s16(vxb0), vb_multiplier);
        vacc1_hi = vmlaq_s32(vacc1_hi, vmovl_high_s16(vxb1), vb_multiplier);
        vacc2_hi = vmlaq_s32(vacc2_hi, vmovl_high_s16(vxb2), vb_multiplier);
        vacc3_hi = vmlaq_s32(vacc3_hi, vmovl_high_s16(vxb3), vb_multiplier);

        /* Shift right and round */
        vacc0_lo =
            vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
        vacc1_lo =
            vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
        vacc2_lo =
            vsraq_n_s32(vacc2_lo, vbicq_s32(vacc2_lo, vzero_shift_mask), 31);
        vacc3_lo =
            vsraq_n_s32(vacc3_lo, vbicq_s32(vacc3_lo, vzero_shift_mask), 31);
        vacc0_hi =
            vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
        vacc1_hi =
            vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);
        vacc2_hi =
            vsraq_n_s32(vacc2_hi, vbicq_s32(vacc2_hi, vzero_shift_mask), 31);
        vacc3_hi =
            vsraq_n_s32(vacc3_hi, vbicq_s32(vacc3_hi, vzero_shift_mask), 31);

        vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
        vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
        vacc2_lo = vrshlq_s32(vacc2_lo, vright_shift);
        vacc3_lo = vrshlq_s32(vacc3_lo, vright_shift);
        vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
        vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);
        vacc2_hi = vrshlq_s32(vacc2_hi, vright_shift);
        vacc3_hi = vrshlq_s32(vacc3_hi, vright_shift);

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
      for (; n >= 16; n -= 16) {
        const uint8x16_t va01 = vld1q_u8(a);
        a += 16;
        const uint8x16_t vb01 = vld1q_u8(b);
        b += 16;

        /* Subtract zero point */
        const int16x8_t vxa0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va01), va_zero_point));
        const int16x8_t vxb0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb01), vb_zero_point));
        const int16x8_t vxa1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va01), va_zero_point));
        const int16x8_t vxb1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb01), vb_zero_point));

        /* Multiply by factors and accumulate products */
        int32x4_t vacc0_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa0)), va_multiplier);
        int32x4_t vacc1_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa1)), va_multiplier);
        int32x4_t vacc0_hi =
            vmulq_s32(vmovl_s16(vget_high_s16(vxa0)), va_multiplier);
        int32x4_t vacc1_hi =
            vmulq_s32(vmovl_s16(vget_high_s16(vxa1)), va_multiplier);

        __builtin_prefetch(a + 640);
        __builtin_prefetch(b + 640);

        vacc0_lo =
            vmlaq_s32(vacc0_lo, vmovl_s16(vget_low_s16(vxb0)), vb_multiplier);
        vacc1_lo =
            vmlaq_s32(vacc1_lo, vmovl_s16(vget_low_s16(vxb1)), vb_multiplier);
        vacc0_hi =
            vmlaq_s32(vacc0_hi, vmovl_s16(vget_high_s16(vxb0)), vb_multiplier);
        vacc1_hi =
            vmlaq_s32(vacc1_hi, vmovl_s16(vget_high_s16(vxb1)), vb_multiplier);

        /* Shift right and round */
        vacc0_lo =
            vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
        vacc1_lo =
            vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
        vacc0_hi =
            vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
        vacc1_hi =
            vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);

        vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
        vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
        vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
        vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);

        /* Pack, saturate, and add output zero point */
        const int16x8_t vacc0 = vqaddq_s16(
            vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi)),
            vy_zero_point);
        const int16x8_t vacc1 = vqaddq_s16(
            vcombine_s16(vqmovn_s32(vacc1_lo), vqmovn_s32(vacc1_hi)),
            vy_zero_point);

        uint8x16_t vy01 = vcombine_u8(vqmovun_s16(vacc0), vqmovun_s16(vacc1));
        vy01 = vmaxq_u8(vy01, vy_min);
        vy01 = vminq_u8(vy01, vy_max);

        vst1q_u8(y, vy01);
        y += 16;
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

        /* Multiply by factors and accumulate products */
        int32x4_t vacc_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa)), va_multiplier);
#ifdef __aarch64__
        int32x4_t vacc_hi = vmulq_s32(vmovl_high_s16(vxa), va_multiplier);
#else
        int32x4_t vacc_hi =
            vmulq_s32(vmovl_s16(vget_high_s16(vxa)), va_multiplier);
#endif

        vacc_lo =
            vmlaq_s32(vacc_lo, vmovl_s16(vget_low_s16(vxb)), vb_multiplier);
#ifdef __aarch64__
        vacc_hi = vmlaq_s32(vacc_hi, vmovl_high_s16(vxb), vb_multiplier);
#else
        vacc_hi =
            vmlaq_s32(vacc_hi, vmovl_s16(vget_high_s16(vxb)), vb_multiplier);
#endif

        /* Shift right and round */
        vacc_lo =
            vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
        vacc_hi =
            vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

        vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
        vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

        /* Pack, saturate, and add output zero point */
#ifdef __aarch64__
        const int16x8_t vacc = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), vy_zero_point);
#else
        const int16x8_t vacc = vqaddq_s16(
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)),
            vy_zero_point);
#endif

        uint8x8_t vy = vqmovun_s16(vacc);
        vy = vmax_u8(vy, vget_low_u8(vy_min));
        vy = vmin_u8(vy, vget_low_u8(vy_max));

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

        /* Multiply by factors and accumulate products */
        int32x4_t vacc_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa)), va_multiplier);
#ifdef __aarch64__
        int32x4_t vacc_hi = vmulq_s32(vmovl_high_s16(vxa), va_multiplier);
#else
        int32x4_t vacc_hi =
            vmulq_s32(vmovl_s16(vget_high_s16(vxa)), va_multiplier);
#endif

        vacc_lo =
            vmlaq_s32(vacc_lo, vmovl_s16(vget_low_s16(vxb)), vb_multiplier);
#ifdef __aarch64__
        vacc_hi = vmlaq_s32(vacc_hi, vmovl_high_s16(vxb), vb_multiplier);
#else
        vacc_hi =
            vmlaq_s32(vacc_hi, vmovl_s16(vget_high_s16(vxb)), vb_multiplier);
#endif

        /* Shift right and round */
        vacc_lo =
            vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
        vacc_hi =
            vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

        vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
        vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

        /* Pack, saturate, and add output zero point */
#ifdef __aarch64__
        const int16x8_t vacc = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), vy_zero_point);
#else
        const int16x8_t vacc = vqaddq_s16(
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)),
            vy_zero_point);
#endif

        uint8x8_t vy = vqmovun_s16(vacc);
        vy = vmax_u8(vy, vget_low_u8(vy_min));
        vy = vmin_u8(vy, vget_low_u8(vy_max));

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

      /* Multiply by factors and accumulate products */
      int32x2_t vacc =
          vmul_s32(vget_low_s32(vmovl_s16(vxa)), vget_low_s32(va_multiplier));
      vacc = vmla_s32(
          vacc, vget_low_s32(vmovl_s16(vxb)), vget_low_s32(vb_multiplier));

      /* Shift right and round */
      vacc =
          vsra_n_s32(vacc, vbic_s32(vacc, vget_low_s32(vzero_shift_mask)), 31);

      vacc = vrshl_s32(vacc, vget_low_s32(vright_shift));

      const int16x4_t vacc16 = vqadd_s16(
          vqmovn_s32(vcombine_s32(vacc, vacc)), vget_low_s16(vy_zero_point));

      /* Pack, saturate, and add output zero point */
      uint8x8_t vy = vqmovun_s16(vcombine_s16(vacc16, vacc16));
      vy = vmin_u8(vy, vget_low_u8(vy_max));
      vy = vmax_u8(vy, vget_low_u8(vy_min));

      vst1_lane_u8(y, vy, 0);
      y += 1;
    }
  }
}
