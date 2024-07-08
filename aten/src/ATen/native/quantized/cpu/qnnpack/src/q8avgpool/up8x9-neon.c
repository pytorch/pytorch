/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <arm_neon.h>

#include <qnnpack/q8avgpool.h>

void pytorch_q8avgpool_ukernel_up8x9__neon(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[restrict static 1]) {
  assert(n != 0);
  assert(ks <= 9);
  assert(kc >= 8);

  const int32x4_t vbias = vld1q_dup_s32(&quantization_params->neon.bias);
  const float32x4_t vscale =
      vdupq_n_f32(quantization_params->neon.scale);
#if defined(__aarch64__)
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  const uint8x8_t voutput_min =
      vld1_dup_u8(&quantization_params->neon.output_min);
  const uint8x8_t voutput_max =
      vld1_dup_u8(&quantization_params->neon.output_max);
#else
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);
#endif

  do {
    const uint8_t* i0 = input[0];
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];
    input = (const uint8_t**)((uintptr_t)input + input_increment);
    if (ks < 2) {
      i1 = zero;
    }
    if (ks <= 2) {
      i2 = zero;
    }
    if (ks < 4) {
      i3 = zero;
    }
    if (ks <= 4) {
      i4 = zero;
    }
    if (ks < 6) {
      i5 = zero;
    }
    if (ks <= 6) {
      i6 = zero;
    }
    if (ks < 8) {
      i7 = zero;
    }
    if (ks <= 8) {
      i8 = zero;
    }

    size_t k = kc;
    while (k >= 8) {
      const uint8x8_t vi0 = vld1_u8(i0);
      i0 += 8;
      const uint8x8_t vi1 = vld1_u8(i1);
      i1 += 8;
      const uint8x8_t vi2 = vld1_u8(i2);
      i2 += 8;
      const uint8x8_t vi3 = vld1_u8(i3);
      i3 += 8;
      const uint8x8_t vi4 = vld1_u8(i4);
      i4 += 8;
      const uint8x8_t vi5 = vld1_u8(i5);
      i5 += 8;
      const uint8x8_t vi6 = vld1_u8(i6);
      i6 += 8;
      const uint8x8_t vi7 = vld1_u8(i7);
      i7 += 8;
      const uint8x8_t vi8 = vld1_u8(i8);
      i8 += 8;

      const uint16x8_t vsum018 = vaddw_u8(vaddl_u8(vi0, vi1), vi8);
      const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
      const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);
      const uint16x8_t vsum67 = vaddl_u8(vi6, vi7);

      const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);
      const uint16x8_t vsum01678 = vaddq_u16(vsum018, vsum67);
      const uint16x8_t vsum = vaddq_u16(vsum2345, vsum01678);

      int32x4_t vacc_lo =
          vaddw_s16(vbias, vreinterpret_s16_u16(vget_low_u16(vsum)));
      int32x4_t vacc_hi =
          vaddw_s16(vbias, vreinterpret_s16_u16(vget_high_u16(vsum)));

      float32x4_t vacc_lo_f = vcvtq_f32_s32(vacc_lo);
      float32x4_t vacc_hi_f = vcvtq_f32_s32(vacc_hi);

      vacc_lo_f = vmulq_f32(vacc_lo_f, vscale);
      vacc_hi_f = vmulq_f32(vacc_hi_f, vscale);

#if defined(__aarch64__)
      vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
      vacc_hi = vcvtnq_s32_f32(vacc_hi_f);
      const int16x8_t vacc = vqaddq_s16(
          vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
      uint8x8_t vout = vqmovun_s16(vacc);
      vout = vmax_u8(vout, voutput_min);
      vout = vmin_u8(vout, voutput_max);
#else
      vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
      vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);

      vacc_lo = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f, vfmagic)), vimagic);
      vacc_hi = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f, vfmagic)), vimagic);
      const int16x8_t vacc =
          vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
      uint8x8_t vout = vqmovun_s16(vacc);
#endif

      vst1_u8(output, vout);
      output += 8;

      k -= 8;
    }
    if (k != 0) {
      const size_t address_increment = k - 8;
      i0 = (const uint8_t*)((uintptr_t)i0 + address_increment);
      i1 = (const uint8_t*)((uintptr_t)i1 + address_increment);
      i2 = (const uint8_t*)((uintptr_t)i2 + address_increment);
      i3 = (const uint8_t*)((uintptr_t)i3 + address_increment);
      i4 = (const uint8_t*)((uintptr_t)i4 + address_increment);
      i5 = (const uint8_t*)((uintptr_t)i5 + address_increment);
      i6 = (const uint8_t*)((uintptr_t)i6 + address_increment);
      i7 = (const uint8_t*)((uintptr_t)i7 + address_increment);
      i8 = (const uint8_t*)((uintptr_t)i8 + address_increment);
      const int64x1_t vshift = vmov_n_s64(8 * address_increment);

      const uint8x8_t vi0 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i0)), vshift));
      const uint8x8_t vi1 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i1)), vshift));
      const uint8x8_t vi2 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i2)), vshift));
      const uint8x8_t vi3 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i3)), vshift));
      const uint8x8_t vi4 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i4)), vshift));
      const uint8x8_t vi5 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i5)), vshift));
      const uint8x8_t vi6 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i6)), vshift));
      const uint8x8_t vi7 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i7)), vshift));
      const uint8x8_t vi8 = vreinterpret_u8_u64(
          vshl_u64(vreinterpret_u64_u8(vld1_u8(i8)), vshift));

      const uint16x8_t vsum018 = vaddw_u8(vaddl_u8(vi0, vi1), vi8);
      const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
      const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);
      const uint16x8_t vsum67 = vaddl_u8(vi6, vi7);

      const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);
      const uint16x8_t vsum01678 = vaddq_u16(vsum018, vsum67);
      const uint16x8_t vsum = vaddq_u16(vsum2345, vsum01678);

      int32x4_t vacc_lo =
          vaddw_s16(vbias, vreinterpret_s16_u16(vget_low_u16(vsum)));
      int32x4_t vacc_hi =
          vaddw_s16(vbias, vreinterpret_s16_u16(vget_high_u16(vsum)));

      float32x4_t vacc_lo_f = vcvtq_f32_s32(vacc_lo);
      float32x4_t vacc_hi_f = vcvtq_f32_s32(vacc_hi);

      vacc_lo_f = vmulq_f32(vacc_lo_f, vscale);
      vacc_hi_f = vmulq_f32(vacc_hi_f, vscale);

#if defined(__aarch64__)
      vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
      vacc_hi = vcvtnq_s32_f32(vacc_hi_f);
      const int16x8_t vacc = vqaddq_s16(
          vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
      uint8x8_t vout = vqmovun_s16(vacc);
      vout = vmax_u8(vout, voutput_min);
      vout = vmin_u8(vout, voutput_max);
#else
      vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
      vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);

      vacc_lo = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f, vfmagic)), vimagic);
      vacc_hi = vsubq_s32(
          vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f, vfmagic)), vimagic);
      const int16x8_t vacc =
          vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
      uint8x8_t vout = vqmovun_s16(vacc);
#endif

      if (k & 4) {
        vst1_lane_u32(
            __builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0);
        output += 4;
        vout = vext_u8(vout, vout, 4);
      }
      if (k & 2) {
        vst1_lane_u16(
            __builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0);
        output += 2;
        vout = vext_u8(vout, vout, 2);
      }
      if (k & 1) {
        vst1_lane_u8(output, vout, 0);
        output += 1;
      }
    }
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--n != 0);
}
