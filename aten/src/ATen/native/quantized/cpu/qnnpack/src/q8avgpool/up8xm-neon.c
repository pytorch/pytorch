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

void pytorch_q8avgpool_ukernel_up8xm__neon(
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
  assert(ks != 0);
  assert(kc < 8);

  const int32x4_t vbias = vld1q_dup_s32(&quantization_params->neon.bias);
  const float32x4_t vscale =
      vdupq_n_f32(quantization_params->neon.scale);
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  const uint8x8_t voutput_min =
      vld1_dup_u8(&quantization_params->neon.output_min);
  const uint8x8_t voutput_max =
      vld1_dup_u8(&quantization_params->neon.output_max);

  do {
    int32x4_t vacc_lo = vbias;
    int32x4_t vacc_hi = vbias;
    const uint8_t** next_input =
        (const uint8_t**)((uintptr_t)input + input_increment);

    size_t m = ks;
    do {
      const uint8_t* i = *input++;
      i += kc;
      uint8x8_t vi = vmov_n_u8(0);
      if (kc & 1) {
        i -= 1;
        vi = vld1_lane_u8(i, vi, 0);
      }
      if (kc & 2) {
        vi = vext_u8(vi, vi, 6);
        i -= 2;
        vi = vreinterpret_u8_u16(vld1_lane_u16(
            __builtin_assume_aligned(i, 1), vreinterpret_u16_u8(vi), 0));
      }
      if (kc & 4) {
        vi = vext_u8(vi, vi, 4);
        i -= 4;
        vi = vreinterpret_u8_u32(vld1_lane_u32(
            __builtin_assume_aligned(i, 1), vreinterpret_u32_u8(vi), 0));
      }

      const uint16x8_t vxi = vmovl_u8(vi);
      vacc_lo = vaddw_s16(vacc_lo, vreinterpret_s16_u16(vget_low_u16(vxi)));
      vacc_hi = vaddw_s16(vacc_hi, vreinterpret_s16_u16(vget_high_u16(vxi)));
    } while (--m != 0);
    input = next_input;

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
    const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
    const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
    const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
    const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);

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

    if (kc & 4) {
      vst1_lane_u32(
          __builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0);
      output += 4;
      vout = vext_u8(vout, vout, 4);
    }
    if (kc & 2) {
      vst1_lane_u16(
          __builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0);
      output += 2;
      vout = vext_u8(vout, vout, 2);
    }
    if (kc & 1) {
      vst1_lane_u8(output, vout, 0);
      output += 1;
    }
    output = (uint8_t*)((uintptr_t)output + output_increment);

  } while (--n != 0);
}
