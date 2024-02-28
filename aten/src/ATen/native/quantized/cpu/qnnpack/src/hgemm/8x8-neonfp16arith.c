/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/hgemm.h>

void pytorch_hgemm_ukernel_8x8__neonfp16arith(
    size_t mr,
    size_t nr,
    size_t k,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t c_stride,
    const struct pytorch_qnnp_fp16_clamping_params
        clamping_params[restrict static 1]) {
  float16x8_t vacc0x01234567 = vld1q_f16(w);
  w = (void*)((uintptr_t)w + sizeof(float16x8_t));
  float16x8_t vacc1x01234567 = vacc0x01234567;
  float16x8_t vacc2x01234567 = vacc0x01234567;
  float16x8_t vacc3x01234567 = vacc0x01234567;
  float16x8_t vacc4x01234567 = vacc0x01234567;
  float16x8_t vacc5x01234567 = vacc0x01234567;
  float16x8_t vacc6x01234567 = vacc0x01234567;
  float16x8_t vacc7x01234567 = vacc0x01234567;

  const __fp16* a0 = a;
  const __fp16* a1 = (const __fp16*)((uintptr_t)a0 + a_stride);
  if (mr < 2) {
    a1 = a0;
  }
  const __fp16* a2 = (const __fp16*)((uintptr_t)a1 + a_stride);
  if (mr <= 2) {
    a2 = a1;
  }
  const __fp16* a3 = (const __fp16*)((uintptr_t)a2 + a_stride);
  if (mr < 4) {
    a3 = a2;
  }
  const __fp16* a4 = (const __fp16*)((uintptr_t)a3 + a_stride);
  if (mr <= 4) {
    a4 = a3;
  }
  const __fp16* a5 = (const __fp16*)((uintptr_t)a4 + a_stride);
  if (mr < 6) {
    a5 = a4;
  }
  const __fp16* a6 = (const __fp16*)((uintptr_t)a5 + a_stride);
  if (mr <= 6) {
    a6 = a5;
  }
  const __fp16* a7 = (const __fp16*)((uintptr_t)a6 + a_stride);
  if (mr != 8) {
    a7 = a6;
  }

  for (; k >= 4; k -= 4) {
    const float16x4_t va0 = vld1_f16(a0);
    a0 += 4;
    const float16x4_t va1 = vld1_f16(a1);
    a1 += 4;
    const float16x4_t va2 = vld1_f16(a2);
    a2 += 4;
    const float16x4_t va3 = vld1_f16(a3);
    a3 += 4;
    const float16x4_t va4 = vld1_f16(a4);
    a4 += 4;
    const float16x4_t va5 = vld1_f16(a5);
    a5 += 4;
    const float16x4_t va6 = vld1_f16(a6);
    a6 += 4;
    const float16x4_t va7 = vld1_f16(a7);
    a7 += 4;

    {
      const float16x8_t vb01234567 = vld1q_f16(w);
      w = (void*)((uintptr_t)w + sizeof(float16x8_t));

      vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 0);
      vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 0);
      vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 0);
      vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 0);
      vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 0);
      vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 0);
      vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 0);
      vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 0);
    }

    {
      const float16x8_t vb01234567 = vld1q_f16(w);
      w = (void*)((uintptr_t)w + sizeof(float16x8_t));

      vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 1);
      vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 1);
      vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 1);
      vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 1);
      vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 1);
      vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 1);
      vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 1);
      vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 1);
    }

    {
      const float16x8_t vb01234567 = vld1q_f16(w);
      w = (void*)((uintptr_t)w + sizeof(float16x8_t));

      vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 2);
      vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 2);
      vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 2);
      vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 2);
      vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 2);
      vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 2);
      vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 2);
      vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 2);
    }

    {
      const float16x8_t vb01234567 = vld1q_f16(w);
      w = (void*)((uintptr_t)w + sizeof(float16x8_t));

      vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 3);
      vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 3);
      vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 3);
      vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 3);
      vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 3);
      vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 3);
      vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 3);
      vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 3);
    }
  }
  if (k != 0) {
    const size_t a_predecrement = 4 - k;
    const int64x1_t va_shift = vmov_n_s64(-16 * a_predecrement);
    const float16x4_t va0 = vreinterpret_f16_u64(vshl_u64(
        vreinterpret_u64_f16(vld1_f16(a0 - a_predecrement)), va_shift));
    const float16x4_t va1 = vreinterpret_f16_u64(vshl_u64(
        vreinterpret_u64_f16(vld1_f16(a1 - a_predecrement)), va_shift));
    const float16x4_t va2 = vreinterpret_f16_u64(vshl_u64(
        vreinterpret_u64_f16(vld1_f16(a2 - a_predecrement)), va_shift));
    const float16x4_t va3 = vreinterpret_f16_u64(vshl_u64(
        vreinterpret_u64_f16(vld1_f16(a3 - a_predecrement)), va_shift));
    const float16x4_t va4 = vreinterpret_f16_u64(vshl_u64(
        vreinterpret_u64_f16(vld1_f16(a4 - a_predecrement)), va_shift));
    const float16x4_t va5 = vreinterpret_f16_u64(vshl_u64(
        vreinterpret_u64_f16(vld1_f16(a5 - a_predecrement)), va_shift));
    const float16x4_t va6 = vreinterpret_f16_u64(vshl_u64(
        vreinterpret_u64_f16(vld1_f16(a6 - a_predecrement)), va_shift));
    const float16x4_t va7 = vreinterpret_f16_u64(vshl_u64(
        vreinterpret_u64_f16(vld1_f16(a7 - a_predecrement)), va_shift));

    {
      const float16x8_t vb01234567 = vld1q_f16(w);
      w = (void*)((uintptr_t)w + sizeof(float16x8_t));

      vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 0);
      vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 0);
      vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 0);
      vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 0);
      vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 0);
      vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 0);
      vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 0);
      vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 0);
    }

    if (k >= 2) {
      const float16x8_t vb01234567 = vld1q_f16(w);
      w = (void*)((uintptr_t)w + sizeof(float16x8_t));

      vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 1);
      vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 1);
      vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 1);
      vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 1);
      vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 1);
      vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 1);
      vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 1);
      vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 1);

      if (k > 2) {
        const float16x8_t vb01234567 = vld1q_f16(w);
        w = (void*)((uintptr_t)w + sizeof(float16x8_t));

        vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 2);
        vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 2);
        vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 2);
        vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 2);
        vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 2);
        vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 2);
        vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 2);
        vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 2);

        if (k >= 4) {
          const float16x8_t vb01234567 = vld1q_f16(w);

          vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 3);
          vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 3);
          vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 3);
          vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 3);
          vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 3);
          vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 3);
          vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 3);
          vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 3);
        }
      }
    }
  }
  const float16x8_t vscale =
      vld1q_dup_f16((const __fp16*)&clamping_params->scale);
  vacc0x01234567 = vmulq_f16(vacc0x01234567, vscale);
  vacc1x01234567 = vmulq_f16(vacc1x01234567, vscale);
  vacc2x01234567 = vmulq_f16(vacc2x01234567, vscale);
  vacc3x01234567 = vmulq_f16(vacc3x01234567, vscale);
  vacc4x01234567 = vmulq_f16(vacc4x01234567, vscale);
  vacc5x01234567 = vmulq_f16(vacc5x01234567, vscale);
  vacc6x01234567 = vmulq_f16(vacc6x01234567, vscale);
  vacc7x01234567 = vmulq_f16(vacc7x01234567, vscale);

  const float16x8_t vmax = vld1q_dup_f16((const __fp16*)&clamping_params->max);
  vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
  vacc1x01234567 = vminq_f16(vacc1x01234567, vmax);
  vacc2x01234567 = vminq_f16(vacc2x01234567, vmax);
  vacc3x01234567 = vminq_f16(vacc3x01234567, vmax);
  vacc4x01234567 = vminq_f16(vacc4x01234567, vmax);
  vacc5x01234567 = vminq_f16(vacc5x01234567, vmax);
  vacc6x01234567 = vminq_f16(vacc6x01234567, vmax);
  vacc7x01234567 = vminq_f16(vacc7x01234567, vmax);

  const float16x8_t vmin = vld1q_dup_f16((const __fp16*)&clamping_params->min);
  vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
  vacc1x01234567 = vmaxq_f16(vacc1x01234567, vmin);
  vacc2x01234567 = vmaxq_f16(vacc2x01234567, vmin);
  vacc3x01234567 = vmaxq_f16(vacc3x01234567, vmin);
  vacc4x01234567 = vmaxq_f16(vacc4x01234567, vmin);
  vacc5x01234567 = vmaxq_f16(vacc5x01234567, vmin);
  vacc6x01234567 = vmaxq_f16(vacc6x01234567, vmin);
  vacc7x01234567 = vmaxq_f16(vacc7x01234567, vmin);

  __fp16* c0 = c;
  __fp16* c1 = (__fp16*)((uintptr_t)c0 + c_stride);
  if (mr < 2) {
    c1 = c0;
  }
  __fp16* c2 = (__fp16*)((uintptr_t)c1 + c_stride);
  if (mr <= 2) {
    c2 = c1;
  }
  __fp16* c3 = (__fp16*)((uintptr_t)c2 + c_stride);
  if (mr < 4) {
    c3 = c2;
  }
  __fp16* c4 = (__fp16*)((uintptr_t)c3 + c_stride);
  if (mr <= 4) {
    c4 = c3;
  }
  __fp16* c5 = (__fp16*)((uintptr_t)c4 + c_stride);
  if (mr < 6) {
    c5 = c4;
  }
  __fp16* c6 = (__fp16*)((uintptr_t)c5 + c_stride);
  if (mr <= 6) {
    c6 = c5;
  }
  __fp16* c7 = (__fp16*)((uintptr_t)c6 + c_stride);
  if (mr != 8) {
    c7 = c6;
  }
  if (nr == 8) {
    vst1q_f16(c0, vacc0x01234567);
    vst1q_f16(c1, vacc1x01234567);
    vst1q_f16(c2, vacc2x01234567);
    vst1q_f16(c3, vacc3x01234567);
    vst1q_f16(c4, vacc4x01234567);
    vst1q_f16(c5, vacc5x01234567);
    vst1q_f16(c6, vacc6x01234567);
    vst1q_f16(c7, vacc7x01234567);
  } else {
    if (nr & 4) {
      vst1_f16(c0, vget_low_f16(vacc0x01234567));
      c0 += 4;
      vst1_f16(c1, vget_low_f16(vacc1x01234567));
      c1 += 4;
      vst1_f16(c2, vget_low_f16(vacc2x01234567));
      c2 += 4;
      vst1_f16(c3, vget_low_f16(vacc3x01234567));
      c3 += 4;
      vst1_f16(c4, vget_low_f16(vacc4x01234567));
      c4 += 4;
      vst1_f16(c5, vget_low_f16(vacc5x01234567));
      c5 += 4;
      vst1_f16(c6, vget_low_f16(vacc6x01234567));
      c6 += 4;
      vst1_f16(c7, vget_low_f16(vacc7x01234567));
      c7 += 4;
      vacc0x01234567 = vextq_f16(vacc0x01234567, vacc0x01234567, 4);
      vacc1x01234567 = vextq_f16(vacc1x01234567, vacc1x01234567, 4);
      vacc2x01234567 = vextq_f16(vacc2x01234567, vacc2x01234567, 4);
      vacc3x01234567 = vextq_f16(vacc3x01234567, vacc3x01234567, 4);
      vacc4x01234567 = vextq_f16(vacc4x01234567, vacc4x01234567, 4);
      vacc5x01234567 = vextq_f16(vacc5x01234567, vacc5x01234567, 4);
      vacc6x01234567 = vextq_f16(vacc6x01234567, vacc6x01234567, 4);
      vacc7x01234567 = vextq_f16(vacc7x01234567, vacc7x01234567, 4);
    }
    if (nr & 2) {
      vst1_lane_u32(
          __builtin_assume_aligned(c0, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc0x01234567)),
          0);
      c0 += 2;
      vst1_lane_u32(
          __builtin_assume_aligned(c1, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc1x01234567)),
          0);
      c1 += 2;
      vst1_lane_u32(
          __builtin_assume_aligned(c2, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc2x01234567)),
          0);
      c2 += 2;
      vst1_lane_u32(
          __builtin_assume_aligned(c3, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc3x01234567)),
          0);
      c3 += 2;
      vst1_lane_u32(
          __builtin_assume_aligned(c4, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc4x01234567)),
          0);
      c4 += 2;
      vst1_lane_u32(
          __builtin_assume_aligned(c5, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc5x01234567)),
          0);
      c5 += 2;
      vst1_lane_u32(
          __builtin_assume_aligned(c6, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc6x01234567)),
          0);
      c6 += 2;
      vst1_lane_u32(
          __builtin_assume_aligned(c7, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc7x01234567)),
          0);
      c7 += 2;
      vacc0x01234567 = vextq_f16(vacc0x01234567, vacc0x01234567, 2);
      vacc1x01234567 = vextq_f16(vacc1x01234567, vacc1x01234567, 2);
      vacc2x01234567 = vextq_f16(vacc2x01234567, vacc2x01234567, 2);
      vacc3x01234567 = vextq_f16(vacc3x01234567, vacc3x01234567, 2);
      vacc4x01234567 = vextq_f16(vacc4x01234567, vacc4x01234567, 2);
      vacc5x01234567 = vextq_f16(vacc5x01234567, vacc5x01234567, 2);
      vacc6x01234567 = vextq_f16(vacc6x01234567, vacc6x01234567, 2);
      vacc7x01234567 = vextq_f16(vacc7x01234567, vacc7x01234567, 2);
    }
    if (nr & 1) {
      vst1q_lane_f16(c0, vacc0x01234567, 0);
      vst1q_lane_f16(c1, vacc1x01234567, 0);
      vst1q_lane_f16(c2, vacc2x01234567, 0);
      vst1q_lane_f16(c3, vacc3x01234567, 0);
      vst1q_lane_f16(c4, vacc4x01234567, 0);
      vst1q_lane_f16(c5, vacc5x01234567, 0);
      vst1q_lane_f16(c6, vacc6x01234567, 0);
      vst1q_lane_f16(c7, vacc7x01234567, 0);
    }
  }
}
