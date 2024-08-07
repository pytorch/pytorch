/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/sgemm.h>

void pytorch_sgemm_ukernel_5x8__neon(
    size_t mr,
    size_t nr,
    size_t k,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t c_stride,
    const struct pytorch_qnnp_fp32_clamping_params
        clamping_params[restrict static 1]) {
  float32x4_t vacc0x0123 = vld1q_f32(w);
  w += 4;
  float32x4_t vacc0x4567 = vld1q_f32(w);
  w += 4;
  float32x4_t vacc1x0123 = vacc0x0123;
  float32x4_t vacc1x4567 = vacc0x4567;
  float32x4_t vacc2x0123 = vacc0x0123;
  float32x4_t vacc2x4567 = vacc0x4567;
  float32x4_t vacc3x0123 = vacc0x0123;
  float32x4_t vacc3x4567 = vacc0x4567;
  float32x4_t vacc4x0123 = vacc0x0123;
  float32x4_t vacc4x4567 = vacc0x4567;

  const float* a0 = a;
  const float* a1 = (const float*)((uintptr_t)a0 + a_stride);
  if (mr < 2) {
    a1 = a0;
  }
  const float* a2 = (const float*)((uintptr_t)a1 + a_stride);
  if (mr <= 2) {
    a2 = a1;
  }
  const float* a3 = (const float*)((uintptr_t)a2 + a_stride);
  if (mr < 4) {
    a3 = a2;
  }
  const float* a4 = (const float*)((uintptr_t)a3 + a_stride);
  if (mr <= 4) {
    a4 = a3;
  }

  for (; k >= 2; k -= 2) {
    const float32x2_t va0 = vld1_f32(a0);
    a0 += 2;
    const float32x2_t va1 = vld1_f32(a1);
    a1 += 2;
    const float32x2_t va2 = vld1_f32(a2);
    a2 += 2;
    const float32x2_t va3 = vld1_f32(a3);
    a3 += 2;
    const float32x2_t va4 = vld1_f32(a4);
    a4 += 2;

    {
      const float32x4_t vb0123 = vld1q_f32(w);
      w += 4;
      const float32x4_t vb4567 = vld1q_f32(w);
      w += 4;

#if defined(__aarch64__)
      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123, va0, 0);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567, va0, 0);
      vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123, va1, 0);
      vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567, va1, 0);
      vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123, va2, 0);
      vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567, va2, 0);
      vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123, va3, 0);
      vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567, va3, 0);
      vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123, va4, 0);
      vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567, va4, 0);
#else
      vacc0x0123 = vmlaq_lane_f32(vacc0x0123, vb0123, va0, 0);
      vacc0x4567 = vmlaq_lane_f32(vacc0x4567, vb4567, va0, 0);
      vacc1x0123 = vmlaq_lane_f32(vacc1x0123, vb0123, va1, 0);
      vacc1x4567 = vmlaq_lane_f32(vacc1x4567, vb4567, va1, 0);
      vacc2x0123 = vmlaq_lane_f32(vacc2x0123, vb0123, va2, 0);
      vacc2x4567 = vmlaq_lane_f32(vacc2x4567, vb4567, va2, 0);
      vacc3x0123 = vmlaq_lane_f32(vacc3x0123, vb0123, va3, 0);
      vacc3x4567 = vmlaq_lane_f32(vacc3x4567, vb4567, va3, 0);
      vacc4x0123 = vmlaq_lane_f32(vacc4x0123, vb0123, va4, 0);
      vacc4x4567 = vmlaq_lane_f32(vacc4x4567, vb4567, va4, 0);
#endif
    }

    {
      const float32x4_t vb0123 = vld1q_f32(w);
      w += 4;
      const float32x4_t vb4567 = vld1q_f32(w);
      w += 4;

#if defined(__aarch64__)
      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123, va0, 1);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567, va0, 1);
      vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123, va1, 1);
      vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567, va1, 1);
      vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123, va2, 1);
      vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567, va2, 1);
      vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123, va3, 1);
      vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567, va3, 1);
      vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123, va4, 1);
      vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567, va4, 1);
#else
      vacc0x0123 = vmlaq_lane_f32(vacc0x0123, vb0123, va0, 1);
      vacc0x4567 = vmlaq_lane_f32(vacc0x4567, vb4567, va0, 1);
      vacc1x0123 = vmlaq_lane_f32(vacc1x0123, vb0123, va1, 1);
      vacc1x4567 = vmlaq_lane_f32(vacc1x4567, vb4567, va1, 1);
      vacc2x0123 = vmlaq_lane_f32(vacc2x0123, vb0123, va2, 1);
      vacc2x4567 = vmlaq_lane_f32(vacc2x4567, vb4567, va2, 1);
      vacc3x0123 = vmlaq_lane_f32(vacc3x0123, vb0123, va3, 1);
      vacc3x4567 = vmlaq_lane_f32(vacc3x4567, vb4567, va3, 1);
      vacc4x0123 = vmlaq_lane_f32(vacc4x0123, vb0123, va4, 1);
      vacc4x4567 = vmlaq_lane_f32(vacc4x4567, vb4567, va4, 1);
#endif
    }
  }
  if (k != 0) {
    const float32x4_t va0 = vld1q_dup_f32(a0);
    const float32x4_t va1 = vld1q_dup_f32(a1);
    const float32x4_t va2 = vld1q_dup_f32(a2);
    const float32x4_t va3 = vld1q_dup_f32(a3);
    const float32x4_t va4 = vld1q_dup_f32(a4);

    const float32x4_t vb0123 = vld1q_f32(w);
    w += 4;
    const float32x4_t vb4567 = vld1q_f32(w);
    w += 4;

#if defined(__aarch64__)
    vacc0x0123 = vfmaq_f32(vacc0x0123, vb0123, va0);
    vacc0x4567 = vfmaq_f32(vacc0x4567, vb4567, va0);
    vacc1x0123 = vfmaq_f32(vacc1x0123, vb0123, va1);
    vacc1x4567 = vfmaq_f32(vacc1x4567, vb4567, va1);
    vacc2x0123 = vfmaq_f32(vacc2x0123, vb0123, va2);
    vacc2x4567 = vfmaq_f32(vacc2x4567, vb4567, va2);
    vacc3x0123 = vfmaq_f32(vacc3x0123, vb0123, va3);
    vacc3x4567 = vfmaq_f32(vacc3x4567, vb4567, va3);
    vacc4x0123 = vfmaq_f32(vacc4x0123, vb0123, va4);
    vacc4x4567 = vfmaq_f32(vacc4x4567, vb4567, va4);
#else
    vacc0x0123 = vmlaq_f32(vacc0x0123, vb0123, va0);
    vacc0x4567 = vmlaq_f32(vacc0x4567, vb4567, va0);
    vacc1x0123 = vmlaq_f32(vacc1x0123, vb0123, va1);
    vacc1x4567 = vmlaq_f32(vacc1x4567, vb4567, va1);
    vacc2x0123 = vmlaq_f32(vacc2x0123, vb0123, va2);
    vacc2x4567 = vmlaq_f32(vacc2x4567, vb4567, va2);
    vacc3x0123 = vmlaq_f32(vacc3x0123, vb0123, va3);
    vacc3x4567 = vmlaq_f32(vacc3x4567, vb4567, va3);
    vacc4x0123 = vmlaq_f32(vacc4x0123, vb0123, va4);
    vacc4x4567 = vmlaq_f32(vacc4x4567, vb4567, va4);
#endif
  }
  const float32x4_t vmax = vld1q_dup_f32(&clamping_params->max);
  vacc0x0123 = vminq_f32(vacc0x0123, vmax);
  vacc0x4567 = vminq_f32(vacc0x4567, vmax);
  vacc1x0123 = vminq_f32(vacc1x0123, vmax);
  vacc1x4567 = vminq_f32(vacc1x4567, vmax);
  vacc2x0123 = vminq_f32(vacc2x0123, vmax);
  vacc2x4567 = vminq_f32(vacc2x4567, vmax);
  vacc3x0123 = vminq_f32(vacc3x0123, vmax);
  vacc3x4567 = vminq_f32(vacc3x4567, vmax);
  vacc4x0123 = vminq_f32(vacc4x0123, vmax);
  vacc4x4567 = vminq_f32(vacc4x4567, vmax);

  const float32x4_t vmin = vld1q_dup_f32(&clamping_params->min);
  vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
  vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);
  vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
  vacc1x4567 = vmaxq_f32(vacc1x4567, vmin);
  vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
  vacc2x4567 = vmaxq_f32(vacc2x4567, vmin);
  vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);
  vacc3x4567 = vmaxq_f32(vacc3x4567, vmin);
  vacc4x0123 = vmaxq_f32(vacc4x0123, vmin);
  vacc4x4567 = vmaxq_f32(vacc4x4567, vmin);

  float* c0 = c;
  float* c1 = (float*)((uintptr_t)c0 + c_stride);
  if (mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*)((uintptr_t)c1 + c_stride);
  if (mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*)((uintptr_t)c2 + c_stride);
  if (mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*)((uintptr_t)c3 + c_stride);
  if (mr <= 4) {
    c4 = c3;
  }
  if (nr == 8) {
    vst1q_f32(c0, vacc0x0123);
    c0 += 4;
    vst1q_f32(c1, vacc1x0123);
    c1 += 4;
    vst1q_f32(c2, vacc2x0123);
    c2 += 4;
    vst1q_f32(c3, vacc3x0123);
    c3 += 4;
    vst1q_f32(c4, vacc4x0123);
    c4 += 4;

    vst1q_f32(c0, vacc0x4567);
    vst1q_f32(c1, vacc1x4567);
    vst1q_f32(c2, vacc2x4567);
    vst1q_f32(c3, vacc3x4567);
    vst1q_f32(c4, vacc4x4567);
  } else {
    if (nr >= 4) {
      vst1q_f32(c0, vacc0x0123);
      c0 += 4;
      vst1q_f32(c1, vacc1x0123);
      c1 += 4;
      vst1q_f32(c2, vacc2x0123);
      c2 += 4;
      vst1q_f32(c3, vacc3x0123);
      c3 += 4;
      vst1q_f32(c4, vacc4x0123);
      c4 += 4;
      vacc0x0123 = vacc0x4567;
      vacc1x0123 = vacc1x4567;
      vacc2x0123 = vacc2x4567;
      vacc3x0123 = vacc3x4567;
      vacc4x0123 = vacc4x4567;
      nr -= 4;
    }
    if (nr >= 2) {
      vst1_f32(c0, vget_low_f32(vacc0x0123));
      c0 += 2;
      vst1_f32(c1, vget_low_f32(vacc1x0123));
      c1 += 2;
      vst1_f32(c2, vget_low_f32(vacc2x0123));
      c2 += 2;
      vst1_f32(c3, vget_low_f32(vacc3x0123));
      c3 += 2;
      vst1_f32(c4, vget_low_f32(vacc4x0123));
      c4 += 2;
      vacc0x0123 = vextq_f32(vacc0x0123, vacc0x0123, 2);
      vacc1x0123 = vextq_f32(vacc1x0123, vacc1x0123, 2);
      vacc2x0123 = vextq_f32(vacc2x0123, vacc2x0123, 2);
      vacc3x0123 = vextq_f32(vacc3x0123, vacc3x0123, 2);
      vacc4x0123 = vextq_f32(vacc4x0123, vacc4x0123, 2);
      nr -= 2;
    }
    if (nr != 0) {
      vst1q_lane_f32(c0, vacc0x0123, 0);
      vst1q_lane_f32(c1, vacc1x0123, 0);
      vst1q_lane_f32(c2, vacc2x0123, 0);
      vst1q_lane_f32(c3, vacc3x0123, 0);
      vst1q_lane_f32(c4, vacc4x0123, 0);
    }
  }
}
