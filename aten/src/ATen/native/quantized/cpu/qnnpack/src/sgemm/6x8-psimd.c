/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <psimd.h>

#include <qnnpack/sgemm.h>

void pytorch_sgemm_ukernel_6x8__psimd(
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
  psimd_f32 vacc0x0123 = psimd_load_f32(w);
  w += 4;
  psimd_f32 vacc0x4567 = psimd_load_f32(w);
  w += 4;
  psimd_f32 vacc1x0123 = vacc0x0123;
  psimd_f32 vacc1x4567 = vacc0x4567;
  psimd_f32 vacc2x0123 = vacc0x0123;
  psimd_f32 vacc2x4567 = vacc0x4567;
  psimd_f32 vacc3x0123 = vacc0x0123;
  psimd_f32 vacc3x4567 = vacc0x4567;
  psimd_f32 vacc4x0123 = vacc0x0123;
  psimd_f32 vacc4x4567 = vacc0x4567;
  psimd_f32 vacc5x0123 = vacc0x0123;
  psimd_f32 vacc5x4567 = vacc0x4567;

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
  const float* a5 = (const float*)((uintptr_t)a4 + a_stride);
  if (mr != 6) {
    a5 = a4;
  }

  do {
    const psimd_f32 va0 = psimd_splat_f32(*a0);
    a0 += 1;
    const psimd_f32 va1 = psimd_splat_f32(*a1);
    a1 += 1;
    const psimd_f32 va2 = psimd_splat_f32(*a2);
    a2 += 1;
    const psimd_f32 va3 = psimd_splat_f32(*a3);
    a3 += 1;
    const psimd_f32 va4 = psimd_splat_f32(*a4);
    a4 += 1;
    const psimd_f32 va5 = psimd_splat_f32(*a5);
    a5 += 1;

    const psimd_f32 vb0123 = psimd_load_f32(w);
    w += 4;
    const psimd_f32 vb4567 = psimd_load_f32(w);
    w += 4;

    vacc0x0123 += vb0123 * va0;
    vacc0x4567 += vb4567 * va0;
    vacc1x0123 += vb0123 * va1;
    vacc1x4567 += vb4567 * va1;
    vacc2x0123 += vb0123 * va2;
    vacc2x4567 += vb4567 * va2;
    vacc3x0123 += vb0123 * va3;
    vacc3x4567 += vb4567 * va3;
    vacc4x0123 += vb0123 * va4;
    vacc4x4567 += vb4567 * va4;
    vacc5x0123 += vb0123 * va5;
    vacc5x4567 += vb4567 * va5;
  } while (--k != 0);

  const psimd_f32 vmax = psimd_splat_f32(clamping_params->max);
  vacc0x0123 = psimd_min_f32(vacc0x0123, vmax);
  vacc0x4567 = psimd_min_f32(vacc0x4567, vmax);
  vacc1x0123 = psimd_min_f32(vacc1x0123, vmax);
  vacc1x4567 = psimd_min_f32(vacc1x4567, vmax);
  vacc2x0123 = psimd_min_f32(vacc2x0123, vmax);
  vacc2x4567 = psimd_min_f32(vacc2x4567, vmax);
  vacc3x0123 = psimd_min_f32(vacc3x0123, vmax);
  vacc3x4567 = psimd_min_f32(vacc3x4567, vmax);
  vacc4x0123 = psimd_min_f32(vacc4x0123, vmax);
  vacc4x4567 = psimd_min_f32(vacc4x4567, vmax);
  vacc5x0123 = psimd_min_f32(vacc5x0123, vmax);
  vacc5x4567 = psimd_min_f32(vacc5x4567, vmax);

  const psimd_f32 vmin = psimd_splat_f32(clamping_params->min);
  vacc0x0123 = psimd_max_f32(vacc0x0123, vmin);
  vacc0x4567 = psimd_max_f32(vacc0x4567, vmin);
  vacc1x0123 = psimd_max_f32(vacc1x0123, vmin);
  vacc1x4567 = psimd_max_f32(vacc1x4567, vmin);
  vacc2x0123 = psimd_max_f32(vacc2x0123, vmin);
  vacc2x4567 = psimd_max_f32(vacc2x4567, vmin);
  vacc3x0123 = psimd_max_f32(vacc3x0123, vmin);
  vacc3x4567 = psimd_max_f32(vacc3x4567, vmin);
  vacc4x0123 = psimd_max_f32(vacc4x0123, vmin);
  vacc4x4567 = psimd_max_f32(vacc4x4567, vmin);
  vacc5x0123 = psimd_max_f32(vacc5x0123, vmin);
  vacc5x4567 = psimd_max_f32(vacc5x4567, vmin);

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
  float* c5 = (float*)((uintptr_t)c4 + c_stride);
  if (mr != 6) {
    c5 = c4;
  }
  if (nr == 8) {
    psimd_store_f32(c0, vacc0x0123);
    c0 += 4;
    psimd_store_f32(c1, vacc1x0123);
    c1 += 4;
    psimd_store_f32(c2, vacc2x0123);
    c2 += 4;
    psimd_store_f32(c3, vacc3x0123);
    c3 += 4;
    psimd_store_f32(c4, vacc4x0123);
    c4 += 4;
    psimd_store_f32(c5, vacc5x0123);
    c5 += 4;

    psimd_store_f32(c0, vacc0x4567);
    psimd_store_f32(c1, vacc1x4567);
    psimd_store_f32(c2, vacc2x4567);
    psimd_store_f32(c3, vacc3x4567);
    psimd_store_f32(c4, vacc4x4567);
    psimd_store_f32(c5, vacc5x4567);
  } else {
    if (nr >= 4) {
      psimd_store_f32(c0, vacc0x0123);
      c0 += 4;
      psimd_store_f32(c1, vacc1x0123);
      c1 += 4;
      psimd_store_f32(c2, vacc2x0123);
      c2 += 4;
      psimd_store_f32(c3, vacc3x0123);
      c3 += 4;
      psimd_store_f32(c4, vacc4x0123);
      c4 += 4;
      psimd_store_f32(c5, vacc5x0123);
      c5 += 4;
      vacc0x0123 = vacc0x4567;
      vacc1x0123 = vacc1x4567;
      vacc2x0123 = vacc2x4567;
      vacc3x0123 = vacc3x4567;
      vacc4x0123 = vacc4x4567;
      vacc5x0123 = vacc5x4567;
      nr -= 4;
    }
    if (nr >= 2) {
      psimd_store2_f32(c0, vacc0x0123);
      c0 += 2;
      psimd_store2_f32(c1, vacc1x0123);
      c1 += 2;
      psimd_store2_f32(c2, vacc2x0123);
      c2 += 2;
      psimd_store2_f32(c3, vacc3x0123);
      c3 += 2;
      psimd_store2_f32(c4, vacc4x0123);
      c4 += 2;
      psimd_store2_f32(c5, vacc5x0123);
      c5 += 2;
      vacc0x0123 = psimd_concat_hi_f32(vacc0x0123, vacc0x0123);
      vacc1x0123 = psimd_concat_hi_f32(vacc1x0123, vacc1x0123);
      vacc2x0123 = psimd_concat_hi_f32(vacc2x0123, vacc2x0123);
      vacc3x0123 = psimd_concat_hi_f32(vacc3x0123, vacc3x0123);
      vacc4x0123 = psimd_concat_hi_f32(vacc4x0123, vacc4x0123);
      vacc5x0123 = psimd_concat_hi_f32(vacc5x0123, vacc5x0123);
      nr -= 2;
    }
    if (nr != 0) {
      psimd_store1_f32(c0, vacc0x0123);
      psimd_store1_f32(c1, vacc1x0123);
      psimd_store1_f32(c2, vacc2x0123);
      psimd_store1_f32(c3, vacc3x0123);
      psimd_store1_f32(c4, vacc4x0123);
      psimd_store1_f32(c5, vacc5x0123);
    }
  }
}
