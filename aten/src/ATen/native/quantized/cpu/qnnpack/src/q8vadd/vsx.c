/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <altivec.h>

#include <qnnpack/common.h>
#include <qnnpack/q8vadd.h>
#include <qnnpack/scalar-utils.h>

void pytorch_q8vadd_ukernel__vsx(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union pytorch_qnnp_add_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  if
    PYTORCH_QNNP_LIKELY(n >= 16) {
      const vector int vzero_point_product =
          vec_splats(quantization_params->vsx.zero_point_product);
      const vector int vremainder_mask =
          vec_splats(quantization_params->vsx.remainder_mask);
      const vector int vremainder_threshold =
          vec_splats(quantization_params->vsx.remainder_threshold);
      const vector short vy_zero_point =
          vec_splats(quantization_params->vsx.y_zero_point);
      const vector unsigned int va_multiplier =
          vec_splats(quantization_params->vsx.a_multiplier);
      const vector unsigned int vb_multiplier =
          vec_splats(quantization_params->vsx.b_multiplier);
      const vector unsigned char vy_max =
          vec_splats(quantization_params->vsx.y_max);
      const vector unsigned char vy_min =
          vec_splats(quantization_params->vsx.y_min);
      const vector unsigned int vshift =
          vec_splats(quantization_params->vsx.shift);
      const vector unsigned char vzero =
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      for (; n >= 16; n -= 16) {
        const vector unsigned char va = vec_xl(0, a);
        a += 16;
        const vector unsigned char vb = vec_xl(0, b);
        b += 16;

        const vector unsigned short vxa0 =
            (vector unsigned short)vec_mergeh(va, vzero);
        const vector unsigned short vxb0 =
            (vector unsigned short)vec_mergeh(vb, vzero);
        const vector unsigned short vxa1 =
            (vector unsigned short)vec_mergel(va, vzero);
        const vector unsigned short vxb1 =
            (vector unsigned short)vec_mergel(vb, vzero);

        // Multiply by factors and accumulate products
        vector int vacc0_hi = (vector int)vec_mul(
            (vector unsigned int)vec_mergeh(vxa0, (vector unsigned short)vzero),
            va_multiplier);
        vector int vacc1_hi = (vector int)vec_mul(
            (vector unsigned int)vec_mergeh(vxa1, (vector unsigned short)vzero),
            va_multiplier);
        vector int vacc0_lo = (vector int)vec_mul(
            (vector unsigned int)vec_mergel(vxa0, (vector unsigned short)vzero),
            va_multiplier);
        vector int vacc1_lo = (vector int)vec_mul(
            (vector unsigned int)vec_mergel(vxa1, (vector unsigned short)vzero),
            va_multiplier);

        vacc0_hi = vec_add(vacc0_hi, (vector int)vec_mul(
            (vector unsigned int)vec_mergeh(vxb0, (vector unsigned short)vzero),
            vb_multiplier));
        vacc1_hi = vec_add(vacc1_hi, (vector int)vec_mul(
            (vector unsigned int)vec_mergeh(vxb1, (vector unsigned short)vzero),
            vb_multiplier));
        vacc0_lo = vec_add(vacc0_lo, (vector int)vec_mul(
            (vector unsigned int)vec_mergel(vxb0, (vector unsigned short)vzero),
            vb_multiplier));
        vacc1_lo = vec_add(vacc1_lo, (vector int)vec_mul(
            (vector unsigned int)vec_mergel(vxb1, (vector unsigned short)vzero),
            vb_multiplier));

        vacc0_hi = vec_add(vacc0_hi, vzero_point_product);
        vacc1_hi = vec_add(vacc1_hi, vzero_point_product);
        vacc0_lo = vec_add(vacc0_lo, vzero_point_product);
        vacc1_lo = vec_add(vacc1_lo, vzero_point_product);

        // Shift right and round
        const vector int vrem0_hi = vec_add(
            vec_and(vacc0_hi, vremainder_mask),
            vec_cmpgt((vector int)vzero, vacc0_hi));
        const vector int vrem1_hi = vec_add(
            vec_and(vacc1_hi, vremainder_mask),
            vec_cmpgt((vector int)vzero, vacc1_hi));
        const vector int vrem0_lo = vec_add(
            vec_and(vacc0_lo, vremainder_mask),
            vec_cmpgt((vector int)vzero, vacc0_lo));
        const vector int vrem1_lo = vec_add(
            vec_and(vacc1_lo, vremainder_mask),
            vec_cmpgt((vector int)vzero, vacc1_lo));

        vacc0_hi = vec_sub(
            vec_sra(vacc0_hi, vshift),
            vec_cmpgt(vrem0_hi, vremainder_threshold));
        vacc1_hi = vec_sub(
            vec_sra(vacc1_hi, vshift),
            vec_cmpgt(vrem1_hi, vremainder_threshold));
        vacc0_lo = vec_sub(
            vec_sra(vacc0_lo, vshift),
            vec_cmpgt(vrem0_lo, vremainder_threshold));
        vacc1_lo = vec_sub(
            vec_sra(vacc1_lo, vshift),
            vec_cmpgt(vrem1_lo, vremainder_threshold));

        // Pack, saturate, and add output zero point
        const vector short vacc0 =
            vec_add(vec_packs(vacc0_hi, vacc0_lo), vy_zero_point);
        const vector short vacc1 =
            vec_add(vec_packs(vacc1_hi, vacc1_lo), vy_zero_point);
        vector unsigned char vy = vec_packsu(vacc0, vacc1);
        vy = vec_max(vy, vy_min);
        vy = vec_min(vy, vy_max);

        vec_xst(vy, 0, y);
        y += 16;
      }
      if (n != 0) {
        const size_t n_decrement = 16 - n;
        const vector unsigned char vload_shift = {
          8 * n_decrement, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const vector unsigned char va =
          vec_sro(vec_xl(-n_decrement, a), vload_shift);
        const vector unsigned char vb =
          vec_sro(vec_xl(-n_decrement, b), vload_shift);

        const vector unsigned short vxa0 =
            (vector unsigned short)vec_mergeh(va, vzero);
        const vector unsigned short vxb0 =
            (vector unsigned short)vec_mergeh(vb, vzero);

        // Multiply by factors and accumulate products
        vector int vacc0_hi = (vector int)vec_mul(
            (vector unsigned int)vec_mergeh(vxa0, (vector unsigned short)vzero),
            va_multiplier);
        vector int vacc0_lo = (vector int)vec_mul(
            (vector unsigned int)vec_mergel(vxa0, (vector unsigned short)vzero),
            va_multiplier);

        vacc0_hi = vec_add(vacc0_hi, (vector int)vec_mul(
            (vector unsigned int)vec_mergeh(vxb0, (vector unsigned short)vzero),
            vb_multiplier));
        vacc0_lo = vec_add(vacc0_lo, (vector int)vec_mul(
            (vector unsigned int)vec_mergel(vxb0, (vector unsigned short)vzero),
            vb_multiplier));

        vacc0_hi = vec_add(vacc0_hi, vzero_point_product);
        vacc0_lo = vec_add(vacc0_lo, vzero_point_product);

        // Shift right and round
        const vector int vrem0_hi = vec_add(
            vec_and(vacc0_hi, vremainder_mask),
            vec_cmpgt((vector int)vzero, vacc0_hi));
        const vector int vrem0_lo = vec_add(
            vec_and(vacc0_lo, vremainder_mask),
            vec_cmpgt((vector int)vzero, vacc0_lo));

        vacc0_hi = vec_sub(
            vec_sra(vacc0_hi, vshift),
            vec_cmpgt(vrem0_hi, vremainder_threshold));
        vacc0_lo = vec_sub(
            vec_sra(vacc0_lo, vshift),
            vec_cmpgt(vrem0_lo, vremainder_threshold));

        // Pack, saturate, and add output zero point
        const vector short vacc0 =
            vec_add(vec_packs(vacc0_hi, vacc0_lo), vy_zero_point);

        vector unsigned char vy;
        if (n > 8) {
          const vector unsigned short vxa1 =
            (vector unsigned short)vec_mergel(va, vzero);
          const vector unsigned short vxb1 =
            (vector unsigned short)vec_mergel(vb, vzero);

          // Multiply by factors and accumulate products
          vector int vacc1_hi = (vector int)vec_mul(
              (vector unsigned int)vec_mergeh(
                  vxa1, (vector unsigned short)vzero),
              va_multiplier);
          vector int vacc1_lo = (vector int)vec_mul(
              (vector unsigned int)vec_mergel(
                  vxa1, (vector unsigned short)vzero),
              va_multiplier);

          vacc1_hi = vec_add(vacc1_hi, (vector int)vec_mul(
              (vector unsigned int)vec_mergeh(vxb1,
              (vector unsigned short)vzero), vb_multiplier));
          vacc1_lo = vec_add(vacc1_lo, (vector int)vec_mul(
              (vector unsigned int)vec_mergel(vxb1,
              (vector unsigned short)vzero), vb_multiplier));

          vacc1_hi = vec_add(vacc1_hi, vzero_point_product);
          vacc1_lo = vec_add(vacc1_lo, vzero_point_product);

          // Shift right and round
          const vector int vrem1_hi = vec_add(
              vec_and(vacc1_hi, vremainder_mask),
              vec_cmpgt((vector int)vzero, vacc1_hi));
          const vector int vrem1_lo = vec_add(
              vec_and(vacc1_lo, vremainder_mask),
              vec_cmpgt((vector int)vzero, vacc1_lo));

          vacc1_hi = vec_sub(
              vec_sra(vacc1_hi, vshift),
              vec_cmpgt(vrem1_hi, vremainder_threshold));
          vacc1_lo = vec_sub(
              vec_sra(vacc1_lo, vshift),
              vec_cmpgt(vrem1_lo, vremainder_threshold));

          // Pack, saturate, and add output zero point
          const vector short vacc1 =
            vec_add(vec_packs(vacc1_hi, vacc1_lo), vy_zero_point);
          vy = vec_packsu(vacc0, vacc1);
        } else {
          vy = vec_packsu(vacc0, vacc0);
        }

        vy = vec_max(vy, vy_min);
        vy = vec_min(vy, vy_max);

        if (n & 8) {
          *(uint64_t*)y = ((vector unsigned long long)vy)[0];
          y += 8;
          const vector unsigned char vshift_8bytes = {
              8 * 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
          vy = vec_sro(vy, vshift_8bytes);
        }
        if (n & 4) {
          *(uint32_t*)y = ((vector unsigned int)vy)[0];
          y += 4;
          const vector unsigned char vshift_4bytes = {
            8 * 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
          vy = vec_sro(vy, vshift_4bytes);
        }
        if (n & 2) {
          *(uint16_t*)y = ((vector unsigned short)vy)[0];
          y += 2;
          const vector unsigned char vshift_2bytes = {
            8 * 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
          vy = vec_sro(vy, vshift_2bytes);
        }
        if (n & 1) {
          y[0] = vy[0];
        }
      }
    }
  else {
    const int32_t vzero_point_product =
        quantization_params->vsx.zero_point_product;
    const uint32_t va_multiplier = quantization_params->vsx.a_multiplier;
    const uint32_t vb_multiplier = quantization_params->vsx.b_multiplier;
    const int32_t vremainder_mask = quantization_params->vsx.remainder_mask;
    const int32_t vremainder_threshold =
        quantization_params->vsx.remainder_threshold;
    const uint32_t vshift = quantization_params->vsx.shift;
    const int32_t vy_zero_point =
        (int32_t)quantization_params->vsx.y_zero_point;
    const int32_t vy_max =
        (int32_t)(uint32_t)quantization_params->vsx.y_max;
    const int32_t vy_min =
        (int32_t)(uint32_t)quantization_params->vsx.y_min;

    while (n-- != 0) {
      const uint32_t vxa = (uint32_t)*a++;
      const uint32_t vxb = (uint32_t)*b++;

      // Multiply by factors and accumulate products
      int32_t vacc = vzero_point_product + (int32_t)(vxa * va_multiplier) +
          (int32_t)(vxb * vb_multiplier);

      // Shift right and round
      const int32_t vrem = (vacc & vremainder_mask) - (int32_t)(vacc < 0);

      vacc = asr_s32(vacc, vshift) + (int32_t)(vrem > vremainder_threshold);

      // Clamp and add output zero point
      int32_t vy = vacc + vy_zero_point;
      vy = vy >= vy_min ? vy : vy_min;
      vy = vy <= vy_max ? vy : vy_max;

      *y++ = (uint8_t)vy;
    }
  }
}
