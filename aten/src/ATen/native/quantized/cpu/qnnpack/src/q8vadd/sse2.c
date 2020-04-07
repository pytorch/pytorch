/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>

#include <qnnpack/common.h>
#include <qnnpack/q8vadd.h>
#include <qnnpack/scalar-utils.h>

#include <math.h>

void pytorch_q8vadd_ukernel__sse2(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union pytorch_qnnp_add_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  if
    PYTORCH_QNNP_LIKELY(n >= 8) {
      const __m128i a_zero_point = _mm_load_si128(
          (const __m128i*)&quantization_params->sse2.a_zero_point);
      const __m128i b_zero_point = _mm_load_si128(
          (const __m128i*)&quantization_params->sse2.b_zero_point);
      const __m128 va_scale = _mm_load_ps(quantization_params->sse2.a_scale);
      const __m128 vb_scale = _mm_load_ps(quantization_params->sse2.b_scale);

      const __m128i vzero = _mm_setzero_si128();
      do {
        const __m128i va = _mm_loadl_epi64((const __m128i*)a);
        a += 8;
        const __m128i vb = _mm_loadl_epi64((const __m128i*)b);
        b += 8;

        const __m128i vxa = _mm_unpacklo_epi8(va, vzero);
        const __m128i vxb = _mm_unpacklo_epi8(vb, vzero);

        __m128i vxa_lo_32x4 = _mm_unpacklo_epi16(vxa, vzero);
        __m128i vxa_hi_32x4 = _mm_unpackhi_epi16(vxa, vzero);
        __m128i vxb_lo_32x4 = _mm_unpacklo_epi16(vxb, vzero);
        __m128i vxb_hi_32x4 = _mm_unpackhi_epi16(vxb, vzero);

        /*
         * Subtract zero point.
         * Reason for changing algo here is that, otherwise it introduces
         * larger error w.r.t other implementation.
         * Neon for example dequantizes the value and then add.
         * So maintain the same for x86 as well.
         */
        vxa_lo_32x4 = _mm_sub_epi32(vxa_lo_32x4, a_zero_point);
        vxa_hi_32x4 = _mm_sub_epi32(vxa_hi_32x4, a_zero_point);
        vxb_lo_32x4 = _mm_sub_epi32(vxb_lo_32x4, b_zero_point);
        vxb_hi_32x4 = _mm_sub_epi32(vxb_hi_32x4, b_zero_point);

        const __m128 vxa_lo_32x4_f =
          _mm_mul_ps(_mm_cvtepi32_ps(vxa_lo_32x4), va_scale);
        const __m128 vxa_hi_32x4_f =
          _mm_mul_ps(_mm_cvtepi32_ps(vxa_hi_32x4), va_scale);
        const __m128 vxb_lo_32x4_f =
          _mm_mul_ps(_mm_cvtepi32_ps(vxb_lo_32x4), vb_scale);
        const __m128 vxb_hi_32x4_f =
          _mm_mul_ps(_mm_cvtepi32_ps(vxb_hi_32x4), vb_scale);

        const __m128 vacc_lo_f = _mm_add_ps(vxa_lo_32x4_f, vxb_lo_32x4_f);
        const __m128 vacc_hi_f = _mm_add_ps(vxa_hi_32x4_f, vxb_hi_32x4_f);

        __m128i vacc_lo = _mm_cvtps_epi32(vacc_lo_f);
        __m128i vacc_hi = _mm_cvtps_epi32(vacc_hi_f);

        /* Pack, saturate, and add output zero point */
        const __m128i vy_zero_point = _mm_load_si128(
            (const __m128i*)quantization_params->sse2.y_zero_point);
        const __m128i vacc =
            _mm_adds_epi16(_mm_packs_epi32(vacc_lo, vacc_hi), vy_zero_point);
        __m128i vy = _mm_packus_epi16(vacc, vacc);
        vy = _mm_max_epu8(
            vy,
            _mm_load_si128((const __m128i*)quantization_params->sse2.y_min));
        vy = _mm_min_epu8(
            vy,
            _mm_load_si128((const __m128i*)quantization_params->sse2.y_max));

        _mm_storel_epi64((__m128i*)y, vy);
        y += 8;

        n -= 8;
      } while (n >= 8);
      if (n != 0) {
        const size_t n_decrement = 8 - n;
        const __m128i vload_shift = _mm_cvtsi32_si128(8 * (int32_t)n_decrement);

        const __m128i va = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(a - n_decrement)), vload_shift);
        const __m128i vb = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(b - n_decrement)), vload_shift);

        const __m128i vxa = _mm_unpacklo_epi8(va, vzero);
        const __m128i vxb = _mm_unpacklo_epi8(vb, vzero);

        __m128i vxa_lo_32x4 = _mm_unpacklo_epi16(vxa, vzero);
        __m128i vxa_hi_32x4 = _mm_unpackhi_epi16(vxa, vzero);
        __m128i vxb_lo_32x4 = _mm_unpacklo_epi16(vxb, vzero);
        __m128i vxb_hi_32x4 = _mm_unpackhi_epi16(vxb, vzero);

        vxa_lo_32x4 = _mm_sub_epi32(vxa_lo_32x4, a_zero_point);
        vxa_hi_32x4 = _mm_sub_epi32(vxa_hi_32x4, a_zero_point);
        vxb_lo_32x4 = _mm_sub_epi32(vxb_lo_32x4, b_zero_point);
        vxb_hi_32x4 = _mm_sub_epi32(vxb_hi_32x4, b_zero_point);

        const __m128 vxa_lo_32x4_f =
          _mm_mul_ps(_mm_cvtepi32_ps(vxa_lo_32x4), va_scale);
        const __m128 vxa_hi_32x4_f =
          _mm_mul_ps(_mm_cvtepi32_ps(vxa_hi_32x4), va_scale);
        const __m128 vxb_lo_32x4_f =
          _mm_mul_ps(_mm_cvtepi32_ps(vxb_lo_32x4), vb_scale);
        const __m128 vxb_hi_32x4_f =
          _mm_mul_ps(_mm_cvtepi32_ps(vxb_hi_32x4), vb_scale);

        const __m128 vacc_lo_f = _mm_add_ps(vxa_lo_32x4_f, vxb_lo_32x4_f);
        const __m128 vacc_hi_f = _mm_add_ps(vxa_hi_32x4_f, vxb_hi_32x4_f);

        __m128i vacc_lo = _mm_cvtps_epi32(vacc_lo_f);
        __m128i vacc_hi = _mm_cvtps_epi32(vacc_hi_f);

        /* Pack, saturate, and add output zero point */
        const __m128i vy_zero_point = _mm_load_si128(
            (const __m128i*)quantization_params->sse2.y_zero_point);
        const __m128i vacc =
            _mm_adds_epi16(_mm_packs_epi32(vacc_lo, vacc_hi), vy_zero_point);
        __m128i vy = _mm_packus_epi16(vacc, vacc);
        vy = _mm_max_epu8(
            vy,
            _mm_load_si128((const __m128i*)quantization_params->sse2.y_min));
        vy = _mm_min_epu8(
            vy,
            _mm_load_si128((const __m128i*)quantization_params->sse2.y_max));

        if (n & 4) {
          *((uint32_t*)y) = (uint32_t)_mm_cvtsi128_si32(vy);
          vy = _mm_shuffle_epi32(vy, _MM_SHUFFLE(3, 2, 1, 1));
          y += 4;
        }
        if (n & 2) {
          *((uint16_t*)y) = (uint16_t)_mm_extract_epi16(vy, 0);
          vy = _mm_srli_epi32(vy, 16);
          y += 2;
        }
        if (n & 1) {
          *((uint8_t*)y) = (uint8_t)_mm_cvtsi128_si32(vy);
        }
      }
    }
  else {
    const int32_t a_zero_point = quantization_params->sse2.a_zero_point[0];
    const int32_t b_zero_point = quantization_params->sse2.b_zero_point[0];
    const float va_multiplier = quantization_params->sse2.a_multiplier;
    const float vb_multiplier = quantization_params->sse2.b_multiplier;
    const int32_t vy_zero_point =
        (int32_t)quantization_params->sse2.y_zero_point[0];
    const int32_t vy_max =
        (int32_t)(uint32_t)quantization_params->sse2.y_max[0];
    const int32_t vy_min =
        (int32_t)(uint32_t)quantization_params->sse2.y_min[0];

    while (n-- != 0) {
      int32_t vxa = (int32_t)*a++;
      int32_t vxb = (int32_t)*b++;
      vxa = vxa - a_zero_point;
      vxb = vxb - b_zero_point;

      /* Multiply by factors and accumulate products */
      int32_t vacc = lrintf(((float)vxa * va_multiplier) + ((float)vxb * vb_multiplier));

      /* Clamp and add output zero point */
      int32_t vy = vacc + vy_zero_point;
      vy = vy >= vy_min ? vy : vy_min;
      vy = vy <= vy_max ? vy : vy_max;

      *y++ = (uint8_t)vy;
    }
  }
}
