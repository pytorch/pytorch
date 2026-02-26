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

void pytorch_q8vadd_ukernel__sse2(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union pytorch_qnnp_add_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  if
    PYTORCH_QNNP_LIKELY(n >= 8) {
      const __m128i vzero_point_product = _mm_load_si128(
          (const __m128i*)&quantization_params->sse2.zero_point_product);
      const __m128i va_multiplier_lo = _mm_load_si128(
          (const __m128i*)&quantization_params->sse2.a_multiplier_lo);
      const __m128i va_multiplier_hi = _mm_load_si128(
          (const __m128i*)&quantization_params->sse2.a_multiplier_hi);
      const __m128i vb_multiplier_lo = _mm_load_si128(
          (const __m128i*)&quantization_params->sse2.b_multiplier_lo);
      const __m128i vb_multiplier_hi = _mm_load_si128(
          (const __m128i*)&quantization_params->sse2.b_multiplier_hi);
      const __m128i vremainder_mask = _mm_load_si128(
          (const __m128i*)quantization_params->sse2.remainder_mask);
      const __m128i vremainder_threshold = _mm_load_si128(
          (const __m128i*)quantization_params->sse2.remainder_threshold);
      const __m128i vshift =
          _mm_cvtsi32_si128((int)quantization_params->sse2.shift);

      const __m128i vzero = _mm_setzero_si128();
      do {
        const __m128i va = _mm_loadl_epi64((const __m128i*)a);
        a += 8;
        const __m128i vb = _mm_loadl_epi64((const __m128i*)b);
        b += 8;

        const __m128i vxa = _mm_unpacklo_epi8(va, vzero);
        const __m128i vxb = _mm_unpacklo_epi8(vb, vzero);

        /* Multiply by factors */
        const __m128i va_product_lo = _mm_mullo_epi16(vxa, va_multiplier_lo);
        const __m128i va_product_hi = _mm_add_epi16(
            _mm_mulhi_epu16(vxa, va_multiplier_lo),
            _mm_mullo_epi16(vxa, va_multiplier_hi));

        const __m128i vb_product_lo = _mm_mullo_epi16(vxb, vb_multiplier_lo);
        const __m128i vb_product_hi = _mm_add_epi16(
            _mm_mulhi_epu16(vxb, vb_multiplier_lo),
            _mm_mullo_epi16(vxb, vb_multiplier_hi));

        /* Accumulate products */
        __m128i vacc_lo = _mm_add_epi32(
            vzero_point_product,
            _mm_unpacklo_epi16(va_product_lo, va_product_hi));
        __m128i vacc_hi = _mm_add_epi32(
            vzero_point_product,
            _mm_unpackhi_epi16(va_product_lo, va_product_hi));

        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vb_product_lo, vb_product_hi));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vb_product_lo, vb_product_hi));

        /* Shift right and round */
        const __m128i vrem_lo = _mm_add_epi32(
            _mm_and_si128(vacc_lo, vremainder_mask),
            _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo));
        const __m128i vrem_hi = _mm_add_epi32(
            _mm_and_si128(vacc_hi, vremainder_mask),
            _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi));

        vacc_lo = _mm_sub_epi32(
            _mm_sra_epi32(vacc_lo, vshift),
            _mm_cmpgt_epi32(vrem_lo, vremainder_threshold));
        vacc_hi = _mm_sub_epi32(
            _mm_sra_epi32(vacc_hi, vshift),
            _mm_cmpgt_epi32(vrem_hi, vremainder_threshold));

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

        /* Multiply by factors */
        const __m128i va_product_lo = _mm_mullo_epi16(vxa, va_multiplier_lo);
        const __m128i va_product_hi = _mm_add_epi16(
            _mm_mulhi_epu16(vxa, va_multiplier_lo),
            _mm_mullo_epi16(vxa, va_multiplier_hi));

        const __m128i vb_product_lo = _mm_mullo_epi16(vxb, vb_multiplier_lo);
        const __m128i vb_product_hi = _mm_add_epi16(
            _mm_mulhi_epu16(vxb, vb_multiplier_lo),
            _mm_mullo_epi16(vxb, vb_multiplier_hi));

        /* Accumulate products */
        __m128i vacc_lo = _mm_add_epi32(
            vzero_point_product,
            _mm_unpacklo_epi16(va_product_lo, va_product_hi));
        __m128i vacc_hi = _mm_add_epi32(
            vzero_point_product,
            _mm_unpackhi_epi16(va_product_lo, va_product_hi));

        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vb_product_lo, vb_product_hi));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vb_product_lo, vb_product_hi));

        /* Shift right and round */
        const __m128i vrem_lo = _mm_add_epi32(
            _mm_and_si128(vacc_lo, vremainder_mask),
            _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo));
        const __m128i vrem_hi = _mm_add_epi32(
            _mm_and_si128(vacc_hi, vremainder_mask),
            _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi));

        vacc_lo = _mm_sub_epi32(
            _mm_sra_epi32(vacc_lo, vshift),
            _mm_cmpgt_epi32(vrem_lo, vremainder_threshold));
        vacc_hi = _mm_sub_epi32(
            _mm_sra_epi32(vacc_hi, vshift),
            _mm_cmpgt_epi32(vrem_hi, vremainder_threshold));

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
    const int32_t vzero_point_product =
        quantization_params->sse2.zero_point_product[0];
    const uint32_t va_multiplier = quantization_params->sse2.a_multiplier;
    const uint32_t vb_multiplier = quantization_params->sse2.b_multiplier;
    const int32_t vremainder_mask = quantization_params->sse2.remainder_mask[0];
    const int32_t vremainder_threshold =
        quantization_params->sse2.remainder_threshold[0];
    const uint32_t vshift = quantization_params->sse2.shift;
    const int32_t vy_zero_point =
        (int32_t)quantization_params->sse2.y_zero_point[0];
    const int32_t vy_max =
        (int32_t)(uint32_t)quantization_params->sse2.y_max[0];
    const int32_t vy_min =
        (int32_t)(uint32_t)quantization_params->sse2.y_min[0];

    while (n-- != 0) {
      const uint32_t vxa = (uint32_t)*a++;
      const uint32_t vxb = (uint32_t)*b++;

      /* Multiply by factors and accumulate products */
      int32_t vacc = vzero_point_product + (int32_t)(vxa * va_multiplier) +
          (int32_t)(vxb * vb_multiplier);

      /* Shift right and round */
      const int32_t vrem = (vacc & vremainder_mask) - (int32_t)(vacc < 0);

      vacc = asr_s32(vacc, vshift) + (int32_t)(vrem > vremainder_threshold);

      /* Clamp and add output zero point */
      int32_t vy = vacc + vy_zero_point;
      vy = vy >= vy_min ? vy : vy_min;
      vy = vy <= vy_max ? vy : vy_max;

      *y++ = (uint8_t)vy;
    }
  }
}
