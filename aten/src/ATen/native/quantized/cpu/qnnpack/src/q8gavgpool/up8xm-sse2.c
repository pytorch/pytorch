/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <emmintrin.h>

#include <qnnpack/q8gavgpool.h>

void pytorch_q8gavgpool_ukernel_up8xm__sse2(
    size_t m,
    size_t n,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  assert(m >= 1);
  assert(n < 8);

  const __m128i vbias =
      _mm_loadu_si128((const __m128i*)&quantization_params->sse2.bias);
  __m128i vacc_lo = vbias;
  __m128i vacc_hi = vbias;
  __m128i vzero = _mm_setzero_si128();
  while (m >= 8) {
    const __m128i vinput = _mm_loadl_epi64((const __m128i*)input);
    const __m128i vxinput = _mm_unpacklo_epi8(vinput, vzero);
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi8(vxinput, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi8(vxinput, vzero));

    input += input_stride;
    m--;
  }
  while (m-- != 0) {
    input += n;
    __m128i vinput = _mm_setzero_si128();
    if (n & 1) {
      input -= 1;
      vinput = _mm_cvtsi32_si128((int)(uint32_t)*input);
    }
    if (n & 2) {
      vinput = _mm_slli_epi32(vinput, 16);
      input -= 2;
      vinput = _mm_insert_epi16(vinput, *((const uint16_t*)input), 0);
    }
    if (n & 4) {
      input -= 4;
      vinput = _mm_unpacklo_epi32(
          _mm_cvtsi32_si128((int)*((const uint32_t*)input)), vinput);
    }
    input += input_stride;

    const __m128i vxinput = _mm_unpacklo_epi8(vinput, vzero);
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi8(vxinput, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi8(vxinput, vzero));
  }

  const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale);

  const __m128 vacc_lo_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_lo), vscale);
  const __m128 vacc_hi_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_hi), vscale);

  const __m128i vscaled_lo = _mm_cvtps_epi32(vacc_lo_f);
  const __m128i vscaled_hi = _mm_cvtps_epi32(vacc_hi_f);

  __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
  vout = _mm_adds_epi16(
      vout,
      _mm_load_si128(
          (const __m128i*)quantization_params->sse2.output_zero_point));
  vout = _mm_packus_epi16(vout, vout);
  vout = _mm_min_epu8(
      vout,
      _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
  vout = _mm_max_epu8(
      vout,
      _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

  if (n & 4) {
    *((uint32_t*)output) = (uint32_t)_mm_cvtsi128_si32(vout);
    output += 4;
    vout = _mm_srli_epi64(vout, 32);
  }
  if (n & 2) {
    *((uint16_t*)output) = (uint16_t)_mm_extract_epi16(vout, 0);
    output += 2;
    vout = _mm_srli_epi32(vout, 16);
  }
  if (n & 1) {
    *((uint8_t*)output) = (uint8_t)_mm_cvtsi128_si32(vout);
  }
}
