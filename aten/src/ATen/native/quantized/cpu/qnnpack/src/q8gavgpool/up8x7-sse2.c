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

void pytorch_q8gavgpool_ukernel_up8x7__sse2(
    size_t m,
    size_t n,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  assert(m >= 1);
  assert(m <= 7);
  assert(n >= 8);

  const uint8_t* i0 = input;
  const uint8_t* i1 = i0 + input_stride;
  if (m < 2) {
    i1 = zero;
  }
  const uint8_t* i2 = i1 + input_stride;
  if (m <= 2) {
    i2 = zero;
  }
  const uint8_t* i3 = i2 + input_stride;
  if (m < 4) {
    i3 = zero;
  }
  const uint8_t* i4 = i3 + input_stride;
  if (m <= 4) {
    i4 = zero;
  }
  const uint8_t* i5 = i4 + input_stride;
  if (m < 6) {
    i5 = zero;
  }
  const uint8_t* i6 = i5 + input_stride;
  if (m <= 6) {
    i6 = zero;
  }
  const __m128i vbias =
      _mm_load_si128((const __m128i*)&quantization_params->sse2.bias);
  const __m128i vzero = _mm_setzero_si128();

  const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale);

  do {
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0);
    i0 += 8;
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1);
    i1 += 8;
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2);
    i2 += 8;
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3);
    i3 += 8;
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*)i4);
    i4 += 8;
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*)i5);
    i5 += 8;
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*)i6);
    i6 += 8;

    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    __m128i vacc_lo = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vxi0, vzero));
    __m128i vacc_hi = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vxi0, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

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

    _mm_storel_epi64((__m128i*)output, vout);
    output += 8;

    n -= 8;
  } while (n >= 8);
  if (n != 0) {
    const size_t address_decrement = 8 - n;
    i0 = (const uint8_t*)((uintptr_t)i0 - address_decrement);
    i1 = (const uint8_t*)((uintptr_t)i1 - address_decrement);
    i2 = (const uint8_t*)((uintptr_t)i2 - address_decrement);
    i3 = (const uint8_t*)((uintptr_t)i3 - address_decrement);
    i4 = (const uint8_t*)((uintptr_t)i4 - address_decrement);
    i5 = (const uint8_t*)((uintptr_t)i5 - address_decrement);
    i6 = (const uint8_t*)((uintptr_t)i6 - address_decrement);
    const __m128i vi_shift = _mm_cvtsi32_si128(8 * address_decrement);

    const __m128i vi0 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i0), vi_shift);
    const __m128i vi1 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i1), vi_shift);
    const __m128i vi2 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i2), vi_shift);
    const __m128i vi3 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i3), vi_shift);
    const __m128i vi4 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i4), vi_shift);
    const __m128i vi5 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i5), vi_shift);
    const __m128i vi6 =
        _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i6), vi_shift);

    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    __m128i vacc_lo = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vxi0, vzero));
    __m128i vacc_hi = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vxi0, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

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
}
