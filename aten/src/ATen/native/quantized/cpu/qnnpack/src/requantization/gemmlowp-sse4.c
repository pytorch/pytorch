/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stdint.h>

#include <smmintrin.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

#include "gemmlowp-sse.h"

void pytorch_qnnp_requantize_gemmlowp__sse4(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 16 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const uint32_t scale_bits = fp32_to_bits(scale);

  /* Compute requantization parameters */
  const uint32_t multiplier =
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7;
  const int32_t exponent = (fp32_to_bits(scale) >> 23) - 127 - 23 - 7;
  const int32_t shift =
      -(32 /* using high 32 bits in VQRDMUL */ - 1 /* doubling in VQRDMUL */ +
        exponent);

  const __m128i vmultiplier = _mm_set1_epi32(multiplier);
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);
  const __m128i vqmin = _mm_set1_epi8((char)qmin);
  const __m128i vqmax = _mm_set1_epi8((char)qmax);
  for (; n != 0; n -= 16) {
    const __m128i x = _mm_loadu_si128((const __m128i*)input);
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));
    input += 16;

    const __m128i x_product = gemmlowp_sse_vqrdmulh_s32(x, vmultiplier);
    const __m128i y_product = gemmlowp_sse_vqrdmulh_s32(y, vmultiplier);
    const __m128i z_product = gemmlowp_sse_vqrdmulh_s32(z, vmultiplier);
    const __m128i w_product = gemmlowp_sse_vqrdmulh_s32(w, vmultiplier);

    const __m128i x_scaled = gemmlowp_sse_rdivbypo2_s32(x_product, shift);
    const __m128i y_scaled = gemmlowp_sse_rdivbypo2_s32(y_product, shift);
    const __m128i z_scaled = gemmlowp_sse_rdivbypo2_s32(z_product, shift);
    const __m128i w_scaled = gemmlowp_sse_rdivbypo2_s32(w_product, shift);

    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);

    _mm_storeu_si128((__m128i*)output, xyzw_clamped);
    output += 16;
  }
}
