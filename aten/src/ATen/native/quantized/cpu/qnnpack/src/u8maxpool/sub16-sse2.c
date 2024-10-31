/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <emmintrin.h>

#include <qnnpack/u8maxpool.h>

void pytorch_u8maxpool_ukernel_sub16__sse2(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_u8_clamping_params params[RESTRICT_STATIC 1]) {
  assert(n != 0);
  assert(ks != 0);
  assert(kc != 0);
  assert(kc < 16);

  const __m128i voutput_max =
      _mm_load_si128((const __m128i*)params->sse2.output_max);
  const __m128i voutput_min =
      _mm_load_si128((const __m128i*)params->sse2.output_min);

  do {
    __m128i vmax = _mm_setzero_si128();

    size_t m = ks;
    do {
      const uint8_t* i = *input++;
      i += kc;
      __m128i vi = vmax;
      if (kc & 1) {
        i -= 1;
        vi = _mm_cvtsi32_si128(*i);
      }
      if (kc & 2) {
        vi = _mm_slli_epi32(vi, 16);
        i -= 2;
        vi = _mm_insert_epi16(vi, *((const uint16_t*)i), 0);
      }
      if (kc & 4) {
        i -= 4;
        vi = _mm_unpacklo_epi32(
            _mm_cvtsi32_si128((int)*((const uint32_t*)i)), vi);
      }
      if (kc & 8) {
        i -= 8;
        vi = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)i), vi);
      }
      vmax = _mm_max_epu8(vmax, vi);
    } while (--m != 0);
    input = (const uint8_t**)((uintptr_t)input + input_increment);
    __m128i vout = _mm_max_epu8(_mm_min_epu8(vmax, voutput_max), voutput_min);

    if (kc & 8) {
      _mm_storel_epi64((__m128i*)output, vout);
      output += 8;
      vout = _mm_unpackhi_epi64(vout, vout);
    }
    if (kc & 4) {
      *((uint32_t*)output) = (uint32_t)_mm_cvtsi128_si32(vout);
      output += 4;
      vout = _mm_srli_epi64(vout, 32);
    }
    if (kc & 2) {
      *((uint16_t*)output) = (uint16_t)_mm_extract_epi16(vout, 0);
      output += 2;
      vout = _mm_srli_epi32(vout, 16);
    }
    if (kc & 1) {
      *((uint8_t*)output) = (uint8_t)_mm_cvtsi128_si32(vout);
      output += 1;
    }
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--n != 0);
}
