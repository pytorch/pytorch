/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits.h>

#include <immintrin.h>

/*
 * The code below is adapted from Google's gemmlowp library.
 * It is only used in QNNPACK unit tests and comparative benchmarks,
 * but not the library itself.
 */

// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

static inline __m128i gemmlowp_sse_rdivbypo2_s32(__m128i x, int exponent) {
  const __m128i mask =
      _mm_set1_epi32((int32_t)((UINT64_C(1) << exponent) - UINT64_C(1)));
  const __m128i remainder = _mm_and_si128(x, mask);
  const __m128i threshold = _mm_sub_epi32(
      _mm_srli_epi32(mask, 1), _mm_cmplt_epi32(x, _mm_setzero_si128()));
  return _mm_sub_epi32(
      _mm_sra_epi32(x, _mm_cvtsi32_si128(exponent)),
      _mm_cmpgt_epi32(remainder, threshold));
}

static inline __m128i gemmlowp_sse_mul_s32(__m128i a, __m128i b) {
#ifdef __SSE4_1__
  return _mm_mul_epi32(a, b);
#else
  __m128i sign, zero, mul_us, a_neg, b_neg, mul_us_neg;
  sign = _mm_xor_si128(a, b);
  sign = _mm_srai_epi32(sign, 31); // promote sign bit to all fields, all fff if
                                   // negative and all 0 if positive
  sign = _mm_shuffle_epi32(
      sign,
      _MM_SHUFFLE(2, 2, 0, 0)); // promote sign bit to 3 and 1st data lanes
  zero = _mm_setzero_si128();
#ifdef __SSSE3__
  a_neg = _mm_abs_epi32(a); // negate a and b
  b_neg = _mm_abs_epi32(b); // negate a and b
#else /* pre-SSSE3 */
  const __m128i a_neg_mask = _mm_cmplt_epi32(a, zero);
  a_neg = _mm_sub_epi32(_mm_xor_si128(a, a_neg_mask), a_neg_mask);
  const __m128i b_neg_mask = _mm_cmplt_epi32(b, zero);
  b_neg = _mm_sub_epi32(_mm_xor_si128(b, b_neg_mask), b_neg_mask);
#endif /* pre-SSSE3 */
  mul_us = _mm_mul_epu32(a_neg, b_neg); // uses 0 and 2nd data lanes, (abs), the
                                        // multiplication gives 64 bit result
  mul_us_neg = _mm_sub_epi64(zero, mul_us);
  mul_us_neg = _mm_and_si128(sign, mul_us_neg);
  mul_us = _mm_andnot_si128(sign, mul_us);
  return _mm_or_si128(mul_us, mul_us_neg);
#endif
}

static inline __m128i gemmlowp_sse_vqrdmulh_s32(__m128i a, __m128i b) {
  // saturation only happen if a == b == INT32_MIN
  const __m128i min = _mm_set1_epi32(INT32_MIN);
  const __m128i saturation_mask =
      _mm_and_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, min));

  // a = a0 | a1 | a2 | a3
  // b = b0 | b1 | b2 | b3
  const __m128i a0_a2 = a;
  const __m128i a1_a3 = _mm_srli_si128(a, 4);
  const __m128i b0_b2 = b;
  const __m128i b1_b3 = _mm_srli_si128(b, 4);

  const __m128i a0b0_a2b2 = gemmlowp_sse_mul_s32(a0_a2, b0_b2);
  const __m128i a1b1_a3b3 = gemmlowp_sse_mul_s32(a1_a3, b1_b3);

  // do the rounding and take into account that it will be doubled
  const __m128i nudge = _mm_set1_epi64x(1 << 30);
  const __m128i a0b0_a2b2_rounded = _mm_add_epi64(a0b0_a2b2, nudge);
  const __m128i a1b1_a3b3_rounded = _mm_add_epi64(a1b1_a3b3, nudge);

  // do the doubling
  const __m128i a0b0_a2b2_rounded_2x = _mm_slli_epi64(a0b0_a2b2_rounded, 1);
  const __m128i a1b1_a3b3_rounded_2x = _mm_slli_epi64(a1b1_a3b3_rounded, 1);

// get the high part of the products
#ifdef __SSE4_1__
  const __m128i result = _mm_blend_epi16(
      _mm_srli_epi64(a0b0_a2b2_rounded_2x, 32), a1b1_a3b3_rounded_2x, 0xCC);
#else
  const __m128i result0213 = _mm_castps_si128(_mm_shuffle_ps(
      _mm_castsi128_ps(a0b0_a2b2_rounded_2x),
      _mm_castsi128_ps(a1b1_a3b3_rounded_2x),
      _MM_SHUFFLE(3, 1, 3, 1)));
  const __m128i result = _mm_shuffle_epi32(result0213, _MM_SHUFFLE(3, 1, 2, 0));
#endif

// saturate those which overflowed
#ifdef __SSE4_1__
  const __m128i saturated_result =
      _mm_blendv_epi8(result, min, saturation_mask);
#else
  const __m128i saturated_result = _mm_or_si128(
      _mm_and_si128(saturation_mask, min),
      _mm_andnot_si128(saturation_mask, result));
#endif
  return saturated_result;
}
