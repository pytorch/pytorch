/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stdint.h>

#include <tmmintrin.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_q31__ssse3(
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

  /* Compute requantization parameters */
  const uint32_t scale_bits = fp32_to_bits(scale);

  /* Multiplier is in [0x40000000, 0x7FFFFF80] range */
  const int32_t multiplier = (int32_t)(
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  /* Shift is in [0, 31] range */
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  const __m128i vmultiplier = _mm_set1_epi32(multiplier);
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);
  const __m128i vqmin = _mm_set1_epi8((char)qmin);
  const __m128i vqmax = _mm_set1_epi8((char)qmax);
  const __m128i vshift = _mm_cvtsi32_si128((int)shift);
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const __m128i vremainder_mask = _mm_set1_epi32((int)remainder_mask);
  const __m128i vthreshold = _mm_set1_epi32((int)(remainder_mask >> 1));
  const __m128i vq31rounding = _mm_set1_epi64x(UINT64_C(0x40000000));
  for (; n != 0; n -= 16) {
    const __m128i x = _mm_loadu_si128((const __m128i*)input);
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));
    input += 16;

    const __m128i x_abs = _mm_abs_epi32(x);
    const __m128i y_abs = _mm_abs_epi32(y);
    const __m128i z_abs = _mm_abs_epi32(z);
    const __m128i w_abs = _mm_abs_epi32(w);

    const __m128i x_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), x);
    const __m128i y_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), y);
    const __m128i z_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), z);
    const __m128i w_neg_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), w);

    const __m128i x_abs_rev = _mm_shuffle_epi32(x_abs, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i y_abs_rev = _mm_shuffle_epi32(y_abs, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i z_abs_rev = _mm_shuffle_epi32(z_abs, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i w_abs_rev = _mm_shuffle_epi32(w_abs, _MM_SHUFFLE(2, 3, 0, 1));

    const __m128i x_abs_product_even = _mm_mul_epu32(x_abs, vmultiplier);
    const __m128i y_abs_product_even = _mm_mul_epu32(y_abs, vmultiplier);
    const __m128i z_abs_product_even = _mm_mul_epu32(z_abs, vmultiplier);
    const __m128i w_abs_product_even = _mm_mul_epu32(w_abs, vmultiplier);

    const __m128i x_neg_mask_even =
        _mm_shuffle_epi32(x_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));
    const __m128i y_neg_mask_even =
        _mm_shuffle_epi32(y_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));
    const __m128i z_neg_mask_even =
        _mm_shuffle_epi32(z_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));
    const __m128i w_neg_mask_even =
        _mm_shuffle_epi32(w_neg_mask, _MM_SHUFFLE(2, 2, 0, 0));

    const __m128i x_product_even = _mm_sub_epi64(
        _mm_xor_si128(x_abs_product_even, x_neg_mask_even), x_neg_mask_even);
    const __m128i y_product_even = _mm_sub_epi64(
        _mm_xor_si128(y_abs_product_even, y_neg_mask_even), y_neg_mask_even);
    const __m128i z_product_even = _mm_sub_epi64(
        _mm_xor_si128(z_abs_product_even, z_neg_mask_even), z_neg_mask_even);
    const __m128i w_product_even = _mm_sub_epi64(
        _mm_xor_si128(w_abs_product_even, w_neg_mask_even), w_neg_mask_even);

    const __m128i x_rounded_product_even =
        _mm_add_epi64(x_product_even, vq31rounding);
    const __m128i y_rounded_product_even =
        _mm_add_epi64(y_product_even, vq31rounding);
    const __m128i z_rounded_product_even =
        _mm_add_epi64(z_product_even, vq31rounding);
    const __m128i w_rounded_product_even =
        _mm_add_epi64(w_product_even, vq31rounding);

    const __m128i x_abs_product_odd = _mm_mul_epu32(x_abs_rev, vmultiplier);
    const __m128i y_abs_product_odd = _mm_mul_epu32(y_abs_rev, vmultiplier);
    const __m128i z_abs_product_odd = _mm_mul_epu32(z_abs_rev, vmultiplier);
    const __m128i w_abs_product_odd = _mm_mul_epu32(w_abs_rev, vmultiplier);

    const __m128i x_neg_mask_odd =
        _mm_shuffle_epi32(x_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i y_neg_mask_odd =
        _mm_shuffle_epi32(y_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i z_neg_mask_odd =
        _mm_shuffle_epi32(z_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i w_neg_mask_odd =
        _mm_shuffle_epi32(w_neg_mask, _MM_SHUFFLE(3, 3, 1, 1));

    const __m128i x_product_odd = _mm_sub_epi64(
        _mm_xor_si128(x_abs_product_odd, x_neg_mask_odd), x_neg_mask_odd);
    const __m128i y_product_odd = _mm_sub_epi64(
        _mm_xor_si128(y_abs_product_odd, y_neg_mask_odd), y_neg_mask_odd);
    const __m128i z_product_odd = _mm_sub_epi64(
        _mm_xor_si128(z_abs_product_odd, z_neg_mask_odd), z_neg_mask_odd);
    const __m128i w_product_odd = _mm_sub_epi64(
        _mm_xor_si128(w_abs_product_odd, w_neg_mask_odd), w_neg_mask_odd);

    const __m128i x_rounded_product_odd =
        _mm_add_epi64(x_product_odd, vq31rounding);
    const __m128i y_rounded_product_odd =
        _mm_add_epi64(y_product_odd, vq31rounding);
    const __m128i z_rounded_product_odd =
        _mm_add_epi64(z_product_odd, vq31rounding);
    const __m128i w_rounded_product_odd =
        _mm_add_epi64(w_product_odd, vq31rounding);

    const __m128i x_q31product_even =
        _mm_srli_epi64(x_rounded_product_even, 31);
    const __m128i x_q31product_odd = _mm_srli_epi64(x_rounded_product_odd, 31);
    const __m128i y_q31product_even =
        _mm_srli_epi64(y_rounded_product_even, 31);
    const __m128i y_q31product_odd = _mm_srli_epi64(y_rounded_product_odd, 31);
    const __m128i z_q31product_even =
        _mm_srli_epi64(z_rounded_product_even, 31);
    const __m128i z_q31product_odd = _mm_srli_epi64(z_rounded_product_odd, 31);
    const __m128i w_q31product_even =
        _mm_srli_epi64(w_rounded_product_even, 31);
    const __m128i w_q31product_odd = _mm_srli_epi64(w_rounded_product_odd, 31);

    const __m128i x_q31product_0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x_q31product_even),
        _mm_castsi128_ps(x_q31product_odd),
        _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i y_q31product_0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(y_q31product_even),
        _mm_castsi128_ps(y_q31product_odd),
        _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i z_q31product_0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(z_q31product_even),
        _mm_castsi128_ps(z_q31product_odd),
        _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i w_q31product_0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(w_q31product_even),
        _mm_castsi128_ps(w_q31product_odd),
        _MM_SHUFFLE(2, 0, 2, 0)));

    const __m128i x_q31product =
        _mm_shuffle_epi32(x_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i y_q31product =
        _mm_shuffle_epi32(y_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i z_q31product =
        _mm_shuffle_epi32(z_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i w_q31product =
        _mm_shuffle_epi32(w_q31product_0213, _MM_SHUFFLE(3, 1, 2, 0));

    const __m128i x_remainder = _mm_add_epi32(
        _mm_and_si128(x_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), x_q31product));
    const __m128i y_remainder = _mm_add_epi32(
        _mm_and_si128(y_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), y_q31product));
    const __m128i z_remainder = _mm_add_epi32(
        _mm_and_si128(z_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), z_q31product));
    const __m128i w_remainder = _mm_add_epi32(
        _mm_and_si128(w_q31product, vremainder_mask),
        _mm_cmpgt_epi32(_mm_setzero_si128(), w_q31product));

    const __m128i x_scaled = _mm_sub_epi32(
        _mm_sra_epi32(x_q31product, vshift),
        _mm_cmpgt_epi32(x_remainder, vthreshold));
    const __m128i y_scaled = _mm_sub_epi32(
        _mm_sra_epi32(y_q31product, vshift),
        _mm_cmpgt_epi32(y_remainder, vthreshold));
    const __m128i z_scaled = _mm_sub_epi32(
        _mm_sra_epi32(z_q31product, vshift),
        _mm_cmpgt_epi32(z_remainder, vthreshold));
    const __m128i w_scaled = _mm_sub_epi32(
        _mm_sra_epi32(w_q31product, vshift),
        _mm_cmpgt_epi32(w_remainder, vthreshold));

    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);

    /*
     * 16x PSHUFD
     * 4x SHUFPS
     * 8x PMULUDQ
     * 8x PXOR (setzero)
     * 8x PXOR
     * 4x PAND
     * 8x PADDQ
     * 4x PADDD
     * 2x PADDW
     * 8x PSUBQ
     * 4x PSUBD
     * 8x PSRLQ (immediate)
     * 4x PSRAD (register)
     * 12x PCMPGTD
     * 4x PABSD
     * 2x PACKSSDW
     * 1x PACKUSWB
     * 1x PMAXUB
     * 1x PMINUB
     * ---------------------
     * 107 instructions total
     */

    _mm_storeu_si128((__m128i*)output, xyzw_clamped);
    output += 16;
  }
}
