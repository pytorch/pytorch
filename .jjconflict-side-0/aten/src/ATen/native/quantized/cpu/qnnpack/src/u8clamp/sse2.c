/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <emmintrin.h>

#include <qnnpack/u8clamp.h>

void pytorch_u8clamp_ukernel__sse2(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union pytorch_qnnp_u8_clamping_params params[RESTRICT_STATIC 1]) {
  assert(n != 0);

  if
    PYTORCH_QNNP_LIKELY(n >= 8) {
      const __m128i voutput_max =
          _mm_load_si128((const __m128i*)&params->sse2.output_max);
      const __m128i voutput_min =
          _mm_load_si128((const __m128i*)&params->sse2.output_min);
      for (; n >= 64; n -= 64) {
        const __m128i vx0 = _mm_loadu_si128((const __m128i*)x);
        const __m128i vx1 = _mm_loadu_si128((const __m128i*)x + 1);
        const __m128i vx2 = _mm_loadu_si128((const __m128i*)x + 2);
        const __m128i vx3 = _mm_loadu_si128((const __m128i*)x + 3);
        x += 64;

        const __m128i vy0 =
            _mm_min_epu8(_mm_max_epu8(vx0, voutput_min), voutput_max);
        const __m128i vy1 =
            _mm_min_epu8(_mm_max_epu8(vx1, voutput_min), voutput_max);
        const __m128i vy2 =
            _mm_min_epu8(_mm_max_epu8(vx2, voutput_min), voutput_max);
        const __m128i vy3 =
            _mm_min_epu8(_mm_max_epu8(vx3, voutput_min), voutput_max);

        __builtin_prefetch(x + 640);

        _mm_storeu_si128((__m128i*)y, vy0);
        _mm_storeu_si128((__m128i*)y + 1, vy1);
        _mm_storeu_si128((__m128i*)y + 2, vy2);
        _mm_storeu_si128((__m128i*)y + 3, vy3);
        y += 64;
      }
      for (; n >= 8; n -= 8) {
        __m128i vout = _mm_loadl_epi64((const __m128i*)x);
        x += 8;
        vout = _mm_min_epu8(vout, voutput_max);
        vout = _mm_max_epu8(vout, voutput_min);
        _mm_storel_epi64((__m128i*)y, vout);
        y += 8;
      }
      if (n != 0) {
        const size_t n_increment = n - 8;
        x = (const uint8_t*)((uintptr_t)x + n_increment);
        y = (uint8_t*)((uintptr_t)y + n_increment);

        __m128i vout = _mm_loadl_epi64((const __m128i*)x);
        vout = _mm_min_epu8(vout, voutput_max);
        vout = _mm_max_epu8(vout, voutput_min);
        _mm_storel_epi64((__m128i*)y, vout);
      }
    }
  else {
    const uint32_t voutput_max = params->sse2.output_max[0];
    const uint32_t voutput_min = params->sse2.output_min[0];
    do {
      uint32_t vout = *x++;
      vout = vout > voutput_max ? voutput_max : vout;
      vout = vout < voutput_min ? voutput_min : vout;
      *y++ = (uint8_t)vout;
    } while (--n != 0);
  }
}
