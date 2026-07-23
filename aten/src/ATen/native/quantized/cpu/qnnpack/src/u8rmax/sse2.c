/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <emmintrin.h>

#include <qnnpack/u8rmax.h>

uint8_t pytorch_u8rmax_ukernel__sse2(size_t n, const uint8_t* x) {
  assert(n != 0);

  if
    PYTORCH_QNNP_LIKELY(n >= 16) {
      __m128i vmax = _mm_setzero_si128();
      do {
        const __m128i vx = _mm_loadu_si128((const __m128i*)x);
        x += 16;
        vmax = _mm_max_epu8(vmax, vx);
        n -= 16;
      } while (n >= 16);
      if (n != 0) {
        const size_t x_increment = n - 16;
        x = (const uint8_t*)((uintptr_t)x + x_increment);
        const __m128i vx = _mm_loadu_si128((const __m128i*)x);
        vmax = _mm_max_epu8(vmax, vx);
      }
      vmax = _mm_max_epu8(vmax, _mm_unpackhi_epi64(vmax, vmax));
      vmax = _mm_max_epu8(vmax, _mm_srli_epi64(vmax, 32));
      vmax = _mm_max_epu8(vmax, _mm_srli_epi32(vmax, 16));
      vmax = _mm_max_epu8(vmax, _mm_srli_epi16(vmax, 8));
      return (uint8_t)_mm_cvtsi128_si32(vmax);
    }
  else {
    uint8_t vmax = 0;
    do {
      const uint8_t vx = *x++;
      vmax = vx > vmax ? vx : vmax;
    } while (--n != 0);
    return vmax;
  }
}
