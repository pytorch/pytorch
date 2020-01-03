/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <emmintrin.h>

#include <qnnpack/x8zip.h>

void pytorch_qnnp_x8zip_x2__sse2(size_t n, const void* input, void* output) {
  const uint8_t* x = input;
  const uint8_t* y = x + n;
  uint8_t* o = output;

  if (n >= 16) {
    do {
      const __m128i vx = _mm_loadu_si128((const __m128i*)x);
      x += 16;
      const __m128i vy = _mm_loadu_si128((const __m128i*)y);
      y += 16;
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
      _mm_storeu_si128((__m128i*)o, vxy_lo);
      _mm_storeu_si128((__m128i*)(o + 16), vxy_hi);
      o = (void*)((uintptr_t)o + 32);
      n -= 16;
    } while (n >= 16);
    if (n != 0) {
      const size_t address_increment = n - 16;
      const __m128i vx =
          _mm_loadu_si128((const __m128i*)((uintptr_t)x + address_increment));
      const __m128i vy =
          _mm_loadu_si128((const __m128i*)((uintptr_t)y + address_increment));
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
      o = (void*)((uintptr_t)o + address_increment * 2);
      _mm_storeu_si128((__m128i*)o, vxy_lo);
      _mm_storeu_si128((__m128i*)o + 1, vxy_hi);
    }
  } else {
    do {
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      o[0] = vx;
      o[1] = vy;
      o += 2;
    } while (--n != 0);
  }
}
