/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <emmintrin.h>

#include <qnnpack/x8zip.h>

void pytorch_qnnp_x8zip_x4__sse2(size_t n, const void* input, void* output) {
  const uint8_t* x = input;
  const uint8_t* y = x + n;
  const uint8_t* z = y + n;
  const uint8_t* w = z + n;
  uint8_t* o = output;

  if (n >= 16) {
    do {
      const __m128i vx = _mm_loadu_si128((const __m128i*)x);
      x += 16;
      const __m128i vy = _mm_loadu_si128((const __m128i*)y);
      y += 16;
      const __m128i vz = _mm_loadu_si128((const __m128i*)z);
      z += 16;
      const __m128i vw = _mm_loadu_si128((const __m128i*)w);
      w += 16;
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
      const __m128i vzw_lo = _mm_unpacklo_epi8(vz, vw);
      const __m128i vzw_hi = _mm_unpackhi_epi8(vz, vw);
      const __m128i vxyzw0 = _mm_unpacklo_epi16(vxy_lo, vzw_lo);
      const __m128i vxyzw1 = _mm_unpackhi_epi16(vxy_lo, vzw_lo);
      const __m128i vxyzw2 = _mm_unpacklo_epi16(vxy_hi, vzw_hi);
      const __m128i vxyzw3 = _mm_unpackhi_epi16(vxy_hi, vzw_hi);
      _mm_storeu_si128((__m128i*)o, vxyzw0);
      _mm_storeu_si128((__m128i*)o + 1, vxyzw1);
      _mm_storeu_si128((__m128i*)o + 2, vxyzw2);
      _mm_storeu_si128((__m128i*)o + 3, vxyzw3);
      o = (void*)((uintptr_t)o + 64);
      n -= 16;
    } while (n >= 16);
    if (n != 0) {
      const size_t address_increment = n - 16;
      const __m128i vx =
          _mm_loadu_si128((const __m128i*)((uintptr_t)x + address_increment));
      const __m128i vy =
          _mm_loadu_si128((const __m128i*)((uintptr_t)y + address_increment));
      const __m128i vz =
          _mm_loadu_si128((const __m128i*)((uintptr_t)z + address_increment));
      const __m128i vw =
          _mm_loadu_si128((const __m128i*)((uintptr_t)w + address_increment));
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
      const __m128i vzw_lo = _mm_unpacklo_epi8(vz, vw);
      const __m128i vzw_hi = _mm_unpackhi_epi8(vz, vw);
      const __m128i vxyzw0 = _mm_unpacklo_epi16(vxy_lo, vzw_lo);
      const __m128i vxyzw1 = _mm_unpackhi_epi16(vxy_lo, vzw_lo);
      const __m128i vxyzw2 = _mm_unpacklo_epi16(vxy_hi, vzw_hi);
      const __m128i vxyzw3 = _mm_unpackhi_epi16(vxy_hi, vzw_hi);
      o = (void*)((uintptr_t)o + address_increment * 4);
      _mm_storeu_si128((__m128i*)o, vxyzw0);
      _mm_storeu_si128((__m128i*)o + 1, vxyzw1);
      _mm_storeu_si128((__m128i*)o + 2, vxyzw2);
      _mm_storeu_si128((__m128i*)o + 3, vxyzw3);
    }
  } else {
    do {
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      const uint8_t vz = *z++;
      const uint8_t vw = *w++;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o[3] = vw;
      o += 4;
    } while (--n != 0);
  }
}
