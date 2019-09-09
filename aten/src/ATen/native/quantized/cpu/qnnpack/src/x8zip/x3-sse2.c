/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <emmintrin.h>

#include <qnnpack/x8zip.h>

void pytorch_qnnp_x8zip_x3__sse2(size_t n, const void* input, void* output) {
  const uint8_t* x = input;
  const uint8_t* y = x + n;
  const uint8_t* z = y + n;
  uint8_t* o = output;

  if (n >= 16) {
    const __m128i vmask0x00FF00FF = _mm_set1_epi16(0x00FF);
    const __m128i vmask0x0000FFFF = _mm_set1_epi32(0x0000FFFF);
    do {
      /* vx  = ( x15, x14, x13, x12, x11, x10,  x9,  x8,  x7,  x6,  x5,  x4, x3,
       * x2, x1, x0 ) */
      const __m128i vx = _mm_loadu_si128((const __m128i*)x);
      x += 16;
      /* vy  = ( y15, y14, y13, y12, y11, y10,  y9,  y8,  y7,  y6,  y5,  y4, y3,
       * y2, y1, y0 ) */
      const __m128i vy = _mm_loadu_si128((const __m128i*)y);
      y += 16;
      /* vz  = ( z15, z14, z13, z12, z11, z10,  z9,  z8,  z7,  z6,  z5,  z4, z3,
       * z2, z1, z0 ) */
      const __m128i vz = _mm_loadu_si128((const __m128i*)z);
      z += 16;

      /* vxeye     = ( y14, x14, y12, x12, y10, x10,  y8,  x8,  y6,  x6,  y4,
       * x4,  y2,  x2,  y0,  x0 ) */
      const __m128i vxeye = _mm_or_si128(
          _mm_and_si128(vx, vmask0x00FF00FF), _mm_slli_epi16(vy, 8));
      /* vyozo     = ( z15, y15, z13, y13, z11, y11,  z9,  y9,  z7,  y7,  z5,
       * y5,  z3,  y3,  z1,  y1 ) */
      const __m128i vyozo = _mm_or_si128(
          _mm_andnot_si128(vmask0x00FF00FF, vz), _mm_srli_epi16(vy, 8));
      /* vzoxo     = ( x15, z14, x13, z12, x11, z10,  x9,  z8,  x7,  z6,  x5,
       * z4,  x3,  z2,  x1,  z0 ) */
      const __m128i vzexo = _mm_or_si128(
          _mm_and_si128(vz, vmask0x00FF00FF),
          _mm_andnot_si128(vmask0x00FF00FF, vx));

      /* vxeyezexo = ( x13, z12, y12, x12,  x9,  z8,  y8,  x8,  x5,  z4,  y4,
       * x4,  x1,  z0,  y0,  x0 ) */
      const __m128i vxeyezexo = _mm_or_si128(
          _mm_and_si128(vxeye, vmask0x0000FFFF), _mm_slli_epi32(vzexo, 16));
      /* vyozoxeye = ( y14, x14, z13, y13, y10, x10,  z9,  y9,  y6,  x6,  z5,
       * y5,  y2,  x2,  z1,  y1 ) */
      const __m128i vyozoxeye = _mm_or_si128(
          _mm_and_si128(vyozo, vmask0x0000FFFF),
          _mm_andnot_si128(vmask0x0000FFFF, vxeye));
      /* vzexoyozo = ( z15, y15, x15, z14, z11, y11, x11, z10,  z7,  y7,  x7,
       * z6,  z3,  y3,  x3,  z2 ) */
      const __m128i vzexoyozo = _mm_or_si128(
          _mm_andnot_si128(vmask0x0000FFFF, vyozo), _mm_srli_epi32(vzexo, 16));

      /* vtemp0    = ( x13, z12, y12, x12,  x5,  z4,  y4,  x4, z11, y11, x11,
       * z10,  z3,  y3,  x3,  z2 ) */
      const __m128i vtemp0 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vzexoyozo),
          _mm_castsi128_ps(vxeyezexo),
          _MM_SHUFFLE(3, 1, 2, 0)));
      /* vtemp1    = ( y10, x10,  z9,  y9,  y2,  x2,  z1,  y1,  x9,  z8,  y8,
       * x8,  x1,  z0,  y0,  x0 ) */
      const __m128i vtemp1 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vxeyezexo),
          _mm_castsi128_ps(vyozoxeye),
          _MM_SHUFFLE(2, 0, 2, 0)));
      /* vtemp2    = ( z15, y15, x15, z14,  z7,  y7,  x7,  z6, y14, x14, z13,
       * y13,  y6,  x6,  z5,  y5 ) */
      const __m128i vtemp2 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vyozoxeye),
          _mm_castsi128_ps(vzexoyozo),
          _MM_SHUFFLE(3, 1, 3, 1)));

      /* vxyz0     = (  x5,  z4,  y4,  x4,  z3,  y3,  x3,  z2,  y2,  x2,  z1,
       * y1,  x1,  z0,  y0,  x0 ) */
      const __m128i vxyz0 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vtemp1),
          _mm_castsi128_ps(vtemp0),
          _MM_SHUFFLE(2, 0, 2, 0)));
      /* vxyz1     = ( y10, x10,  z9,  y9,  x9,  z8,  y8,  x8,  z7,  y7,  x7,
       * z6,  y6,  x6,  z5,  y5 ) */
      const __m128i vxyz1 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vtemp2),
          _mm_castsi128_ps(vtemp1),
          _MM_SHUFFLE(3, 1, 2, 0)));
      /* vxyz2     = ( z15, y15, x15, z14, y14, x14, z13, y13, x13, z12, y12,
       * x12, z11, y11, x11, z10 ) */
      const __m128i vxyz2 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vtemp0),
          _mm_castsi128_ps(vtemp2),
          _MM_SHUFFLE(3, 1, 3, 1)));

      _mm_storeu_si128((__m128i*)o, vxyz0);
      _mm_storeu_si128((__m128i*)o + 1, vxyz1);
      _mm_storeu_si128((__m128i*)o + 2, vxyz2);
      o += 48;
      n -= 16;
    } while (n >= 16);
    if (n != 0) {
      const size_t address_increment = n - 16;
      /* vx  = ( x15, x14, x13, x12, x11, x10,  x9,  x8,  x7,  x6,  x5,  x4, x3,
       * x2, x1, x0 ) */
      const __m128i vx =
          _mm_loadu_si128((const __m128i*)((uintptr_t)x + address_increment));
      /* vy  = ( y15, y14, y13, y12, y11, y10,  y9,  y8,  y7,  y6,  y5,  y4, y3,
       * y2, y1, y0 ) */
      const __m128i vy =
          _mm_loadu_si128((const __m128i*)((uintptr_t)y + address_increment));
      /* vz  = ( z15, z14, z13, z12, z11, z10,  z9,  z8,  z7,  z6,  z5,  z4, z3,
       * z2, z1, z0 ) */
      const __m128i vz =
          _mm_loadu_si128((const __m128i*)((uintptr_t)z + address_increment));

      /* vxeye     = ( y14, x14, y12, x12, y10, x10,  y8,  x8,  y6,  x6,  y4,
       * x4,  y2,  x2,  y0,  x0 ) */
      const __m128i vxeye = _mm_or_si128(
          _mm_and_si128(vx, vmask0x00FF00FF), _mm_slli_epi16(vy, 8));
      /* vyozo     = ( z15, y15, z13, y13, z11, y11,  z9,  y9,  z7,  y7,  z5,
       * y5,  z3,  y3,  z1,  y1 ) */
      const __m128i vyozo = _mm_or_si128(
          _mm_andnot_si128(vmask0x00FF00FF, vz), _mm_srli_epi16(vy, 8));
      /* vzoxo     = ( x15, z14, x13, z12, x11, z10,  x9,  z8,  x7,  z6,  x5,
       * z4,  x3,  z2,  x1,  z0 ) */
      const __m128i vzexo = _mm_or_si128(
          _mm_and_si128(vz, vmask0x00FF00FF),
          _mm_andnot_si128(vmask0x00FF00FF, vx));

      /* vxeyezexo = ( x13, z12, y12, x12,  x9,  z8,  y8,  x8,  x5,  z4,  y4,
       * x4,  x1,  z0,  y0,  x0 ) */
      const __m128i vxeyezexo = _mm_or_si128(
          _mm_and_si128(vxeye, vmask0x0000FFFF), _mm_slli_epi32(vzexo, 16));
      /* vyozoxeye = ( y14, x14, z13, y13, y10, x10,  z9,  y9,  y6,  x6,  z5,
       * y5,  y2,  x2,  z1,  y1 ) */
      const __m128i vyozoxeye = _mm_or_si128(
          _mm_and_si128(vyozo, vmask0x0000FFFF),
          _mm_andnot_si128(vmask0x0000FFFF, vxeye));
      /* vzexoyozo = ( z15, y15, x15, z14, z11, y11, x11, z10,  z7,  y7,  x7,
       * z6,  z3,  y3,  x3,  z2 ) */
      const __m128i vzexoyozo = _mm_or_si128(
          _mm_andnot_si128(vmask0x0000FFFF, vyozo), _mm_srli_epi32(vzexo, 16));

      /* vtemp0    = ( x13, z12, y12, x12,  x5,  z4,  y4,  x4, z11, y11, x11,
       * z10,  z3,  y3,  x3,  z2 ) */
      const __m128i vtemp0 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vzexoyozo),
          _mm_castsi128_ps(vxeyezexo),
          _MM_SHUFFLE(3, 1, 2, 0)));
      /* vtemp1    = ( y10, x10,  z9,  y9,  y2,  x2,  z1,  y1,  x9,  z8,  y8,
       * x8,  x1,  z0,  y0,  x0 ) */
      const __m128i vtemp1 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vxeyezexo),
          _mm_castsi128_ps(vyozoxeye),
          _MM_SHUFFLE(2, 0, 2, 0)));
      /* vtemp2    = ( z15, y15, x15, z14,  z7,  y7,  x7,  z6, y14, x14, z13,
       * y13,  y6,  x6,  z5,  y5 ) */
      const __m128i vtemp2 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vyozoxeye),
          _mm_castsi128_ps(vzexoyozo),
          _MM_SHUFFLE(3, 1, 3, 1)));

      /* vxyz0     = (  x5,  z4,  y4,  x4,  z3,  y3,  x3,  z2,  y2,  x2,  z1,
       * y1,  x1,  z0,  y0,  x0 ) */
      const __m128i vxyz0 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vtemp1),
          _mm_castsi128_ps(vtemp0),
          _MM_SHUFFLE(2, 0, 2, 0)));
      /* vxyz1     = ( y10, x10,  z9,  y9,  x9,  z8,  y8,  x8,  z7,  y7,  x7,
       * z6,  y6,  x6,  z5,  y5 ) */
      const __m128i vxyz1 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vtemp2),
          _mm_castsi128_ps(vtemp1),
          _MM_SHUFFLE(3, 1, 2, 0)));
      /* vxyz2     = ( z15, y15, x15, z14, y14, x14, z13, y13, x13, z12, y12,
       * x12, z11, y11, x11, z10 ) */
      const __m128i vxyz2 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vtemp0),
          _mm_castsi128_ps(vtemp2),
          _MM_SHUFFLE(3, 1, 3, 1)));

      o = (uint8_t*)((uintptr_t)o + address_increment * 3);
      _mm_storeu_si128((__m128i*)o, vxyz0);
      _mm_storeu_si128((__m128i*)o + 1, vxyz1);
      _mm_storeu_si128((__m128i*)o + 2, vxyz2);
    }
  } else {
    do {
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      const uint8_t vz = *z++;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o += 3;
    } while (--n != 0);
  }
}
