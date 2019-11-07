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

void pytorch_u8maxpool_ukernel_16x9p8q__sse2(
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
  assert(kc >= 16);

  const __m128i voutput_max =
      _mm_load_si128((const __m128i*)params->sse2.output_max);
  const __m128i voutput_min =
      _mm_load_si128((const __m128i*)params->sse2.output_min);

  do {
    uint8_t* o = output;
    {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      const uint8_t* i8 = *input++;
      if (ks < 2) {
        i1 = i0;
      }
      if (ks <= 2) {
        i2 = i0;
      }
      if (ks < 4) {
        i3 = i0;
      }
      if (ks <= 4) {
        i4 = i0;
      }
      if (ks < 6) {
        i5 = i0;
      }
      if (ks <= 6) {
        i6 = i0;
      }
      if (ks < 8) {
        i7 = i0;
      }
      if (ks <= 8) {
        i8 = i0;
      }

      size_t k = kc;
      while (k >= 16) {
        const __m128i vi0 = _mm_loadu_si128((const __m128i*)i0);
        i0 += 16;
        const __m128i vi1 = _mm_loadu_si128((const __m128i*)i1);
        i1 += 16;
        const __m128i vi2 = _mm_loadu_si128((const __m128i*)i2);
        i2 += 16;
        const __m128i vi3 = _mm_loadu_si128((const __m128i*)i3);
        i3 += 16;
        const __m128i vi4 = _mm_loadu_si128((const __m128i*)i4);
        i4 += 16;
        const __m128i vi5 = _mm_loadu_si128((const __m128i*)i5);
        i5 += 16;
        const __m128i vi6 = _mm_loadu_si128((const __m128i*)i6);
        i6 += 16;
        const __m128i vi7 = _mm_loadu_si128((const __m128i*)i7);
        i7 += 16;
        const __m128i vi8 = _mm_loadu_si128((const __m128i*)i8);
        i8 += 16;

        const __m128i vmax018 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vi8);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax01678 = _mm_max_epu8(vmax018, vmax67);
        const __m128i vmax = _mm_max_epu8(vmax2345, vmax01678);
        const __m128i vout =
            _mm_max_epu8(_mm_min_epu8(vmax, voutput_max), voutput_min);

        _mm_storeu_si128((__m128i*)o, vout);
        o += 16;

        k -= 16;
      }
      if (k != 0) {
        const size_t address_increment = k - 16;
        i0 = (const uint8_t*)((uintptr_t)i0 + address_increment);
        i1 = (const uint8_t*)((uintptr_t)i1 + address_increment);
        i2 = (const uint8_t*)((uintptr_t)i2 + address_increment);
        i3 = (const uint8_t*)((uintptr_t)i3 + address_increment);
        i4 = (const uint8_t*)((uintptr_t)i4 + address_increment);
        i5 = (const uint8_t*)((uintptr_t)i5 + address_increment);
        i6 = (const uint8_t*)((uintptr_t)i6 + address_increment);
        i7 = (const uint8_t*)((uintptr_t)i7 + address_increment);
        i8 = (const uint8_t*)((uintptr_t)i8 + address_increment);
        o = (uint8_t*)((uintptr_t)o + address_increment);

        const __m128i vi0 = _mm_loadu_si128((const __m128i*)i0);
        const __m128i vi1 = _mm_loadu_si128((const __m128i*)i1);
        const __m128i vi2 = _mm_loadu_si128((const __m128i*)i2);
        const __m128i vi3 = _mm_loadu_si128((const __m128i*)i3);
        const __m128i vi4 = _mm_loadu_si128((const __m128i*)i4);
        const __m128i vi5 = _mm_loadu_si128((const __m128i*)i5);
        const __m128i vi6 = _mm_loadu_si128((const __m128i*)i6);
        const __m128i vi7 = _mm_loadu_si128((const __m128i*)i7);
        const __m128i vi8 = _mm_loadu_si128((const __m128i*)i8);

        const __m128i vmax018 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vi8);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax01678 = _mm_max_epu8(vmax018, vmax67);
        const __m128i vmax = _mm_max_epu8(vmax2345, vmax01678);
        const __m128i vout =
            _mm_max_epu8(_mm_min_epu8(vmax, voutput_max), voutput_min);

        _mm_storeu_si128((__m128i*)o, vout);
        o += 16;
      }
    }

    for (ptrdiff_t m = (ptrdiff_t)ks - 9; m > 0; m -= 8) {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      if (m < 2) {
        i1 = i0;
      }
      if (m <= 2) {
        i2 = i0;
      }
      if (m < 4) {
        i3 = i0;
      }
      if (m <= 4) {
        i4 = i0;
      }
      if (m < 6) {
        i5 = i0;
      }
      if (m <= 6) {
        i6 = i0;
      }
      if (m < 8) {
        i7 = i0;
      }

      o = output;
      size_t k = kc;
      while (k >= 16) {
        const __m128i vi0 = _mm_loadu_si128((const __m128i*)i0);
        i0 += 16;
        const __m128i vi1 = _mm_loadu_si128((const __m128i*)i1);
        i1 += 16;
        const __m128i vi2 = _mm_loadu_si128((const __m128i*)i2);
        i2 += 16;
        const __m128i vi3 = _mm_loadu_si128((const __m128i*)i3);
        i3 += 16;
        const __m128i vi4 = _mm_loadu_si128((const __m128i*)i4);
        i4 += 16;
        const __m128i vi5 = _mm_loadu_si128((const __m128i*)i5);
        i5 += 16;
        const __m128i vi6 = _mm_loadu_si128((const __m128i*)i6);
        i6 += 16;
        const __m128i vi7 = _mm_loadu_si128((const __m128i*)i7);
        i7 += 16;
        const __m128i vo = _mm_loadu_si128((const __m128i*)o);

        const __m128i vmax01 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vo);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax0167 = _mm_max_epu8(vmax01, vmax67);
        const __m128i vmax = _mm_max_epu8(vmax2345, vmax0167);
        const __m128i vout =
            _mm_max_epu8(_mm_min_epu8(vmax, voutput_max), voutput_min);

        _mm_storeu_si128((__m128i*)o, vout);
        o += 16;

        k -= 16;
      }
      if (k != 0) {
        const size_t address_increment = k - 16;
        i0 = (const uint8_t*)((uintptr_t)i0 + address_increment);
        i1 = (const uint8_t*)((uintptr_t)i1 + address_increment);
        i2 = (const uint8_t*)((uintptr_t)i2 + address_increment);
        i3 = (const uint8_t*)((uintptr_t)i3 + address_increment);
        i4 = (const uint8_t*)((uintptr_t)i4 + address_increment);
        i5 = (const uint8_t*)((uintptr_t)i5 + address_increment);
        i6 = (const uint8_t*)((uintptr_t)i6 + address_increment);
        i7 = (const uint8_t*)((uintptr_t)i7 + address_increment);
        o = (uint8_t*)((uintptr_t)o + address_increment);

        const __m128i vi0 = _mm_loadu_si128((const __m128i*)i0);
        const __m128i vi1 = _mm_loadu_si128((const __m128i*)i1);
        const __m128i vi2 = _mm_loadu_si128((const __m128i*)i2);
        const __m128i vi3 = _mm_loadu_si128((const __m128i*)i3);
        const __m128i vi4 = _mm_loadu_si128((const __m128i*)i4);
        const __m128i vi5 = _mm_loadu_si128((const __m128i*)i5);
        const __m128i vi6 = _mm_loadu_si128((const __m128i*)i6);
        const __m128i vi7 = _mm_loadu_si128((const __m128i*)i7);
        const __m128i vo = _mm_loadu_si128((const __m128i*)o);

        const __m128i vmax01 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vo);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax0167 = _mm_max_epu8(vmax01, vmax67);
        const __m128i vmax = _mm_max_epu8(vmax2345, vmax0167);
        const __m128i vout =
            _mm_max_epu8(_mm_min_epu8(vmax, voutput_max), voutput_min);

        _mm_storeu_si128((__m128i*)o, vout);
        o += 16;
      }
    }
    input = (const uint8_t**)((uintptr_t)input + input_increment);
    output = (uint8_t*)((uintptr_t)o + output_increment);
  } while (--n != 0);
}
