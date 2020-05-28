/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>

#include <qnnpack/q8conv.h>
#include <requantization/runtime-sse2.h>

void pytorch_q8conv_ukernel_4x4c2__sse2(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  __m128i vacc0x0123 = _mm_loadu_si128((const __m128i*)w);
  __m128i vacc1x0123 = vacc0x0123;
  __m128i vacc2x0123 = vacc0x0123;
  __m128i vacc3x0123 = vacc0x0123;
  w = (const void*)((uintptr_t)w + 16);

  const __m128i va_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.input_zero_point);
  const int16_t vb_zero_point_0 =
    quantization_params->sse2.kernel_zero_points[output_channel_index];
  const int16_t vb_zero_point_1 =
      quantization_params->sse2.kernel_zero_points[output_channel_index + 1];
  const int16_t vb_zero_point_2 =
      quantization_params->sse2.kernel_zero_points[output_channel_index + 2];
  const int16_t vb_zero_point_3 =
      quantization_params->sse2.kernel_zero_points[output_channel_index + 3];

  const __m128i vb_zero_point = _mm_set_epi16(vb_zero_point_3,
                                              vb_zero_point_3,
                                              vb_zero_point_2,
                                              vb_zero_point_2,
                                              vb_zero_point_1,
                                              vb_zero_point_1,
                                              vb_zero_point_0,
                                              vb_zero_point_0
                                              );
  const __m128i vzero = _mm_setzero_si128();
  do {
    const uint8_t* restrict a0 = *a++;
    const uint8_t* restrict a1 = *a++;
    const uint8_t* restrict a2 = *a++;
    const uint8_t* restrict a3 = *a++;

    size_t k = kc;
    for (; k >= 8; k -= 8) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*)a0);
      const __m128i vxa0 =
          sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*)a1);
      const __m128i vxa1 =
          sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
      a1 += 8;
      const __m128i va2 = _mm_loadl_epi64((const __m128i*)a2);
      const __m128i vxa2 =
          sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
      a2 += 8;
      const __m128i va3 = _mm_loadl_epi64((const __m128i*)a3);
      const __m128i vxa3 =
          sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);
      a3 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*)w);
      const __m128i vxb0 =
          _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);
      vacc0x0123 = _mm_add_epi32(
          vacc0x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
      vacc1x0123 = _mm_add_epi32(
          vacc1x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
      vacc2x0123 = _mm_add_epi32(
          vacc2x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
      vacc3x0123 = _mm_add_epi32(
          vacc3x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));

      const __m128i vb1 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
      const __m128i vxb1 =
          _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);
      vacc0x0123 = _mm_add_epi32(
          vacc0x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
      vacc1x0123 = _mm_add_epi32(
          vacc1x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
      vacc2x0123 = _mm_add_epi32(
          vacc2x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
      vacc3x0123 = _mm_add_epi32(
          vacc3x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));

      const __m128i vb2 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
      const __m128i vxb2 =
          _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);
      vacc0x0123 = _mm_add_epi32(
          vacc0x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
      vacc1x0123 = _mm_add_epi32(
          vacc1x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
      vacc2x0123 = _mm_add_epi32(
          vacc2x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
      vacc3x0123 = _mm_add_epi32(
          vacc3x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));

      const __m128i vb3 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
      const __m128i vxb3 =
          _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);
      vacc0x0123 = _mm_add_epi32(
          vacc0x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
      vacc1x0123 = _mm_add_epi32(
          vacc1x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
      vacc2x0123 = _mm_add_epi32(
          vacc2x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
      vacc3x0123 = _mm_add_epi32(
          vacc3x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));

      w = (void*)((uintptr_t)w + 32);
    }
    if (k != 0) {
      const size_t a_predecrement = 8 - k;
      const __m128i va_shift = _mm_cvtsi32_si128(8 * a_predecrement);

      const __m128i va0 = _mm_srl_epi64(
          _mm_loadl_epi64((const __m128i*)(a0 - a_predecrement)), va_shift);
      const __m128i vxa0 =
          sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
      const __m128i va1 = _mm_srl_epi64(
          _mm_loadl_epi64((const __m128i*)(a1 - a_predecrement)), va_shift);
      const __m128i vxa1 =
          sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
      const __m128i va2 = _mm_srl_epi64(
          _mm_loadl_epi64((const __m128i*)(a2 - a_predecrement)), va_shift);
      const __m128i vxa2 =
          sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
      const __m128i va3 = _mm_srl_epi64(
          _mm_loadl_epi64((const __m128i*)(a3 - a_predecrement)), va_shift);
      const __m128i vxa3 =
          sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*)w);
      const __m128i vxb0 =
          _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);
      w = (void*)((uintptr_t)w + 8);

      vacc0x0123 = _mm_add_epi32(
          vacc0x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
      vacc1x0123 = _mm_add_epi32(
          vacc1x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
      vacc2x0123 = _mm_add_epi32(
          vacc2x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
      vacc3x0123 = _mm_add_epi32(
          vacc3x0123,
          _mm_madd_epi16(
              _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));

      if (k > 2) {
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*)w);
        const __m128i vxb1 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);
        w = (void*)((uintptr_t)w + 8);

        vacc0x0123 = _mm_add_epi32(
            vacc0x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
        vacc1x0123 = _mm_add_epi32(
            vacc1x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
        vacc2x0123 = _mm_add_epi32(
            vacc2x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
        vacc3x0123 = _mm_add_epi32(
            vacc3x0123,
            _mm_madd_epi16(
                _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));

        if (k > 4) {
          const __m128i vb2 = _mm_loadl_epi64((const __m128i*)w);
          const __m128i vxb2 =
              _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);
          w = (void*)((uintptr_t)w + 8);

          vacc0x0123 = _mm_add_epi32(
              vacc0x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
          vacc1x0123 = _mm_add_epi32(
              vacc1x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
          vacc2x0123 = _mm_add_epi32(
              vacc2x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
          vacc3x0123 = _mm_add_epi32(
              vacc3x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));

          if (k > 6) {
            const __m128i vb3 = _mm_loadl_epi64((const __m128i*)w);
            const __m128i vxb3 =
                _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);
            w = (void*)((uintptr_t)w + 8);

            vacc0x0123 = _mm_add_epi32(
                vacc0x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
            vacc1x0123 = _mm_add_epi32(
                vacc1x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
            vacc2x0123 = _mm_add_epi32(
                vacc2x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
            vacc3x0123 = _mm_add_epi32(
                vacc3x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
          }
        }
      }
    }
  } while (--ks != 0);

  const __m128 vmultiplier =
      _mm_loadu_ps(&quantization_params->sse2.requantization_scales
          [output_channel_index]);

  vacc0x0123 = _mm_cvtps_epi32(
                _mm_mul_ps(
                  _mm_cvtepi32_ps(vacc0x0123),
                  vmultiplier
                  )
                );
  vacc1x0123 = _mm_cvtps_epi32(
                _mm_mul_ps(
                  _mm_cvtepi32_ps(vacc1x0123),
                  vmultiplier
                  )
                );
  vacc2x0123 = _mm_cvtps_epi32(
                _mm_mul_ps(
                  _mm_cvtepi32_ps(vacc2x0123),
                  vmultiplier
                  )
                );
  vacc3x0123 = _mm_cvtps_epi32(
                _mm_mul_ps(
                  _mm_cvtepi32_ps(vacc3x0123),
                  vmultiplier
                  )
                );

  const __m128i voutput_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.output_zero_point);
  const __m128i vacc01x0123 = _mm_adds_epi16(
      _mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
  const __m128i vacc23x0123 = _mm_adds_epi16(
      _mm_packs_epi32(vacc2x0123, vacc3x0123), voutput_zero_point);
  __m128i vout = _mm_packus_epi16(vacc01x0123, vacc23x0123);
  vout = _mm_min_epu8(
      vout,
      _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
  vout = _mm_max_epu8(
      vout,
      _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*)((uintptr_t)c0 + c_stride);
  if (mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*)((uintptr_t)c1 + c_stride);
  if (mr <= 2) {
    c2 = c1;
  }
  uint8_t* c3 = (uint8_t*)((uintptr_t)c2 + c_stride);
  if (mr != 4) {
    c3 = c2;
  }
  if (nr == 4) {
    *((uint32_t*)c0) = (uint32_t)_mm_cvtsi128_si32(vout);
    *((uint32_t*)c1) = (uint32_t)_mm_cvtsi128_si32(_mm_srli_epi64(vout, 32));
    *((uint32_t*)c2) =
        (uint32_t)_mm_cvtsi128_si32(_mm_unpackhi_epi32(vout, vout));
    *((uint32_t*)c3) = (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(vout, 12));
  } else {
    if (nr >= 2) {
      *((uint16_t*)c0) = (uint16_t)_mm_extract_epi16(vout, 0);
      c0 += 2;
      *((uint16_t*)c1) = (uint16_t)_mm_extract_epi16(vout, 2);
      c1 += 2;
      *((uint16_t*)c2) = (uint16_t)_mm_extract_epi16(vout, 4);
      c2 += 2;
      *((uint16_t*)c3) = (uint16_t)_mm_extract_epi16(vout, 6);
      c3 += 2;
      vout = _mm_srli_epi32(vout, 16);
      nr -= 2;
    }
    if (nr != 0) {
      *((uint8_t*)c0) = (uint8_t)_mm_cvtsi128_si32(vout);
      *((uint8_t*)c1) = (uint8_t)_mm_extract_epi16(vout, 2);
      *((uint8_t*)c2) = (uint8_t)_mm_extract_epi16(vout, 4);
      *((uint8_t*)c3) = (uint8_t)_mm_extract_epi16(vout, 6);
    }
  }
}
