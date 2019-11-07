/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>

#include <qnnpack/q8dwconv.h>

void pytorch_q8dwconv_ukernel_mp8x25__sse2(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    int32_t* outacc32,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  const __m128i vinput_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.input_zero_point);
  const __m128i vkernel_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.kernel_zero_point);
  const __m128i vzero = _mm_setzero_si128();

  do {
    int32_t* outacc = outacc32;
    const void* w = weights;
    {
      const uint8_t* i00 = input[0];
      const uint8_t* i01 = input[1];
      const uint8_t* i02 = input[2];
      const uint8_t* i10 = input[3];
      const uint8_t* i11 = input[4];
      const uint8_t* i12 = input[5];
      const uint8_t* i20 = input[6];
      const uint8_t* i21 = input[7];
      const uint8_t* i22 = input[8];
      const uint8_t* i23 = input[9];

      size_t c = channels;
      for (; c >= 8; c -= 8) {
        __m128i vacc_lo = _mm_loadu_si128((const __m128i*)w);
        __m128i vacc_hi = _mm_loadu_si128((const __m128i*)((uintptr_t)w + 16));

        const __m128i vi00 = _mm_loadl_epi64((const __m128i*)i00);
        i00 += 8;
        const __m128i vxi00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi00, vzero), vinput_zero_point);
        const __m128i vk00 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 32));
        const __m128i vxk00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk00, vzero), vkernel_zero_point);
        const __m128i vprod00_odd = _mm_mullo_epi16(vxi00, vxk00);
        const __m128i vprod00_even = _mm_mulhi_epi16(vxi00, vxk00);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod00_odd, vprod00_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod00_odd, vprod00_even));

        const __m128i vi01 = _mm_loadl_epi64((const __m128i*)i01);
        i01 += 8;
        const __m128i vxi01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi01, vzero), vinput_zero_point);
        const __m128i vk01 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 40));
        const __m128i vxk01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk01, vzero), vkernel_zero_point);
        const __m128i vprod01_odd = _mm_mullo_epi16(vxi01, vxk01);
        const __m128i vprod01_even = _mm_mulhi_epi16(vxi01, vxk01);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod01_odd, vprod01_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod01_odd, vprod01_even));

        const __m128i vi02 = _mm_loadl_epi64((const __m128i*)i02);
        i02 += 8;
        const __m128i vxi02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi02, vzero), vinput_zero_point);
        const __m128i vk02 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 48));
        const __m128i vxk02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk02, vzero), vkernel_zero_point);
        const __m128i vprod02_odd = _mm_mullo_epi16(vxi02, vxk02);
        const __m128i vprod02_even = _mm_mulhi_epi16(vxi02, vxk02);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod02_odd, vprod02_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod02_odd, vprod02_even));

        const __m128i vi10 = _mm_loadl_epi64((const __m128i*)i10);
        i10 += 8;
        const __m128i vxi10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi10, vzero), vinput_zero_point);
        const __m128i vk10 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 56));
        const __m128i vxk10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk10, vzero), vkernel_zero_point);
        const __m128i vprod10_odd = _mm_mullo_epi16(vxi10, vxk10);
        const __m128i vprod10_even = _mm_mulhi_epi16(vxi10, vxk10);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod10_odd, vprod10_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod10_odd, vprod10_even));

        const __m128i vi11 = _mm_loadl_epi64((const __m128i*)i11);
        i11 += 8;
        const __m128i vxi11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi11, vzero), vinput_zero_point);
        const __m128i vk11 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 64));
        const __m128i vxk11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk11, vzero), vkernel_zero_point);
        const __m128i vprod11_odd = _mm_mullo_epi16(vxi11, vxk11);
        const __m128i vprod11_even = _mm_mulhi_epi16(vxi11, vxk11);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod11_odd, vprod11_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod11_odd, vprod11_even));

        const __m128i vi12 = _mm_loadl_epi64((const __m128i*)i12);
        i12 += 8;
        const __m128i vxi12 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi12, vzero), vinput_zero_point);
        const __m128i vk12 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 72));
        const __m128i vxk12 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk12, vzero), vkernel_zero_point);
        const __m128i vprod12_odd = _mm_mullo_epi16(vxi12, vxk12);
        const __m128i vprod12_even = _mm_mulhi_epi16(vxi12, vxk12);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod12_odd, vprod12_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod12_odd, vprod12_even));

        const __m128i vi20 = _mm_loadl_epi64((const __m128i*)i20);
        i20 += 8;
        const __m128i vxi20 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi20, vzero), vinput_zero_point);
        const __m128i vk20 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 80));
        const __m128i vxk20 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk20, vzero), vkernel_zero_point);
        const __m128i vprod20_odd = _mm_mullo_epi16(vxi20, vxk20);
        const __m128i vprod20_even = _mm_mulhi_epi16(vxi20, vxk20);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod20_odd, vprod20_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod20_odd, vprod20_even));

        const __m128i vi21 = _mm_loadl_epi64((const __m128i*)i21);
        i21 += 8;
        const __m128i vxi21 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi21, vzero), vinput_zero_point);
        const __m128i vk21 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 88));
        const __m128i vxk21 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk21, vzero), vkernel_zero_point);
        const __m128i vprod21_odd = _mm_mullo_epi16(vxi21, vxk21);
        const __m128i vprod21_even = _mm_mulhi_epi16(vxi21, vxk21);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod21_odd, vprod21_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod21_odd, vprod21_even));

        const __m128i vi22 = _mm_loadl_epi64((const __m128i*)i22);
        i22 += 8;
        const __m128i vxi22 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi22, vzero), vinput_zero_point);
        const __m128i vk22 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 96));
        const __m128i vxk22 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk22, vzero), vkernel_zero_point);
        const __m128i vprod22_odd = _mm_mullo_epi16(vxi22, vxk22);
        const __m128i vprod22_even = _mm_mulhi_epi16(vxi22, vxk22);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod22_odd, vprod22_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod22_odd, vprod22_even));

        const __m128i vi23 = _mm_loadl_epi64((const __m128i*)i23);
        i23 += 8;
        const __m128i vxi23 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi23, vzero), vinput_zero_point);
        const __m128i vk23 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 104));
        const __m128i vxk23 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk23, vzero), vkernel_zero_point);
        const __m128i vprod23_odd = _mm_mullo_epi16(vxi23, vxk23);
        const __m128i vprod23_even = _mm_mulhi_epi16(vxi23, vxk23);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod23_odd, vprod23_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod23_odd, vprod23_even));

        w = (const void*)((uintptr_t)w + 112);
        _mm_storeu_si128((__m128i*)outacc, vacc_lo);
        outacc += 4;
        _mm_storeu_si128((__m128i*)outacc, vacc_hi);
        outacc += 4;
      }
      if (c != 0) {
        const size_t i_predecrement = 8 - c;
        const __m128i vi_shift = _mm_cvtsi32_si128(8 * i_predecrement);
        i00 -= i_predecrement;
        i01 -= i_predecrement;
        i02 -= i_predecrement;
        i10 -= i_predecrement;
        i11 -= i_predecrement;
        i12 -= i_predecrement;
        i20 -= i_predecrement;
        i21 -= i_predecrement;
        i22 -= i_predecrement;
        i23 -= i_predecrement;

        __m128i vacc_lo = _mm_loadu_si128((const __m128i*)w);
        __m128i vacc_hi = _mm_loadu_si128((const __m128i*)((uintptr_t)w + 16));

        const __m128i vi00 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i00), vi_shift);
        const __m128i vxi00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi00, vzero), vinput_zero_point);
        const __m128i vk00 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 32));
        const __m128i vxk00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk00, vzero), vkernel_zero_point);
        const __m128i vprod00_odd = _mm_mullo_epi16(vxi00, vxk00);
        const __m128i vprod00_even = _mm_mulhi_epi16(vxi00, vxk00);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod00_odd, vprod00_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod00_odd, vprod00_even));

        const __m128i vi01 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i01), vi_shift);
        const __m128i vxi01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi01, vzero), vinput_zero_point);
        const __m128i vk01 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 40));
        const __m128i vxk01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk01, vzero), vkernel_zero_point);
        const __m128i vprod01_odd = _mm_mullo_epi16(vxi01, vxk01);
        const __m128i vprod01_even = _mm_mulhi_epi16(vxi01, vxk01);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod01_odd, vprod01_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod01_odd, vprod01_even));

        const __m128i vi02 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i02), vi_shift);
        const __m128i vxi02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi02, vzero), vinput_zero_point);
        const __m128i vk02 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 48));
        const __m128i vxk02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk02, vzero), vkernel_zero_point);
        const __m128i vprod02_odd = _mm_mullo_epi16(vxi02, vxk02);
        const __m128i vprod02_even = _mm_mulhi_epi16(vxi02, vxk02);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod02_odd, vprod02_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod02_odd, vprod02_even));

        const __m128i vi10 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i10), vi_shift);
        const __m128i vxi10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi10, vzero), vinput_zero_point);
        const __m128i vk10 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 56));
        const __m128i vxk10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk10, vzero), vkernel_zero_point);
        const __m128i vprod10_odd = _mm_mullo_epi16(vxi10, vxk10);
        const __m128i vprod10_even = _mm_mulhi_epi16(vxi10, vxk10);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod10_odd, vprod10_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod10_odd, vprod10_even));

        const __m128i vi11 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i11), vi_shift);
        const __m128i vxi11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi11, vzero), vinput_zero_point);
        const __m128i vk11 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 64));
        const __m128i vxk11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk11, vzero), vkernel_zero_point);
        const __m128i vprod11_odd = _mm_mullo_epi16(vxi11, vxk11);
        const __m128i vprod11_even = _mm_mulhi_epi16(vxi11, vxk11);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod11_odd, vprod11_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod11_odd, vprod11_even));

        const __m128i vi12 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i12), vi_shift);
        const __m128i vxi12 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi12, vzero), vinput_zero_point);
        const __m128i vk12 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 72));
        const __m128i vxk12 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk12, vzero), vkernel_zero_point);
        const __m128i vprod12_odd = _mm_mullo_epi16(vxi12, vxk12);
        const __m128i vprod12_even = _mm_mulhi_epi16(vxi12, vxk12);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod12_odd, vprod12_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod12_odd, vprod12_even));

        const __m128i vi20 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i20), vi_shift);
        const __m128i vxi20 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi20, vzero), vinput_zero_point);
        const __m128i vk20 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 80));
        const __m128i vxk20 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk20, vzero), vkernel_zero_point);
        const __m128i vprod20_odd = _mm_mullo_epi16(vxi20, vxk20);
        const __m128i vprod20_even = _mm_mulhi_epi16(vxi20, vxk20);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod20_odd, vprod20_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod20_odd, vprod20_even));

        const __m128i vi21 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i21), vi_shift);
        const __m128i vxi21 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi21, vzero), vinput_zero_point);
        const __m128i vk21 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 88));
        const __m128i vxk21 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk21, vzero), vkernel_zero_point);
        const __m128i vprod21_odd = _mm_mullo_epi16(vxi21, vxk21);
        const __m128i vprod21_even = _mm_mulhi_epi16(vxi21, vxk21);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod21_odd, vprod21_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod21_odd, vprod21_even));

        const __m128i vi22 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i22), vi_shift);
        const __m128i vxi22 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi22, vzero), vinput_zero_point);
        const __m128i vk22 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 96));
        const __m128i vxk22 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk22, vzero), vkernel_zero_point);
        const __m128i vprod22_odd = _mm_mullo_epi16(vxi22, vxk22);
        const __m128i vprod22_even = _mm_mulhi_epi16(vxi22, vxk22);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod22_odd, vprod22_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod22_odd, vprod22_even));

        const __m128i vi23 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i23), vi_shift);
        const __m128i vxi23 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi23, vzero), vinput_zero_point);
        const __m128i vk23 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 104));
        const __m128i vxk23 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk23, vzero), vkernel_zero_point);
        const __m128i vprod23_odd = _mm_mullo_epi16(vxi23, vxk23);
        const __m128i vprod23_even = _mm_mulhi_epi16(vxi23, vxk23);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod23_odd, vprod23_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod23_odd, vprod23_even));

        w = (const void*)((uintptr_t)w + 112);
        _mm_storeu_si128((__m128i*)outacc, vacc_lo);
        outacc += 4;
        _mm_storeu_si128((__m128i*)outacc, vacc_hi);
        outacc += 4;
      }
    }
    {
      const uint8_t* i00 = input[10];
      const uint8_t* i01 = input[11];
      const uint8_t* i02 = input[12];
      const uint8_t* i10 = input[13];
      const uint8_t* i11 = input[14];
      const uint8_t* i12 = input[15];
      const uint8_t* i20 = input[16];
      const uint8_t* i21 = input[17];
      const uint8_t* i22 = input[18];
      const uint8_t* i23 = input[19];
      outacc = outacc32;

      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const __m128i vi00 = _mm_loadl_epi64((const __m128i*)i00);
        i00 += 8;
        const __m128i vxi00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi00, vzero), vinput_zero_point);
        const __m128i vk00 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w));
        const __m128i vxk00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk00, vzero), vkernel_zero_point);
        const __m128i vprod00_odd = _mm_mullo_epi16(vxi00, vxk00);
        const __m128i vprod00_even = _mm_mulhi_epi16(vxi00, vxk00);
        __m128i vacc_lo = _mm_unpacklo_epi16(vprod00_odd, vprod00_even);
        __m128i vacc_hi = _mm_unpackhi_epi16(vprod00_odd, vprod00_even);

        const __m128i vi01 = _mm_loadl_epi64((const __m128i*)i01);
        i01 += 8;
        const __m128i vxi01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi01, vzero), vinput_zero_point);
        const __m128i vk01 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
        const __m128i vxk01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk01, vzero), vkernel_zero_point);
        const __m128i vprod01_odd = _mm_mullo_epi16(vxi01, vxk01);
        const __m128i vprod01_even = _mm_mulhi_epi16(vxi01, vxk01);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod01_odd, vprod01_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod01_odd, vprod01_even));

        const __m128i vi02 = _mm_loadl_epi64((const __m128i*)i02);
        i02 += 8;
        const __m128i vxi02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi02, vzero), vinput_zero_point);
        const __m128i vk02 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
        const __m128i vxk02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk02, vzero), vkernel_zero_point);
        const __m128i vprod02_odd = _mm_mullo_epi16(vxi02, vxk02);
        const __m128i vprod02_even = _mm_mulhi_epi16(vxi02, vxk02);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod02_odd, vprod02_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod02_odd, vprod02_even));

        const __m128i vi10 = _mm_loadl_epi64((const __m128i*)i10);
        i10 += 8;
        const __m128i vxi10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi10, vzero), vinput_zero_point);
        const __m128i vk10 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
        const __m128i vxk10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk10, vzero), vkernel_zero_point);
        const __m128i vprod10_odd = _mm_mullo_epi16(vxi10, vxk10);
        const __m128i vprod10_even = _mm_mulhi_epi16(vxi10, vxk10);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod10_odd, vprod10_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod10_odd, vprod10_even));

        const __m128i vi11 = _mm_loadl_epi64((const __m128i*)i11);
        i11 += 8;
        const __m128i vxi11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi11, vzero), vinput_zero_point);
        const __m128i vk11 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 32));
        const __m128i vxk11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk11, vzero), vkernel_zero_point);
        const __m128i vprod11_odd = _mm_mullo_epi16(vxi11, vxk11);
        const __m128i vprod11_even = _mm_mulhi_epi16(vxi11, vxk11);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod11_odd, vprod11_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod11_odd, vprod11_even));

        const __m128i vi12 = _mm_loadl_epi64((const __m128i*)i12);
        i12 += 8;
        const __m128i vxi12 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi12, vzero), vinput_zero_point);
        const __m128i vk12 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 40));
        const __m128i vxk12 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk12, vzero), vkernel_zero_point);
        const __m128i vprod12_odd = _mm_mullo_epi16(vxi12, vxk12);
        const __m128i vprod12_even = _mm_mulhi_epi16(vxi12, vxk12);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod12_odd, vprod12_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod12_odd, vprod12_even));

        const __m128i vi20 = _mm_loadl_epi64((const __m128i*)i20);
        i20 += 8;
        const __m128i vxi20 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi20, vzero), vinput_zero_point);
        const __m128i vk20 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 48));
        const __m128i vxk20 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk20, vzero), vkernel_zero_point);
        const __m128i vprod20_odd = _mm_mullo_epi16(vxi20, vxk20);
        const __m128i vprod20_even = _mm_mulhi_epi16(vxi20, vxk20);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod20_odd, vprod20_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod20_odd, vprod20_even));

        const __m128i vi21 = _mm_loadl_epi64((const __m128i*)i21);
        i21 += 8;
        const __m128i vxi21 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi21, vzero), vinput_zero_point);
        const __m128i vk21 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 56));
        const __m128i vxk21 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk21, vzero), vkernel_zero_point);
        const __m128i vprod21_odd = _mm_mullo_epi16(vxi21, vxk21);
        const __m128i vprod21_even = _mm_mulhi_epi16(vxi21, vxk21);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod21_odd, vprod21_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod21_odd, vprod21_even));

        const __m128i vi22 = _mm_loadl_epi64((const __m128i*)i22);
        i22 += 8;
        const __m128i vxi22 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi22, vzero), vinput_zero_point);
        const __m128i vk22 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 64));
        const __m128i vxk22 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk22, vzero), vkernel_zero_point);
        const __m128i vprod22_odd = _mm_mullo_epi16(vxi22, vxk22);
        const __m128i vprod22_even = _mm_mulhi_epi16(vxi22, vxk22);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod22_odd, vprod22_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod22_odd, vprod22_even));

        const __m128i vi23 = _mm_loadl_epi64((const __m128i*)i23);
        i23 += 8;
        const __m128i vxi23 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi23, vzero), vinput_zero_point);
        const __m128i vk23 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 72));
        const __m128i vxk23 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk23, vzero), vkernel_zero_point);
        const __m128i vprod23_odd = _mm_mullo_epi16(vxi23, vxk23);
        const __m128i vprod23_even = _mm_mulhi_epi16(vxi23, vxk23);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod23_odd, vprod23_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod23_odd, vprod23_even));

        w = (const void*)((uintptr_t)w + 80);
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_loadu_si128((__m128i*)outacc));
        vacc_hi =
            _mm_add_epi32(vacc_hi, _mm_loadu_si128((__m128i*)(outacc + 4)));
        _mm_storeu_si128((__m128i*)outacc, vacc_lo);
        outacc += 4;
        _mm_storeu_si128((__m128i*)outacc, vacc_hi);
        outacc += 4;
      }
      if (c != 0) {
        const size_t i_predecrement = 8 - c;
        const __m128i vi_shift = _mm_cvtsi32_si128(8 * i_predecrement);
        i00 -= i_predecrement;
        i01 -= i_predecrement;
        i02 -= i_predecrement;
        i10 -= i_predecrement;
        i11 -= i_predecrement;
        i12 -= i_predecrement;
        i20 -= i_predecrement;
        i21 -= i_predecrement;
        i22 -= i_predecrement;
        i23 -= i_predecrement;

        const __m128i vi00 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i00), vi_shift);
        const __m128i vxi00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi00, vzero), vinput_zero_point);
        const __m128i vk00 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w));
        const __m128i vxk00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk00, vzero), vkernel_zero_point);
        const __m128i vprod00_odd = _mm_mullo_epi16(vxi00, vxk00);
        const __m128i vprod00_even = _mm_mulhi_epi16(vxi00, vxk00);
        __m128i vacc_lo = _mm_unpacklo_epi16(vprod00_odd, vprod00_even);
        __m128i vacc_hi = _mm_unpackhi_epi16(vprod00_odd, vprod00_even);

        const __m128i vi01 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i01), vi_shift);
        const __m128i vxi01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi01, vzero), vinput_zero_point);
        const __m128i vk01 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
        const __m128i vxk01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk01, vzero), vkernel_zero_point);
        const __m128i vprod01_odd = _mm_mullo_epi16(vxi01, vxk01);
        const __m128i vprod01_even = _mm_mulhi_epi16(vxi01, vxk01);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod01_odd, vprod01_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod01_odd, vprod01_even));

        const __m128i vi02 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i02), vi_shift);
        const __m128i vxi02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi02, vzero), vinput_zero_point);
        const __m128i vk02 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
        const __m128i vxk02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk02, vzero), vkernel_zero_point);
        const __m128i vprod02_odd = _mm_mullo_epi16(vxi02, vxk02);
        const __m128i vprod02_even = _mm_mulhi_epi16(vxi02, vxk02);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod02_odd, vprod02_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod02_odd, vprod02_even));

        const __m128i vi10 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i10), vi_shift);
        const __m128i vxi10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi10, vzero), vinput_zero_point);
        const __m128i vk10 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
        const __m128i vxk10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk10, vzero), vkernel_zero_point);
        const __m128i vprod10_odd = _mm_mullo_epi16(vxi10, vxk10);
        const __m128i vprod10_even = _mm_mulhi_epi16(vxi10, vxk10);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod10_odd, vprod10_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod10_odd, vprod10_even));

        const __m128i vi11 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i11), vi_shift);
        const __m128i vxi11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi11, vzero), vinput_zero_point);
        const __m128i vk11 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 32));
        const __m128i vxk11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk11, vzero), vkernel_zero_point);
        const __m128i vprod11_odd = _mm_mullo_epi16(vxi11, vxk11);
        const __m128i vprod11_even = _mm_mulhi_epi16(vxi11, vxk11);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod11_odd, vprod11_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod11_odd, vprod11_even));

        const __m128i vi12 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i12), vi_shift);
        const __m128i vxi12 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi12, vzero), vinput_zero_point);
        const __m128i vk12 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 40));
        const __m128i vxk12 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk12, vzero), vkernel_zero_point);
        const __m128i vprod12_odd = _mm_mullo_epi16(vxi12, vxk12);
        const __m128i vprod12_even = _mm_mulhi_epi16(vxi12, vxk12);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod12_odd, vprod12_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod12_odd, vprod12_even));

        const __m128i vi20 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i20), vi_shift);
        const __m128i vxi20 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi20, vzero), vinput_zero_point);
        const __m128i vk20 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 48));
        const __m128i vxk20 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk20, vzero), vkernel_zero_point);
        const __m128i vprod20_odd = _mm_mullo_epi16(vxi20, vxk20);
        const __m128i vprod20_even = _mm_mulhi_epi16(vxi20, vxk20);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod20_odd, vprod20_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod20_odd, vprod20_even));

        const __m128i vi21 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i21), vi_shift);
        const __m128i vxi21 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi21, vzero), vinput_zero_point);
        const __m128i vk21 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 56));
        const __m128i vxk21 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk21, vzero), vkernel_zero_point);
        const __m128i vprod21_odd = _mm_mullo_epi16(vxi21, vxk21);
        const __m128i vprod21_even = _mm_mulhi_epi16(vxi21, vxk21);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod21_odd, vprod21_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod21_odd, vprod21_even));

        const __m128i vi22 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i22), vi_shift);
        const __m128i vxi22 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi22, vzero), vinput_zero_point);
        const __m128i vk22 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 64));
        const __m128i vxk22 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk22, vzero), vkernel_zero_point);
        const __m128i vprod22_odd = _mm_mullo_epi16(vxi22, vxk22);
        const __m128i vprod22_even = _mm_mulhi_epi16(vxi22, vxk22);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod22_odd, vprod22_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod22_odd, vprod22_even));

        const __m128i vi23 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i23), vi_shift);
        const __m128i vxi23 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi23, vzero), vinput_zero_point);
        const __m128i vk23 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 72));
        const __m128i vxk23 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk23, vzero), vkernel_zero_point);
        const __m128i vprod23_odd = _mm_mullo_epi16(vxi23, vxk23);
        const __m128i vprod23_even = _mm_mulhi_epi16(vxi23, vxk23);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod23_odd, vprod23_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod23_odd, vprod23_even));

        w = (const void*)((uintptr_t)w + 80);
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_loadu_si128((__m128i*)outacc));
        vacc_hi =
            _mm_add_epi32(vacc_hi, _mm_loadu_si128((__m128i*)(outacc + 4)));
        _mm_storeu_si128((__m128i*)outacc, vacc_lo);
        outacc += 4;
        _mm_storeu_si128((__m128i*)outacc, vacc_hi);
        outacc += 4;
      }
    }
    {
      const uint8_t* i00 = input[20];
      const uint8_t* i01 = input[21];
      const uint8_t* i02 = input[22];
      const uint8_t* i10 = input[23];
      const uint8_t* i11 = input[24];
      input = (const uint8_t**)((uintptr_t)input + input_stride);
      outacc = outacc32;
      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const __m128i vi00 = _mm_loadl_epi64((const __m128i*)i00);
        i00 += 8;
        const __m128i vxi00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi00, vzero), vinput_zero_point);
        const __m128i vk00 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w));
        const __m128i vxk00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk00, vzero), vkernel_zero_point);
        const __m128i vprod00_odd = _mm_mullo_epi16(vxi00, vxk00);
        const __m128i vprod00_even = _mm_mulhi_epi16(vxi00, vxk00);
        __m128i vacc_lo = _mm_unpacklo_epi16(vprod00_odd, vprod00_even);
        __m128i vacc_hi = _mm_unpackhi_epi16(vprod00_odd, vprod00_even);

        const __m128i vi01 = _mm_loadl_epi64((const __m128i*)i01);
        i01 += 8;
        const __m128i vxi01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi01, vzero), vinput_zero_point);
        const __m128i vk01 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
        const __m128i vxk01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk01, vzero), vkernel_zero_point);
        const __m128i vprod01_odd = _mm_mullo_epi16(vxi01, vxk01);
        const __m128i vprod01_even = _mm_mulhi_epi16(vxi01, vxk01);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod01_odd, vprod01_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod01_odd, vprod01_even));

        const __m128i vi02 = _mm_loadl_epi64((const __m128i*)i02);
        i02 += 8;
        const __m128i vxi02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi02, vzero), vinput_zero_point);
        const __m128i vk02 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
        const __m128i vxk02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk02, vzero), vkernel_zero_point);
        const __m128i vprod02_odd = _mm_mullo_epi16(vxi02, vxk02);
        const __m128i vprod02_even = _mm_mulhi_epi16(vxi02, vxk02);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod02_odd, vprod02_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod02_odd, vprod02_even));

        const __m128i vi10 = _mm_loadl_epi64((const __m128i*)i10);
        i10 += 8;
        const __m128i vxi10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi10, vzero), vinput_zero_point);
        const __m128i vk10 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
        const __m128i vxk10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk10, vzero), vkernel_zero_point);
        const __m128i vprod10_odd = _mm_mullo_epi16(vxi10, vxk10);
        const __m128i vprod10_even = _mm_mulhi_epi16(vxi10, vxk10);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod10_odd, vprod10_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod10_odd, vprod10_even));

        const __m128i vi11 = _mm_loadl_epi64((const __m128i*)i11);
        i11 += 8;
        const __m128i vxi11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi11, vzero), vinput_zero_point);
        const __m128i vk11 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 32));
        const __m128i vxk11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk11, vzero), vkernel_zero_point);
        const __m128i vprod11_odd = _mm_mullo_epi16(vxi11, vxk11);
        const __m128i vprod11_even = _mm_mulhi_epi16(vxi11, vxk11);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod11_odd, vprod11_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod11_odd, vprod11_even));

        w = (const void*)((uintptr_t)w + 40);

        vacc_lo = _mm_add_epi32(vacc_lo, _mm_loadu_si128((__m128i*)outacc));
        vacc_hi =
            _mm_add_epi32(vacc_hi, _mm_loadu_si128((__m128i*)(outacc + 4)));
        outacc += 8;

        const __m128i vmultiplier = _mm_load_si128(
            (const __m128i*)quantization_params->sse2.multiplier);
        const __m128i vrounding =
            _mm_load_si128((const __m128i*)quantization_params->sse2.rounding);

        const __m128i vnmask_lo0123 =
            _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo);
        const __m128i vnmask_hi0123 =
            _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi);

        const __m128i vabsacc_lo0123 =
            _mm_sub_epi32(_mm_xor_si128(vacc_lo, vnmask_lo0123), vnmask_lo0123);
        const __m128i vabsacc_hi0123 =
            _mm_sub_epi32(_mm_xor_si128(vacc_hi, vnmask_hi0123), vnmask_hi0123);

        const __m128i vabsacc_lo1032 =
            _mm_shuffle_epi32(vabsacc_lo0123, _MM_SHUFFLE(2, 3, 0, 1));
        const __m128i vabsacc_hi1032 =
            _mm_shuffle_epi32(vabsacc_hi0123, _MM_SHUFFLE(2, 3, 0, 1));

        const __m128i vabsprod_lo02 =
            _mm_mul_epu32(vabsacc_lo0123, vmultiplier);
        const __m128i vabsprod_hi02 =
            _mm_mul_epu32(vabsacc_hi0123, vmultiplier);

        const __m128i vnmask_lo02 =
            _mm_shuffle_epi32(vnmask_lo0123, _MM_SHUFFLE(2, 2, 0, 0));
        const __m128i vnmask_hi02 =
            _mm_shuffle_epi32(vnmask_hi0123, _MM_SHUFFLE(2, 2, 0, 0));

        const __m128i vprod_lo02 = _mm_sub_epi64(
            _mm_xor_si128(vabsprod_lo02, vnmask_lo02), vnmask_lo02);
        const __m128i vprod_hi02 = _mm_sub_epi64(
            _mm_xor_si128(vabsprod_hi02, vnmask_hi02), vnmask_hi02);

        const __m128i vq31prod_lo02 =
            _mm_srli_epi64(_mm_add_epi64(vprod_lo02, vrounding), 31);
        const __m128i vq31prod_hi02 =
            _mm_srli_epi64(_mm_add_epi64(vprod_hi02, vrounding), 31);

        const __m128i vabsprod_lo13 =
            _mm_mul_epu32(vabsacc_lo1032, vmultiplier);
        const __m128i vabsprod_hi13 =
            _mm_mul_epu32(vabsacc_hi1032, vmultiplier);

        const __m128i vnmask_lo13 =
            _mm_shuffle_epi32(vnmask_lo0123, _MM_SHUFFLE(3, 3, 1, 1));
        const __m128i vnmask_hi13 =
            _mm_shuffle_epi32(vnmask_hi0123, _MM_SHUFFLE(3, 3, 1, 1));

        const __m128i vprod_lo13 = _mm_sub_epi64(
            _mm_xor_si128(vabsprod_lo13, vnmask_lo13), vnmask_lo13);
        const __m128i vprod_hi13 = _mm_sub_epi64(
            _mm_xor_si128(vabsprod_hi13, vnmask_hi13), vnmask_hi13);

        const __m128i vq31prod_lo13 =
            _mm_srli_epi64(_mm_add_epi64(vprod_lo13, vrounding), 31);
        const __m128i vq31prod_hi13 =
            _mm_srli_epi64(_mm_add_epi64(vprod_hi13, vrounding), 31);

        const __m128i vq31prod_lo0213 = _mm_castps_si128(_mm_shuffle_ps(
            _mm_castsi128_ps(vq31prod_lo02),
            _mm_castsi128_ps(vq31prod_lo13),
            _MM_SHUFFLE(2, 0, 2, 0)));
        const __m128i vq31prod_hi0213 = _mm_castps_si128(_mm_shuffle_ps(
            _mm_castsi128_ps(vq31prod_hi02),
            _mm_castsi128_ps(vq31prod_hi13),
            _MM_SHUFFLE(2, 0, 2, 0)));

        const __m128i vq31prod_lo0123 =
            _mm_shuffle_epi32(vq31prod_lo0213, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vq31prod_hi0123 =
            _mm_shuffle_epi32(vq31prod_hi0213, _MM_SHUFFLE(3, 1, 2, 0));

        const __m128i vremainder_mask = _mm_load_si128(
            (const __m128i*)quantization_params->sse2.remainder_mask);

        const __m128i vrem_lo0123 = _mm_add_epi32(
            _mm_and_si128(vq31prod_lo0123, vremainder_mask),
            _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod_lo0123));
        const __m128i vrem_hi0123 = _mm_add_epi32(
            _mm_and_si128(vq31prod_hi0123, vremainder_mask),
            _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod_hi0123));

        const __m128i vremainder_threshold = _mm_load_si128(
            (const __m128i*)quantization_params->sse2.remainder_threshold);
        const __m128i vshift =
            _mm_load_si128((const __m128i*)quantization_params->sse2.shift);

        const __m128i vout_lo = _mm_sub_epi32(
            _mm_sra_epi32(vq31prod_lo0123, vshift),
            _mm_cmpgt_epi32(vrem_lo0123, vremainder_threshold));
        const __m128i vout_hi = _mm_sub_epi32(
            _mm_sra_epi32(vq31prod_hi0123, vshift),
            _mm_cmpgt_epi32(vrem_hi0123, vremainder_threshold));

        const __m128i voutput_zero_point = _mm_load_si128(
            (const __m128i*)quantization_params->sse2.output_zero_point);
        __m128i vout = _mm_adds_epi16(
            _mm_packs_epi32(vout_lo, vout_hi), voutput_zero_point);
        vout = _mm_packus_epi16(vout, vout);
        vout = _mm_max_epu8(
            vout,
            _mm_load_si128(
                (const __m128i*)quantization_params->sse2.output_min));
        vout = _mm_min_epu8(
            vout,
            _mm_load_si128(
                (const __m128i*)quantization_params->sse2.output_max));

        _mm_storel_epi64((__m128i*)output, vout);
        output += 8;
      }
      if (c != 0) {
        const size_t i_predecrement = 8 - c;
        const __m128i vi_shift = _mm_cvtsi32_si128(8 * i_predecrement);
        i00 -= i_predecrement;
        i01 -= i_predecrement;
        i02 -= i_predecrement;
        i10 -= i_predecrement;
        i11 -= i_predecrement;

        const __m128i vi00 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i00), vi_shift);
        const __m128i vxi00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi00, vzero), vinput_zero_point);
        const __m128i vk00 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w));
        const __m128i vxk00 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk00, vzero), vkernel_zero_point);
        const __m128i vprod00_odd = _mm_mullo_epi16(vxi00, vxk00);
        const __m128i vprod00_even = _mm_mulhi_epi16(vxi00, vxk00);
        __m128i vacc_lo = _mm_unpacklo_epi16(vprod00_odd, vprod00_even);
        __m128i vacc_hi = _mm_unpackhi_epi16(vprod00_odd, vprod00_even);

        const __m128i vi01 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i01), vi_shift);
        const __m128i vxi01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi01, vzero), vinput_zero_point);
        const __m128i vk01 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
        const __m128i vxk01 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk01, vzero), vkernel_zero_point);
        const __m128i vprod01_odd = _mm_mullo_epi16(vxi01, vxk01);
        const __m128i vprod01_even = _mm_mulhi_epi16(vxi01, vxk01);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod01_odd, vprod01_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod01_odd, vprod01_even));

        const __m128i vi02 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i02), vi_shift);
        const __m128i vxi02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi02, vzero), vinput_zero_point);
        const __m128i vk02 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
        const __m128i vxk02 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk02, vzero), vkernel_zero_point);
        const __m128i vprod02_odd = _mm_mullo_epi16(vxi02, vxk02);
        const __m128i vprod02_even = _mm_mulhi_epi16(vxi02, vxk02);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod02_odd, vprod02_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod02_odd, vprod02_even));

        const __m128i vi10 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i10), vi_shift);
        const __m128i vxi10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi10, vzero), vinput_zero_point);
        const __m128i vk10 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
        const __m128i vxk10 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk10, vzero), vkernel_zero_point);
        const __m128i vprod10_odd = _mm_mullo_epi16(vxi10, vxk10);
        const __m128i vprod10_even = _mm_mulhi_epi16(vxi10, vxk10);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod10_odd, vprod10_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod10_odd, vprod10_even));

        const __m128i vi11 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i11), vi_shift);
        const __m128i vxi11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vi11, vzero), vinput_zero_point);
        const __m128i vk11 =
            _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 32));
        const __m128i vxk11 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vk11, vzero), vkernel_zero_point);
        const __m128i vprod11_odd = _mm_mullo_epi16(vxi11, vxk11);
        const __m128i vprod11_even = _mm_mulhi_epi16(vxi11, vxk11);
        vacc_lo = _mm_add_epi32(
            vacc_lo, _mm_unpacklo_epi16(vprod11_odd, vprod11_even));
        vacc_hi = _mm_add_epi32(
            vacc_hi, _mm_unpackhi_epi16(vprod11_odd, vprod11_even));

        vacc_lo = _mm_add_epi32(vacc_lo, _mm_loadu_si128((__m128i*)outacc));
        vacc_hi =
            _mm_add_epi32(vacc_hi, _mm_loadu_si128((__m128i*)(outacc + 4)));
        outacc += 8;

        const __m128i vmultiplier = _mm_load_si128(
            (const __m128i*)quantization_params->sse2.multiplier);
        const __m128i vrounding =
            _mm_load_si128((const __m128i*)quantization_params->sse2.rounding);

        const __m128i vnmask_lo0123 =
            _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo);
        const __m128i vnmask_hi0123 =
            _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi);

        const __m128i vabsacc_lo0123 =
            _mm_sub_epi32(_mm_xor_si128(vacc_lo, vnmask_lo0123), vnmask_lo0123);
        const __m128i vabsacc_hi0123 =
            _mm_sub_epi32(_mm_xor_si128(vacc_hi, vnmask_hi0123), vnmask_hi0123);

        const __m128i vabsacc_lo1032 =
            _mm_shuffle_epi32(vabsacc_lo0123, _MM_SHUFFLE(2, 3, 0, 1));
        const __m128i vabsacc_hi1032 =
            _mm_shuffle_epi32(vabsacc_hi0123, _MM_SHUFFLE(2, 3, 0, 1));

        const __m128i vabsprod_lo02 =
            _mm_mul_epu32(vabsacc_lo0123, vmultiplier);
        const __m128i vabsprod_hi02 =
            _mm_mul_epu32(vabsacc_hi0123, vmultiplier);

        const __m128i vnmask_lo02 =
            _mm_shuffle_epi32(vnmask_lo0123, _MM_SHUFFLE(2, 2, 0, 0));
        const __m128i vnmask_hi02 =
            _mm_shuffle_epi32(vnmask_hi0123, _MM_SHUFFLE(2, 2, 0, 0));

        const __m128i vprod_lo02 = _mm_sub_epi64(
            _mm_xor_si128(vabsprod_lo02, vnmask_lo02), vnmask_lo02);
        const __m128i vprod_hi02 = _mm_sub_epi64(
            _mm_xor_si128(vabsprod_hi02, vnmask_hi02), vnmask_hi02);

        const __m128i vq31prod_lo02 =
            _mm_srli_epi64(_mm_add_epi64(vprod_lo02, vrounding), 31);
        const __m128i vq31prod_hi02 =
            _mm_srli_epi64(_mm_add_epi64(vprod_hi02, vrounding), 31);

        const __m128i vabsprod_lo13 =
            _mm_mul_epu32(vabsacc_lo1032, vmultiplier);
        const __m128i vabsprod_hi13 =
            _mm_mul_epu32(vabsacc_hi1032, vmultiplier);

        const __m128i vnmask_lo13 =
            _mm_shuffle_epi32(vnmask_lo0123, _MM_SHUFFLE(3, 3, 1, 1));
        const __m128i vnmask_hi13 =
            _mm_shuffle_epi32(vnmask_hi0123, _MM_SHUFFLE(3, 3, 1, 1));

        const __m128i vprod_lo13 = _mm_sub_epi64(
            _mm_xor_si128(vabsprod_lo13, vnmask_lo13), vnmask_lo13);
        const __m128i vprod_hi13 = _mm_sub_epi64(
            _mm_xor_si128(vabsprod_hi13, vnmask_hi13), vnmask_hi13);

        const __m128i vq31prod_lo13 =
            _mm_srli_epi64(_mm_add_epi64(vprod_lo13, vrounding), 31);
        const __m128i vq31prod_hi13 =
            _mm_srli_epi64(_mm_add_epi64(vprod_hi13, vrounding), 31);

        const __m128i vq31prod_lo0213 = _mm_castps_si128(_mm_shuffle_ps(
            _mm_castsi128_ps(vq31prod_lo02),
            _mm_castsi128_ps(vq31prod_lo13),
            _MM_SHUFFLE(2, 0, 2, 0)));
        const __m128i vq31prod_hi0213 = _mm_castps_si128(_mm_shuffle_ps(
            _mm_castsi128_ps(vq31prod_hi02),
            _mm_castsi128_ps(vq31prod_hi13),
            _MM_SHUFFLE(2, 0, 2, 0)));

        const __m128i vq31prod_lo0123 =
            _mm_shuffle_epi32(vq31prod_lo0213, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vq31prod_hi0123 =
            _mm_shuffle_epi32(vq31prod_hi0213, _MM_SHUFFLE(3, 1, 2, 0));

        const __m128i vremainder_mask = _mm_load_si128(
            (const __m128i*)quantization_params->sse2.remainder_mask);

        const __m128i vrem_lo0123 = _mm_add_epi32(
            _mm_and_si128(vq31prod_lo0123, vremainder_mask),
            _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod_lo0123));
        const __m128i vrem_hi0123 = _mm_add_epi32(
            _mm_and_si128(vq31prod_hi0123, vremainder_mask),
            _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod_hi0123));

        const __m128i vremainder_threshold = _mm_load_si128(
            (const __m128i*)quantization_params->sse2.remainder_threshold);
        const __m128i vshift =
            _mm_load_si128((const __m128i*)quantization_params->sse2.shift);

        const __m128i vout_lo = _mm_sub_epi32(
            _mm_sra_epi32(vq31prod_lo0123, vshift),
            _mm_cmpgt_epi32(vrem_lo0123, vremainder_threshold));
        const __m128i vout_hi = _mm_sub_epi32(
            _mm_sra_epi32(vq31prod_hi0123, vshift),
            _mm_cmpgt_epi32(vrem_hi0123, vremainder_threshold));

        const __m128i voutput_zero_point = _mm_load_si128(
            (const __m128i*)quantization_params->sse2.output_zero_point);
        __m128i vout = _mm_adds_epi16(
            _mm_packs_epi32(vout_lo, vout_hi), voutput_zero_point);
        vout = _mm_packus_epi16(vout, vout);
        vout = _mm_max_epu8(
            vout,
            _mm_load_si128(
                (const __m128i*)quantization_params->sse2.output_min));
        vout = _mm_min_epu8(
            vout,
            _mm_load_si128(
                (const __m128i*)quantization_params->sse2.output_max));

        if (c & 4) {
          *((uint32_t*)output) = (uint32_t)_mm_cvtsi128_si32(vout);
          output += 4;
          vout = _mm_srli_epi64(vout, 32);
        }
        if (c & 2) {
          *((uint16_t*)output) = (uint16_t)_mm_extract_epi16(vout, 0);
          output += 2;
          vout = _mm_srli_epi32(vout, 16);
        }
        if (c & 1) {
          *((uint8_t*)output) = (uint8_t)_mm_cvtsi128_si32(vout);
          output += 1;
        }
      }
    }
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--output_width != 0);
}
