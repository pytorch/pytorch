/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <altivec.h>

#include <qnnpack/q8dwconv.h>

void pytorch_q8dwconv_ukernel_mp16x25__vsx(
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
  const vector short va_zero_point =
      vec_splats(quantization_params->vsx.input_zero_point);
  const vector short vkernel_zero_point =
      vec_splats((int16_t)quantization_params->vsx.kernel_zero_points[0]);
  const vector float vmultiplier =
      vec_splats(quantization_params->vsx.requantization_scales[0]);
  const vector short voutput_zero_point =
      vec_splats(quantization_params->vsx.output_zero_point);
  const vector unsigned char vmin =
      vec_splats(quantization_params->vsx.output_min);
  const vector unsigned char vmax =
      vec_splats(quantization_params->vsx.output_max);

  const vector unsigned char vzero = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Iterate over the output columns
  do {
    int32_t* outacc = outacc32;
    const void* w = weights;
    //Compute the first 10 elements of the filters (out of 25)
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
      // Iterate over 16 input channels per iteration
      for (; c >= 16; c -= 16) {
        // Initialize the accumulators with bias
        vector int vacc_hi_hi = vec_xl(0, (int32_t*)w);
        vector int vacc_hi_lo = vec_xl(16, (int32_t*)w);
        vector int vacc_lo_hi = vec_xl(32, (int32_t*)w);
        vector int vacc_lo_lo = vec_xl(48, (int32_t*)w);

        /* [1/25] Load 16 input elements over the channels, add zero point to
         * the input and weight vectors, and accumulate the products */
        const vector unsigned char vi00 = vec_xl(0, i00);
        i00 += 16;
        const vector short vxi00_hi =
            vec_sub((vector short)vec_mergeh(vi00, vzero), va_zero_point);
        const vector short vxi00_lo =
            vec_sub((vector short)vec_mergel(vi00, vzero), va_zero_point);
        const vector unsigned char vk00 = vec_xl(64, (uint8_t*)w);
        const vector short vxk00_hi =
            vec_sub((vector short)vec_mergeh(vk00, vzero), vkernel_zero_point);
        const vector short vxk00_lo =
            vec_sub((vector short)vec_mergel(vk00, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi00_hi), vec_unpackh(vxk00_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi00_hi), vec_unpackl(vxk00_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi00_lo), vec_unpackh(vxk00_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi00_lo), vec_unpackl(vxk00_lo)));

        // [2/25] Next inputs/filters
        const vector unsigned char vi01 = vec_xl(0, i01);
        i01 += 16;
        const vector short vxi01_hi =
            vec_sub((vector short)vec_mergeh(vi01, vzero), va_zero_point);
        const vector short vxi01_lo =
            vec_sub((vector short)vec_mergel(vi01, vzero), va_zero_point);
        const vector unsigned char vk01 = vec_xl(80, (uint8_t*)w);
        const vector short vxk01_hi =
            vec_sub((vector short)vec_mergeh(vk01, vzero), vkernel_zero_point);
        const vector short vxk01_lo =
            vec_sub((vector short)vec_mergel(vk01, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi01_hi), vec_unpackh(vxk01_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi01_hi), vec_unpackl(vxk01_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi01_lo), vec_unpackh(vxk01_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi01_lo), vec_unpackl(vxk01_lo)));

        // [3/25] Next inputs/filters
        const vector unsigned char vi02 = vec_xl(0, i02);
        i02 += 16;
        const vector short vxi02_hi =
            vec_sub((vector short)vec_mergeh(vi02, vzero), va_zero_point);
        const vector short vxi02_lo =
            vec_sub((vector short)vec_mergel(vi02, vzero), va_zero_point);
        const vector unsigned char vk02 = vec_xl(96, (uint8_t*)w);
        const vector short vxk02_hi =
            vec_sub((vector short)vec_mergeh(vk02, vzero), vkernel_zero_point);
        const vector short vxk02_lo =
            vec_sub((vector short)vec_mergel(vk02, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi02_hi), vec_unpackh(vxk02_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi02_hi), vec_unpackl(vxk02_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi02_lo), vec_unpackh(vxk02_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi02_lo), vec_unpackl(vxk02_lo)));

        // [4/25] Next inputs/filters
        const vector unsigned char vi10 = vec_xl(0, i10);
        i10 += 16;
        const vector short vxi10_hi =
            vec_sub((vector short)vec_mergeh(vi10, vzero), va_zero_point);
        const vector short vxi10_lo =
            vec_sub((vector short)vec_mergel(vi10, vzero), va_zero_point);
        const vector unsigned char vk10 = vec_xl(112, (uint8_t*)w);
        const vector short vxk10_hi =
            vec_sub((vector short)vec_mergeh(vk10, vzero), vkernel_zero_point);
        const vector short vxk10_lo =
            vec_sub((vector short)vec_mergel(vk10, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi10_hi), vec_unpackh(vxk10_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi10_hi), vec_unpackl(vxk10_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi10_lo), vec_unpackh(vxk10_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi10_lo), vec_unpackl(vxk10_lo)));

        // [5/25] Next inputs/filters
        const vector unsigned char vi11 = vec_xl(0, i11);
        i11 += 16;
        const vector short vxi11_hi =
            vec_sub((vector short)vec_mergeh(vi11, vzero), va_zero_point);
        const vector short vxi11_lo =
            vec_sub((vector short)vec_mergel(vi11, vzero), va_zero_point);
        const vector unsigned char vk11 = vec_xl(128, (uint8_t*)w);
        const vector short vxk11_hi =
            vec_sub((vector short)vec_mergeh(vk11, vzero), vkernel_zero_point);
        const vector short vxk11_lo =
            vec_sub((vector short)vec_mergel(vk11, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi11_hi), vec_unpackh(vxk11_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi11_hi), vec_unpackl(vxk11_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi11_lo), vec_unpackh(vxk11_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi11_lo), vec_unpackl(vxk11_lo)));

        // [6/25] Next inputs/filters
        const vector unsigned char vi12 = vec_xl(0, i12);
        i12 += 16;
        const vector short vxi12_hi =
            vec_sub((vector short)vec_mergeh(vi12, vzero), va_zero_point);
        const vector short vxi12_lo =
            vec_sub((vector short)vec_mergel(vi12, vzero), va_zero_point);
        const vector unsigned char vk12 = vec_xl(144, (uint8_t*)w);
        const vector short vxk12_hi =
            vec_sub((vector short)vec_mergeh(vk12, vzero), vkernel_zero_point);
        const vector short vxk12_lo =
            vec_sub((vector short)vec_mergel(vk12, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi12_hi), vec_unpackh(vxk12_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi12_hi), vec_unpackl(vxk12_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi12_lo), vec_unpackh(vxk12_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi12_lo), vec_unpackl(vxk12_lo)));

        // [7/25] Next inputs/filters
        const vector unsigned char vi20 = vec_xl(0, i20);
        i20 += 16;
        const vector short vxi20_hi =
            vec_sub((vector short)vec_mergeh(vi20, vzero), va_zero_point);
        const vector short vxi20_lo =
            vec_sub((vector short)vec_mergel(vi20, vzero), va_zero_point);
        const vector unsigned char vk20 = vec_xl(160, (uint8_t*)w);
        const vector short vxk20_hi =
            vec_sub((vector short)vec_mergeh(vk20, vzero), vkernel_zero_point);
        const vector short vxk20_lo =
            vec_sub((vector short)vec_mergel(vk20, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi20_hi), vec_unpackh(vxk20_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi20_hi), vec_unpackl(vxk20_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi20_lo), vec_unpackh(vxk20_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi20_lo), vec_unpackl(vxk20_lo)));

        // [8/25] Next inputs/filters
        const vector unsigned char vi21 = vec_xl(0, i21);
        i21 += 16;
        const vector short vxi21_hi =
            vec_sub((vector short)vec_mergeh(vi21, vzero), va_zero_point);
        const vector short vxi21_lo =
            vec_sub((vector short)vec_mergel(vi21, vzero), va_zero_point);
        const vector unsigned char vk21 = vec_xl(176, (uint8_t*)w);
        const vector short vxk21_hi =
            vec_sub((vector short)vec_mergeh(vk21, vzero), vkernel_zero_point);
        const vector short vxk21_lo =
            vec_sub((vector short)vec_mergel(vk21, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi21_hi), vec_unpackh(vxk21_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi21_hi), vec_unpackl(vxk21_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi21_lo), vec_unpackh(vxk21_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi21_lo), vec_unpackl(vxk21_lo)));

        // [9/25] Next inputs/filters
        const vector unsigned char vi22 = vec_xl(0, i22);
        i22 += 16;
        const vector short vxi22_hi =
            vec_sub((vector short)vec_mergeh(vi22, vzero), va_zero_point);
        const vector short vxi22_lo =
            vec_sub((vector short)vec_mergel(vi22, vzero), va_zero_point);
        const vector unsigned char vk22 = vec_xl(192, (uint8_t*)w);
        const vector short vxk22_hi =
            vec_sub((vector short)vec_mergeh(vk22, vzero), vkernel_zero_point);
        const vector short vxk22_lo =
            vec_sub((vector short)vec_mergel(vk22, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi22_hi), vec_unpackh(vxk22_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi22_hi), vec_unpackl(vxk22_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi22_lo), vec_unpackh(vxk22_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi22_lo), vec_unpackl(vxk22_lo)));

        // [10/25] Next inputs/filters
        const vector unsigned char vi23 = vec_xl(0, i23);
        i23 += 16;
        const vector short vxi23_hi =
            vec_sub((vector short)vec_mergeh(vi23, vzero), va_zero_point);
        const vector short vxi23_lo =
            vec_sub((vector short)vec_mergel(vi23, vzero), va_zero_point);
        const vector unsigned char vk23 = vec_xl(208, (uint8_t*)w);
        const vector short vxk23_hi =
            vec_sub((vector short)vec_mergeh(vk23, vzero), vkernel_zero_point);
        const vector short vxk23_lo =
            vec_sub((vector short)vec_mergel(vk23, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi23_hi), vec_unpackh(vxk23_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi23_hi), vec_unpackl(vxk23_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi23_lo), vec_unpackh(vxk23_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi23_lo), vec_unpackl(vxk23_lo)));

        w = (const void*)((uintptr_t)w + 224);

        // Store the partial results of the accumulators
        vec_xst(vacc_hi_hi, 0, outacc);
        vec_xst(vacc_hi_lo, 16, outacc);
        vec_xst(vacc_lo_hi, 32, outacc);
        vec_xst(vacc_lo_lo, 48, outacc);
        outacc += 16;
      }
      if (c != 0) {
        // Compute the remaining channels
        const size_t i_predecrement = 16 - c;
        const vector unsigned char vi_shift = {
            8 * i_predecrement, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        vector int vacc_hi_hi = vec_xl(0, (int32_t*)w);
        vector int vacc_hi_lo = vec_xl(16, (int32_t*)w);

        /* Load the remaining input elements over the channels, shift the
         * vector, add zero point to the input and weight vectors, and
         * accumulate the products */
        const vector unsigned char vi00 =
            vec_sro(vec_xl(-i_predecrement, i00), vi_shift);
        i00 += 16;
        const vector short vxi00_hi =
            vec_sub((vector short)vec_mergeh(vi00, vzero), va_zero_point);
        const vector unsigned char vk00 = vec_xl(64, (uint8_t*)w);
        const vector short vxk00_hi =
            vec_sub((vector short)vec_mergeh(vk00, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi00_hi), vec_unpackh(vxk00_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi00_hi), vec_unpackl(vxk00_hi)));

        const vector unsigned char vi01 =
            vec_sro(vec_xl(-i_predecrement, i01), vi_shift);
        i01 += 16;
        const vector short vxi01_hi =
            vec_sub((vector short)vec_mergeh(vi01, vzero), va_zero_point);
        const vector unsigned char vk01 = vec_xl(80, (uint8_t*)w);
        const vector short vxk01_hi =
            vec_sub((vector short)vec_mergeh(vk01, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi01_hi), vec_unpackh(vxk01_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi01_hi), vec_unpackl(vxk01_hi)));

        const vector unsigned char vi02 =
            vec_sro(vec_xl(-i_predecrement, i02), vi_shift);
        i02 += 16;
        const vector short vxi02_hi =
            vec_sub((vector short)vec_mergeh(vi02, vzero), va_zero_point);
        const vector unsigned char vk02 = vec_xl(96, (uint8_t*)w);
        const vector short vxk02_hi =
            vec_sub((vector short)vec_mergeh(vk02, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi02_hi), vec_unpackh(vxk02_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi02_hi), vec_unpackl(vxk02_hi)));

        const vector unsigned char vi10 =
            vec_sro(vec_xl(-i_predecrement, i10), vi_shift);
        i10 += 16;
        const vector short vxi10_hi =
            vec_sub((vector short)vec_mergeh(vi10, vzero), va_zero_point);
        const vector unsigned char vk10 = vec_xl(112, (uint8_t*)w);
        const vector short vxk10_hi =
            vec_sub((vector short)vec_mergeh(vk10, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi10_hi), vec_unpackh(vxk10_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi10_hi), vec_unpackl(vxk10_hi)));

        const vector unsigned char vi11 =
            vec_sro(vec_xl(-i_predecrement, i11), vi_shift);
        i11 += 16;
        const vector short vxi11_hi =
            vec_sub((vector short)vec_mergeh(vi11, vzero), va_zero_point);
        const vector unsigned char vk11 = vec_xl(128, (uint8_t*)w);
        const vector short vxk11_hi =
            vec_sub((vector short)vec_mergeh(vk11, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi11_hi), vec_unpackh(vxk11_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi11_hi), vec_unpackl(vxk11_hi)));

        const vector unsigned char vi12 =
            vec_sro(vec_xl(-i_predecrement, i12), vi_shift);
        i12 += 16;
        const vector short vxi12_hi =
            vec_sub((vector short)vec_mergeh(vi12, vzero), va_zero_point);
        const vector unsigned char vk12 = vec_xl(144, (uint8_t*)w);
        const vector short vxk12_hi =
            vec_sub((vector short)vec_mergeh(vk12, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi12_hi), vec_unpackh(vxk12_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi12_hi), vec_unpackl(vxk12_hi)));

        const vector unsigned char vi20 =
            vec_sro(vec_xl(-i_predecrement, i20), vi_shift);
        i20 += 16;
        const vector short vxi20_hi =
            vec_sub((vector short)vec_mergeh(vi20, vzero), va_zero_point);
        const vector unsigned char vk20 = vec_xl(160, (uint8_t*)w);
        const vector short vxk20_hi =
            vec_sub((vector short)vec_mergeh(vk20, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi20_hi), vec_unpackh(vxk20_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi20_hi), vec_unpackl(vxk20_hi)));

        const vector unsigned char vi21 =
            vec_sro(vec_xl(-i_predecrement, i21), vi_shift);
        i21 += 16;
        const vector short vxi21_hi =
            vec_sub((vector short)vec_mergeh(vi21, vzero), va_zero_point);
        const vector unsigned char vk21 = vec_xl(176, (uint8_t*)w);
        const vector short vxk21_hi =
            vec_sub((vector short)vec_mergeh(vk21, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi21_hi), vec_unpackh(vxk21_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi21_hi), vec_unpackl(vxk21_hi)));

        const vector unsigned char vi22 =
            vec_sro(vec_xl(-i_predecrement, i22), vi_shift);
        i22 += 16;
        const vector short vxi22_hi =
            vec_sub((vector short)vec_mergeh(vi22, vzero), va_zero_point);
        const vector unsigned char vk22 = vec_xl(192, (uint8_t*)w);
        const vector short vxk22_hi =
            vec_sub((vector short)vec_mergeh(vk22, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi22_hi), vec_unpackh(vxk22_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi22_hi), vec_unpackl(vxk22_hi)));

        const vector unsigned char vi23 =
            vec_sro(vec_xl(-i_predecrement, i23), vi_shift);
        i23 += 16;
        const vector short vxi23_hi =
            vec_sub((vector short)vec_mergeh(vi23, vzero), va_zero_point);
        const vector unsigned char vk23 = vec_xl(208, (uint8_t*)w);
        const vector short vxk23_hi =
            vec_sub((vector short)vec_mergeh(vk23, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi23_hi), vec_unpackh(vxk23_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi23_hi), vec_unpackl(vxk23_hi)));

        vec_xst(vacc_hi_hi, 0, outacc);
        vec_xst(vacc_hi_lo, 16, outacc);

        if (c > 8) {
          vector int vacc_lo_hi = vec_xl(32, (int32_t*)w);
          vector int vacc_lo_lo = vec_xl(48, (int32_t*)w);

          const vector short vxi00_lo =
              vec_sub((vector short)vec_mergel(vi00, vzero), va_zero_point);
          const vector short vxk00_lo = vec_sub(
              (vector short)vec_mergel(vk00, vzero), vkernel_zero_point);
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi00_lo), vec_unpackl(vxk00_lo)));
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi00_lo), vec_unpackh(vxk00_lo)));

          const vector short vxi01_lo =
              vec_sub((vector short)vec_mergel(vi01, vzero), va_zero_point);
          const vector short vxk01_lo = vec_sub(
              (vector short)vec_mergel(vk01, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi01_lo), vec_unpackh(vxk01_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi01_lo), vec_unpackl(vxk01_lo)));

          const vector short vxi02_lo =
              vec_sub((vector short)vec_mergel(vi02, vzero), va_zero_point);
          const vector short vxk02_lo = vec_sub(
              (vector short)vec_mergel(vk02, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi02_lo), vec_unpackh(vxk02_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi02_lo), vec_unpackl(vxk02_lo)));

          const vector short vxi10_lo =
              vec_sub((vector short)vec_mergel(vi10, vzero), va_zero_point);
          const vector short vxk10_lo = vec_sub(
              (vector short)vec_mergel(vk10, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi10_lo), vec_unpackh(vxk10_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi10_lo), vec_unpackl(vxk10_lo)));

          const vector short vxi11_lo =
              vec_sub((vector short)vec_mergel(vi11, vzero), va_zero_point);
          const vector short vxk11_lo = vec_sub(
              (vector short)vec_mergel(vk11, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi11_lo), vec_unpackh(vxk11_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi11_lo), vec_unpackl(vxk11_lo)));

          const vector short vxi12_lo =
              vec_sub((vector short)vec_mergel(vi12, vzero), va_zero_point);
          const vector short vxk12_lo = vec_sub(
              (vector short)vec_mergel(vk12, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi12_lo), vec_unpackh(vxk12_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi12_lo), vec_unpackl(vxk12_lo)));

          const vector short vxi20_lo =
              vec_sub((vector short)vec_mergel(vi20, vzero), va_zero_point);
          const vector short vxk20_lo = vec_sub(
              (vector short)vec_mergel(vk20, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi20_lo), vec_unpackh(vxk20_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi20_lo), vec_unpackl(vxk20_lo)));

          const vector short vxi21_lo =
              vec_sub((vector short)vec_mergel(vi21, vzero), va_zero_point);
          const vector short vxk21_lo = vec_sub(
              (vector short)vec_mergel(vk21, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi21_lo), vec_unpackh(vxk21_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi21_lo), vec_unpackl(vxk21_lo)));

          const vector short vxi22_lo =
              vec_sub((vector short)vec_mergel(vi22, vzero), va_zero_point);
          const vector short vxk22_lo = vec_sub(
              (vector short)vec_mergel(vk22, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi22_lo), vec_unpackh(vxk22_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi22_lo), vec_unpackl(vxk22_lo)));

          const vector short vxi23_lo =
              vec_sub((vector short)vec_mergel(vi23, vzero), va_zero_point);
          const vector short vxk23_lo = vec_sub(
              (vector short)vec_mergel(vk23, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi23_lo), vec_unpackh(vxk23_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi23_lo), vec_unpackl(vxk23_lo)));

          vec_xst(vacc_lo_hi, 32, outacc);
          vec_xst(vacc_lo_lo, 48, outacc);
        }

        w = (const void*)((uintptr_t)w + 224);
        outacc += 16;
      }
    }
    // Compute the next 10 elements of the filters (out of 25)
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
      for (; c >= 16; c -= 16) {
        /* Load the partial result into the accumulators to continue
         * accumulating the next products on them */
        vector int vacc_hi_hi = vec_xl(0, (int32_t*)outacc);
        vector int vacc_hi_lo = vec_xl(16, (int32_t*)outacc);
        vector int vacc_lo_hi = vec_xl(32, (int32_t*)outacc);
        vector int vacc_lo_lo = vec_xl(48, (int32_t*)outacc);

        /* [11/25] Load 16 input elements over the channels, add zero point to
         * the input and weight vectors and accumulate the products */
        const vector unsigned char vi00 = vec_xl(0, i00);
        i00 += 16;
        const vector short vxi00_hi =
            vec_sub((vector short)vec_mergeh(vi00, vzero), va_zero_point);
        const vector short vxi00_lo =
            vec_sub((vector short)vec_mergel(vi00, vzero), va_zero_point);
        const vector unsigned char vk00 = vec_xl(0, (uint8_t*)w);
        const vector short vxk00_hi =
            vec_sub((vector short)vec_mergeh(vk00, vzero), vkernel_zero_point);
        const vector short vxk00_lo =
            vec_sub((vector short)vec_mergel(vk00, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi00_hi), vec_unpackh(vxk00_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi00_hi), vec_unpackl(vxk00_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi00_lo), vec_unpackh(vxk00_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi00_lo), vec_unpackl(vxk00_lo)));

        // [12/25] Next inputs/filters
        const vector unsigned char vi01 = vec_xl(0, i01);
        i01 += 16;
        const vector short vxi01_hi =
            vec_sub((vector short)vec_mergeh(vi01, vzero), va_zero_point);
        const vector short vxi01_lo =
            vec_sub((vector short)vec_mergel(vi01, vzero), va_zero_point);
        const vector unsigned char vk01 = vec_xl(16, (uint8_t*)w);
        const vector short vxk01_hi =
            vec_sub((vector short)vec_mergeh(vk01, vzero), vkernel_zero_point);
        const vector short vxk01_lo =
            vec_sub((vector short)vec_mergel(vk01, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi01_hi), vec_unpackh(vxk01_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi01_hi), vec_unpackl(vxk01_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi01_lo), vec_unpackh(vxk01_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi01_lo), vec_unpackl(vxk01_lo)));

        // [13/25] Next inputs/filters
        const vector unsigned char vi02 = vec_xl(0, i02);
        i02 += 16;
        const vector short vxi02_hi =
            vec_sub((vector short)vec_mergeh(vi02, vzero), va_zero_point);
        const vector short vxi02_lo =
            vec_sub((vector short)vec_mergel(vi02, vzero), va_zero_point);
        const vector unsigned char vk02 = vec_xl(32, (uint8_t*)w);
        const vector short vxk02_hi =
            vec_sub((vector short)vec_mergeh(vk02, vzero), vkernel_zero_point);
        const vector short vxk02_lo =
            vec_sub((vector short)vec_mergel(vk02, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi02_hi), vec_unpackh(vxk02_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi02_hi), vec_unpackl(vxk02_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi02_lo), vec_unpackh(vxk02_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi02_lo), vec_unpackl(vxk02_lo)));

        // [14/25] Next inputs/filters
        const vector unsigned char vi10 = vec_xl(0, i10);
        i10 += 16;
        const vector short vxi10_hi =
            vec_sub((vector short)vec_mergeh(vi10, vzero), va_zero_point);
        const vector short vxi10_lo =
            vec_sub((vector short)vec_mergel(vi10, vzero), va_zero_point);
        const vector unsigned char vk10 = vec_xl(48, (uint8_t*)w);
        const vector short vxk10_hi =
            vec_sub((vector short)vec_mergeh(vk10, vzero), vkernel_zero_point);
        const vector short vxk10_lo =
            vec_sub((vector short)vec_mergel(vk10, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi10_hi), vec_unpackh(vxk10_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi10_hi), vec_unpackl(vxk10_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi10_lo), vec_unpackh(vxk10_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi10_lo), vec_unpackl(vxk10_lo)));

        // [15/25] Next inputs/filters
        const vector unsigned char vi11 = vec_xl(0, i11);
        i11 += 16;
        const vector short vxi11_hi =
            vec_sub((vector short)vec_mergeh(vi11, vzero), va_zero_point);
        const vector short vxi11_lo =
            vec_sub((vector short)vec_mergel(vi11, vzero), va_zero_point);
        const vector unsigned char vk11 = vec_xl(64, (uint8_t*)w);
        const vector short vxk11_hi =
            vec_sub((vector short)vec_mergeh(vk11, vzero), vkernel_zero_point);
        const vector short vxk11_lo =
            vec_sub((vector short)vec_mergel(vk11, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi11_hi), vec_unpackh(vxk11_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi11_hi), vec_unpackl(vxk11_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi11_lo), vec_unpackh(vxk11_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi11_lo), vec_unpackl(vxk11_lo)));

        // [16/25] Next inputs/filters
        const vector unsigned char vi12 = vec_xl(0, i12);
        i12 += 16;
        const vector short vxi12_hi =
            vec_sub((vector short)vec_mergeh(vi12, vzero), va_zero_point);
        const vector short vxi12_lo =
            vec_sub((vector short)vec_mergel(vi12, vzero), va_zero_point);
        const vector unsigned char vk12 = vec_xl(80, (uint8_t*)w);
        const vector short vxk12_hi =
            vec_sub((vector short)vec_mergeh(vk12, vzero), vkernel_zero_point);
        const vector short vxk12_lo =
            vec_sub((vector short)vec_mergel(vk12, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi12_hi), vec_unpackh(vxk12_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi12_hi), vec_unpackl(vxk12_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi12_lo), vec_unpackh(vxk12_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi12_lo), vec_unpackl(vxk12_lo)));

        // [17/25] Next inputs/filters
        const vector unsigned char vi20 = vec_xl(0, i20);
        i20 += 16;
        const vector short vxi20_hi =
            vec_sub((vector short)vec_mergeh(vi20, vzero), va_zero_point);
        const vector short vxi20_lo =
            vec_sub((vector short)vec_mergel(vi20, vzero), va_zero_point);
        const vector unsigned char vk20 = vec_xl(96, (uint8_t*)w);
        const vector short vxk20_hi =
            vec_sub((vector short)vec_mergeh(vk20, vzero), vkernel_zero_point);
        const vector short vxk20_lo =
            vec_sub((vector short)vec_mergel(vk20, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi20_hi), vec_unpackh(vxk20_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi20_hi), vec_unpackl(vxk20_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi20_lo), vec_unpackh(vxk20_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi20_lo), vec_unpackl(vxk20_lo)));

        // [18/25] Next inputs/filters
        const vector unsigned char vi21 = vec_xl(0, i21);
        i21 += 16;
        const vector short vxi21_hi =
            vec_sub((vector short)vec_mergeh(vi21, vzero), va_zero_point);
        const vector short vxi21_lo =
            vec_sub((vector short)vec_mergel(vi21, vzero), va_zero_point);
        const vector unsigned char vk21 = vec_xl(112, (uint8_t*)w);
        const vector short vxk21_hi =
            vec_sub((vector short)vec_mergeh(vk21, vzero), vkernel_zero_point);
        const vector short vxk21_lo =
            vec_sub((vector short)vec_mergel(vk21, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi21_hi), vec_unpackh(vxk21_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi21_hi), vec_unpackl(vxk21_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi21_lo), vec_unpackh(vxk21_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi21_lo), vec_unpackl(vxk21_lo)));

        // [19/25] Next inputs/filters
        const vector unsigned char vi22 = vec_xl(0, i22);
        i22 += 16;
        const vector short vxi22_hi =
            vec_sub((vector short)vec_mergeh(vi22, vzero), va_zero_point);
        const vector short vxi22_lo =
            vec_sub((vector short)vec_mergel(vi22, vzero), va_zero_point);
        const vector unsigned char vk22 = vec_xl(128, (uint8_t*)w);
        const vector short vxk22_hi =
            vec_sub((vector short)vec_mergeh(vk22, vzero), vkernel_zero_point);
        const vector short vxk22_lo =
            vec_sub((vector short)vec_mergel(vk22, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi22_hi), vec_unpackh(vxk22_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi22_hi), vec_unpackl(vxk22_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi22_lo), vec_unpackh(vxk22_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi22_lo), vec_unpackl(vxk22_lo)));

        // [20/25] Next inputs/filters
        const vector unsigned char vi23 = vec_xl(0, i23);
        i23 += 16;
        const vector short vxi23_hi =
            vec_sub((vector short)vec_mergeh(vi23, vzero), va_zero_point);
        const vector short vxi23_lo =
            vec_sub((vector short)vec_mergel(vi23, vzero), va_zero_point);
        const vector unsigned char vk23 = vec_xl(144, (uint8_t*)w);
        const vector short vxk23_hi =
            vec_sub((vector short)vec_mergeh(vk23, vzero), vkernel_zero_point);
        const vector short vxk23_lo =
            vec_sub((vector short)vec_mergel(vk23, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi23_hi), vec_unpackh(vxk23_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi23_hi), vec_unpackl(vxk23_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi23_lo), vec_unpackh(vxk23_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi23_lo), vec_unpackl(vxk23_lo)));

        w = (const void*)((uintptr_t)w + 160);

        // Store the partial results of the accumulators
        vec_xst(vacc_hi_hi, 0, outacc);
        vec_xst(vacc_hi_lo, 16, outacc);
        vec_xst(vacc_lo_hi, 32, outacc);
        vec_xst(vacc_lo_lo, 48, outacc);
        outacc += 16;
      }
      if (c != 0) {
        // Compute remaining channels
        const size_t i_predecrement = 16 - c;
        const vector unsigned char vi_shift = {
            8 * i_predecrement, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        vector int vacc_hi_hi = vec_xl(0, (int32_t*)outacc);
        vector int vacc_hi_lo = vec_xl(16, (int32_t*)outacc);

        const vector unsigned char vi00 =
            vec_sro(vec_xl(-i_predecrement, i00), vi_shift);
        i00 += 16;
        const vector short vxi00_hi =
            vec_sub((vector short)vec_mergeh(vi00, vzero), va_zero_point);
        const vector unsigned char vk00 = vec_xl(0, (uint8_t*)w);
        const vector short vxk00_hi =
            vec_sub((vector short)vec_mergeh(vk00, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi00_hi), vec_unpackh(vxk00_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi00_hi), vec_unpackl(vxk00_hi)));

        const vector unsigned char vi01 =
            vec_sro(vec_xl(-i_predecrement, i01), vi_shift);
        i01 += 16;
        const vector short vxi01_hi =
            vec_sub((vector short)vec_mergeh(vi01, vzero), va_zero_point);
        const vector unsigned char vk01 = vec_xl(16, (uint8_t*)w);
        const vector short vxk01_hi =
            vec_sub((vector short)vec_mergeh(vk01, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi01_hi), vec_unpackh(vxk01_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi01_hi), vec_unpackl(vxk01_hi)));

        const vector unsigned char vi02 =
            vec_sro(vec_xl(-i_predecrement, i02), vi_shift);
        i02 += 16;
        const vector short vxi02_hi =
            vec_sub((vector short)vec_mergeh(vi02, vzero), va_zero_point);
        const vector unsigned char vk02 = vec_xl(32, (uint8_t*)w);
        const vector short vxk02_hi =
            vec_sub((vector short)vec_mergeh(vk02, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi02_hi), vec_unpackh(vxk02_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi02_hi), vec_unpackl(vxk02_hi)));

        const vector unsigned char vi10 =
            vec_sro(vec_xl(-i_predecrement, i10), vi_shift);
        i10 += 16;
        const vector short vxi10_hi =
            vec_sub((vector short)vec_mergeh(vi10, vzero), va_zero_point);
        const vector unsigned char vk10 = vec_xl(48, (uint8_t*)w);
        const vector short vxk10_hi =
            vec_sub((vector short)vec_mergeh(vk10, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi10_hi), vec_unpackh(vxk10_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi10_hi), vec_unpackl(vxk10_hi)));

        const vector unsigned char vi11 =
            vec_sro(vec_xl(-i_predecrement, i11), vi_shift);
        i11 += 16;
        const vector short vxi11_hi =
            vec_sub((vector short)vec_mergeh(vi11, vzero), va_zero_point);
        const vector unsigned char vk11 = vec_xl(64, (uint8_t*)w);
        const vector short vxk11_hi =
            vec_sub((vector short)vec_mergeh(vk11, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi11_hi), vec_unpackh(vxk11_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi11_hi), vec_unpackl(vxk11_hi)));

        const vector unsigned char vi12 =
            vec_sro(vec_xl(-i_predecrement, i12), vi_shift);
        i12 += 16;
        const vector short vxi12_hi =
            vec_sub((vector short)vec_mergeh(vi12, vzero), va_zero_point);
        const vector unsigned char vk12 = vec_xl(80, (uint8_t*)w);
        const vector short vxk12_hi =
            vec_sub((vector short)vec_mergeh(vk12, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi12_hi), vec_unpackh(vxk12_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi12_hi), vec_unpackl(vxk12_hi)));

        const vector unsigned char vi20 =
            vec_sro(vec_xl(-i_predecrement, i20), vi_shift);
        i20 += 16;
        const vector short vxi20_hi =
            vec_sub((vector short)vec_mergeh(vi20, vzero), va_zero_point);
        const vector unsigned char vk20 = vec_xl(96, (uint8_t*)w);
        const vector short vxk20_hi =
            vec_sub((vector short)vec_mergeh(vk20, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi20_hi), vec_unpackh(vxk20_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi20_hi), vec_unpackl(vxk20_hi)));

        const vector unsigned char vi21 =
            vec_sro(vec_xl(-i_predecrement, i21), vi_shift);
        i21 += 16;
        const vector short vxi21_hi =
            vec_sub((vector short)vec_mergeh(vi21, vzero), va_zero_point);
        const vector unsigned char vk21 = vec_xl(112, (uint8_t*)w);
        const vector short vxk21_hi =
            vec_sub((vector short)vec_mergeh(vk21, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi21_hi), vec_unpackh(vxk21_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi21_hi), vec_unpackl(vxk21_hi)));

        const vector unsigned char vi22 =
            vec_sro(vec_xl(-i_predecrement, i22), vi_shift);
        i22 += 16;
        const vector short vxi22_hi =
            vec_sub((vector short)vec_mergeh(vi22, vzero), va_zero_point);
        const vector unsigned char vk22 = vec_xl(128, (uint8_t*)w);
        const vector short vxk22_hi =
            vec_sub((vector short)vec_mergeh(vk22, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi22_hi), vec_unpackh(vxk22_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi22_hi), vec_unpackl(vxk22_hi)));

        const vector unsigned char vi23 =
            vec_sro(vec_xl(-i_predecrement, i23), vi_shift);
        i23 += 16;
        const vector short vxi23_hi =
            vec_sub((vector short)vec_mergeh(vi23, vzero), va_zero_point);
        const vector unsigned char vk23 = vec_xl(144, (uint8_t*)w);
        const vector short vxk23_hi =
            vec_sub((vector short)vec_mergeh(vk23, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi23_hi), vec_unpackh(vxk23_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi23_hi), vec_unpackl(vxk23_hi)));

        vec_xst(vacc_hi_hi, 0, outacc);
        vec_xst(vacc_hi_lo, 16, outacc);

        if (c > 8) {
          vector int vacc_lo_hi = vec_xl(32, (int32_t*)outacc);
          vector int vacc_lo_lo = vec_xl(48, (int32_t*)outacc);

          const vector short vxi00_lo =
              vec_sub((vector short)vec_mergel(vi00, vzero), va_zero_point);
          const vector short vxk00_lo = vec_sub(
              (vector short)vec_mergel(vk00, vzero), vkernel_zero_point);
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi00_lo), vec_unpackl(vxk00_lo)));
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi00_lo), vec_unpackh(vxk00_lo)));

          const vector short vxi01_lo =
              vec_sub((vector short)vec_mergel(vi01, vzero), va_zero_point);
          const vector short vxk01_lo = vec_sub(
              (vector short)vec_mergel(vk01, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi01_lo), vec_unpackh(vxk01_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi01_lo), vec_unpackl(vxk01_lo)));

          const vector short vxi02_lo =
              vec_sub((vector short)vec_mergel(vi02, vzero), va_zero_point);
          const vector short vxk02_lo = vec_sub(
              (vector short)vec_mergel(vk02, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi02_lo), vec_unpackh(vxk02_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi02_lo), vec_unpackl(vxk02_lo)));

          const vector short vxi10_lo =
              vec_sub((vector short)vec_mergel(vi10, vzero), va_zero_point);
          const vector short vxk10_lo = vec_sub(
              (vector short)vec_mergel(vk10, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi10_lo), vec_unpackh(vxk10_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi10_lo), vec_unpackl(vxk10_lo)));

          const vector short vxi11_lo =
              vec_sub((vector short)vec_mergel(vi11, vzero), va_zero_point);
          const vector short vxk11_lo = vec_sub(
              (vector short)vec_mergel(vk11, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi11_lo), vec_unpackh(vxk11_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi11_lo), vec_unpackl(vxk11_lo)));

          const vector short vxi12_lo =
              vec_sub((vector short)vec_mergel(vi12, vzero), va_zero_point);
          const vector short vxk12_lo = vec_sub(
              (vector short)vec_mergel(vk12, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi12_lo), vec_unpackh(vxk12_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi12_lo), vec_unpackl(vxk12_lo)));

          const vector short vxi20_lo =
              vec_sub((vector short)vec_mergel(vi20, vzero), va_zero_point);
          const vector short vxk20_lo = vec_sub(
              (vector short)vec_mergel(vk20, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi20_lo), vec_unpackh(vxk20_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi20_lo), vec_unpackl(vxk20_lo)));

          const vector short vxi21_lo =
              vec_sub((vector short)vec_mergel(vi21, vzero), va_zero_point);
          const vector short vxk21_lo = vec_sub(
              (vector short)vec_mergel(vk21, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi21_lo), vec_unpackh(vxk21_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi21_lo), vec_unpackl(vxk21_lo)));

          const vector short vxi22_lo =
              vec_sub((vector short)vec_mergel(vi22, vzero), va_zero_point);
          const vector short vxk22_lo = vec_sub(
              (vector short)vec_mergel(vk22, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi22_lo), vec_unpackh(vxk22_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi22_lo), vec_unpackl(vxk22_lo)));

          const vector short vxi23_lo =
              vec_sub((vector short)vec_mergel(vi23, vzero), va_zero_point);
          const vector short vxk23_lo = vec_sub(
              (vector short)vec_mergel(vk23, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi23_lo), vec_unpackh(vxk23_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi23_lo), vec_unpackl(vxk23_lo)));

          vec_xst(vacc_lo_hi, 32, outacc);
          vec_xst(vacc_lo_lo, 48, outacc);
        }

        w = (const void*)((uintptr_t)w + 160);
        outacc += 16;
      }
    }
    // Compute the last 5 elements of the filters (out of 25)
    {
      const uint8_t* i00 = input[20];
      const uint8_t* i01 = input[21];
      const uint8_t* i02 = input[22];
      const uint8_t* i10 = input[23];
      const uint8_t* i11 = input[24];
      input = (const uint8_t**)((uintptr_t)input + input_stride);
      outacc = outacc32;
      size_t c = channels;
      for (; c >= 16; c -= 16) {
        /* Load the partial result into the accumulators to continue
         * accumulating the next products on them */
        vector int vacc_hi_hi = vec_xl(0, (int32_t*)outacc);
        vector int vacc_hi_lo = vec_xl(16, (int32_t*)outacc);
        vector int vacc_lo_hi = vec_xl(32, (int32_t*)outacc);
        vector int vacc_lo_lo = vec_xl(48, (int32_t*)outacc);
        outacc += 16;

        /* [21/25] Load 16 input elements over the channels, add zero point to
         * the input and weight vectors, and accumulate the products */
        const vector unsigned char vi00 = vec_xl(0, i00);
        i00 += 16;
        const vector short vxi00_hi =
            vec_sub((vector short)vec_mergeh(vi00, vzero), va_zero_point);
        const vector short vxi00_lo =
            vec_sub((vector short)vec_mergel(vi00, vzero), va_zero_point);
        const vector unsigned char vk00 = vec_xl(0, (uint8_t*)w);
        const vector short vxk00_hi =
            vec_sub((vector short)vec_mergeh(vk00, vzero), vkernel_zero_point);
        const vector short vxk00_lo =
            vec_sub((vector short)vec_mergel(vk00, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi00_hi), vec_unpackh(vxk00_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi00_hi), vec_unpackl(vxk00_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi00_lo), vec_unpackh(vxk00_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi00_lo), vec_unpackl(vxk00_lo)));

        // [22/25] Next inputs/filters
        const vector unsigned char vi01 = vec_xl(0, i01);
        i01 += 16;
        const vector short vxi01_hi =
            vec_sub((vector short)vec_mergeh(vi01, vzero), va_zero_point);
        const vector short vxi01_lo =
            vec_sub((vector short)vec_mergel(vi01, vzero), va_zero_point);
        const vector unsigned char vk01 = vec_xl(16, (uint8_t*)w);
        const vector short vxk01_hi =
            vec_sub((vector short)vec_mergeh(vk01, vzero), vkernel_zero_point);
        const vector short vxk01_lo =
            vec_sub((vector short)vec_mergel(vk01, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi01_hi), vec_unpackh(vxk01_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi01_hi), vec_unpackl(vxk01_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi01_lo), vec_unpackh(vxk01_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi01_lo), vec_unpackl(vxk01_lo)));

        // [23/25] Next inputs/filters
        const vector unsigned char vi02 = vec_xl(0, i02);
        i02 += 16;
        const vector short vxi02_hi =
            vec_sub((vector short)vec_mergeh(vi02, vzero), va_zero_point);
        const vector short vxi02_lo =
            vec_sub((vector short)vec_mergel(vi02, vzero), va_zero_point);
        const vector unsigned char vk02 = vec_xl(32, (uint8_t*)w);
        const vector short vxk02_hi =
            vec_sub((vector short)vec_mergeh(vk02, vzero), vkernel_zero_point);
        const vector short vxk02_lo =
            vec_sub((vector short)vec_mergel(vk02, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi02_hi), vec_unpackh(vxk02_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi02_hi), vec_unpackl(vxk02_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi02_lo), vec_unpackh(vxk02_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi02_lo), vec_unpackl(vxk02_lo)));

        // [24/25] Next inputs/filters
        const vector unsigned char vi10 = vec_xl(0, i10);
        i10 += 16;
        const vector short vxi10_hi =
            vec_sub((vector short)vec_mergeh(vi10, vzero), va_zero_point);
        const vector short vxi10_lo =
            vec_sub((vector short)vec_mergel(vi10, vzero), va_zero_point);
        const vector unsigned char vk10 = vec_xl(48, (uint8_t*)w);
        const vector short vxk10_hi =
            vec_sub((vector short)vec_mergeh(vk10, vzero), vkernel_zero_point);
        const vector short vxk10_lo =
            vec_sub((vector short)vec_mergel(vk10, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi10_hi), vec_unpackh(vxk10_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi10_hi), vec_unpackl(vxk10_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi10_lo), vec_unpackh(vxk10_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi10_lo), vec_unpackl(vxk10_lo)));

        // [25/25] Next inputs/filters
        const vector unsigned char vi11 = vec_xl(0, i11);
        i11 += 16;
        const vector short vxi11_hi =
            vec_sub((vector short)vec_mergeh(vi11, vzero), va_zero_point);
        const vector short vxi11_lo =
            vec_sub((vector short)vec_mergel(vi11, vzero), va_zero_point);
        const vector unsigned char vk11 = vec_xl(64, (uint8_t*)w);
        const vector short vxk11_hi =
            vec_sub((vector short)vec_mergeh(vk11, vzero), vkernel_zero_point);
        const vector short vxk11_lo =
            vec_sub((vector short)vec_mergel(vk11, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi11_hi), vec_unpackh(vxk11_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi11_hi), vec_unpackl(vxk11_hi)));
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi11_lo), vec_unpackh(vxk11_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi11_lo), vec_unpackl(vxk11_lo)));

        w = (const void*)((uintptr_t)w + 80);

        // Multiply the accumulators by the scale and add the output zero-point
        vacc_hi_hi =
            vec_signed(vec_round(vec_mul(vmultiplier, vec_float(vacc_hi_hi))));
        vacc_hi_lo =
            vec_signed(vec_round(vec_mul(vmultiplier, vec_float(vacc_hi_lo))));
        vacc_lo_hi =
            vec_signed(vec_round(vec_mul(vmultiplier, vec_float(vacc_lo_hi))));
        vacc_lo_lo =
            vec_signed(vec_round(vec_mul(vmultiplier, vec_float(vacc_lo_lo))));

        // Pack the accumulators into a uint8 vector and check min/max ranges
        vector short vout_hi =
            vec_add(vec_packs(vacc_hi_hi, vacc_hi_lo), voutput_zero_point);
        vector short vout_lo =
            vec_add(vec_packs(vacc_lo_hi, vacc_lo_lo), voutput_zero_point);

        vector unsigned char vout = vec_packsu(vout_hi, vout_lo);

        vout = vec_min(vout, vmax);
        vout = vec_max(vout, vmin);

        // Store the final results
        vec_xst(vout, 0, output);
        output += 16;
      }
      if (c != 0) {
        // Compute remaining channels
        const size_t i_predecrement = 16 - c;
        const vector unsigned char vi_shift = {
            8 * i_predecrement, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        vector int vacc_hi_hi = vec_xl(0, (int32_t*)outacc);
        vector int vacc_hi_lo = vec_xl(16, (int32_t*)outacc);

        const vector unsigned char vi00 =
            vec_sro(vec_xl(-i_predecrement, i00), vi_shift);
        i00 += 16;
        const vector short vxi00_hi =
            vec_sub((vector short)vec_mergeh(vi00, vzero), va_zero_point);
        const vector unsigned char vk00 = vec_xl(0, (uint8_t*)w);
        const vector short vxk00_hi =
            vec_sub((vector short)vec_mergeh(vk00, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi00_hi), vec_unpackh(vxk00_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi00_hi), vec_unpackl(vxk00_hi)));

        const vector unsigned char vi01 =
            vec_sro(vec_xl(-i_predecrement, i01), vi_shift);
        i01 += 16;
        const vector short vxi01_hi =
            vec_sub((vector short)vec_mergeh(vi01, vzero), va_zero_point);
        const vector unsigned char vk01 = vec_xl(16, (uint8_t*)w);
        const vector short vxk01_hi =
            vec_sub((vector short)vec_mergeh(vk01, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi01_hi), vec_unpackh(vxk01_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi01_hi), vec_unpackl(vxk01_hi)));

        const vector unsigned char vi02 =
            vec_sro(vec_xl(-i_predecrement, i02), vi_shift);
        i02 += 16;
        const vector short vxi02_hi =
            vec_sub((vector short)vec_mergeh(vi02, vzero), va_zero_point);
        const vector unsigned char vk02 = vec_xl(32, (uint8_t*)w);
        const vector short vxk02_hi =
            vec_sub((vector short)vec_mergeh(vk02, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi02_hi), vec_unpackh(vxk02_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi02_hi), vec_unpackl(vxk02_hi)));

        const vector unsigned char vi10 =
            vec_sro(vec_xl(-i_predecrement, i10), vi_shift);
        i10 += 16;
        const vector short vxi10_hi =
            vec_sub((vector short)vec_mergeh(vi10, vzero), va_zero_point);
        const vector unsigned char vk10 = vec_xl(48, (uint8_t*)w);
        const vector short vxk10_hi =
            vec_sub((vector short)vec_mergeh(vk10, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi10_hi), vec_unpackh(vxk10_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi10_hi), vec_unpackl(vxk10_hi)));

        const vector unsigned char vi11 =
            vec_sro(vec_xl(-i_predecrement, i11), vi_shift);
        i11 += 16;
        const vector short vxi11_hi =
            vec_sub((vector short)vec_mergeh(vi11, vzero), va_zero_point);
        const vector unsigned char vk11 = vec_xl(64, (uint8_t*)w);
        const vector short vxk11_hi =
            vec_sub((vector short)vec_mergeh(vk11, vzero), vkernel_zero_point);
        vacc_hi_hi = vec_add(
            vacc_hi_hi, vec_mul(vec_unpackh(vxi11_hi), vec_unpackh(vxk11_hi)));
        vacc_hi_lo = vec_add(
            vacc_hi_lo, vec_mul(vec_unpackl(vxi11_hi), vec_unpackl(vxk11_hi)));

        vacc_hi_hi =
            vec_signed(vec_round(vec_mul(vmultiplier, vec_float(vacc_hi_hi))));
        vacc_hi_lo =
            vec_signed(vec_round(vec_mul(vmultiplier, vec_float(vacc_hi_lo))));

        vector short vout_hi =
            vec_add(vec_packs(vacc_hi_hi, vacc_hi_lo), voutput_zero_point);

        vector unsigned char vout;
        if (c > 8) {
          vector int vacc_lo_hi = vec_xl(32, (int32_t*)outacc);
          vector int vacc_lo_lo = vec_xl(48, (int32_t*)outacc);

          const vector short vxi00_lo =
              vec_sub((vector short)vec_mergel(vi00, vzero), va_zero_point);
          const vector short vxk00_lo = vec_sub(
              (vector short)vec_mergel(vk00, vzero), vkernel_zero_point);
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi00_lo), vec_unpackl(vxk00_lo)));
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi00_lo), vec_unpackh(vxk00_lo)));

          const vector short vxi01_lo =
              vec_sub((vector short)vec_mergel(vi01, vzero), va_zero_point);
          const vector short vxk01_lo = vec_sub(
              (vector short)vec_mergel(vk01, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi01_lo), vec_unpackh(vxk01_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi01_lo), vec_unpackl(vxk01_lo)));

          const vector short vxi02_lo =
              vec_sub((vector short)vec_mergel(vi02, vzero), va_zero_point);
          const vector short vxk02_lo = vec_sub(
              (vector short)vec_mergel(vk02, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi02_lo), vec_unpackh(vxk02_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi02_lo), vec_unpackl(vxk02_lo)));

          const vector short vxi10_lo =
              vec_sub((vector short)vec_mergel(vi10, vzero), va_zero_point);
          const vector short vxk10_lo = vec_sub(
              (vector short)vec_mergel(vk10, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi10_lo), vec_unpackh(vxk10_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi10_lo), vec_unpackl(vxk10_lo)));

          const vector short vxi11_lo =
              vec_sub((vector short)vec_mergel(vi11, vzero), va_zero_point);
          const vector short vxk11_lo = vec_sub(
              (vector short)vec_mergel(vk11, vzero), vkernel_zero_point);
          vacc_lo_hi = vec_add(
              vacc_lo_hi,
              vec_mul(vec_unpackh(vxi11_lo), vec_unpackh(vxk11_lo)));
          vacc_lo_lo = vec_add(
              vacc_lo_lo,
              vec_mul(vec_unpackl(vxi11_lo), vec_unpackl(vxk11_lo)));

          vacc_lo_hi = vec_signed(
              vec_round(vec_mul(vmultiplier, vec_float(vacc_lo_hi))));
          vacc_lo_lo = vec_signed(
              vec_round(vec_mul(vmultiplier, vec_float(vacc_lo_lo))));

          vector short vout_lo =
              vec_add(vec_packs(vacc_lo_hi, vacc_lo_lo), voutput_zero_point);

          vout = vec_packsu(vout_hi, vout_lo);
        } else {
          vout = vec_packsu(vout_hi, vout_hi);
        }

        outacc += 16;

        vout = vec_min(vout, vmax);
        vout = vec_max(vout, vmin);

        // Store the results of the remaining channels
        if (c & 8) {
          *(uint64_t*)output = ((vector unsigned long long)vout)[0];
          output += 8;
          const vector unsigned char vshift_8bytes = {
              8 * 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
          vout = vec_sro(vout, vshift_8bytes);
        }
        if (c & 4) {
          *(uint32_t*)output = ((vector unsigned int)vout)[0];
          output += 4;
          const vector unsigned char vshift_4bytes = {
              8 * 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
          vout = vec_sro(vout, vshift_4bytes);
        }
        if (c & 2) {
          *(uint16_t*)output = ((vector unsigned short)vout)[0];
          output += 2;
          const vector unsigned char vshift_2bytes = {
              8 * 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
          vout = vec_sro(vout, vshift_2bytes);
        }
        if (c & 1) {
          output[0] = vout[0];
          output += 1;
        }
      }
    }

    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--output_width != 0);
}
