/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <altivec.h>

#include <qnnpack/q8dwconv.h>
#include <requantization/runtime-vsx.h>

void pytorch_q8dwconv_ukernel_up16x9_per_channel__vsx(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  const vector short va_zero_point =
      vec_splats(quantization_params->vsx.input_zero_point);
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
    const uint8_t* i0 = input[0];
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];

    input = (const uint8_t**)((uintptr_t)input + input_stride);

    // Iterate over 16 input channels computing kernels of size 3x3
    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      vector int vacc_hi_hi = vec_xl(0, (int32_t*)w);
      vector int vacc_hi_lo = vec_xl(16, (int32_t*)w);
      vector int vacc_lo_hi = vec_xl(32, (int32_t*)w);
      vector int vacc_lo_lo = vec_xl(48, (int32_t*)w);

      // Load the zero points of each channel of the filters
      const vector unsigned char vkernel_zero_point =
          vec_xl(0, &quantization_params->vsx.kernel_zero_points[channels - c]);
      const vector short vkernel_zero_point_hi =
          (vector short)vec_mergeh(vkernel_zero_point, vzero);
      const vector short vkernel_zero_point_lo =
          (vector short)vec_mergel(vkernel_zero_point, vzero);

      /* [1/9] Load 16 input elements over the channels, add zero point to the
       * input and weight vectors, and accumulate the products */
      const vector unsigned char vi0 = vec_xl(0, i0);
      i0 += 16;
      const vector short vxi0_hi =
          sub_zero_point((vector short)vec_mergeh(vi0, vzero), va_zero_point);
      const vector short vxi0_lo =
          sub_zero_point((vector short)vec_mergel(vi0, vzero), va_zero_point);
      const vector unsigned char vk0 = vec_xl(64, (uint8_t*)w);
      const vector short vxk0_hi =
          vec_sub((vector short)vec_mergeh(vk0, vzero), vkernel_zero_point_hi);
      const vector short vxk0_lo =
          vec_sub((vector short)vec_mergel(vk0, vzero), vkernel_zero_point_lo);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi0_hi), vec_unpackh(vxk0_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi0_hi), vec_unpackl(vxk0_hi)));
      vacc_lo_hi = vec_add(
          vacc_lo_hi, vec_mul(vec_unpackh(vxi0_lo), vec_unpackh(vxk0_lo)));
      vacc_lo_lo = vec_add(
          vacc_lo_lo, vec_mul(vec_unpackl(vxi0_lo), vec_unpackl(vxk0_lo)));

      // [2/9] Next inputs/filters
      const vector unsigned char vi1 = vec_xl(0, i1);
      i1 += 16;
      const vector short vxi1_hi =
          sub_zero_point((vector short)vec_mergeh(vi1, vzero), va_zero_point);
      const vector short vxi1_lo =
          sub_zero_point((vector short)vec_mergel(vi1, vzero), va_zero_point);
      const vector unsigned char vk1 = vec_xl(80, (uint8_t*)w);
      const vector short vxk1_hi =
          vec_sub((vector short)vec_mergeh(vk1, vzero), vkernel_zero_point_hi);
      const vector short vxk1_lo =
          vec_sub((vector short)vec_mergel(vk1, vzero), vkernel_zero_point_lo);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi1_hi), vec_unpackh(vxk1_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi1_hi), vec_unpackl(vxk1_hi)));
      vacc_lo_hi = vec_add(
          vacc_lo_hi, vec_mul(vec_unpackh(vxi1_lo), vec_unpackh(vxk1_lo)));
      vacc_lo_lo = vec_add(
          vacc_lo_lo, vec_mul(vec_unpackl(vxi1_lo), vec_unpackl(vxk1_lo)));

      // [3/9] Next inputs/filters
      const vector unsigned char vi2 = vec_xl(0, i2);
      i2 += 16;
      const vector short vxi2_hi =
          sub_zero_point((vector short)vec_mergeh(vi2, vzero), va_zero_point);
      const vector short vxi2_lo =
          sub_zero_point((vector short)vec_mergel(vi2, vzero), va_zero_point);
      const vector unsigned char vk2 = vec_xl(96, (uint8_t*)w);
      const vector short vxk2_hi =
          vec_sub((vector short)vec_mergeh(vk2, vzero), vkernel_zero_point_hi);
      const vector short vxk2_lo =
          vec_sub((vector short)vec_mergel(vk2, vzero), vkernel_zero_point_lo);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi2_hi), vec_unpackh(vxk2_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi2_hi), vec_unpackl(vxk2_hi)));
      vacc_lo_hi = vec_add(
          vacc_lo_hi, vec_mul(vec_unpackh(vxi2_lo), vec_unpackh(vxk2_lo)));
      vacc_lo_lo = vec_add(
          vacc_lo_lo, vec_mul(vec_unpackl(vxi2_lo), vec_unpackl(vxk2_lo)));

      // [4/9] Next inputs/filters
      const vector unsigned char vi3 = vec_xl(0, i3);
      i3 += 16;
      const vector short vxi3_hi =
          sub_zero_point((vector short)vec_mergeh(vi3, vzero), va_zero_point);
      const vector short vxi3_lo =
          sub_zero_point((vector short)vec_mergel(vi3, vzero), va_zero_point);
      const vector unsigned char vk3 = vec_xl(112, (uint8_t*)w);
      const vector short vxk3_hi =
          vec_sub((vector short)vec_mergeh(vk3, vzero), vkernel_zero_point_hi);
      const vector short vxk3_lo =
          vec_sub((vector short)vec_mergel(vk3, vzero), vkernel_zero_point_lo);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi3_hi), vec_unpackh(vxk3_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi3_hi), vec_unpackl(vxk3_hi)));
      vacc_lo_hi = vec_add(
          vacc_lo_hi, vec_mul(vec_unpackh(vxi3_lo), vec_unpackh(vxk3_lo)));
      vacc_lo_lo = vec_add(
          vacc_lo_lo, vec_mul(vec_unpackl(vxi3_lo), vec_unpackl(vxk3_lo)));

      // [5/9] Next inputs/filters
      const vector unsigned char vi4 = vec_xl(0, i4);
      i4 += 16;
      const vector short vxi4_hi =
          sub_zero_point((vector short)vec_mergeh(vi4, vzero), va_zero_point);
      const vector short vxi4_lo =
          sub_zero_point((vector short)vec_mergel(vi4, vzero), va_zero_point);
      const vector unsigned char vk4 = vec_xl(128, (uint8_t*)w);
      const vector short vxk4_hi =
          vec_sub((vector short)vec_mergeh(vk4, vzero), vkernel_zero_point_hi);
      const vector short vxk4_lo =
          vec_sub((vector short)vec_mergel(vk4, vzero), vkernel_zero_point_lo);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi4_hi), vec_unpackh(vxk4_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi4_hi), vec_unpackl(vxk4_hi)));
      vacc_lo_hi = vec_add(
          vacc_lo_hi, vec_mul(vec_unpackh(vxi4_lo), vec_unpackh(vxk4_lo)));
      vacc_lo_lo = vec_add(
          vacc_lo_lo, vec_mul(vec_unpackl(vxi4_lo), vec_unpackl(vxk4_lo)));

      // [6/9] Next inputs/filters
      const vector unsigned char vi5 = vec_xl(0, i5);
      i5 += 16;
      const vector short vxi5_hi =
          sub_zero_point((vector short)vec_mergeh(vi5, vzero), va_zero_point);
      const vector short vxi5_lo =
          sub_zero_point((vector short)vec_mergel(vi5, vzero), va_zero_point);
      const vector unsigned char vk5 = vec_xl(144, (uint8_t*)w);
      const vector short vxk5_hi =
          vec_sub((vector short)vec_mergeh(vk5, vzero), vkernel_zero_point_hi);
      const vector short vxk5_lo =
          vec_sub((vector short)vec_mergel(vk5, vzero), vkernel_zero_point_lo);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi5_hi), vec_unpackh(vxk5_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi5_hi), vec_unpackl(vxk5_hi)));
      vacc_lo_hi = vec_add(
          vacc_lo_hi, vec_mul(vec_unpackh(vxi5_lo), vec_unpackh(vxk5_lo)));
      vacc_lo_lo = vec_add(
          vacc_lo_lo, vec_mul(vec_unpackl(vxi5_lo), vec_unpackl(vxk5_lo)));

      // [7/9] Next inputs/filters
      const vector unsigned char vi6 = vec_xl(0, i6);
      i6 += 16;
      const vector short vxi6_hi =
          sub_zero_point((vector short)vec_mergeh(vi6, vzero), va_zero_point);
      const vector short vxi6_lo =
          sub_zero_point((vector short)vec_mergel(vi6, vzero), va_zero_point);
      const vector unsigned char vk6 = vec_xl(160, (uint8_t*)w);
      const vector short vxk6_hi =
          vec_sub((vector short)vec_mergeh(vk6, vzero), vkernel_zero_point_hi);
      const vector short vxk6_lo =
          vec_sub((vector short)vec_mergel(vk6, vzero), vkernel_zero_point_lo);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi6_hi), vec_unpackh(vxk6_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi6_hi), vec_unpackl(vxk6_hi)));
      vacc_lo_hi = vec_add(
          vacc_lo_hi, vec_mul(vec_unpackh(vxi6_lo), vec_unpackh(vxk6_lo)));
      vacc_lo_lo = vec_add(
          vacc_lo_lo, vec_mul(vec_unpackl(vxi6_lo), vec_unpackl(vxk6_lo)));

      // [8/9] Next inputs/filters
      const vector unsigned char vi7 = vec_xl(0, i7);
      i7 += 16;
      const vector short vxi7_hi =
          sub_zero_point((vector short)vec_mergeh(vi7, vzero), va_zero_point);
      const vector short vxi7_lo =
          sub_zero_point((vector short)vec_mergel(vi7, vzero), va_zero_point);
      const vector unsigned char vk7 = vec_xl(176, (uint8_t*)w);
      const vector short vxk7_hi =
          vec_sub((vector short)vec_mergeh(vk7, vzero), vkernel_zero_point_hi);
      const vector short vxk7_lo =
          vec_sub((vector short)vec_mergel(vk7, vzero), vkernel_zero_point_lo);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi7_hi), vec_unpackh(vxk7_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi7_hi), vec_unpackl(vxk7_hi)));
      vacc_lo_hi = vec_add(
          vacc_lo_hi, vec_mul(vec_unpackh(vxi7_lo), vec_unpackh(vxk7_lo)));
      vacc_lo_lo = vec_add(
          vacc_lo_lo, vec_mul(vec_unpackl(vxi7_lo), vec_unpackl(vxk7_lo)));

      // [9/9] Next inputs/filters
      const vector unsigned char vi8 = vec_xl(0, i8);
      i8 += 16;
      const vector short vxi8_hi =
          sub_zero_point((vector short)vec_mergeh(vi8, vzero), va_zero_point);
      const vector short vxi8_lo =
          sub_zero_point((vector short)vec_mergel(vi8, vzero), va_zero_point);
      const vector unsigned char vk8 = vec_xl(192, (uint8_t*)w);
      const vector short vxk8_hi =
          vec_sub((vector short)vec_mergeh(vk8, vzero), vkernel_zero_point_hi);
      const vector short vxk8_lo =
          vec_sub((vector short)vec_mergel(vk8, vzero), vkernel_zero_point_lo);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi8_hi), vec_unpackh(vxk8_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi8_hi), vec_unpackl(vxk8_hi)));
      vacc_lo_hi = vec_add(
          vacc_lo_hi, vec_mul(vec_unpackh(vxi8_lo), vec_unpackh(vxk8_lo)));
      vacc_lo_lo = vec_add(
          vacc_lo_lo, vec_mul(vec_unpackl(vxi8_lo), vec_unpackl(vxk8_lo)));

      w = (void*)((uintptr_t)w + 208);

      // Load the scale of each channel (per_channel quantization)
      const vector float vmultiplier_hi_hi = vec_xl(
          0, &quantization_params->vsx.requantization_scales[channels - c]);
      const vector float vmultiplier_hi_lo = vec_xl(
          0, &quantization_params->vsx.requantization_scales[channels - c + 4]);
      const vector float vmultiplier_lo_hi = vec_xl(
          0, &quantization_params->vsx.requantization_scales[channels - c + 8]);
      const vector float vmultiplier_lo_lo = vec_xl(
          0,
          &quantization_params->vsx.requantization_scales[channels - c + 12]);

      /* Multiply the accumulators by the scales (each channel has its own
       * scale factor), and add the output zero point */
      vacc_hi_hi = vec_signed(
          vec_round(vec_mul(vmultiplier_hi_hi, vec_float(vacc_hi_hi))));
      vacc_hi_lo = vec_signed(
          vec_round(vec_mul(vmultiplier_hi_lo, vec_float(vacc_hi_lo))));
      vacc_lo_hi = vec_signed(
          vec_round(vec_mul(vmultiplier_lo_hi, vec_float(vacc_lo_hi))));
      vacc_lo_lo = vec_signed(
          vec_round(vec_mul(vmultiplier_lo_lo, vec_float(vacc_lo_lo))));

      vector short vout_hi =
          vec_add(vec_packs(vacc_hi_hi, vacc_hi_lo), voutput_zero_point);
      vector short vout_lo =
          vec_add(vec_packs(vacc_lo_hi, vacc_lo_lo), voutput_zero_point);

      // Pack the accumulators into a uint8 vector and check min/max ranges
      vector unsigned char vout = vec_packsu(vout_hi, vout_lo);

      vout = vec_min(vout, vmax);
      vout = vec_max(vout, vmin);

      // Store the packed vector
      vec_xst(vout, 0, output);
      output += 16;
    }
    if (c != 0) {
      // Compute remaining channels
      const size_t i_predecrement = 16 - c;
      const vector unsigned char vi_shift = {
          8 * i_predecrement, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      const vector unsigned char vkernel_zero_point =
          vec_xl(0, &quantization_params->vsx.kernel_zero_points[channels - c]);
      const vector short vkernel_zero_point_hi =
          (vector short)vec_mergeh(vkernel_zero_point, vzero);
      const vector short vkernel_zero_point_lo =
          (vector short)vec_mergel(vkernel_zero_point, vzero);

      vector int vacc_hi_hi = vec_xl(0, (int32_t*)w);
      vector int vacc_hi_lo = vec_xl(16, (int32_t*)w);

      // Shift the elements in order to zero the lower lanes
      const vector unsigned char vi0 =
          vec_sro(vec_xl(-i_predecrement, i0), vi_shift);
      const vector short vxi0_hi =
          sub_zero_point((vector short)vec_mergeh(vi0, vzero), va_zero_point);
      const vector unsigned char vk0 = vec_xl(64, (uint8_t*)w);
      const vector short vxk0_hi =
          vec_sub((vector short)vec_mergeh(vk0, vzero), vkernel_zero_point_hi);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi0_hi), vec_unpackh(vxk0_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi0_hi), vec_unpackl(vxk0_hi)));

      const vector unsigned char vi1 =
          vec_sro(vec_xl(-i_predecrement, i1), vi_shift);
      const vector short vxi1_hi =
          sub_zero_point((vector short)vec_mergeh(vi1, vzero), va_zero_point);
      const vector unsigned char vk1 = vec_xl(80, (uint8_t*)w);
      const vector short vxk1_hi =
          vec_sub((vector short)vec_mergeh(vk1, vzero), vkernel_zero_point_hi);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi1_hi), vec_unpackh(vxk1_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi1_hi), vec_unpackl(vxk1_hi)));

      const vector unsigned char vi2 =
          vec_sro(vec_xl(-i_predecrement, i2), vi_shift);
      const vector short vxi2_hi =
          sub_zero_point((vector short)vec_mergeh(vi2, vzero), va_zero_point);
      const vector unsigned char vk2 = vec_xl(96, (uint8_t*)w);
      const vector short vxk2_hi =
          vec_sub((vector short)vec_mergeh(vk2, vzero), vkernel_zero_point_hi);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi2_hi), vec_unpackh(vxk2_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi2_hi), vec_unpackl(vxk2_hi)));

      const vector unsigned char vi3 =
          vec_sro(vec_xl(-i_predecrement, i3), vi_shift);
      const vector short vxi3_hi =
          sub_zero_point((vector short)vec_mergeh(vi3, vzero), va_zero_point);
      const vector unsigned char vk3 = vec_xl(112, (uint8_t*)w);
      const vector short vxk3_hi =
          vec_sub((vector short)vec_mergeh(vk3, vzero), vkernel_zero_point_hi);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi3_hi), vec_unpackh(vxk3_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi3_hi), vec_unpackl(vxk3_hi)));

      const vector unsigned char vi4 =
          vec_sro(vec_xl(-i_predecrement, i4), vi_shift);
      const vector short vxi4_hi =
          sub_zero_point((vector short)vec_mergeh(vi4, vzero), va_zero_point);
      const vector unsigned char vk4 = vec_xl(128, (uint8_t*)w);
      const vector short vxk4_hi =
          vec_sub((vector short)vec_mergeh(vk4, vzero), vkernel_zero_point_hi);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi4_hi), vec_unpackh(vxk4_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi4_hi), vec_unpackl(vxk4_hi)));

      const vector unsigned char vi5 =
          vec_sro(vec_xl(-i_predecrement, i5), vi_shift);
      const vector short vxi5_hi =
          sub_zero_point((vector short)vec_mergeh(vi5, vzero), va_zero_point);
      const vector unsigned char vk5 = vec_xl(144, (uint8_t*)w);
      const vector short vxk5_hi =
          vec_sub((vector short)vec_mergeh(vk5, vzero), vkernel_zero_point_hi);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi5_hi), vec_unpackh(vxk5_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi5_hi), vec_unpackl(vxk5_hi)));

      const vector unsigned char vi6 =
          vec_sro(vec_xl(-i_predecrement, i6), vi_shift);
      const vector short vxi6_hi =
          sub_zero_point((vector short)vec_mergeh(vi6, vzero), va_zero_point);
      const vector unsigned char vk6 = vec_xl(160, (uint8_t*)w);
      const vector short vxk6_hi =
          vec_sub((vector short)vec_mergeh(vk6, vzero), vkernel_zero_point_hi);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi6_hi), vec_unpackh(vxk6_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi6_hi), vec_unpackl(vxk6_hi)));

      const vector unsigned char vi7 =
          vec_sro(vec_xl(-i_predecrement, i7), vi_shift);
      const vector short vxi7_hi =
          sub_zero_point((vector short)vec_mergeh(vi7, vzero), va_zero_point);
      const vector unsigned char vk7 = vec_xl(176, (uint8_t*)w);
      const vector short vxk7_hi =
          vec_sub((vector short)vec_mergeh(vk7, vzero), vkernel_zero_point_hi);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi7_hi), vec_unpackh(vxk7_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi7_hi), vec_unpackl(vxk7_hi)));

      const vector unsigned char vi8 =
          vec_sro(vec_xl(-i_predecrement, i8), vi_shift);
      const vector short vxi8_hi =
          sub_zero_point((vector short)vec_mergeh(vi8, vzero), va_zero_point);
      const vector unsigned char vk8 = vec_xl(192, (uint8_t*)w);
      const vector short vxk8_hi =
          vec_sub((vector short)vec_mergeh(vk8, vzero), vkernel_zero_point_hi);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, vec_mul(vec_unpackh(vxi8_hi), vec_unpackh(vxk8_hi)));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, vec_mul(vec_unpackl(vxi8_hi), vec_unpackl(vxk8_hi)));

      const vector float vmultiplier_hi_hi = vec_xl(
          0, &quantization_params->vsx.requantization_scales[channels - c]);
      const vector float vmultiplier_hi_lo = vec_xl(
          0, &quantization_params->vsx.requantization_scales[channels - c + 4]);

      vacc_hi_hi = vec_signed(
          vec_round(vec_mul(vmultiplier_hi_hi, vec_float(vacc_hi_hi))));
      vacc_hi_lo = vec_signed(
          vec_round(vec_mul(vmultiplier_hi_lo, vec_float(vacc_hi_lo))));

      vector short vout_hi =
          vec_add(vec_packs(vacc_hi_hi, vacc_hi_lo), voutput_zero_point);

      vector unsigned char vout;
      if (c > 8) {
        vector int vacc_lo_hi = vec_xl(32, (int32_t*)w);
        vector int vacc_lo_lo = vec_xl(48, (int32_t*)w);

        const vector short vxi0_lo =
            sub_zero_point((vector short)vec_mergel(vi0, vzero), va_zero_point);
        const vector short vxk0_lo = vec_sub(
            (vector short)vec_mergel(vk0, vzero), vkernel_zero_point_lo);
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi0_lo), vec_unpackh(vxk0_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi0_lo), vec_unpackl(vxk0_lo)));

        const vector short vxi1_lo =
            sub_zero_point((vector short)vec_mergel(vi1, vzero), va_zero_point);
        const vector short vxk1_lo = vec_sub(
            (vector short)vec_mergel(vk1, vzero), vkernel_zero_point_lo);
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi1_lo), vec_unpackh(vxk1_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi1_lo), vec_unpackl(vxk1_lo)));

        const vector short vxi2_lo =
            sub_zero_point((vector short)vec_mergel(vi2, vzero), va_zero_point);
        const vector short vxk2_lo = vec_sub(
            (vector short)vec_mergel(vk2, vzero), vkernel_zero_point_lo);
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi2_lo), vec_unpackh(vxk2_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi2_lo), vec_unpackl(vxk2_lo)));

        const vector short vxi3_lo =
            sub_zero_point((vector short)vec_mergel(vi3, vzero), va_zero_point);
        const vector short vxk3_lo = vec_sub(
            (vector short)vec_mergel(vk3, vzero), vkernel_zero_point_lo);
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi3_lo), vec_unpackh(vxk3_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi3_lo), vec_unpackl(vxk3_lo)));

        const vector short vxi4_lo =
            sub_zero_point((vector short)vec_mergel(vi4, vzero), va_zero_point);
        const vector short vxk4_lo = vec_sub(
            (vector short)vec_mergel(vk4, vzero), vkernel_zero_point_lo);
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi4_lo), vec_unpackh(vxk4_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi4_lo), vec_unpackl(vxk4_lo)));

        const vector short vxi5_lo =
            sub_zero_point((vector short)vec_mergel(vi5, vzero), va_zero_point);
        const vector short vxk5_lo = vec_sub(
            (vector short)vec_mergel(vk5, vzero), vkernel_zero_point_lo);
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi5_lo), vec_unpackh(vxk5_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi5_lo), vec_unpackl(vxk5_lo)));

        const vector short vxi6_lo =
            sub_zero_point((vector short)vec_mergel(vi6, vzero), va_zero_point);
        const vector short vxk6_lo = vec_sub(
            (vector short)vec_mergel(vk6, vzero), vkernel_zero_point_lo);
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi6_lo), vec_unpackh(vxk6_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi6_lo), vec_unpackl(vxk6_lo)));

        const vector short vxi7_lo =
            sub_zero_point((vector short)vec_mergel(vi7, vzero), va_zero_point);
        const vector short vxk7_lo = vec_sub(
            (vector short)vec_mergel(vk7, vzero), vkernel_zero_point_lo);
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi7_lo), vec_unpackh(vxk7_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi7_lo), vec_unpackl(vxk7_lo)));

        const vector short vxi8_lo =
            sub_zero_point((vector short)vec_mergel(vi8, vzero), va_zero_point);
        const vector short vxk8_lo = vec_sub(
            (vector short)vec_mergel(vk8, vzero), vkernel_zero_point_lo);
        vacc_lo_hi = vec_add(
            vacc_lo_hi, vec_mul(vec_unpackh(vxi8_lo), vec_unpackh(vxk8_lo)));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, vec_mul(vec_unpackl(vxi8_lo), vec_unpackl(vxk8_lo)));

        const vector float vmultiplier_lo_hi = vec_xl(
            0,
            &quantization_params->vsx.requantization_scales[channels - c + 8]);
        const vector float vmultiplier_lo_lo = vec_xl(
            0,
            &quantization_params->vsx.requantization_scales[channels - c + 12]);
        vacc_lo_hi = vec_signed(
            vec_round(vec_mul(vmultiplier_lo_hi, vec_float(vacc_lo_hi))));
        vacc_lo_lo = vec_signed(
            vec_round(vec_mul(vmultiplier_lo_lo, vec_float(vacc_lo_lo))));

        vector short vout_lo =
            vec_add(vec_packs(vacc_lo_hi, vacc_lo_lo), voutput_zero_point);

        vout = vec_packsu(vout_hi, vout_lo);
      } else {
        vout = vec_packsu(vout_hi, vout_hi);
      }

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

    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--output_width != 0);
}
