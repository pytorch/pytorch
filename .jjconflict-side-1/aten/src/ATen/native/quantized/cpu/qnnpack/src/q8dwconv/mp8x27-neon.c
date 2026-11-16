/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/q8dwconv.h>

/**
 * Kernel for performing depthwise convolution with a kernel of weights of
 * size 27 (ex. 3x3x3). The general strategy is
 * For each output row:
 *   For each output pixel in that row:
 *     For each of the three groupings of 9 of the 27 weights (ex. one yz slice for a 3x3x3 kernel):
 *       For each grouping of 8 channels:
 *          Load input pixel values from the indirection buffer and the weights,
 *          multiply and add them, and keep track of a running total of these
 *          products and the bias in a temporary buffer
 *       Perform requantization to obtain final output
 *     Advance indirection buffer to next pixel in the row
 *   Advance indirection buffer to next row
 */
void pytorch_q8dwconv_ukernel_mp8x27__neon(
    size_t channels,
    size_t output_height,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    int32_t* outacc32,
    uint8_t* output,
    size_t input_row_stride, // Stride between rows in the indirection buffer
    size_t input_col_stride, // Stride between columns in the indirection buffer
    size_t
        output_increment, // Padding between output pixels in the output buffer
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[restrict static 1]) {
  const uint8x8_t vinput_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);
#ifdef __aarch64__
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&quantization_params->neon.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&quantization_params->neon.output_max);
#else
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);
#endif

  for (size_t output_y = 0; output_y < output_height; output_y++) {
    const uint8_t** input_row_start = input;
    for (size_t output_x = 0; output_x < output_width; output_x++) {
      uint8_t* output_start = output;
      int32_t* outacc = outacc32;
      const void* w = weights;
      {
        const uint8_t* i0 = input[0];
        const uint8_t* i1 = input[1];
        const uint8_t* i2 = input[2];
        const uint8_t* i3 = input[3];
        const uint8_t* i4 = input[4];
        const uint8_t* i5 = input[5];
        const uint8_t* i6 = input[6];
        const uint8_t* i7 = input[7];
        const uint8_t* i8 = input[8];

        size_t c = channels;
        const uint8_t* kernel_zero_points_ptr =
            quantization_params->neon.kernel_zero_points + channels;
        for (; c >= 8; c -= 8) {
          const uint8x8_t vkernel_zero_point =
              vld1_u8(kernel_zero_points_ptr - c);
          int32x4_t vaccX1_lo = vld1q_s32(w);
          w = (void*)((uintptr_t)w + sizeof(int32x4_t));
          int32x4_t vaccX1_hi = vld1q_s32(w);
          w = (void*)((uintptr_t)w + sizeof(int32x4_t));

          const uint8x8_t vk0 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi0 = vld1_u8(i0);
          i0 += 8;
          const int16x8_t vxk0 =
              vreinterpretq_s16_u16(vsubl_u8(vk0, vkernel_zero_point));
          const int16x8_t vxi0 =
              vreinterpretq_s16_u16(vsubl_u8(vi0, vinput_zero_point));
          int32x4_t vaccX0_lo =
              vmull_s16(vget_low_s16(vxk0), vget_low_s16(vxi0));
          int32x4_t vaccX0_hi =
              vmull_s16(vget_high_s16(vxk0), vget_high_s16(vxi0));

          const uint8x8_t vk1 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi1 = vld1_u8(i1);
          i1 += 8;
          const int16x8_t vxk1 =
              vreinterpretq_s16_u16(vsubl_u8(vk1, vkernel_zero_point));
          const int16x8_t vxi1 =
              vreinterpretq_s16_u16(vsubl_u8(vi1, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk1), vget_low_s16(vxi1));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk1), vget_high_s16(vxi1));

          const uint8x8_t vk2 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi2 = vld1_u8(i2);
          i2 += 8;
          const int16x8_t vxk2 =
              vreinterpretq_s16_u16(vsubl_u8(vk2, vkernel_zero_point));
          const int16x8_t vxi2 =
              vreinterpretq_s16_u16(vsubl_u8(vi2, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk2), vget_low_s16(vxi2));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk2), vget_high_s16(vxi2));

          const uint8x8_t vk3 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi3 = vld1_u8(i3);
          i3 += 8;
          const int16x8_t vxk3 =
              vreinterpretq_s16_u16(vsubl_u8(vk3, vkernel_zero_point));
          const int16x8_t vxi3 =
              vreinterpretq_s16_u16(vsubl_u8(vi3, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk3), vget_low_s16(vxi3));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk3), vget_high_s16(vxi3));

          const uint8x8_t vk4 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi4 = vld1_u8(i4);
          i4 += 8;
          const int16x8_t vxk4 =
              vreinterpretq_s16_u16(vsubl_u8(vk4, vkernel_zero_point));
          const int16x8_t vxi4 =
              vreinterpretq_s16_u16(vsubl_u8(vi4, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk4), vget_low_s16(vxi4));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk4), vget_high_s16(vxi4));

          const uint8x8_t vk5 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi5 = vld1_u8(i5);
          i5 += 8;
          const int16x8_t vxk5 =
              vreinterpretq_s16_u16(vsubl_u8(vk5, vkernel_zero_point));
          const int16x8_t vxi5 =
              vreinterpretq_s16_u16(vsubl_u8(vi5, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk5), vget_low_s16(vxi5));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk5), vget_high_s16(vxi5));

          const uint8x8_t vk6 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi6 = vld1_u8(i6);
          i6 += 8;
          const int16x8_t vxk6 =
              vreinterpretq_s16_u16(vsubl_u8(vk6, vkernel_zero_point));
          const int16x8_t vxi6 =
              vreinterpretq_s16_u16(vsubl_u8(vi6, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk6), vget_low_s16(vxi6));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk6), vget_high_s16(vxi6));

          const uint8x8_t vk7 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi7 = vld1_u8(i7);
          i7 += 8;
          const int16x8_t vxk7 =
              vreinterpretq_s16_u16(vsubl_u8(vk7, vkernel_zero_point));
          const int16x8_t vxi7 =
              vreinterpretq_s16_u16(vsubl_u8(vi7, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk7), vget_low_s16(vxi7));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk7), vget_high_s16(vxi7));

          const uint8x8_t vk8 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi8 = vld1_u8(i8);
          i8 += 8;
          const int16x8_t vxk8 =
              vreinterpretq_s16_u16(vsubl_u8(vk8, vkernel_zero_point));
          const int16x8_t vxi8 =
              vreinterpretq_s16_u16(vsubl_u8(vi8, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk8), vget_low_s16(vxi8));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk8), vget_high_s16(vxi8));

          int32x4_t vacc_lo = vaddq_s32(vaccX0_lo, vaccX1_lo);
          int32x4_t vacc_hi = vaddq_s32(vaccX0_hi, vaccX1_hi);

          vst1q_s32(outacc, vacc_lo);
          outacc += 4;
          vst1q_s32(outacc, vacc_hi);
          outacc += 4;
        }
        if (c != 0) {
          const size_t c_predecrement = 8 - c;
          const int64x1_t vi_shift = vmov_n_s64(-8 * c_predecrement);
          i0 -= c_predecrement;
          i1 -= c_predecrement;
          i2 -= c_predecrement;
          i3 -= c_predecrement;
          i4 -= c_predecrement;
          i5 -= c_predecrement;
          i6 -= c_predecrement;
          i7 -= c_predecrement;
          i8 -= c_predecrement;

          const uint8x8_t vkernel_zero_point = vld1_u8(
              &quantization_params->neon.kernel_zero_points[channels - c]);
          int32x4_t vaccX1_lo = vld1q_s32(w);
          w = (void*)((uintptr_t)w + sizeof(int32x4_t));
          int32x4_t vaccX1_hi = vld1q_s32(w);
          w = (void*)((uintptr_t)w + sizeof(int32x4_t));

          const uint8x8_t vk0 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi0 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i0)), vi_shift));
          const int16x8_t vxk0 =
              vreinterpretq_s16_u16(vsubl_u8(vk0, vkernel_zero_point));
          const int16x8_t vxi0 =
              vreinterpretq_s16_u16(vsubl_u8(vi0, vinput_zero_point));
          int32x4_t vaccX0_lo =
              vmull_s16(vget_low_s16(vxk0), vget_low_s16(vxi0));
          int32x4_t vaccX0_hi =
              vmull_s16(vget_high_s16(vxk0), vget_high_s16(vxi0));

          const uint8x8_t vk1 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi1 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i1)), vi_shift));
          const int16x8_t vxk1 =
              vreinterpretq_s16_u16(vsubl_u8(vk1, vkernel_zero_point));
          const int16x8_t vxi1 =
              vreinterpretq_s16_u16(vsubl_u8(vi1, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk1), vget_low_s16(vxi1));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk1), vget_high_s16(vxi1));

          const uint8x8_t vk2 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi2 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i2)), vi_shift));
          const int16x8_t vxk2 =
              vreinterpretq_s16_u16(vsubl_u8(vk2, vkernel_zero_point));
          const int16x8_t vxi2 =
              vreinterpretq_s16_u16(vsubl_u8(vi2, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk2), vget_low_s16(vxi2));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk2), vget_high_s16(vxi2));

          const uint8x8_t vk3 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi3 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i3)), vi_shift));
          const int16x8_t vxk3 =
              vreinterpretq_s16_u16(vsubl_u8(vk3, vkernel_zero_point));
          const int16x8_t vxi3 =
              vreinterpretq_s16_u16(vsubl_u8(vi3, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk3), vget_low_s16(vxi3));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk3), vget_high_s16(vxi3));

          const uint8x8_t vk4 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi4 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i4)), vi_shift));
          const int16x8_t vxk4 =
              vreinterpretq_s16_u16(vsubl_u8(vk4, vkernel_zero_point));
          const int16x8_t vxi4 =
              vreinterpretq_s16_u16(vsubl_u8(vi4, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk4), vget_low_s16(vxi4));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk4), vget_high_s16(vxi4));

          const uint8x8_t vk5 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi5 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i5)), vi_shift));
          const int16x8_t vxk5 =
              vreinterpretq_s16_u16(vsubl_u8(vk5, vkernel_zero_point));
          const int16x8_t vxi5 =
              vreinterpretq_s16_u16(vsubl_u8(vi5, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk5), vget_low_s16(vxi5));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk5), vget_high_s16(vxi5));

          const uint8x8_t vk6 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi6 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i6)), vi_shift));
          const int16x8_t vxk6 =
              vreinterpretq_s16_u16(vsubl_u8(vk6, vkernel_zero_point));
          const int16x8_t vxi6 =
              vreinterpretq_s16_u16(vsubl_u8(vi6, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk6), vget_low_s16(vxi6));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk6), vget_high_s16(vxi6));

          const uint8x8_t vk7 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi7 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i7)), vi_shift));
          const int16x8_t vxk7 =
              vreinterpretq_s16_u16(vsubl_u8(vk7, vkernel_zero_point));
          const int16x8_t vxi7 =
              vreinterpretq_s16_u16(vsubl_u8(vi7, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk7), vget_low_s16(vxi7));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk7), vget_high_s16(vxi7));

          const uint8x8_t vk8 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi8 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i8)), vi_shift));
          const int16x8_t vxk8 =
              vreinterpretq_s16_u16(vsubl_u8(vk8, vkernel_zero_point));
          const int16x8_t vxi8 =
              vreinterpretq_s16_u16(vsubl_u8(vi8, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk8), vget_low_s16(vxi8));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk8), vget_high_s16(vxi8));

          int32x4_t vacc_lo = vaddq_s32(vaccX0_lo, vaccX1_lo);
          int32x4_t vacc_hi = vaddq_s32(vaccX0_hi, vaccX1_hi);

          vst1q_s32(outacc, vacc_lo);
          outacc += 4;
          vst1q_s32(outacc, vacc_hi);
          outacc += 4;
        }
      }
      {
        const uint8_t* i0 = input[9];
        const uint8_t* i1 = input[10];
        const uint8_t* i2 = input[11];
        const uint8_t* i3 = input[12];
        const uint8_t* i4 = input[13];
        const uint8_t* i5 = input[14];
        const uint8_t* i6 = input[15];
        const uint8_t* i7 = input[16];
        const uint8_t* i8 = input[17];
        output = output_start;
        outacc = outacc32;

        size_t c = channels;
        const uint8_t* kernel_zero_points_ptr =
            quantization_params->neon.kernel_zero_points + channels;
        for (; c >= 8; c -= 8) {
          const uint8x8_t vkernel_zero_point =
              vld1_u8(kernel_zero_points_ptr - c);
          const uint8x8_t vk0 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi0 = vld1_u8(i0);
          i0 += 8;
          const int16x8_t vxk0 =
              vreinterpretq_s16_u16(vsubl_u8(vk0, vkernel_zero_point));
          const int16x8_t vxi0 =
              vreinterpretq_s16_u16(vsubl_u8(vi0, vinput_zero_point));
          int32x4_t vaccX0_lo =
              vmull_s16(vget_low_s16(vxk0), vget_low_s16(vxi0));
          int32x4_t vaccX0_hi =
              vmull_s16(vget_high_s16(vxk0), vget_high_s16(vxi0));

          const uint8x8_t vk1 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi1 = vld1_u8(i1);
          i1 += 8;
          const int16x8_t vxk1 =
              vreinterpretq_s16_u16(vsubl_u8(vk1, vkernel_zero_point));
          const int16x8_t vxi1 =
              vreinterpretq_s16_u16(vsubl_u8(vi1, vinput_zero_point));
          int32x4_t vaccX1_lo =
              vmull_s16(vget_low_s16(vxk1), vget_low_s16(vxi1));
          int32x4_t vaccX1_hi =
              vmull_s16(vget_high_s16(vxk1), vget_high_s16(vxi1));

          const uint8x8_t vk2 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi2 = vld1_u8(i2);
          i2 += 8;
          const int16x8_t vxk2 =
              vreinterpretq_s16_u16(vsubl_u8(vk2, vkernel_zero_point));
          const int16x8_t vxi2 =
              vreinterpretq_s16_u16(vsubl_u8(vi2, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk2), vget_low_s16(vxi2));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk2), vget_high_s16(vxi2));

          const uint8x8_t vk3 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi3 = vld1_u8(i3);
          i3 += 8;
          const int16x8_t vxk3 =
              vreinterpretq_s16_u16(vsubl_u8(vk3, vkernel_zero_point));
          const int16x8_t vxi3 =
              vreinterpretq_s16_u16(vsubl_u8(vi3, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk3), vget_low_s16(vxi3));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk3), vget_high_s16(vxi3));

          const uint8x8_t vk4 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi4 = vld1_u8(i4);
          i4 += 8;
          const int16x8_t vxk4 =
              vreinterpretq_s16_u16(vsubl_u8(vk4, vkernel_zero_point));
          const int16x8_t vxi4 =
              vreinterpretq_s16_u16(vsubl_u8(vi4, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk4), vget_low_s16(vxi4));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk4), vget_high_s16(vxi4));

          const uint8x8_t vk5 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi5 = vld1_u8(i5);
          i5 += 8;
          const int16x8_t vxk5 =
              vreinterpretq_s16_u16(vsubl_u8(vk5, vkernel_zero_point));
          const int16x8_t vxi5 =
              vreinterpretq_s16_u16(vsubl_u8(vi5, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk5), vget_low_s16(vxi5));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk5), vget_high_s16(vxi5));

          const uint8x8_t vk6 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi6 = vld1_u8(i6);
          i6 += 8;
          const int16x8_t vxk6 =
              vreinterpretq_s16_u16(vsubl_u8(vk6, vkernel_zero_point));
          const int16x8_t vxi6 =
              vreinterpretq_s16_u16(vsubl_u8(vi6, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk6), vget_low_s16(vxi6));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk6), vget_high_s16(vxi6));

          const uint8x8_t vk7 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi7 = vld1_u8(i7);
          i7 += 8;
          const int16x8_t vxk7 =
              vreinterpretq_s16_u16(vsubl_u8(vk7, vkernel_zero_point));
          const int16x8_t vxi7 =
              vreinterpretq_s16_u16(vsubl_u8(vi7, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk7), vget_low_s16(vxi7));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk7), vget_high_s16(vxi7));

          const uint8x8_t vk8 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi8 = vld1_u8(i8);
          i8 += 8;
          const int16x8_t vxk8 =
              vreinterpretq_s16_u16(vsubl_u8(vk8, vkernel_zero_point));
          const int16x8_t vxi8 =
              vreinterpretq_s16_u16(vsubl_u8(vi8, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk8), vget_low_s16(vxi8));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk8), vget_high_s16(vxi8));

          int32x4_t vacc_lo = vaddq_s32(vaccX0_lo, vaccX1_lo);
          int32x4_t vacc_hi = vaddq_s32(vaccX0_hi, vaccX1_hi);

          const int32x4_t vacc_lo_old = vld1q_s32(outacc);
          const int32x4_t vacc_hi_old = vld1q_s32(outacc + 4);
          vacc_lo = vaddq_s32(vacc_lo, vacc_lo_old);
          vacc_hi = vaddq_s32(vacc_hi, vacc_hi_old);
          vst1q_s32(outacc, vacc_lo);
          outacc += 4;
          vst1q_s32(outacc, vacc_hi);
          outacc += 4;
        }
        if (c != 0) {
          const size_t c_predecrement = 8 - c;
          const int64x1_t vi_shift = vmov_n_s64(-8 * c_predecrement);
          i0 -= c_predecrement;
          i1 -= c_predecrement;
          i2 -= c_predecrement;
          i3 -= c_predecrement;
          i4 -= c_predecrement;
          i5 -= c_predecrement;
          i6 -= c_predecrement;
          i7 -= c_predecrement;
          i8 -= c_predecrement;

          const uint8x8_t vkernel_zero_point = vld1_u8(
              &quantization_params->neon.kernel_zero_points[channels - c]);
          const uint8x8_t vk0 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi0 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i0)), vi_shift));
          const int16x8_t vxk0 =
              vreinterpretq_s16_u16(vsubl_u8(vk0, vkernel_zero_point));
          const int16x8_t vxi0 =
              vreinterpretq_s16_u16(vsubl_u8(vi0, vinput_zero_point));
          int32x4_t vaccX0_lo =
              vmull_s16(vget_low_s16(vxk0), vget_low_s16(vxi0));
          int32x4_t vaccX0_hi =
              vmull_s16(vget_high_s16(vxk0), vget_high_s16(vxi0));

          const uint8x8_t vk1 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi1 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i1)), vi_shift));
          const int16x8_t vxk1 =
              vreinterpretq_s16_u16(vsubl_u8(vk1, vkernel_zero_point));
          const int16x8_t vxi1 =
              vreinterpretq_s16_u16(vsubl_u8(vi1, vinput_zero_point));
          int32x4_t vaccX1_lo =
              vmull_s16(vget_low_s16(vxk1), vget_low_s16(vxi1));
          int32x4_t vaccX1_hi =
              vmull_s16(vget_high_s16(vxk1), vget_high_s16(vxi1));

          const uint8x8_t vk2 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi2 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i2)), vi_shift));
          const int16x8_t vxk2 =
              vreinterpretq_s16_u16(vsubl_u8(vk2, vkernel_zero_point));
          const int16x8_t vxi2 =
              vreinterpretq_s16_u16(vsubl_u8(vi2, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk2), vget_low_s16(vxi2));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk2), vget_high_s16(vxi2));

          const uint8x8_t vk3 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi3 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i3)), vi_shift));
          const int16x8_t vxk3 =
              vreinterpretq_s16_u16(vsubl_u8(vk3, vkernel_zero_point));
          const int16x8_t vxi3 =
              vreinterpretq_s16_u16(vsubl_u8(vi3, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk3), vget_low_s16(vxi3));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk3), vget_high_s16(vxi3));

          const uint8x8_t vk4 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi4 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i4)), vi_shift));
          const int16x8_t vxk4 =
              vreinterpretq_s16_u16(vsubl_u8(vk4, vkernel_zero_point));
          const int16x8_t vxi4 =
              vreinterpretq_s16_u16(vsubl_u8(vi4, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk4), vget_low_s16(vxi4));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk4), vget_high_s16(vxi4));

          const uint8x8_t vk5 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi5 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i5)), vi_shift));
          const int16x8_t vxk5 =
              vreinterpretq_s16_u16(vsubl_u8(vk5, vkernel_zero_point));
          const int16x8_t vxi5 =
              vreinterpretq_s16_u16(vsubl_u8(vi5, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk5), vget_low_s16(vxi5));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk5), vget_high_s16(vxi5));

          const uint8x8_t vk6 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi6 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i6)), vi_shift));
          const int16x8_t vxk6 =
              vreinterpretq_s16_u16(vsubl_u8(vk6, vkernel_zero_point));
          const int16x8_t vxi6 =
              vreinterpretq_s16_u16(vsubl_u8(vi6, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk6), vget_low_s16(vxi6));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk6), vget_high_s16(vxi6));

          const uint8x8_t vk7 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi7 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i7)), vi_shift));
          const int16x8_t vxk7 =
              vreinterpretq_s16_u16(vsubl_u8(vk7, vkernel_zero_point));
          const int16x8_t vxi7 =
              vreinterpretq_s16_u16(vsubl_u8(vi7, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk7), vget_low_s16(vxi7));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk7), vget_high_s16(vxi7));

          const uint8x8_t vk8 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi8 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i8)), vi_shift));
          const int16x8_t vxk8 =
              vreinterpretq_s16_u16(vsubl_u8(vk8, vkernel_zero_point));
          const int16x8_t vxi8 =
              vreinterpretq_s16_u16(vsubl_u8(vi8, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk8), vget_low_s16(vxi8));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk8), vget_high_s16(vxi8));

          int32x4_t vacc_lo = vaddq_s32(vaccX0_lo, vaccX1_lo);
          int32x4_t vacc_hi = vaddq_s32(vaccX0_hi, vaccX1_hi);

          const int32x4_t vacc_lo_old = vld1q_s32(outacc);
          const int32x4_t vacc_hi_old = vld1q_s32(outacc + 4);
          vacc_lo = vaddq_s32(vacc_lo, vacc_lo_old);
          vacc_hi = vaddq_s32(vacc_hi, vacc_hi_old);
          vst1q_s32(outacc, vacc_lo);
          outacc += 4;
          vst1q_s32(outacc, vacc_hi);
          outacc += 4;
        }
      }

      {
        const uint8_t* i0 = input[18];
        const uint8_t* i1 = input[19];
        const uint8_t* i2 = input[20];
        const uint8_t* i3 = input[21];
        const uint8_t* i4 = input[22];
        const uint8_t* i5 = input[23];
        const uint8_t* i6 = input[24];
        const uint8_t* i7 = input[25];
        const uint8_t* i8 = input[26];
        input = (const uint8_t**)((uintptr_t)input + input_col_stride);
        output = output_start;
        outacc = outacc32;

        size_t c = channels;
        const uint8_t* kernel_zero_points_ptr =
            quantization_params->neon.kernel_zero_points + channels;
        const float* requantization_scales_ptr =
            quantization_params->neon.requantization_scales + channels;
        for (; c >= 8; c -= 8) {
          const uint8x8_t vkernel_zero_point =
              vld1_u8(kernel_zero_points_ptr - c);
          const uint8x8_t vk0 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi0 = vld1_u8(i0);
          i0 += 8;
          const int16x8_t vxk0 =
              vreinterpretq_s16_u16(vsubl_u8(vk0, vkernel_zero_point));
          const int16x8_t vxi0 =
              vreinterpretq_s16_u16(vsubl_u8(vi0, vinput_zero_point));
          int32x4_t vaccX0_lo =
              vmull_s16(vget_low_s16(vxk0), vget_low_s16(vxi0));
          int32x4_t vaccX0_hi =
              vmull_s16(vget_high_s16(vxk0), vget_high_s16(vxi0));

          const uint8x8_t vk1 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi1 = vld1_u8(i1);
          i1 += 8;
          const int16x8_t vxk1 =
              vreinterpretq_s16_u16(vsubl_u8(vk1, vkernel_zero_point));
          const int16x8_t vxi1 =
              vreinterpretq_s16_u16(vsubl_u8(vi1, vinput_zero_point));
          int32x4_t vaccX1_lo =
              vmull_s16(vget_low_s16(vxk1), vget_low_s16(vxi1));
          int32x4_t vaccX1_hi =
              vmull_s16(vget_high_s16(vxk1), vget_high_s16(vxi1));

          const uint8x8_t vk2 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi2 = vld1_u8(i2);
          i2 += 8;
          const int16x8_t vxk2 =
              vreinterpretq_s16_u16(vsubl_u8(vk2, vkernel_zero_point));
          const int16x8_t vxi2 =
              vreinterpretq_s16_u16(vsubl_u8(vi2, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk2), vget_low_s16(vxi2));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk2), vget_high_s16(vxi2));

          const uint8x8_t vk3 = vld1_u8(w);
          w += 8;
          const uint8x8_t vi3 = vld1_u8(i3);
          i3 += 8;
          const int16x8_t vxk3 =
              vreinterpretq_s16_u16(vsubl_u8(vk3, vkernel_zero_point));
          const int16x8_t vxi3 =
              vreinterpretq_s16_u16(vsubl_u8(vi3, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk3), vget_low_s16(vxi3));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk3), vget_high_s16(vxi3));

          const uint8x8_t vk4 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi4 = vld1_u8(i4);
          i4 += 8;
          const int16x8_t vxk4 =
              vreinterpretq_s16_u16(vsubl_u8(vk4, vkernel_zero_point));
          const int16x8_t vxi4 =
              vreinterpretq_s16_u16(vsubl_u8(vi4, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk4), vget_low_s16(vxi4));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk4), vget_high_s16(vxi4));

          const uint8x8_t vk5 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi5 = vld1_u8(i5);
          i5 += 8;
          const int16x8_t vxk5 =
              vreinterpretq_s16_u16(vsubl_u8(vk5, vkernel_zero_point));
          const int16x8_t vxi5 =
              vreinterpretq_s16_u16(vsubl_u8(vi5, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk5), vget_low_s16(vxi5));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk5), vget_high_s16(vxi5));

          const uint8x8_t vk6 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi6 = vld1_u8(i6);
          i6 += 8;
          const int16x8_t vxk6 =
              vreinterpretq_s16_u16(vsubl_u8(vk6, vkernel_zero_point));
          const int16x8_t vxi6 =
              vreinterpretq_s16_u16(vsubl_u8(vi6, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk6), vget_low_s16(vxi6));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk6), vget_high_s16(vxi6));

          const uint8x8_t vk7 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi7 = vld1_u8(i7);
          i7 += 8;
          const int16x8_t vxk7 =
              vreinterpretq_s16_u16(vsubl_u8(vk7, vkernel_zero_point));
          const int16x8_t vxi7 =
              vreinterpretq_s16_u16(vsubl_u8(vi7, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk7), vget_low_s16(vxi7));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk7), vget_high_s16(vxi7));

          const uint8x8_t vk8 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi8 = vld1_u8(i8);
          i8 += 8;
          const int16x8_t vxk8 =
              vreinterpretq_s16_u16(vsubl_u8(vk8, vkernel_zero_point));
          const int16x8_t vxi8 =
              vreinterpretq_s16_u16(vsubl_u8(vi8, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk8), vget_low_s16(vxi8));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk8), vget_high_s16(vxi8));

          int32x4_t vacc_lo = vaddq_s32(vaccX0_lo, vaccX1_lo);
          int32x4_t vacc_hi = vaddq_s32(vaccX0_hi, vaccX1_hi);

          const int32x4_t vacc_lo_old = vld1q_s32(outacc);
          outacc += 4;
          const int32x4_t vacc_hi_old = vld1q_s32(outacc);
          outacc += 4;
          vacc_lo = vaddq_s32(vacc_lo, vacc_lo_old);
          vacc_hi = vaddq_s32(vacc_hi, vacc_hi_old);

          const float32x4_t requantization_scale_v_lo =
              vld1q_f32(requantization_scales_ptr - c);
          const float32x4_t requantization_scale_v_hi =
              vld1q_f32(requantization_scales_ptr - c + 4);

          const float32x4_t vacc_lo_f =
              vmulq_f32(vcvtq_f32_s32(vacc_lo), requantization_scale_v_lo);
          const float32x4_t vacc_hi_f =
              vmulq_f32(vcvtq_f32_s32(vacc_hi), requantization_scale_v_hi);

#ifdef __aarch64__
          vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
          vacc_hi = vcvtnq_s32_f32(vacc_hi_f);

          const int16x8_t vacc = vqaddq_s16(
              vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi),
              voutput_zero_point);

          uint8x8_t vout = vqmovun_s16(vacc);
          vout = vmax_u8(vout, voutput_min);
          vout = vmin_u8(vout, voutput_max);
#else
          const float32x4_t vacc_lo_f_clamped =
              vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
          const float32x4_t vacc_hi_f_clamped =
              vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
          vacc_lo = vsubq_s32(
              vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f_clamped, vfmagic)),
              vimagic);
          vacc_hi = vsubq_s32(
              vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f_clamped, vfmagic)),
              vimagic);
          const int16x8_t vacc =
              vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

          uint8x8_t vout = vqmovun_s16(vacc);
#endif

          vst1_u8(output, vout);
          output += 8;
        }
        if (c != 0) {
          const size_t c_predecrement = 8 - c;
          const int64x1_t vi_shift = vmov_n_s64(-8 * c_predecrement);
          i0 -= c_predecrement;
          i1 -= c_predecrement;
          i2 -= c_predecrement;
          i3 -= c_predecrement;
          i4 -= c_predecrement;
          i5 -= c_predecrement;
          i6 -= c_predecrement;
          i7 -= c_predecrement;
          i8 -= c_predecrement;

          const uint8x8_t vkernel_zero_point = vld1_u8(
              &quantization_params->neon.kernel_zero_points[channels - c]);
          const uint8x8_t vk0 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi0 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i0)), vi_shift));
          const int16x8_t vxk0 =
              vreinterpretq_s16_u16(vsubl_u8(vk0, vkernel_zero_point));
          const int16x8_t vxi0 =
              vreinterpretq_s16_u16(vsubl_u8(vi0, vinput_zero_point));
          int32x4_t vaccX0_lo =
              vmull_s16(vget_low_s16(vxk0), vget_low_s16(vxi0));
          int32x4_t vaccX0_hi =
              vmull_s16(vget_high_s16(vxk0), vget_high_s16(vxi0));

          const uint8x8_t vk1 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi1 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i1)), vi_shift));
          const int16x8_t vxk1 =
              vreinterpretq_s16_u16(vsubl_u8(vk1, vkernel_zero_point));
          const int16x8_t vxi1 =
              vreinterpretq_s16_u16(vsubl_u8(vi1, vinput_zero_point));
          int32x4_t vaccX1_lo =
              vmull_s16(vget_low_s16(vxk1), vget_low_s16(vxi1));
          int32x4_t vaccX1_hi =
              vmull_s16(vget_high_s16(vxk1), vget_high_s16(vxi1));

          const uint8x8_t vk2 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi2 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i2)), vi_shift));
          const int16x8_t vxk2 =
              vreinterpretq_s16_u16(vsubl_u8(vk2, vkernel_zero_point));
          const int16x8_t vxi2 =
              vreinterpretq_s16_u16(vsubl_u8(vi2, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk2), vget_low_s16(vxi2));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk2), vget_high_s16(vxi2));

          const uint8x8_t vk3 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi3 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i3)), vi_shift));
          const int16x8_t vxk3 =
              vreinterpretq_s16_u16(vsubl_u8(vk3, vkernel_zero_point));
          const int16x8_t vxi3 =
              vreinterpretq_s16_u16(vsubl_u8(vi3, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk3), vget_low_s16(vxi3));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk3), vget_high_s16(vxi3));

          const uint8x8_t vk4 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi4 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i4)), vi_shift));
          const int16x8_t vxk4 =
              vreinterpretq_s16_u16(vsubl_u8(vk4, vkernel_zero_point));
          const int16x8_t vxi4 =
              vreinterpretq_s16_u16(vsubl_u8(vi4, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk4), vget_low_s16(vxi4));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk4), vget_high_s16(vxi4));

          const uint8x8_t vk5 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi5 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i5)), vi_shift));
          const int16x8_t vxk5 =
              vreinterpretq_s16_u16(vsubl_u8(vk5, vkernel_zero_point));
          const int16x8_t vxi5 =
              vreinterpretq_s16_u16(vsubl_u8(vi5, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk5), vget_low_s16(vxi5));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk5), vget_high_s16(vxi5));

          const uint8x8_t vk6 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi6 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i6)), vi_shift));
          const int16x8_t vxk6 =
              vreinterpretq_s16_u16(vsubl_u8(vk6, vkernel_zero_point));
          const int16x8_t vxi6 =
              vreinterpretq_s16_u16(vsubl_u8(vi6, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk6), vget_low_s16(vxi6));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk6), vget_high_s16(vxi6));

          const uint8x8_t vk7 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi7 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i7)), vi_shift));
          const int16x8_t vxk7 =
              vreinterpretq_s16_u16(vsubl_u8(vk7, vkernel_zero_point));
          const int16x8_t vxi7 =
              vreinterpretq_s16_u16(vsubl_u8(vi7, vinput_zero_point));
          vaccX1_lo =
              vmlal_s16(vaccX1_lo, vget_low_s16(vxk7), vget_low_s16(vxi7));
          vaccX1_hi =
              vmlal_s16(vaccX1_hi, vget_high_s16(vxk7), vget_high_s16(vxi7));

          const uint8x8_t vk8 = vld1_u8(w);
          w = (void*)((uintptr_t)w + sizeof(uint8x8_t));
          const uint8x8_t vi8 = vreinterpret_u8_u64(
              vshl_u64(vreinterpret_u64_u8(vld1_u8(i8)), vi_shift));
          const int16x8_t vxk8 =
              vreinterpretq_s16_u16(vsubl_u8(vk8, vkernel_zero_point));
          const int16x8_t vxi8 =
              vreinterpretq_s16_u16(vsubl_u8(vi8, vinput_zero_point));
          vaccX0_lo =
              vmlal_s16(vaccX0_lo, vget_low_s16(vxk8), vget_low_s16(vxi8));
          vaccX0_hi =
              vmlal_s16(vaccX0_hi, vget_high_s16(vxk8), vget_high_s16(vxi8));

          int32x4_t vacc_lo = vaddq_s32(vaccX0_lo, vaccX1_lo);
          int32x4_t vacc_hi = vaddq_s32(vaccX0_hi, vaccX1_hi);

          const int32x4_t vacc_lo_old = vld1q_s32(outacc);
          const int32x4_t vacc_hi_old = vld1q_s32(outacc + 4);
          vacc_lo = vaddq_s32(vacc_lo, vacc_lo_old);
          vacc_hi = vaddq_s32(vacc_hi, vacc_hi_old);

          const float32x4_t requantization_scale_v_lo = vld1q_f32(
              &quantization_params->neon.requantization_scales[channels - c]);
          const float32x4_t requantization_scale_v_hi =
              vld1q_f32(&quantization_params->neon
                             .requantization_scales[channels - c + 4]);

          const float32x4_t vacc_lo_f =
              vmulq_f32(vcvtq_f32_s32(vacc_lo), requantization_scale_v_lo);
          const float32x4_t vacc_hi_f =
              vmulq_f32(vcvtq_f32_s32(vacc_hi), requantization_scale_v_hi);

#ifdef __aarch64__
          vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
          vacc_hi = vcvtnq_s32_f32(vacc_hi_f);

          const int16x8_t vacc = vqaddq_s16(
              vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi),
              voutput_zero_point);

          uint8x8_t vout = vqmovun_s16(vacc);
          vout = vmax_u8(vout, voutput_min);
          vout = vmin_u8(vout, voutput_max);
#else
          const float32x4_t vacc_lo_f_clamped =
              vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
          const float32x4_t vacc_hi_f_clamped =
              vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
          vacc_lo = vsubq_s32(
              vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f_clamped, vfmagic)),
              vimagic);
          vacc_hi = vsubq_s32(
              vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f_clamped, vfmagic)),
              vimagic);
          const int16x8_t vacc =
              vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

          uint8x8_t vout = vqmovun_s16(vacc);
#endif

          if (c & 4) {
            vst1_lane_u32(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u32_u8(vout),
                0);
            output += 4;
            vout = vext_u8(vout, vout, 4);
          }
          if (c & 2) {
            vst1_lane_u16(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u16_u8(vout),
                0);
            output += 2;
            vout = vext_u8(vout, vout, 2);
          }
          if (c & 1) {
            vst1_lane_u8(__builtin_assume_aligned(output, 1), vout, 0);
            output++;
          }
        }
      }

      output = (uint8_t*)((uintptr_t)output + output_increment);
    }
    input = (const uint8_t**)((uintptr_t)input_row_start + input_row_stride);
  }
}
