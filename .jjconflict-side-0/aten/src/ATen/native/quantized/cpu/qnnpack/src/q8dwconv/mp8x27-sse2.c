/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>

#include <math.h>
#include <qnnpack/q8dwconv.h>

void pytorch_q8dwconv_ukernel_mp8x27__sse2(
    size_t channels,
    size_t output_height,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    int32_t* outacc32,
    uint8_t* output,
    size_t input_row_stride,
    size_t input_col_stride,
    size_t output_increment,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  const int16_t input_zero_point =
      quantization_params->sse2.input_zero_point[0];
  const uint8_t* kernel_zero_points =
      quantization_params->sse2.kernel_zero_points;
  const float* requantization_scales =
      quantization_params->sse2.requantization_scales;
  const int16_t output_zero_point =
      quantization_params->sse2.output_zero_point[0];
  const uint8_t output_min = quantization_params->sse2.output_min[0];
  const uint8_t output_max = quantization_params->sse2.output_max[0];

  union {
    const uint8_t* as_uint8_ptr;
    const int32_t* as_int32_ptr;
  } weights_ptr = {weights};

  const size_t cr_block = 8;
  const size_t kernel_depth = 3;
  const size_t kernel_height = 3;
  const size_t kernel_width = 3;

  const size_t num_groups = ((channels - 1) / cr_block) + 1;

  const size_t yz_block = kernel_depth * kernel_height;
  const size_t yz_bias_size = (cr_block * sizeof(int32_t));
  const size_t yz_weight_size = yz_block * cr_block;

  for (size_t output_y = 0; output_y < output_height; output_y++) {
    const uint8_t** input_row_start = input;
    for (size_t output_x = 0; output_x < output_width; output_x++) {
      for (size_t c = 0; c < channels; c++) {
        int32_t accumulator =
            (weights_ptr.as_int32_ptr +
             ((c / cr_block) * (yz_bias_size + yz_weight_size) /
              sizeof(int32_t)))[c % cr_block];
        for (int x = 0; x < kernel_width; x++) {
          for (int y = 0; y < kernel_height; y++) {
            for (int z = 0; z < kernel_depth; z++) {
              int32_t input_val =
                  (int32_t)(input
                                [z + kernel_depth * y +
                                 kernel_depth * kernel_height * x][c]);
/*
 * The weights are setup as follows
 * (where Wzyx means the weight for kernel position Z=z, Y=y, X=x, and cn means
 * channel n)
 *
 *  x = 0 (first yz slice) region:
 *  0_______________32______________40______________48    96______________104
 *  |     BIAS      |     W000      |     W100      |     |     W220      |
 *  | c0 | ... | c8 | c0 | ... | c8 | c0 | ... | c8 | ... | c0 | ... | c8 |
 *   -----------------------------------------------       ---------------
 *    (4 bytes x 8)    (1 byte x 8)    (1 byte x 8)          (1 byte x 8)
 *
 *  104_____________136_____________144_____________152   200_____________208
 *  |     BIAS      |     W000      |     W100      |     |     W220      |
 *  | c8 | ... | c15| c8 | ... | c15| c8 | ... | c15| ... | c8 | ... | c15|
 *   -----------------------------------------------       ---------------
 *
 *  ... Repeat the above arrangement over all chunks of 8 channels, then ...
 *
 *  x = 1 (second yz slice) region:
 *  +0_______________+8_____________+16    +64_____________+72
 *  |     W001      |     W101      |     |     W221      |
 *  | c0 | ... | c7 | c0 | ... | c7 | ... | c0 | ... | c7 |
 *   -------------------------------       ---------------
 *  +72_____________+80____________+88    +136____________+144
 *  |     W001      |     W101      |     |     W221      |
 *  | c8 | ... | c15| c8 | ... | c15| ... | c8 | ... | c15|
 *   -------------------------------       ---------------
 *
 *  ... Repeat the above arrangement over all chunks of 8 channels, then ...
 *
 *  x = 2 (third yz slice) region:
 *  +0_______________+8_____________+16    +64_____________+72
 *  |     W002      |     W102      |     |     W222      |
 *  | c0 | ... | c7 | c0 | ... | c7 | ... | c0 | ... | c7 |
 *   -------------------------------       ---------------
 *   +72____________+80____________+88    +136____________+144
 *  |     W002      |     W102      |     |     W222      |
 *  | c8 | ... | c15| c8 | ... | c15| ... | c8 | ... | c15|
 *   -------------------------------       ---------------
 *
 *  ... Repeat the above arrangement over all chunks of 8 channels
 */
              size_t yz_slice_advance_per_group = 0; // Get to yz slice
              size_t channel_chunk_advance = 0; // Get to 8-channel chunk
              size_t bias_advance = 0; // Get past bias
              if (x == 0) {
                channel_chunk_advance = yz_bias_size + yz_weight_size;
                bias_advance = yz_bias_size;
              } else {
                yz_slice_advance_per_group = yz_bias_size + x * yz_weight_size;
                channel_chunk_advance = yz_weight_size;
              }
              const size_t yz_position_advance =
                  ((kernel_depth * y + z) * cr_block); // Get to y and z
              const uint8_t* w_zyxc_ptr = weights_ptr.as_uint8_ptr +
                  yz_slice_advance_per_group * num_groups +
                  channel_chunk_advance * (c / cr_block) +
                  bias_advance +
                  yz_position_advance;
              int32_t w = (int32_t)(w_zyxc_ptr[c % cr_block]);
              int32_t kernel_zero_point =
                  (int32_t)(kernel_zero_points[c % channels]);
              accumulator +=
                  (w - kernel_zero_point) * (input_val - input_zero_point);
            }
          }
        }

        // Requantization
        // 1) Convert to float and multiply by scale
        double scaled_accumulator =
            accumulator * ((double)(requantization_scales[c]));
        // 2) Cast to int
        int32_t int_accumulator = (int32_t)(nearbyint(scaled_accumulator));
        // 3) Add zero point
        int32_t shifted_accumulator = int_accumulator + output_zero_point;
        // 4) Clip to [output_min, output_max]
        if (shifted_accumulator > output_max) {
          shifted_accumulator = output_max;
        } else if (shifted_accumulator < output_min) {
          shifted_accumulator = output_min;
        }
        output[c] = (uint8_t)(shifted_accumulator);
      }
      input = (const uint8_t**)((uint8_t*)input + input_col_stride);
      output += channels + output_increment;
    }
    input = (const uint8_t**)((uint8_t*)input_row_start + input_row_stride);
  }
}
