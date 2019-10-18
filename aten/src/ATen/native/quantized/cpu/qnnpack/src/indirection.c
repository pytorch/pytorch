/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

#include <fxdiv.h>

#include <qnnpack/indirection.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>

void pytorch_qnnp_indirection_init_conv2d(
    pytorch_qnnp_operator_t op,
    size_t output_tile_size,
    size_t tiled_output_size) {
  const void** indirection_buffer = op->indirection_buffer;
  const void* input = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride;
  const void* zero = op->zero_pointer;
  const size_t groups = op->groups;
  const size_t group_input_channels = op->group_input_channels;
  const size_t batch_size = op->batch_size;
  const size_t input_height = op->input_height;
  const size_t input_width = op->input_width;
  const size_t output_height = op->output_height;
  const size_t output_width = op->output_width;
  const size_t kernel_height = op->kernel_height;
  const size_t kernel_width = op->kernel_width;
  const size_t stride_height = op->stride_height;
  const size_t stride_width = op->stride_width;
  const size_t dilation_height = op->dilation_height;
  const size_t dilation_width = op->dilation_width;
  const size_t input_padding_top = op->input_padding_top;
  const size_t input_padding_left = op->input_padding_left;

  const size_t output_size = output_height * output_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const struct fxdiv_divisor_size_t output_width_divisor =
      fxdiv_init_size_t(output_width);
  for (size_t group = 0; group < groups; group++) {
    for (size_t image = 0; image < batch_size; image++) {
      for (size_t output_tile_start = 0; output_tile_start < tiled_output_size;
           output_tile_start += output_tile_size) {
        for (size_t output_tile_offset = 0;
             output_tile_offset < output_tile_size;
             output_tile_offset++) {
          const size_t tiled_output_index =
              output_tile_start + output_tile_offset;
          const size_t output_index = min(tiled_output_index, output_size - 1);
          const struct fxdiv_result_size_t output_index_components =
              fxdiv_divide_size_t(output_index, output_width_divisor);
          const size_t output_y = output_index_components.quotient;
          const size_t output_x = output_index_components.remainder;
          for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
            const size_t input_y = output_y * stride_height +
                kernel_y * dilation_height - input_padding_top;
            if (input_y < input_height) {
              for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                const size_t input_x = output_x * stride_width +
                    kernel_x * dilation_width - input_padding_left;
                const size_t index = (group * batch_size + image) *
                        tiled_output_size * kernel_size +
                    output_tile_start * kernel_size +
                    (kernel_y * kernel_width + kernel_x) * output_tile_size +
                    output_tile_offset;
                if (input_x < input_width) {
                  indirection_buffer[index] = (char*)input +
                      ((image * input_height + input_y) * input_width +
                       input_x) *
                          input_pixel_stride +
                      group * group_input_channels;
                } else {
                  indirection_buffer[index] = zero;
                }
              }
            } else {
              for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                const size_t index = (group * batch_size + image) *
                        tiled_output_size * kernel_size +
                    output_tile_start * kernel_size +
                    (kernel_y * kernel_width + kernel_x) * output_tile_size +
                    output_tile_offset;
                indirection_buffer[index] = zero;
              }
            }
          }
        }
      }
    }
  }
}

void pytorch_qnnp_indirection_init_dwconv2d(
    pytorch_qnnp_operator_t op,
    size_t batch_start,
    size_t step_height,
    size_t step_width) {
  const void** indirection_buffer = op->indirection_buffer;
  const void* input = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride;
  const void* zero = op->zero_pointer;
  const size_t batch_size = op->batch_size;
  const size_t input_height = op->input_height;
  const size_t input_width = op->input_width;
  const size_t output_height = op->output_height;
  const size_t output_width = op->output_width;
  const size_t kernel_height = op->kernel_height;
  const size_t kernel_width = op->kernel_width;
  const size_t stride_height = op->stride_height;
  const size_t stride_width = op->stride_width;
  const size_t dilation_height = op->dilation_height;
  const size_t dilation_width = op->dilation_width;
  const size_t input_padding_top = op->input_padding_top;
  const size_t input_padding_left = op->input_padding_left;

  for (size_t image = batch_start; image < batch_size; image++) {
    for (size_t output_y = 0; output_y < output_height; output_y++) {
      for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
        const size_t input_y = output_y * stride_height +
            kernel_y * dilation_height - input_padding_top;
        if (input_y < input_height) {
          for (size_t output_x = 0; output_x < output_width; output_x++) {
            for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
              const size_t input_x = output_x * stride_width +
                  kernel_x * dilation_width - input_padding_left;
              const size_t index =
                  (image * output_height + output_y) * step_height +
                  output_x * step_width * kernel_height +
                  kernel_x * kernel_height + kernel_y;
              if (input_x < input_width) {
                indirection_buffer[index] = (char*)input +
                    ((image * input_height + input_y) * input_width + input_x) *
                        input_pixel_stride;
              } else {
                indirection_buffer[index] = zero;
              }
            }
          }
        } else {
          for (size_t output_x = 0; output_x < output_width; output_x++) {
            for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
              const size_t index =
                  (image * output_height + output_y) * step_height +
                  output_x * step_width * kernel_height +
                  kernel_x * kernel_height + kernel_y;
              indirection_buffer[index] = zero;
            }
          }
        }
      }
    }
  }
}

void pytorch_qnnp_indirection_init_deconv2d(
    pytorch_qnnp_operator_t op,
    size_t output_tile_size,
    size_t tiled_output_size) {
  const void** indirection_buffer = op->indirection_buffer;
  const void* input = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride;
  const void* zero = op->zero_pointer;
  const size_t groups = op->groups;
  const size_t group_input_channels = op->group_input_channels;
  const size_t batch_size = op->batch_size;
  const size_t input_height = op->input_height;
  const size_t input_width = op->input_width;
  const size_t output_height = op->output_height;
  const size_t output_width = op->output_width;
  const size_t kernel_height = op->kernel_height;
  const size_t kernel_width = op->kernel_width;
  const size_t stride_height = op->stride_height;
  const size_t stride_width = op->stride_width;
  const size_t dilation_height = op->dilation_height;
  const size_t dilation_width = op->dilation_width;
  const size_t input_padding_top = op->input_padding_top;
  const size_t input_padding_left = op->input_padding_left;

  const size_t output_size = output_height * output_width;
  const size_t kernel_size = kernel_height * kernel_width;

  for (size_t group = 0; group < groups; group++) {
    for (size_t image = 0; image < batch_size; image++) {
      for (size_t output_tile_start = 0; output_tile_start < tiled_output_size;
           output_tile_start += output_tile_size) {
        for (size_t output_tile_offset = 0;
             output_tile_offset < output_tile_size;
             output_tile_offset++) {
          const size_t tiled_output_index =
              output_tile_start + output_tile_offset;
          const size_t output_index = min(tiled_output_index, output_size - 1);
          const size_t output_y = output_index / output_width;
          const size_t output_x = output_index % output_width;
          for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
            const size_t y =
                output_y + input_padding_top - kernel_y * dilation_height;
            const size_t input_y = y / stride_height;
            for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
              const size_t x =
                  output_x + input_padding_left - kernel_x * dilation_width;
              const size_t input_x = x / stride_width;
              const size_t index = (group * batch_size + image) *
                      tiled_output_size * kernel_size +
                  output_tile_start * kernel_size +
                  (kernel_y * kernel_width + kernel_x) * output_tile_size +
                  output_tile_offset;
              if (input_y * stride_height == y && input_y < input_height &&
                  input_x * stride_width == x && input_x < input_width) {
                indirection_buffer[index] = (char*)input +
                    ((image * input_height + input_y) * input_width + input_x) *
                        input_pixel_stride +
                    group * group_input_channels;
              } else {
                indirection_buffer[index] = zero;
              }
            }
          }
        }
      }
    }
  }
}

void pytorch_qnnp_indirection_init_maxpool2d(
    pytorch_qnnp_operator_t op,
    size_t batch_start,
    size_t step_height,
    size_t step_width) {
  const void** indirection_buffer = op->indirection_buffer;
  const void* input = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride;
  const size_t batch_size = op->batch_size;
  const size_t input_height = op->input_height;
  const size_t input_width = op->input_width;
  const size_t output_height = op->output_height;
  const size_t output_width = op->output_width;
  const size_t pooling_height = op->kernel_height;
  const size_t pooling_width = op->kernel_width;
  const size_t stride_height = op->stride_height;
  const size_t stride_width = op->stride_width;
  const size_t dilation_height = op->dilation_height;
  const size_t dilation_width = op->dilation_width;
  const size_t input_padding_top = op->input_padding_top;
  const size_t input_padding_left = op->input_padding_left;

  for (size_t image = batch_start; image < batch_size; image++) {
    for (size_t output_y = 0; output_y < output_height; output_y++) {
      for (size_t pooling_y = 0; pooling_y < pooling_height; pooling_y++) {
        const size_t input_y =
            doz(output_y * stride_height + pooling_y * dilation_height,
                input_padding_top);
        const size_t clamped_input_y = min(input_y, input_height - 1);
        for (size_t output_x = 0; output_x < output_width; output_x++) {
          for (size_t pooling_x = 0; pooling_x < pooling_width; pooling_x++) {
            const size_t input_x =
                doz(output_x * stride_width + pooling_x * dilation_width,
                    input_padding_left);
            const size_t clamped_input_x = min(input_x, input_width - 1);
            const size_t index =
                (image * output_height + output_y) * step_height +
                output_x * step_width * pooling_height +
                pooling_x * pooling_height + pooling_y;
            indirection_buffer[index] = (char*)input +
                ((image * input_height + clamped_input_y) * input_width +
                 clamped_input_x) *
                    input_pixel_stride;
          }
        }
      }
    }
  }
}
