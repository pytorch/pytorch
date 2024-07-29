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

void pytorch_qnnp_indirection_init_conv3d(
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
  const size_t input_depth = op->input_depth;
  const size_t input_height = op->input_height;
  const size_t input_width = op->input_width;
  const size_t output_depth = op->output_depth;
  const size_t output_height = op->output_height;
  const size_t output_width = op->output_width;
  const size_t kernel_depth = op->kernel_depth;
  const size_t kernel_height = op->kernel_height;
  const size_t kernel_width = op->kernel_width;
  const size_t stride_depth = op->stride_depth;
  const size_t stride_height = op->stride_height;
  const size_t stride_width = op->stride_width;
  const size_t dilation_depth = op->dilation_depth;
  const size_t dilation_height = op->dilation_height;
  const size_t dilation_width = op->dilation_width;
  const size_t input_padding_depth = op->input_padding_depth;
  const size_t input_padding_height = op->input_padding_height;
  const size_t input_padding_width = op->input_padding_width;

  const size_t output_size = output_depth * output_height * output_width;
  const size_t kernel_size = kernel_depth * kernel_height * kernel_width;
  const struct fxdiv_divisor_size_t output_yx_divisor =
      fxdiv_init_size_t(output_height * output_width);
  const struct fxdiv_divisor_size_t output_x_divisor =
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
          const struct fxdiv_result_size_t z_yx =
              fxdiv_divide_size_t(output_index, output_yx_divisor);
          const struct fxdiv_result_size_t y_x =
              fxdiv_divide_size_t(z_yx.remainder, output_x_divisor);
          const size_t output_z = z_yx.quotient;
          const size_t output_y = y_x.quotient;
          const size_t output_x = y_x.remainder;

          for (size_t kernel_z = 0; kernel_z < kernel_depth; kernel_z++) {
            const size_t input_z = output_z * stride_depth +
                kernel_z * dilation_depth - input_padding_depth;
            if (input_z < input_depth) {
              for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                const size_t input_y = output_y * stride_height +
                    kernel_y * dilation_height - input_padding_height;
                if (input_y < input_height) {
                  for (size_t kernel_x = 0; kernel_x < kernel_width;
                       kernel_x++) {
                    const size_t input_x = output_x * stride_width +
                        kernel_x * dilation_width - input_padding_width;
                    const size_t index = (group * batch_size + image) *
                            tiled_output_size * kernel_size +
                        output_tile_start * kernel_size +
                        ((kernel_height * kernel_z + kernel_y) * kernel_width +
                         kernel_x) *
                            output_tile_size +
                        output_tile_offset;
                    if (input_x < input_width) {
                      indirection_buffer[index] = (char*)input +
                          (((image * input_depth + input_z) * input_height +
                            input_y) *
                               input_width +
                           input_x) *
                              input_pixel_stride +
                          group * group_input_channels;
                    } else {
                      indirection_buffer[index] = zero;
                    }
                  }
                } else {
                  for (size_t kernel_x = 0; kernel_x < kernel_width;
                       kernel_x++) {
                    const size_t index = (group * batch_size + image) *
                            tiled_output_size * kernel_size +
                        output_tile_start * kernel_size +
                        ((kernel_height * kernel_z + kernel_y) * kernel_width +
                         kernel_x) *
                            output_tile_size +
                        output_tile_offset;
                    indirection_buffer[index] = zero;
                  }
                }
              }
            } else {
              for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                  const size_t index = (group * batch_size + image) *
                          tiled_output_size * kernel_size +
                      output_tile_start * kernel_size +
                      ((kernel_height * kernel_z + kernel_y) * kernel_width +
                       kernel_x) *
                          output_tile_size +
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
}

/**
 * Imagine a we want to do dw conv or avgpooling with these parameters:
 * kernel_width/height=3 stride=2
 * Input is:
 *  ---------------
 *  |0|1|2|3|4|5|6|
 *  ---------------       -------
 *  | | | | | | | |   to  |0|1|2|
 *  ---------------       -------
 *  | | | | | | | |       | | | |
 *  ---------------       -------
 *  | | | | | | | |
 *  ---------------
 *  | | | | | | | |
 *  ---------------
 *
 *  Thus we are going from width=7 height=5 input to width=3 height=2
 *  Convince yourself that input 5x7 with pooling params of 3x3 kernel
 *  with 2x2 stride gets you to 2x3 output.
 *  Now for each output place (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
 *  we have 3x3 input.
 *  For just the first row of output this will look like as follows:
 *  pixel:0   pixel:1  pixel:2
 *  -------   -------  -------
 *  |0|1|2|   |2|3|4|  |4|5|6|
 *  -------   -------  -------
 *  | | | |   | | | |  | | | |
 *  -------   -------  -------
 *  | | | |   | | | |  | | | |
 *  -------   -------  -------
 *  As you can see there is some overlap in the input needed for each
 *  output pixel.
 *  What is indirection buffer:
 *  Indirection buffer just stores the pointer to the underlying data.
 *  In this case pointer for a particular input position will point to
 *  all the input channels of that position in NHWC format.
 *  So one option for the aforemnetioned storage would be:
 *  For each output position: store a 3x3 array of pointers. Thus we
 *  would have 3x3 * 3 (3 output pixel of the first row) = 27 pointers
 *  stored.
 *  Now instead we store the pointer in this format:
 *  ---------------
 *  |0|1|2|3|4|5|6|
 *  ---------------
 *  | | | | | | | |
 *  ---------------
 *  | | | | | | | |
 *  ---------------
 *  Then we have all the pointers needed as before, but with less duplication.
 *  So instead of 27 pointers now we have:
 *  (3 (# of output pixels) - 1) * (stride) * 3 (kernel height) * + 3 * 3 (kernel h*w)
 *  = 4 * 3 + 9
 *  = 21 pointers.
 *  which is the equation below.
 *  Now in order for this to work the kernel has to be adjusted.
 *  Here the kernel produced output worth of entire width. Thus as you move from one
 *  pixel to the next, the jump in the indirection buffer has to be not 3*3 = 9
 *  but kernel height (3) * stride (2) = 6.
 *  This you will see operator-run.c
 *
 * step_width: The number of yz slices of the kernel to traverse to move from
 *   the starting input index of an output pixel in the indirection buffer to
 *   that of the output pixel directly after it in the same row.
 *   i.e. if indirection_buffer[j] points to the first input pixel used to
 *   compute the i'th output pixel, then
 *   indirection_buffer[j + (kernel_depth * kernel_height * step_width)]
 *   points to the first input pixel used to compute the (i + 1)'th output
 *   pixel if in the same row
 *   When dilation is 1 (for convolution): if neighboring output pixels use
 *   overlapping regions of the input, this overlap is not included in the
 *   indirection buffer (saving some space), hence step width is set to stride
 *   width
 *
 * step_height: The number of pointers to traverse to move from an output
 *   pixel's first input's index in the indirection buffer to that of the
 *   output pixel one ROW (one output y) after it.
 *   i.e. if indirection_buffer[j] points to the first input pixel used to
 *   compute the i'th output pixel, then
 *   indirection_buffer[j + step_height] points to the first
 *   input pixel used to compute the output pixel one row below-
 *   the (i + output_width)'th output pixel
 *
 * step_depth: Same as step height but for an xy slice rather than a row
 *
 * The input operator's step dimensions must have been set up before calling
 * this function.
 */
void pytorch_qnnp_indirection_init_dwconv(
    pytorch_qnnp_operator_t op,
    size_t batch_start) {
  const void** indirection_buffer = op->indirection_buffer;
  const void* input = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride;
  const void* zero = op->zero_pointer;
  const size_t batch_size = op->batch_size;
  const size_t input_depth = op->input_depth;
  const size_t input_height = op->input_height;
  const size_t input_width = op->input_width;
  const size_t output_depth = op->output_depth;
  const size_t output_height = op->output_height;
  const size_t output_width = op->output_width;
  const size_t kernel_depth = op->kernel_depth;
  const size_t kernel_height = op->kernel_height;
  const size_t kernel_width = op->kernel_width;
  const size_t stride_depth = op->stride_depth;
  const size_t stride_height = op->stride_height;
  const size_t stride_width = op->stride_width;
  const size_t dilation_depth = op->dilation_depth;
  const size_t dilation_height = op->dilation_height;
  const size_t dilation_width = op->dilation_width;
  const size_t input_padding_depth = op->input_padding_depth;
  const size_t input_padding_height = op->input_padding_height;
  const size_t input_padding_width = op->input_padding_width;
  const size_t step_depth = op->step_depth;
  const size_t step_height = op->step_height;
  const size_t step_width = op->step_width;

#define DW_CONV_3D_INDEX(oz, oy, ox, kz, ky, kx)                              \
  /* Output Pixel */                                                          \
  (image * output_depth + oz) * step_depth + /* slice */                      \
  oy * step_height + /* row */                                                \
  ox * step_width * kernel_height * kernel_depth + /* column */               \
  /* Kernel */                                                                \
  kx * kernel_depth * kernel_height + /* column */                            \
  ky * kernel_depth + /* row */                                               \
  kz /* slice */

  for (size_t image = batch_start; image < batch_size; image++) {
    for (size_t output_z = 0; output_z < output_depth; output_z++) {
      for (size_t kernel_z = 0; kernel_z < kernel_depth; kernel_z++) {
        const size_t input_z = output_z * stride_depth +
            kernel_z * dilation_depth - input_padding_depth;
        if (input_z < input_depth) {
          for (size_t output_y = 0; output_y < output_height; output_y++) {
            for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
              const size_t input_y = output_y * stride_height +
                  kernel_y * dilation_height - input_padding_height;
              if (input_y < input_height) {
                for (size_t output_x = 0; output_x < output_width; output_x++) {
                  for (size_t kernel_x = 0; kernel_x < kernel_width;
                       kernel_x++) {
                    const size_t input_x = output_x * stride_width +
                        kernel_x * dilation_width - input_padding_width;
                    const size_t index = DW_CONV_3D_INDEX(
                        output_z,
                        output_y,
                        output_x,
                        kernel_z,
                        kernel_y,
                        kernel_x);
                    if (input_x < input_width) {
                      indirection_buffer[index] = (char*)input +
                          ((image * input_depth + input_z) * input_height *
                               input_width + // slice
                           input_y * input_width + // row
                           input_x // column
                           ) * input_pixel_stride;
                    } else {
                      indirection_buffer[index] = zero;
                    }
                  }
                }
              } else {
                for (size_t output_x = 0; output_x < output_width; output_x++) {
                  for (size_t kernel_x = 0; kernel_x < kernel_width;
                       kernel_x++) {
                    const size_t index = DW_CONV_3D_INDEX(
                        output_z,
                        output_y,
                        output_x,
                        kernel_z,
                        kernel_y,
                        kernel_x);
                    indirection_buffer[index] = zero;
                  }
                }
              }
            }
          }
        } else {
          for (size_t output_y = 0; output_y < output_height; output_y++) {
            for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
              for (size_t output_x = 0; output_x < output_width; output_x++) {
                for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                  const size_t index = DW_CONV_3D_INDEX(
                      output_z,
                      output_y,
                      output_x,
                      kernel_z,
                      kernel_y,
                      kernel_x);
                  indirection_buffer[index] = zero;
                }
              }
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
  const size_t input_padding_height = op->input_padding_height;
  const size_t input_padding_width = op->input_padding_width;

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
                output_y + input_padding_height - kernel_y * dilation_height;
            const size_t input_y = y / stride_height;
            for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
              const size_t x =
                  output_x + input_padding_width - kernel_x * dilation_width;
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
    size_t batch_start) {
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
  const size_t input_padding_height = op->input_padding_height;
  const size_t input_padding_width = op->input_padding_width;
  const size_t step_height = op->step_height;
  const size_t step_width = op->step_width;

  for (size_t image = batch_start; image < batch_size; image++) {
    for (size_t output_y = 0; output_y < output_height; output_y++) {
      for (size_t pooling_y = 0; pooling_y < pooling_height; pooling_y++) {
        const size_t input_y =
            doz(output_y * stride_height + pooling_y * dilation_height,
                input_padding_height);
        const size_t clamped_input_y = min(input_y, input_height - 1);
        for (size_t output_x = 0; output_x < output_width; output_x++) {
          for (size_t pooling_x = 0; pooling_x < pooling_width; pooling_x++) {
            const size_t input_x =
                doz(output_x * stride_width + pooling_x * dilation_width,
                    input_padding_width);
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

void pytorch_qnnp_indirection_set_step_dimensions(pytorch_qnnp_operator_t op) {
  const size_t original_kernel_depth = op->kernel_depth;
  const size_t kernel_depth =
      (original_kernel_depth != 0) ? original_kernel_depth : 1;
  const size_t kernel_height = op->kernel_height;
  const size_t kernel_width = op->kernel_width;
  const size_t kernel_size = kernel_depth * kernel_height * kernel_width;
  const size_t output_height = op->output_height;
  const size_t output_width = op->output_width;

  size_t step_width = 0;
  switch (op->ukernel_type) {
    case pytorch_qnnp_ukernel_type_dwconv:
      step_width = op->dilation_width == 1 ? op->stride_width : kernel_width;
      break;
    case pytorch_qnnp_ukernel_type_average_pooling:
      step_width = min(op->stride_width, kernel_width);
      break;
    case pytorch_qnnp_ukernel_type_max_pooling:
      step_width = op->dilation_width > 1 ? kernel_width
                                          : min(op->stride_width, kernel_width);
      break;
    default:
      PYTORCH_QNNP_UNREACHABLE;
  }

  const size_t step_height = kernel_size +
      (output_width - 1) * step_width * kernel_height * kernel_depth;

  const size_t step_depth = step_height * output_height;

  op->step_depth = step_depth;
  op->step_height = step_height;
  op->step_width = step_width;
}
