/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/common.h>
#include <qnnpack/indirection.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>
#include <qnnpack/params.h>

static inline size_t compute_output_dimension(
    size_t padded_input_dimension,
    size_t pooling_dimension,
    size_t stride_dimension) {
  return (padded_input_dimension - pooling_dimension) / stride_dimension + 1;
}

enum pytorch_qnnp_status pytorch_qnnp_create_average_pooling2d_nhwc_q8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* average_pooling_out) {
  pytorch_qnnp_operator_t average_pooling = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_average_pooling2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with %" PRIu32 "x%" PRIu32
        " pooling size: "
        "pooling size dimensions must be non-zero",
        pooling_width,
        pooling_height);
    goto error;
  }

  if (pooling_size == 1) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with 1 pooling element: "
        "1x1 pooling is meaningless");
    goto error;
  }

  if (stride_height == 0 || stride_width == 0) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with %" PRIu32 "x%" PRIu32
        " stride: "
        "stride dimensions must be non-zero",
        stride_width,
        stride_height);
    goto error;
  }

  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with %zu channels: "
        "number of channels must be non-zero",
        channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with %.7g input scale: "
        "scale must be finite and positive",
        input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with %.7g output scale: "
        "scale must be finite and positive",
        output_scale);
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter;

  const float input_output_scale = input_scale / output_scale;
  if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with %.7g input scale and %.7g output scale: "
        "input-to-output scale ratio (%.7f) must be in [2**-8, 2**8) range",
        input_scale,
        output_scale,
        input_output_scale);
    goto error;
  }

  if (pooling_size >= 16777216) {
    pytorch_qnnp_log_error(
        "failed to create average pooling with %" PRIu32 " (%" PRIu32
        "x%" PRIu32
        ") pooling elements: "
        "the number of elements in the pooling area must be below 2**24",
        pooling_size,
        pooling_width,
        pooling_height);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;

  average_pooling = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (average_pooling == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  const bool any_padding = (input_padding_left | input_padding_top |
                            input_padding_right | input_padding_bottom) != 0;
  const uint32_t kr = pytorch_qnnp_params.q8avgpool.kr;
  const uint32_t mr = pytorch_qnnp_params.q8avgpool.mr;
  const uint32_t qr = pytorch_qnnp_params.q8avgpool.qr;
  if (any_padding || (channels >= kr || (pooling_size - mr) % qr != 0)) {
    void* zero_buffer = malloc(channels);
    if (zero_buffer == NULL) {
      pytorch_qnnp_log_error(
          "failed to allocate %zu bytes for zero padding", channels);
      goto error;
    }
    memset(zero_buffer, input_zero_point, channels);
    average_pooling->zero_buffer = zero_buffer;
    average_pooling->zero_pointer = zero_buffer;
  }

  average_pooling->input_padding_top = input_padding_top;
  average_pooling->input_padding_right = input_padding_right;
  average_pooling->input_padding_bottom = input_padding_bottom;
  average_pooling->input_padding_left = input_padding_left;

  average_pooling->kernel_height = pooling_height;
  average_pooling->kernel_width = pooling_width;
  average_pooling->stride_height = stride_height;
  average_pooling->stride_width = stride_width;
  average_pooling->dilation_height = 1;
  average_pooling->dilation_width = 1;
  average_pooling->channels = channels;

  size_t nrows = pooling_height * pooling_width;
  if (channels >= pytorch_qnnp_params.q8avgpool.kr) {
    if (nrows <= mr) {
      nrows = mr;
    } else {
      nrows = round_up(nrows - mr, qr) + mr;
    }
  }

  average_pooling->avgpool_quantization_params =
      pytorch_qnnp_compute_avgpool_quantization_params(
          (int32_t) - ((uint32_t)input_zero_point * (uint32_t)nrows),
          input_scale / (output_scale * (float)pooling_size),
          output_zero_point,
          output_min,
          output_max);

  average_pooling->ukernel_type = pytorch_qnnp_ukernel_type_average_pooling;
  average_pooling->format = pytorch_qnnp_format_quint8;

  *average_pooling_out = average_pooling;
  return pytorch_qnnp_status_success;

error:
  pytorch_qnnp_delete_operator(average_pooling);
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_average_pooling2d_nhwc_q8(
    pytorch_qnnp_operator_t average_pooling,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_pixel_stride,
    uint8_t* output,
    size_t output_pixel_stride,
    pthreadpool_t threadpool) {
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_average_pooling2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    average_pooling->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  if (input_width == 0 || input_height == 0) {
    pytorch_qnnp_log_error(
        "failed to setup average pooling with %zux%zu input: input dimensions must be non-zero",
        input_width,
        input_height);
    return pytorch_qnnp_status_invalid_parameter;
  }

  average_pooling->batch_size = batch_size;
  average_pooling->input_height = input_height;
  average_pooling->input_width = input_width;
  average_pooling->input = input;
  average_pooling->input_pixel_stride = input_pixel_stride;

  average_pooling->output_height = compute_output_dimension(
      average_pooling->input_padding_top + input_height +
          average_pooling->input_padding_bottom,
      average_pooling->kernel_height,
      average_pooling->stride_height);
  average_pooling->output_width = compute_output_dimension(
      average_pooling->input_padding_left + input_width +
          average_pooling->input_padding_right,
      average_pooling->kernel_width,
      average_pooling->stride_width);
  average_pooling->output = output;
  average_pooling->output_pixel_stride = output_pixel_stride;

  size_t valid_batch_size = 0;
  if (input == average_pooling->last_input &&
      input_height == average_pooling->last_input_height &&
      input_width == average_pooling->last_input_width) {
    valid_batch_size = average_pooling->valid_batch_size;
    if (batch_size <= valid_batch_size) {
      return pytorch_qnnp_status_success;
    }
  }

  const size_t pooling_height = average_pooling->kernel_height;
  const size_t pooling_width = average_pooling->kernel_width;
  const size_t pooling_size = pooling_height * pooling_width;
  const size_t output_height = average_pooling->output_height;
  const size_t output_width = average_pooling->output_width;
  /* Micro-kernel may read up to (mr - 1) elements after the end of indirection
   * buffer */
  const uint32_t mr = pytorch_qnnp_params.q8avgpool.mr;

  /*
   * Indirection buffer:
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
   */
  const size_t step_width = min(average_pooling->stride_width, pooling_width);
  const size_t step_height =
      pooling_size + (output_width * step_width - 1) * pooling_height;
  const size_t indirection_buffer_size =
      sizeof(void*) * ((mr - 1) + batch_size * output_height * step_height);

  const void** indirection_buffer = (const void**)realloc(
      average_pooling->indirection_buffer, indirection_buffer_size);
  if (indirection_buffer == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for indirection buffer",
        indirection_buffer_size);
    return pytorch_qnnp_status_out_of_memory;
  }
  average_pooling->indirection_buffer = indirection_buffer;

  pytorch_qnnp_indirection_init_dwconv2d(
      average_pooling, valid_batch_size, step_height, step_width);

  average_pooling->last_input = input;
  average_pooling->last_input_height = input_height;
  average_pooling->last_input_width = input_width;
  average_pooling->valid_batch_size = max(valid_batch_size, batch_size);

  return pytorch_qnnp_status_success;
}
