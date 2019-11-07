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
#include <string.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/indirection.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

static inline size_t compute_output_dimension(
    size_t input_dimension,
    size_t input_padding_dimension,
    size_t adjustment_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t stride_dimension) {
  const size_t effective_kernel_dimension =
      (kernel_dimension - 1) * dilation_dimension + 1;
  return stride_dimension * (input_dimension - 1) + adjustment_dimension +
      effective_kernel_dimension - input_padding_dimension;
}

enum pytorch_qnnp_status pytorch_qnnp_create_deconvolution2d_nhwc_q8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t adjustment_height,
    uint32_t adjustment_width,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t kernel_zero_point,
    float kernel_scale,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* deconvolution_out) {
  pytorch_qnnp_operator_t deconvolution = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_deconvolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    pytorch_qnnp_log_error(
        "failed to create deconvolution with %" PRIu32 "x%" PRIu32
        " kernel: kernel dimensions must be non-zero",
        kernel_width,
        kernel_height);
    goto error;
  }

  if (stride_width == 0 || stride_height == 0) {
    pytorch_qnnp_log_error(
        "failed to create deconvolution with %" PRIu32 "x%" PRIu32
        " stride: "
        "stride dimensions must be non-zero",
        stride_width,
        stride_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    pytorch_qnnp_log_error(
        "failed to create deconvolution with %" PRIu32 "x%" PRIu32
        " dilation: "
        "dilation dimensions must be non-zero",
        dilation_width,
        dilation_height);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create deconvolution with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    pytorch_qnnp_log_error(
        "failed to create deconvolution with %.7g kernel scale: scale must be finite and positive",
        kernel_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create deconvolution with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter;

  const float deconvolution_scale = input_scale * kernel_scale / output_scale;
  if (deconvolution_scale >= 1.0f) {
    pytorch_qnnp_log_error(
        "failed to create deconvolution with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
        "deconvolution scale %.7g is greater or equal to 1.0",
        input_scale,
        kernel_scale,
        output_scale,
        deconvolution_scale);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;

  deconvolution = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (deconvolution == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
  const uint32_t kr = pytorch_qnnp_params.q8conv.kr;

  const uint32_t n_stride = (group_output_channels + (nr - 1)) & -nr;
  const uint32_t k_stride = (group_input_channels + (kr - 1)) & -kr;
  const uint32_t kernel_size = kernel_height * kernel_width;
  const size_t packed_group_weights_size =
      (sizeof(uint8_t) * kernel_size * k_stride + sizeof(int32_t)) * n_stride;
  deconvolution->packed_weights = malloc(packed_group_weights_size * groups);
  if (deconvolution->packed_weights == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for packed weights",
        packed_group_weights_size * groups);
    goto error;
  }
  memset(
      deconvolution->packed_weights,
      kernel_zero_point,
      packed_group_weights_size * groups);

  for (uint32_t group = 0; group < groups; group++) {
    pytorch_pack_q8deconv_w(
        group_output_channels,
        kernel_size,
        group_input_channels,
        nr,
        kr,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
        input_zero_point,
        kernel_zero_point,
#endif
        kernel +
            group * group_output_channels * kernel_size * group_input_channels,
        bias + group * group_output_channels,
        (void*)((uintptr_t)deconvolution->packed_weights + group * packed_group_weights_size));
  }

  size_t zero_size = sizeof(uint8_t) * k_stride;
  size_t zero_offset = 0;
  if (group_input_channels < 8) {
    zero_size += 8;
    zero_offset = 8;
  }

  void* zero_buffer = malloc(zero_size);
  if (zero_buffer == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for zero padding", zero_size);
    goto error;
  }
  memset(zero_buffer, input_zero_point, zero_size);
  deconvolution->zero_buffer = zero_buffer;
  deconvolution->zero_pointer = (void*)((uintptr_t)zero_buffer + zero_offset);

  deconvolution->input_padding_top = input_padding_top;
  deconvolution->input_padding_right = input_padding_right;
  deconvolution->input_padding_bottom = input_padding_bottom;
  deconvolution->input_padding_left = input_padding_left;
  deconvolution->adjustment_height = adjustment_height;
  deconvolution->adjustment_width = adjustment_width;

  deconvolution->kernel_height = kernel_height;
  deconvolution->kernel_width = kernel_width;
  deconvolution->stride_height = stride_height;
  deconvolution->stride_width = stride_width;
  deconvolution->dilation_height = dilation_height;
  deconvolution->dilation_width = dilation_width;
  deconvolution->groups = groups;
  deconvolution->group_input_channels = group_input_channels;
  deconvolution->group_output_channels = group_output_channels;

  deconvolution->kernel_zero_point = kernel_zero_point;

  deconvolution->conv_quantization_params =
      pytorch_qnnp_compute_conv_quantization_params(
          input_zero_point,
          kernel_zero_point,
          deconvolution_scale,
          output_zero_point,
          output_min,
          output_max);

  deconvolution->ukernel_type = pytorch_qnnp_ukernel_type_conv;
  deconvolution->format = pytorch_qnnp_format_quint8;

  *deconvolution_out = deconvolution;
  return pytorch_qnnp_status_success;

error:
  pytorch_qnnp_delete_operator(deconvolution);
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_deconvolution2d_nhwc_q8(
    pytorch_qnnp_operator_t deconvolution,
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
        "pytorch_qnnp_setup_deconvolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    deconvolution->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  if (input_width == 0 || input_height == 0) {
    pytorch_qnnp_log_error(
        "failed to setup deconvolution with %zux%zu input: input dimensions must be non-zero",
        input_width,
        input_height);
    return pytorch_qnnp_status_invalid_parameter;
  }

  deconvolution->batch_size = batch_size;
  deconvolution->input_height = input_height;
  deconvolution->input_width = input_width;
  deconvolution->input = input;
  deconvolution->input_pixel_stride = input_pixel_stride;
  deconvolution->output = output;
  deconvolution->output_pixel_stride = output_pixel_stride;

  const size_t kernel_height = deconvolution->kernel_height;
  const size_t kernel_width = deconvolution->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t stride_height = deconvolution->stride_height;
  const size_t stride_width = deconvolution->stride_width;
  const size_t output_height = deconvolution->output_height =
      compute_output_dimension(
          input_height,
          deconvolution->input_padding_top +
              deconvolution->input_padding_bottom,
          deconvolution->adjustment_height,
          kernel_height,
          deconvolution->dilation_height,
          stride_height);
  const size_t output_width = deconvolution->output_width =
      compute_output_dimension(
          input_width,
          deconvolution->input_padding_left +
              deconvolution->input_padding_right,
          deconvolution->adjustment_width,
          kernel_width,
          deconvolution->dilation_width,
          stride_width);

  const size_t groups = deconvolution->groups;
  const size_t output_size = output_height * output_width;
  const size_t output_tile_size = pytorch_qnnp_params.q8conv.mr;
  const size_t tiled_output_size = round_up(output_size, output_tile_size);
  const size_t indirection_buffer_size =
      sizeof(void*) * batch_size * groups * tiled_output_size * kernel_size;

  const void** indirection_buffer = (const void**)realloc(
      deconvolution->indirection_buffer, indirection_buffer_size);
  if (indirection_buffer == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for indirection buffer",
        indirection_buffer_size);
    return pytorch_qnnp_status_out_of_memory;
  }
  deconvolution->indirection_buffer = indirection_buffer;

  pytorch_qnnp_indirection_init_deconv2d(
      deconvolution, output_tile_size, tiled_output_size);

  return pytorch_qnnp_status_success;
}
