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

#include <fxdiv.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/common.h>
#include <qnnpack/indirection.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>

static inline size_t compute_output_dimension(
    size_t padded_input_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t subsampling_dimension) {
  const size_t effective_kernel_dimension =
      (kernel_dimension - 1) * dilation_dimension + 1;
  return (padded_input_dimension - effective_kernel_dimension) /
      subsampling_dimension +
      1;
}

/**
 * Not exposed in header file
 */
static enum pytorch_qnnp_status pytorch_qnnp_create_convolution_ndhwc_q8(
    uint32_t input_padding_depth,
    uint32_t input_padding_height,
    uint32_t input_padding_width,
    uint32_t kernel_depth,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_depth,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
    uint32_t dilation_depth,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    bool per_channel,
    pytorch_qnnp_operator_t* convolution_out,
    bool is_2d /* true: 2d, false: 3d */) {
  pytorch_qnnp_operator_t convolution = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_convolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    pytorch_qnnp_log_error(
        "failed to create convolution with %" PRIu32 "x%" PRIu32
        " kernel: kernel dimensions must be non-zero",
        kernel_width,
        kernel_height);
    goto error;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    pytorch_qnnp_log_error(
        "failed to create convolution with %" PRIu32 "x%" PRIu32
        " subsampling: "
        "subsampling dimensions must be non-zero",
        subsampling_width,
        subsampling_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    pytorch_qnnp_log_error(
        "failed to create convolution with %" PRIu32 "x%" PRIu32
        " dilation: "
        "dilation dimensions must be non-zero",
        dilation_width,
        dilation_height);
    goto error;
  }

  status = pytorch_qnnp_status_unsupported_parameter;

  if (subsampling_height > kernel_height) {
    pytorch_qnnp_log_info(
        "inefficiency in convolution with %" PRIu32 "x%" PRIu32
        " kernel and %" PRIu32 "x%" PRIu32
        " subsampling: "
        "height subsampling is greater than kernel height; subsampling should be performed before the convolution",
        kernel_width,
        kernel_height,
        subsampling_width,
        subsampling_height);
  }

  if (subsampling_width > kernel_width) {
    pytorch_qnnp_log_info(
        "inefficiency in convolution with %" PRIu32 "x%" PRIu32
        " kernel and %" PRIu32 "x%" PRIu32
        " subsampling: "
        "width subsampling is greater than kernel width; subsampling should be performed before the convolution",
        kernel_width,
        kernel_height,
        subsampling_width,
        subsampling_height);
  }

  if (input_padding_depth >= kernel_depth) {
    pytorch_qnnp_log_info(
        "inefficiency in convolution with %" PRIu32 "x%" PRIu32 "x%" PRIu32
        " kernel and %" PRIu32 "+%" PRIu32
        " depth padding: "
        "input depth padding is greater or equal to kernel depth",
        kernel_depth,
        kernel_height,
        kernel_width,
        input_padding_depth,
        input_padding_depth);
  }

  if (input_padding_height >= kernel_height) {
    pytorch_qnnp_log_info(
        "inefficiency in convolution with %" PRIu32 "x%" PRIu32 "x%" PRIu32
        " kernel and %" PRIu32 "+%" PRIu32
        " height padding: "
        "input height padding is greater or equal to kernel height",
        kernel_depth,
        kernel_height,
        kernel_width,
        input_padding_height,
        input_padding_height);
  }

  if (input_padding_width >= kernel_width) {
    pytorch_qnnp_log_info(
        "inefficiency in convolution with %" PRIu32 "x%" PRIu32 "x%" PRIu32
        " kernel and %" PRIu32 "+%" PRIu32
        " width padding: "
        "input width padding is greater or equal to kernel width",
        kernel_depth,
        kernel_height,
        kernel_width,
        input_padding_width,
        input_padding_width);
  }

  for (int i = 0; i < groups * group_output_channels; ++i) {
    if (requantization_scales[i] <= 0.0f ||
        !isnormal(requantization_scales[i])) {
      pytorch_qnnp_log_error(
          "failed to create fully connected operator with %.7g requantization scale: scale must be finite and positive",
          requantization_scales[i]);
      goto error;
    }
  }

  status = pytorch_qnnp_status_out_of_memory;

  convolution = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (convolution == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  const size_t kernel_size = kernel_height * kernel_width * kernel_depth;

  enum pytorch_qnnp_ukernel_type ukernel_type = pytorch_qnnp_ukernel_type_none;
  const bool any_padding =
      (input_padding_depth | input_padding_height | input_padding_width) != 0;

  const bool has_depthwise_dimensions =
      (is_2d &&
       ((kernel_height == 3 && kernel_width == 3) ||
        (kernel_height == 5 && kernel_width == 5))) ||
      (!is_2d && kernel_height == 3 && kernel_width == 3 && kernel_depth == 3);
  const bool has_depthwise_grouping =
      group_input_channels == 1 && group_output_channels == 1 && groups > 1;
  if (has_depthwise_dimensions && has_depthwise_grouping) {
    ukernel_type = pytorch_qnnp_ukernel_type_dwconv;
  } else if (
      kernel_size == 1 && subsampling_height == 1 && subsampling_width == 1 &&
      !any_padding) {
    ukernel_type =
        group_input_channels >= pytorch_qnnp_params.q8conv_xzp.kthreshold
        ? pytorch_qnnp_ukernel_type_xzp_gemm
        : pytorch_qnnp_ukernel_type_gemm;
  } else {
    ukernel_type = pytorch_qnnp_ukernel_type_conv;
  }
  size_t zero_size = 0, zero_offset = 0;

  switch (ukernel_type) {
    // This also covers the case of dwconv_per_channel
    // since the weight packing is shared between the two.
    case pytorch_qnnp_ukernel_type_dwconv: {
      const uint32_t cr = pytorch_qnnp_params.q8dw9.cr;
      const uint32_t c_stride = (groups + (cr - 1)) & -cr;
      convolution->group_stride = c_stride;
      const size_t packed_weights_size =
          (sizeof(uint8_t) * kernel_size + sizeof(int32_t)) * c_stride;
      convolution->packed_weights = malloc(packed_weights_size);
      if (convolution->packed_weights == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for packed weights",
            packed_weights_size);
        goto error;
      }

      switch (kernel_size) {
        case 9:
          pytorch_pack_q8dw_w(
              kernel_height,
              kernel_width,
              groups,
              cr,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
              input_zero_point,
              kernel_zero_points[0],
#endif
              kernel,
              bias,
              convolution->packed_weights);
          break;
        case 25:
          /* change this later */
          pytorch_pack_q8dw_2d_w_dilation(
              kernel_height,
              kernel_width,
              groups,
              cr,
              0,
              kernel_height,
              0,
              2,
              kernel,
              bias,
              convolution->packed_weights,
              true);
          pytorch_pack_q8dw_2d_w_dilation(
              kernel_height,
              kernel_width,
              groups,
              cr,
              0,
              kernel_height,
              2,
              4,
              kernel,
              bias,
              (char*)convolution->packed_weights +
                  (10 + sizeof(int32_t) / sizeof(uint8_t)) * c_stride,
              false);
          pytorch_pack_q8dw_2d_w_dilation(
              kernel_height,
              kernel_width,
              groups,
              cr,
              0,
              kernel_height,
              4,
              5,
              kernel,
              bias,
              (char*)convolution->packed_weights +
                  (20 + sizeof(int32_t) / sizeof(uint8_t)) * c_stride,
              false);
          break;
        case 27:
          pytorch_pack_q8dw_3d_w_dilation(
              kernel_depth,
              kernel_height,
              kernel_width,
              groups,
              cr,
              0,
              kernel_depth,
              0,
              kernel_height,
              0,
              1,
              kernel,
              bias,
              convolution->packed_weights,
              true);
          pytorch_pack_q8dw_3d_w_dilation(
              kernel_depth,
              kernel_height,
              kernel_width,
              groups,
              cr,
              0,
              kernel_depth,
              0,
              kernel_height,
              1,
              2,
              kernel,
              bias,
              (char*)convolution->packed_weights +
                  (kernel_depth * kernel_height +
                   sizeof(int32_t) / sizeof(uint8_t)) *
                      c_stride,
              false);
          pytorch_pack_q8dw_3d_w_dilation(
              kernel_depth,
              kernel_height,
              kernel_width,
              groups,
              cr,
              0,
              kernel_depth,
              0,
              kernel_height,
              2,
              3,
              kernel,
              bias,
              (char*)convolution->packed_weights +
                  (2 * kernel_depth * kernel_height +
                   sizeof(int32_t) / sizeof(uint8_t)) *
                      c_stride,
              false);
          break;
        default:
          PYTORCH_QNNP_UNREACHABLE;
      }

      if (groups >= 8) {
        zero_size = sizeof(uint8_t) * c_stride;
        zero_offset = 0;
      } else {
        zero_size = sizeof(uint8_t) * c_stride + 8;
        zero_offset = sizeof(uint8_t) * 8;
      }
      break;
    }
    case pytorch_qnnp_ukernel_type_xzp_gemm: {
      // TODO: XZP kernels won't be supporting per channel quantization.
      // For now we dont use XZP kernels anywhere. Probably deprecate it for now
      // and ressurrect later if needed.
      const uint32_t nr = pytorch_qnnp_params.q8conv_xzp.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv_xzp.kr;
      const uint32_t sr = pytorch_qnnp_params.q8conv_xzp.kc;
      const uint32_t n_stride = (group_output_channels + (nr - 1)) & -nr;
      const uint32_t k_stride = (group_input_channels + (kr - 1)) & -kr;

      const size_t packed_group_weights_size =
          (sizeof(uint8_t) * kernel_size * k_stride + sizeof(int32_t)) *
          n_stride;
      convolution->packed_weights = malloc(packed_group_weights_size * groups);
      if (convolution->packed_weights == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for packed weights",
            packed_group_weights_size * groups);
        goto error;
      }
      /* The XZP ukernel needs the padding to be 0 */
      memset(
          convolution->packed_weights, 0, packed_group_weights_size * groups);

      for (uint32_t group = 0; group < groups; group++) {
        pytorch_pack_swizzle_q8gemm_b(
            group_output_channels,
            group_input_channels,
            nr,
            kr,
            sr,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
            input_zero_point,
            kernel_zero_points[0],
#endif
            kernel + group * group_output_channels * group_input_channels,
            bias + group * group_output_channels,
            (void*)((uintptr_t)convolution->packed_weights + group * packed_group_weights_size));
      }
      break;
    }
    case pytorch_qnnp_ukernel_type_gemm:
    case pytorch_qnnp_ukernel_type_conv: {
      const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
      const uint32_t n_stride = (group_output_channels + (nr - 1)) & -nr;
      const uint32_t k_stride = (group_input_channels + (kr - 1)) & -kr;

      const size_t packed_group_weights_size =
          (sizeof(uint8_t) * kernel_size * k_stride + sizeof(int32_t)) *
          n_stride;
      convolution->packed_weights = malloc(packed_group_weights_size * groups);
      if (convolution->packed_weights == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for packed weights",
            packed_group_weights_size * groups);
        goto error;
      }
      memset(
          convolution->packed_weights,
          kernel_zero_points[0],
          packed_group_weights_size * groups);

      switch (ukernel_type) {
        case pytorch_qnnp_ukernel_type_gemm:
          for (uint32_t group = 0; group < groups; group++) {
            pytorch_pack_q8gemm_w(
                group_output_channels,
                group_input_channels,
                nr,
                nr,
                kr,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
                input_zero_point,
                kernel_zero_points[0],
#endif
                kernel + group * group_output_channels * group_input_channels,
                bias + group * group_output_channels,
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
                kernel_zero_points + group * group_output_channels,
#endif
                (void*)((uintptr_t)convolution->packed_weights + group * packed_group_weights_size));
          }
          break;
        case pytorch_qnnp_ukernel_type_conv:
          for (uint32_t group = 0; group < groups; group++) {
            pytorch_pack_q8conv_w(
                group_output_channels,
                kernel_size,
                group_input_channels,
                nr,
                kr,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
                input_zero_point,
                kernel_zero_points[0],
#endif
                kernel +
                    group * group_output_channels * kernel_size *
                        group_input_channels,
                bias + group * group_output_channels,
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
                kernel_zero_points + group * group_output_channels,
#endif
                (void*)((uintptr_t)convolution->packed_weights + group * packed_group_weights_size));
          }
          break;
        default:
          PYTORCH_QNNP_UNREACHABLE;
      }

      if (group_input_channels >= 8) {
        zero_size = sizeof(uint8_t) * k_stride;
        zero_offset = 0;
      } else {
        zero_size = sizeof(uint8_t) * k_stride + 8;
        zero_offset = 8;
      }
      break;
    }
    default:
      PYTORCH_QNNP_UNREACHABLE;
  }

  if (any_padding) {
    void* zero_buffer = malloc(zero_size);
    if (zero_buffer == NULL) {
      pytorch_qnnp_log_error(
          "failed to allocate %zu bytes for zero padding", zero_size);
      goto error;
    }
    memset(zero_buffer, input_zero_point, zero_size);
    convolution->zero_buffer = zero_buffer;
    convolution->zero_pointer = (void*)((uintptr_t)zero_buffer + zero_offset);
  }

  convolution->input_padding_depth = input_padding_depth;
  convolution->input_padding_height = input_padding_height;
  convolution->input_padding_width = input_padding_width;
  convolution->kernel_depth = kernel_depth;
  convolution->kernel_height = kernel_height;
  convolution->kernel_width = kernel_width;
  convolution->stride_depth = subsampling_depth;
  convolution->stride_height = subsampling_height;
  convolution->stride_width = subsampling_width;
  convolution->dilation_depth = dilation_depth;
  convolution->dilation_height = dilation_height;
  convolution->dilation_width = dilation_width;
  convolution->groups = groups;
  convolution->group_input_channels = group_input_channels;
  convolution->group_output_channels = group_output_channels;

  convolution->kernel_zero_point = kernel_zero_points[0];

  if (ukernel_type == pytorch_qnnp_ukernel_type_xzp_gemm) {
    convolution->requantization_params =
        pytorch_qnnp_compute_requantization_params(
            requantization_scales[0], output_zero_point, output_min, output_max);
  } else {
    convolution->conv_quantization_params =
        pytorch_qnnp_compute_conv_quantization_params(
            input_zero_point,
            kernel_zero_points,
            requantization_scales,
            output_zero_point,
            output_min,
            output_max);
  }

  convolution->ukernel_type = ukernel_type;
  convolution->format = pytorch_qnnp_format_quint8;

  convolution->per_channel = per_channel;

  *convolution_out = convolution;
  return pytorch_qnnp_status_success;

error:
  pytorch_qnnp_delete_operator(convolution);
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_create_convolution2d_nhwc_q8(
    uint32_t input_padding_height,
    uint32_t input_padding_width,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    bool per_channel,
    pytorch_qnnp_operator_t* convolution_out) {
  return pytorch_qnnp_create_convolution_ndhwc_q8(
      0,
      input_padding_height,
      input_padding_width,
      1,
      kernel_height,
      kernel_width,
      1,
      subsampling_height,
      subsampling_width,
      1,
      dilation_height,
      dilation_width,
      groups,
      group_input_channels,
      group_output_channels,
      input_zero_point,
      kernel_zero_points,
      kernel,
      bias,
      output_zero_point,
      output_min,
      output_max,
      flags,
      requantization_scales,
      per_channel,
      convolution_out,
      true /* is_2d? */);
}

enum pytorch_qnnp_status pytorch_qnnp_create_convolution3d_ndhwc_q8(
    uint32_t input_padding_depth,
    uint32_t input_padding_height,
    uint32_t input_padding_width,
    uint32_t kernel_depth,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_depth,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
    uint32_t dilation_depth,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    bool per_channel,
    pytorch_qnnp_operator_t* convolution_out) {
  return pytorch_qnnp_create_convolution_ndhwc_q8(
      input_padding_depth,
      input_padding_height,
      input_padding_width,
      kernel_depth,
      kernel_height,
      kernel_width,
      subsampling_depth,
      subsampling_height,
      subsampling_width,
      dilation_depth,
      dilation_height,
      dilation_width,
      groups,
      group_input_channels,
      group_output_channels,
      input_zero_point,
      kernel_zero_points,
      kernel,
      bias,
      output_zero_point,
      output_min,
      output_max,
      flags,
      requantization_scales,
      per_channel,
      convolution_out,
      false /* is_2d? */);
}

enum pytorch_qnnp_status pytorch_qnnp_setup_convolution2d_nhwc_q8(
    pytorch_qnnp_operator_t convolution,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_pixel_stride,
    uint8_t* output,
    size_t output_pixel_stride,
    pthreadpool_t threadpool) {
  return pytorch_qnnp_setup_convolution_ndhwc_q8(
      convolution,
      batch_size,
      1,
      input_height,
      input_width,
      input,
      input_pixel_stride,
      output,
      output_pixel_stride,
      threadpool);
}

enum pytorch_qnnp_status pytorch_qnnp_setup_convolution_ndhwc_q8(
    pytorch_qnnp_operator_t convolution,
    size_t batch_size,
    size_t input_depth,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_pixel_stride,
    uint8_t* output,
    size_t output_pixel_stride,
    pthreadpool_t threadpool) {
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_convolution_ndhwc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    convolution->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  if (input_width == 0 || input_height == 0 || input_depth == 0) {
    pytorch_qnnp_log_error(
        "failed to setup convolution with %zux%zux%zu input: input dimensions must be non-zero",
        input_width,
        input_height,
        input_depth);
    return pytorch_qnnp_status_invalid_parameter;
  }

  convolution->batch_size = batch_size;
  convolution->input_depth = input_depth;
  convolution->input_height = input_height;
  convolution->input_width = input_width;
  convolution->input = input;
  convolution->input_pixel_stride = input_pixel_stride;

  convolution->output_depth = compute_output_dimension(
      input_depth + convolution->input_padding_depth * 2,
      convolution->kernel_depth,
      convolution->dilation_depth,
      convolution->stride_depth);
  convolution->output_height = compute_output_dimension(
      input_height + convolution->input_padding_height * 2,
      convolution->kernel_height,
      convolution->dilation_height,
      convolution->stride_height);
  convolution->output_width = compute_output_dimension(
      input_width + convolution->input_padding_width * 2,
      convolution->kernel_width,
      convolution->dilation_width,
      convolution->stride_width);
  convolution->output = output;
  convolution->output_pixel_stride = output_pixel_stride;

  switch (convolution->ukernel_type) {
    case pytorch_qnnp_ukernel_type_gemm:
      /* Convolution maps directly to GEMM and doesn't use indirection buffer */
      return pytorch_qnnp_status_success;
    case pytorch_qnnp_ukernel_type_xzp_gemm: {
      const size_t groups = convolution->groups;
      const size_t input_size = input_depth * input_height * input_width;
      void* a_sum = (void*)realloc(
          convolution->a_sum,
          sizeof(int32_t) * batch_size * groups * input_size);
      if (a_sum == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for row sum data",
            sizeof(int32_t) * batch_size * groups * input_size);
        return pytorch_qnnp_status_out_of_memory;
      }
      convolution->a_sum = a_sum;
      return pytorch_qnnp_status_success;
    }
    case pytorch_qnnp_ukernel_type_conv: {
      const size_t groups = convolution->groups;
      const size_t kernel_depth = convolution->kernel_depth;
      const size_t kernel_height = convolution->kernel_height;
      const size_t kernel_width = convolution->kernel_width;
      const size_t kernel_size = kernel_depth * kernel_height * kernel_width;
      const size_t output_depth = convolution->output_depth;
      const size_t output_height = convolution->output_height;
      const size_t output_width = convolution->output_width;
      const size_t output_size = output_depth * output_height * output_width;
      const size_t output_tile_size = pytorch_qnnp_params.q8conv.mr;
      const size_t tiled_output_size = round_up(output_size, output_tile_size);
      const size_t indirection_buffer_size =
          sizeof(void*) * batch_size * groups * tiled_output_size * kernel_size;

      const void** indirection_buffer = (const void**)realloc(
          convolution->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for indirection buffer",
            indirection_buffer_size);
        return pytorch_qnnp_status_out_of_memory;
      }
      convolution->indirection_buffer = indirection_buffer;
      pytorch_qnnp_indirection_init_conv3d(
          convolution, output_tile_size, tiled_output_size);
      return pytorch_qnnp_status_success;
    }
    case pytorch_qnnp_ukernel_type_dwconv: {
      pytorch_qnnp_indirection_set_step_dimensions(convolution);

      const size_t indirection_buffer_size = sizeof(void*) * batch_size *
          convolution->output_depth * convolution->step_depth;

      const void** indirection_buffer = (const void**)realloc(
          convolution->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for indirection buffer",
            indirection_buffer_size);
        return pytorch_qnnp_status_out_of_memory;
      }
      convolution->indirection_buffer = indirection_buffer;

      pytorch_qnnp_indirection_init_dwconv(convolution, 0);
      return pytorch_qnnp_status_success;
    }
    default:
      PYTORCH_QNNP_UNREACHABLE;
  }
}
