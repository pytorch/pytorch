/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/requantization.h>

enum pytorch_qnnp_format {
  pytorch_qnnp_format_quint8 = 0x02000000,
  pytorch_qnnp_format_float32 = 0x02020202,
  pytorch_qnnp_format_float16 = 0x01010101,
};

enum pytorch_qnnp_ukernel_type {
  pytorch_qnnp_ukernel_type_none = 0,
  pytorch_qnnp_ukernel_type_add,
  pytorch_qnnp_ukernel_type_average_pooling,
  pytorch_qnnp_ukernel_type_channel_shuffle,
  pytorch_qnnp_ukernel_type_clamp,
  pytorch_qnnp_ukernel_type_conv,
  pytorch_qnnp_ukernel_type_dwconv,
  pytorch_qnnp_ukernel_type_gemm,
  pytorch_qnnp_ukernel_type_gemm_sparse_dq,
  pytorch_qnnp_ukernel_type_gemm_prepackA_sparse_dq,
  pytorch_qnnp_ukernel_type_global_average_pooling,
  pytorch_qnnp_ukernel_type_lut,
  pytorch_qnnp_ukernel_type_max_pooling,
  pytorch_qnnp_ukernel_type_softargmax,
  pytorch_qnnp_ukernel_type_xzp_gemm,
};

typedef struct {
  const uint32_t* col_indices;
  const uint32_t* row_values;
  const uint8_t* values;
  uint32_t col_block_size;
} sparse_matrix_t;

struct pytorch_qnnp_operator {
  size_t batch_size;
  uint32_t input_padding_top;
  uint32_t input_padding_right;
  uint32_t input_padding_bottom;
  uint32_t input_padding_left;
  uint32_t adjustment_height;
  uint32_t adjustment_width;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t stride_height;
  uint32_t stride_width;
  uint32_t dilation_height;
  uint32_t dilation_width;
  uint32_t groups;
  size_t group_stride;
  size_t group_channels;
  size_t group_input_channels;
  size_t group_output_channels;
  size_t channels;

  size_t input_height;
  size_t input_width;
  size_t input_pixel_stride;
  const void* input;
  const void** indirection_buffer;
  void* a_sum;

  size_t input2_pixel_stride;
  const void* input2;

  size_t output_height;
  size_t output_width;
  size_t output_pixel_stride;
  void* output;

  void* packed_weights;
  float input_scale;
  float output_scale;
  uint8_t input_zero_point;
  uint8_t kernel_zero_point;
  uint8_t output_zero_point;
  uint8_t output_min;
  uint8_t output_max;

  size_t valid_batch_size;
  size_t last_input_height;
  size_t last_input_width;
  const void* last_input;

  void* zero_buffer;
  void* zero_pointer;
  void* lookup_table;

  union {
    union pytorch_qnnp_q31_requantization_params requantization_params;
    union pytorch_qnnp_conv_quantization_params conv_quantization_params;
    union pytorch_qnnp_add_quantization_params add_quantization_params;
    union pytorch_qnnp_avgpool_quantization_params avgpool_quantization_params;
    union pytorch_qnnp_u8_clamping_params u8_clamping_params;
  };
  enum pytorch_qnnp_ukernel_type ukernel_type;
  enum pytorch_qnnp_format format;

  bool per_channel;

  // Sparsity support
  sparse_matrix_t sparse_matrix;
  const void* bias;
  struct pytorch_qnnp_conv_dynamic_quantization_params dynamic_conv_quantization_params;
  uint8_t* prepacked_a;
};

static inline uint32_t pytorch_qnnp_operator_get_log2_output_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  return (uint32_t)(convolution->format & UINT32_C(0xFF));
}

static inline uint32_t pytorch_qnnp_operator_get_log2_input_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  return (uint32_t)((convolution->format >> 8) & UINT32_C(0xFF));
}

static inline uint32_t pytorch_qnnp_operator_get_log2_kernel_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  return (uint32_t)((convolution->format >> 16) & UINT32_C(0xFF));
}

static inline uint32_t pytorch_qnnp_operator_get_log2_bias_element_size(
    const struct pytorch_qnnp_operator* convolution) {
  return (uint32_t)((convolution->format >> 24) & UINT32_C(0xFF));
}
