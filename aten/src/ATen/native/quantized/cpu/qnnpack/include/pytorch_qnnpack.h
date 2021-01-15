/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <pthreadpool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status code for any QNNPACK function call.
 */
enum pytorch_qnnp_status {
  /** The call succeeded, and all output arguments now contain valid data. */
  pytorch_qnnp_status_success = 0,
  pytorch_qnnp_status_uninitialized = 1,
  pytorch_qnnp_status_invalid_parameter = 2,
  pytorch_qnnp_status_unsupported_parameter = 3,
  pytorch_qnnp_status_unsupported_hardware = 4,
  pytorch_qnnp_status_out_of_memory = 5,
};

enum pytorch_qnnp_status pytorch_qnnp_initialize(void);

enum pytorch_qnnp_status pytorch_qnnp_deinitialize(void);

typedef struct pytorch_qnnp_operator* pytorch_qnnp_operator_t;

enum pytorch_qnnp_status pytorch_qnnp_create_convolution2d_nhwc_q8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
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
    pytorch_qnnp_operator_t* convolution);

enum pytorch_qnnp_status pytorch_qnnp_setup_convolution2d_nhwc_q8(
    pytorch_qnnp_operator_t convolution,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride,
    pthreadpool_t threadpool);

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
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    pytorch_qnnp_operator_t* deconvolution);

enum pytorch_qnnp_status pytorch_qnnp_setup_deconvolution2d_nhwc_q8(
    pytorch_qnnp_operator_t deconvolution,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride,
    pthreadpool_t threadpool);

enum pytorch_qnnp_status pytorch_qnnp_create_fully_connected_nc_q8(
    size_t input_channels,
    size_t output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    pytorch_qnnp_operator_t* fully_connected);

enum pytorch_qnnp_status pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8(
    size_t input_channels,
    size_t output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint32_t* kernel_col_indices,
    const uint32_t* kernel_row_values,
    const uint8_t* kernel_values,
    const uint32_t kernel_col_block_size,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    bool use_prepack_kernel,
    pytorch_qnnp_operator_t* fully_connected);

enum pytorch_qnnp_status pytorch_qnnp_setup_fully_connected_nc_q8(
    pytorch_qnnp_operator_t fully_connected,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum pytorch_qnnp_status pytorch_qnnp_setup_fully_connected_sparse_dq_nc_q8(
    pytorch_qnnp_operator_t fully_connected,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    const float* bias,
    float* output,
    size_t output_stride);

enum pytorch_qnnp_status pytorch_qnnp_create_global_average_pooling_nwc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* global_average_pooling);

enum pytorch_qnnp_status pytorch_qnnp_setup_global_average_pooling_nwc_q8(
    pytorch_qnnp_operator_t global_average_pooling,
    size_t batch_size,
    size_t width,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

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
    pytorch_qnnp_operator_t* average_pooling);

enum pytorch_qnnp_status pytorch_qnnp_setup_average_pooling2d_nhwc_q8(
    pytorch_qnnp_operator_t average_pooling,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride,
    pthreadpool_t threadpool);

enum pytorch_qnnp_status pytorch_qnnp_create_max_pooling2d_nhwc_u8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    size_t channels,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* max_pooling);

enum pytorch_qnnp_status pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
    pytorch_qnnp_operator_t max_pooling,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride,
    pthreadpool_t threadpool);

enum pytorch_qnnp_status pytorch_qnnp_create_channel_shuffle_nc_x8(
    size_t groups,
    size_t group_channels,
    uint32_t flags,
    pytorch_qnnp_operator_t* channel_shuffle);

enum pytorch_qnnp_status pytorch_qnnp_setup_channel_shuffle_nc_x8(
    pytorch_qnnp_operator_t channel_shuffle,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum pytorch_qnnp_status pytorch_qnnp_create_add_nc_q8(
    size_t channels,
    uint8_t a_zero_point,
    float a_scale,
    uint8_t b_zero_point,
    float b_scale,
    uint8_t sum_zero_point,
    float sum_scale,
    uint8_t sum_min,
    uint8_t sum_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* add);

enum pytorch_qnnp_status pytorch_qnnp_setup_add_nc_q8(
    pytorch_qnnp_operator_t add,
    size_t batch_size,
    const uint8_t* a,
    size_t a_stride,
    const uint8_t* b,
    size_t b_stride,
    uint8_t* sum,
    size_t sum_stride);

enum pytorch_qnnp_status pytorch_qnnp_create_clamp_nc_u8(
    size_t channels,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* clamp);

enum pytorch_qnnp_status pytorch_qnnp_setup_clamp_nc_u8(
    pytorch_qnnp_operator_t clamp,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum pytorch_qnnp_status pytorch_qnnp_create_sigmoid_nc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* sigmoid);

enum pytorch_qnnp_status pytorch_qnnp_setup_sigmoid_nc_q8(
    pytorch_qnnp_operator_t sigmoid,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum pytorch_qnnp_status pytorch_qnnp_create_leaky_relu_nc_q8(
    size_t channels,
    float negative_slope,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* leaky_relu);

enum pytorch_qnnp_status pytorch_qnnp_setup_leaky_relu_nc_q8(
    pytorch_qnnp_operator_t leaky_relu,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum pytorch_qnnp_status pytorch_qnnp_create_softargmax_nc_q8(
    size_t channels,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint32_t flags,
    pytorch_qnnp_operator_t* softargmax);

enum pytorch_qnnp_status pytorch_qnnp_setup_softargmax_nc_q8(
    pytorch_qnnp_operator_t softargmax,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum pytorch_qnnp_status pytorch_qnnp_create_tanh_nc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* tanh);

enum pytorch_qnnp_status pytorch_qnnp_setup_tanh_nc_q8(
    pytorch_qnnp_operator_t tanh,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum pytorch_qnnp_status pytorch_qnnp_create_hardsigmoid_nc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* hardsigmoid);

enum pytorch_qnnp_status pytorch_qnnp_setup_hardsigmoid_nc_q8(
    pytorch_qnnp_operator_t hardsigmoid,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum pytorch_qnnp_status pytorch_qnnp_create_hardswish_nc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* hardswish);

enum pytorch_qnnp_status pytorch_qnnp_setup_hardswish_nc_q8(
    pytorch_qnnp_operator_t hardswish,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride);

enum pytorch_qnnp_status pytorch_qnnp_run_operator(
    pytorch_qnnp_operator_t op,
    pthreadpool_t threadpool);

enum pytorch_qnnp_status pytorch_qnnp_delete_operator(
    pytorch_qnnp_operator_t op);

#ifdef __cplusplus
} /* extern "C" */
#endif
