/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/operator.h>

enum pytorch_qnnp_status pytorch_qnnp_create_hardswish_nc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    pytorch_qnnp_operator_t* hardswish_out) {
  pytorch_qnnp_operator_t hardswish_op = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_hardswish_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  if (channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create Hardswish operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Hardswish operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    pytorch_qnnp_log_error(
        "failed to create Hardswish operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    pytorch_qnnp_log_error(
        "failed to create Hardswish operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;

  hardswish_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (hardswish_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  hardswish_op->lookup_table = malloc(256 * sizeof(uint8_t));
  if (hardswish_op->lookup_table == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate 256 bytes for Hardswish lookup table");
    goto error;
  }

  uint8_t* lookup_table = hardswish_op->lookup_table;
  const float scaled_min = (float)(int32_t)output_min;
  const float scaled_max = (float)(int32_t)output_max;
  const float inv_output_scale = 1.0f / output_scale;
  for (int32_t i = 0; i < 256; i++) {
    float x =
        input_scale * (float)(i - (int32_t)(uint32_t)input_zero_point);
    // hardswish, no min/max functions in C
    float x2 = x + 3.0f;
    x2 = x2 > 0.0f ? x2 : 0.0f;
    x2 = x2 < 6.0f ? x2 : 6.0f;
    x2 = x * x2 / 6.0f;
    float scaled_hardswish_x = inv_output_scale * x2 + output_zero_point;
    if (scaled_hardswish_x < scaled_min) {
      scaled_hardswish_x = scaled_min;
    }
    if (scaled_hardswish_x > scaled_max) {
      scaled_hardswish_x = scaled_max;
    }
    lookup_table[(uint32_t)i] = (uint8_t)lrintf(scaled_hardswish_x);
  }

  hardswish_op->channels = channels;

  hardswish_op->ukernel_type = pytorch_qnnp_ukernel_type_lut;
  hardswish_op->format = pytorch_qnnp_format_quint8;

  *hardswish_out = hardswish_op;
  return pytorch_qnnp_status_success;

error:
  pytorch_qnnp_delete_operator(hardswish_op);
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_hardswish_nc_q8(
    pytorch_qnnp_operator_t hardswish,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_hardswish_nc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    hardswish->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  hardswish->batch_size = batch_size;
  hardswish->input = input;
  hardswish->input_pixel_stride = input_stride;
  hardswish->output = output;
  hardswish->output_pixel_stride = output_stride;

  return pytorch_qnnp_status_success;
}
