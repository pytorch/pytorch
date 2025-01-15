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
#include <qnnpack/params.h>

enum pytorch_qnnp_status pytorch_qnnp_create_channel_shuffle_nc_x8(
    size_t groups,
    size_t group_channels,
    uint32_t flags,
    pytorch_qnnp_operator_t* channel_shuffle_out) {
  pytorch_qnnp_operator_t channel_shuffle_op = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_channel_shuffle_nc_x8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = pytorch_qnnp_status_invalid_parameter;

  if (groups <= 1) {
    pytorch_qnnp_log_error(
        "failed to create channel shuffle operator with %zu groups: "
        "at least two groups required",
        groups);
    goto error;
  }

  if (group_channels == 0) {
    pytorch_qnnp_log_error(
        "failed to create channel shuffle operator with %zu group channels: "
        "number of group channels must be non-zero",
        group_channels);
    goto error;
  }

  status = pytorch_qnnp_status_out_of_memory;

  channel_shuffle_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (channel_shuffle_op == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  channel_shuffle_op->groups = groups;
  channel_shuffle_op->group_channels = group_channels;

  channel_shuffle_op->ukernel_type = pytorch_qnnp_ukernel_type_channel_shuffle;
  channel_shuffle_op->format = pytorch_qnnp_format_quint8;

  *channel_shuffle_out = channel_shuffle_op;
  return pytorch_qnnp_status_success;

error:
  pytorch_qnnp_delete_operator(channel_shuffle_op);
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_channel_shuffle_nc_x8(
    pytorch_qnnp_operator_t channel_shuffle_op,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_channel_shuffle_nc_x8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    channel_shuffle_op->batch_size = 0;
    return pytorch_qnnp_status_success;
  }

  channel_shuffle_op->batch_size = batch_size;
  channel_shuffle_op->input = input;
  channel_shuffle_op->input_pixel_stride = input_stride;
  channel_shuffle_op->output = output;
  channel_shuffle_op->output_pixel_stride = output_stride;

  return pytorch_qnnp_status_success;
}
