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

#include<pytorch_qnnpack.h>
#include <qnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif

  PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_conv2d(
      pytorch_qnnp_operator_t op,
      size_t output_tile_size,
      size_t tiled_output_size);

  PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_dwconv2d(
      pytorch_qnnp_operator_t convolution,
      size_t batch_start,
      size_t step_height,
      size_t step_width);

  PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_deconv2d(
      pytorch_qnnp_operator_t op,
      size_t output_tile_size,
      size_t tiled_output_size);

  PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_maxpool2d(
      pytorch_qnnp_operator_t op,
      size_t batch_start,
      size_t step_height,
      size_t step_width);

#ifdef __cplusplus
} /* extern "C" */
#endif
