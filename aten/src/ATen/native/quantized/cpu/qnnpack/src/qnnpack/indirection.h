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

PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_conv3d(
    pytorch_qnnp_operator_t op,
    size_t output_tile_size,
    size_t tiled_output_size);

PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_dwconv(
    pytorch_qnnp_operator_t op,
    size_t batch_start);

PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_deconv2d(
    pytorch_qnnp_operator_t op,
    size_t output_tile_size,
    size_t tiled_output_size);

PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_init_maxpool2d(
    pytorch_qnnp_operator_t op,
    size_t batch_start);

PYTORCH_QNNP_INTERNAL void pytorch_qnnp_indirection_set_step_dimensions(
    pytorch_qnnp_operator_t op);

#ifdef __cplusplus
} /* extern "C" */
#endif
