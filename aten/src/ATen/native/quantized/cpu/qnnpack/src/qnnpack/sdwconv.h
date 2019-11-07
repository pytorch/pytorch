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

#include <qnnpack/common.h>
#include <qnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_PYTORCH_SUPDWCONV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(               \
      size_t channels,                              \
      size_t output_width,                          \
      const float** input,                          \
      const float* weights,                         \
      float* output,                                \
      size_t input_stride,                          \
      size_t output_increment,                      \
      const struct pytorch_qnnp_fp32_clamping_params* clamping_params);

DECLARE_PYTORCH_SUPDWCONV_UKERNEL_FUNCTION(pytorch_sdwconv_ukernel_up4x9__psimd)

#define DECLARE_PYTORCH_SMPDWCONV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(               \
      size_t channels,                              \
      size_t output_width,                          \
      const uint8_t** input,                        \
      const void* weights,                          \
      int32_t* buffer,                              \
      uint8_t* output,                              \
      size_t input_stride,                          \
      size_t output_increment,                      \
      const struct pytorch_qnnp_fp32_clamping_params* clamping_params);

#ifdef __cplusplus
} /* extern "C" */
#endif
