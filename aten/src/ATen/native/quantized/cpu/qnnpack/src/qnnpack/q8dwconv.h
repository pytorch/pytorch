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

#define DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(                \
      size_t channels,                               \
      size_t output_width,                           \
      const uint8_t** input,                         \
      const void* weights,                           \
      uint8_t* output,                               \
      size_t input_stride,                           \
      size_t output_increment,                       \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(pytorch_q8dwconv_ukernel_up8x9__neon)
DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_up8x9_per_channel__neon)
DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon)
DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon)
DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(pytorch_q8dwconv_ukernel_up8x9__sse2)
DECLARE_PYTORCH_Q8UPDWCONV_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2)

#define DECLARE_PYTORCH_Q8MPDWCONV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(                \
      size_t channels,                               \
      size_t output_width,                           \
      const uint8_t** input,                         \
      const void* weights,                           \
      int32_t* buffer,                               \
      uint8_t* output,                               \
      size_t input_stride,                           \
      size_t output_increment,                       \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8MPDWCONV_UKERNEL_FUNCTION(pytorch_q8dwconv_ukernel_mp8x25__neon)
DECLARE_PYTORCH_Q8MPDWCONV_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon)
DECLARE_PYTORCH_Q8MPDWCONV_UKERNEL_FUNCTION(pytorch_q8dwconv_ukernel_mp8x25__sse2)
DECLARE_PYTORCH_Q8MPDWCONV_UKERNEL_FUNCTION(
    pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
