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

#define DECLARE_PYTORCH_Q8CONV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      size_t kc,                                 \
      size_t ks,                                 \
      const uint8_t** a,                         \
      const void* w,                             \
      uint8_t* c,                                \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8CONV_UKERNEL_FUNCTION(pytorch_q8conv_ukernel_4x8__neon)
DECLARE_PYTORCH_Q8CONV_UKERNEL_FUNCTION(pytorch_q8conv_ukernel_4x8__aarch32_neon)
DECLARE_PYTORCH_Q8CONV_UKERNEL_FUNCTION(pytorch_q8conv_ukernel_8x8__aarch64_neon)
DECLARE_PYTORCH_Q8CONV_UKERNEL_FUNCTION(pytorch_q8conv_ukernel_8x8__neon)
DECLARE_PYTORCH_Q8CONV_UKERNEL_FUNCTION(pytorch_q8conv_ukernel_4x4c2__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
