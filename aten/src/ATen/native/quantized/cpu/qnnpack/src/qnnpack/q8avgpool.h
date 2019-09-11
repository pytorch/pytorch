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

#define PYTORCH_DECLARE_Q8MPAVGPOOL_UKERNEL_FUNCTION(fn_name)       \
  PYTORCH_QNNP_INTERNAL void fn_name(                       \
      size_t n,                                             \
      size_t ks,                                            \
      size_t kc,                                            \
      const uint8_t** x,                                    \
      const uint8_t* zero,                                  \
      int32_t* buffer,                                      \
      uint8_t* y,                                           \
      size_t x_increment,                                   \
      size_t y_increment,                                   \
      const union pytorch_qnnp_avgpool_quantization_params* \
          quantization_params);

PYTORCH_DECLARE_Q8MPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_mp8x9p8q__neon)
PYTORCH_DECLARE_Q8MPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2)

#define PYTORCH_DECLARE_Q8UPAVGPOOL_UKERNEL_FUNCTION(fn_name)       \
  PYTORCH_QNNP_INTERNAL void fn_name(                       \
      size_t n,                                             \
      size_t ks,                                            \
      size_t kc,                                            \
      const uint8_t** x,                                    \
      const uint8_t* zero,                                  \
      uint8_t* y,                                           \
      size_t x_increment,                                   \
      size_t y_increment,                                   \
      const union pytorch_qnnp_avgpool_quantization_params* \
          quantization_params);

PYTORCH_DECLARE_Q8UPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_up8x9__neon)
PYTORCH_DECLARE_Q8UPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_up8xm__neon)
PYTORCH_DECLARE_Q8UPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_up8x9__sse2)
PYTORCH_DECLARE_Q8UPAVGPOOL_UKERNEL_FUNCTION(pytorch_q8avgpool_ukernel_up8xm__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
