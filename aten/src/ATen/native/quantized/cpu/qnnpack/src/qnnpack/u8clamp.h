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

#define DECLARE_PYTORCH_U8CLAMP_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(             \
      size_t n,                                   \
      const uint8_t* x,                           \
      uint8_t* y,                                 \
      const union pytorch_qnnp_u8_clamping_params* params);

DECLARE_PYTORCH_U8CLAMP_UKERNEL_FUNCTION(pytorch_u8clamp_ukernel__neon)
DECLARE_PYTORCH_U8CLAMP_UKERNEL_FUNCTION(pytorch_u8clamp_ukernel__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
