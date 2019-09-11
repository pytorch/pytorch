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

#define PYTORCH_DECLARE_U8RMAX_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL uint8_t fn_name(size_t n, const uint8_t* x);

PYTORCH_DECLARE_U8RMAX_UKERNEL_FUNCTION(pytorch_u8rmax_ukernel__neon)
PYTORCH_DECLARE_U8RMAX_UKERNEL_FUNCTION(pytorch_u8rmax_ukernel__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
