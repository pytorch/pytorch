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

#define DECLARE_PYTORCH_X8LUT32NORM_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(                 \
      size_t n, const uint8_t* x, const uint32_t* t, uint8_t* y);

DECLARE_PYTORCH_X8LUT32NORM_UKERNEL_FUNCTION(pytorch_u8lut32norm_ukernel__scalar)

#ifdef __cplusplus
} /* extern "C" */
#endif
