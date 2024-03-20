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

#define DECLARE_PYTORCH_SCONV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(           \
      size_t mr,                                \
      size_t nr,                                \
      size_t kc,                                \
      size_t ks,                                \
      const float** a,                          \
      const float* w,                           \
      float* c,                                 \
      size_t c_stride,                          \
      const struct pytorch_qnnp_fp32_clamping_params* params);

DECLARE_PYTORCH_SCONV_UKERNEL_FUNCTION(pytorch_sconv_ukernel_6x8__psimd)

#ifdef __cplusplus
} /* extern "C" */
#endif
