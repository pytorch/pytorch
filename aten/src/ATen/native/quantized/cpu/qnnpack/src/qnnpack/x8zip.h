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

#define PYTORCH_DECLARE_XZIPC_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(size_t n, const void* x, void* y);

PYTORCH_DECLARE_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x2__neon)
PYTORCH_DECLARE_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x2__sse2)
PYTORCH_DECLARE_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x3__neon)
PYTORCH_DECLARE_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x3__sse2)
PYTORCH_DECLARE_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x4__neon)
PYTORCH_DECLARE_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x4__sse2)

#define PYTORCH_DECLARE_XZIPV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(           \
      size_t n, size_t m, const void* x, void* y);

PYTORCH_DECLARE_XZIPV_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_xm__neon)
PYTORCH_DECLARE_XZIPV_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_xm__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
