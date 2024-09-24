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

#define DECLARE_PYTORCH_XZIPC_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(size_t n, const void* x, void* y);

DECLARE_PYTORCH_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x2__neon)
DECLARE_PYTORCH_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x2__sse2)
DECLARE_PYTORCH_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x3__neon)
DECLARE_PYTORCH_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x3__sse2)
DECLARE_PYTORCH_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x4__neon)
DECLARE_PYTORCH_XZIPC_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_x4__sse2)

#define DECLARE_PYTORCH_XZIPV_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(           \
      size_t n, size_t m, const void* x, void* y);

DECLARE_PYTORCH_XZIPV_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_xm__neon)
DECLARE_PYTORCH_XZIPV_UKERNEL_FUNCTION(pytorch_qnnp_x8zip_xm__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
