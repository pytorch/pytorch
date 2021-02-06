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

#define DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      const uint8_t* a,                          \
      size_t a_stride,                           \
      const uint8_t* packed_w,                   \
      const uint32_t* w_row_ptr,                 \
      const uint32_t* w_block_ids_ptr,           \
      uint8_t* c,                                \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const union pytorch_qnnp_conv_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_4x8__neon)
DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_8x8__neon)

DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_4x8__aarch32_neon)

DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_8x8__aarch64_neon)

DECLARE_PYTORCH_Q8GEMM_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_sparse_1x4_ukernel_4x4c2__sse2)

#define DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      const uint8_t* a,                          \
      size_t a_stride,                           \
      const uint8_t* packed_w,                   \
      const uint32_t* w_row_ptr,                 \
      const uint32_t* w_block_ids_ptr,           \
      const float* b,                            \
      float* c,                                  \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch64_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_UKERNEL_FUNCTION(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2)

#define DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      size_t mr,                                 \
      size_t nr,                                 \
      const uint8_t* a_packed,                   \
      const uint8_t* packed_w,                   \
      const uint32_t* w_row_ptr,                 \
      const uint32_t* w_block_ids_ptr,           \
      const float* b,                            \
      float* c,                                  \
      size_t c_stride,                           \
      size_t output_channel_index,               \
      const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params);

DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA__aarch64_neon)
DECLARE_PYTORCH_Q8GEMM_DYNAMIC_QUANTIZATION_SPARSE_PACKEDA_UKERNEL_FUNCTION(
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2)

#define DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(fn_name) \
  PYTORCH_QNNP_INTERNAL void fn_name(            \
      const size_t mr,                           \
      const size_t K,                            \
      const uint8_t* a,                          \
      const size_t a_stride,                     \
      uint8_t* a_packed);

DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon)
DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon)
DECLARE_PYTORCH_Q8GEMM_PARSE_PACKA_UKERNEL_FUNCTION(
    pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2)

#ifdef __cplusplus
} /* extern "C" */
#endif
