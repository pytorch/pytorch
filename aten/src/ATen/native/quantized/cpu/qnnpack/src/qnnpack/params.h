/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <qnnpack/common.h>

#include <cpuinfo.h>

struct pytorch_qnnp_fp16_clamping_params {
  uint16_t scale;
  uint16_t max;
  uint16_t min;
};

struct pytorch_qnnp_fp32_clamping_params {
  float max;
  float min;
};

union pytorch_qnnp_fp32_requantization_params {
  struct {
    float scale;
    float min_less_zero_point;
    float max_less_zero_point;
    float magic;
    int32_t magic_less_zero_point;
  } scalar;
  struct {
    float scale;
    float max;
    float min;
    float magic;
    int32_t magic_less_zero_point;
  } neon;
  struct {
    float scale;
    int16_t zero_point;
    uint8_t max;
    uint8_t min;
  } neonv8;
  struct {
    PYTORCH_QNNP_ALIGN(16) float scale[4];
    PYTORCH_QNNP_ALIGN(16) int16_t zero_point[8];
    PYTORCH_QNNP_ALIGN(16) uint8_t max[16];
    PYTORCH_QNNP_ALIGN(16) uint8_t min[16];
  } sse2;
  struct {
    PYTORCH_QNNP_ALIGN(16) float scale[4];
    PYTORCH_QNNP_ALIGN(16) float min_less_zero_point[4];
    PYTORCH_QNNP_ALIGN(16) float max_less_zero_point[4];
    PYTORCH_QNNP_ALIGN(16) float magic[4];
    PYTORCH_QNNP_ALIGN(16) int32_t magic_less_zero_point[4];
  } psimd;
};

union pytorch_qnnp_precise_requantization_params {
  struct {
    uint32_t multiplier;
    uint32_t rounding_lo;
    uint32_t rounding_hi;
    uint32_t shift_less_32;
    int32_t min_less_zero_point;
    int32_t max_less_zero_point;
    int32_t zero_point;
  } scalar;
  struct {
    int32_t multiplier;
    int32_t right_shift;
    int16_t zero_point;
    uint8_t max;
    uint8_t min;
  } neon;
  struct {
    PYTORCH_QNNP_ALIGN(16) uint32_t multiplier[4];
    PYTORCH_QNNP_ALIGN(16) uint64_t rounding[2];
    PYTORCH_QNNP_ALIGN(16) uint32_t shift[4];
    PYTORCH_QNNP_ALIGN(16) int16_t zero_point[8];
    PYTORCH_QNNP_ALIGN(16) uint8_t max[16];
    PYTORCH_QNNP_ALIGN(16) uint8_t min[16];
  } sse2;
};

union pytorch_qnnp_q31_requantization_params {
  struct {
    int32_t multiplier;
    int32_t remainder_mask;
    int32_t remainder_threshold;
    uint32_t shift;
    int32_t min_less_zero_point;
    int32_t max_less_zero_point;
    int32_t zero_point;
  } scalar;
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    int32_t multiplier;
    int32_t right_shift;
    int16_t zero_point;
    uint8_t max;
    uint8_t min;
  } neon;
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    PYTORCH_QNNP_ALIGN(16) uint32_t multiplier[4];
    PYTORCH_QNNP_ALIGN(16) uint64_t rounding[2];
    PYTORCH_QNNP_ALIGN(16) int32_t remainder_mask[4];
    PYTORCH_QNNP_ALIGN(16) int32_t remainder_threshold[4];
    PYTORCH_QNNP_ALIGN(16) uint64_t shift[2];
    PYTORCH_QNNP_ALIGN(16) int16_t zero_point[8];
    PYTORCH_QNNP_ALIGN(16) uint8_t max[16];
    PYTORCH_QNNP_ALIGN(16) uint8_t min[16];
  } sse2;
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
};

union pytorch_qnnp_conv_quantization_params {
  struct {
    int32_t kernel_zero_point;
    int32_t input_zero_point;
    int32_t multiplier;
    int32_t remainder_mask;
    int32_t remainder_threshold;
    uint32_t shift;
    int32_t output_min_less_zero_point;
    int32_t output_max_less_zero_point;
    int32_t output_zero_point;
  } scalar;
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    int16_t kernel_zero_point;
    int16_t input_zero_point;
    int32_t multiplier;
    int32_t right_shift;
    int16_t output_zero_point;
    uint8_t output_max;
    uint8_t output_min;
  } neon;
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    PYTORCH_QNNP_ALIGN(16) int16_t kernel_zero_point[8];
    PYTORCH_QNNP_ALIGN(16) int16_t input_zero_point[8];
    PYTORCH_QNNP_ALIGN(16) uint32_t multiplier[4];
    PYTORCH_QNNP_ALIGN(16) uint64_t rounding[2];
    PYTORCH_QNNP_ALIGN(16) int32_t remainder_mask[4];
    PYTORCH_QNNP_ALIGN(16) int32_t remainder_threshold[4];
    PYTORCH_QNNP_ALIGN(16) uint64_t shift[2];
    PYTORCH_QNNP_ALIGN(16) int16_t output_zero_point[8];
    PYTORCH_QNNP_ALIGN(16) uint8_t output_max[16];
    PYTORCH_QNNP_ALIGN(16) uint8_t output_min[16];
  } sse2;
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
};

union pytorch_qnnp_requantization_params {
  union pytorch_qnnp_precise_requantization_params precise;
  union pytorch_qnnp_fp32_requantization_params fp32;
  union pytorch_qnnp_q31_requantization_params q31;
};

union pytorch_qnnp_add_quantization_params {
  struct {
    int32_t zero_point_product;
    uint32_t a_multiplier;
    uint32_t b_multiplier;
    uint32_t shift;
    int32_t remainder_mask;
    int32_t remainder_threshold;
    int32_t y_zero_point;
    int32_t y_max;
    int32_t y_min;
  } scalar;
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    uint8_t a_zero_point;
    uint8_t b_zero_point;
    int16_t y_zero_point;
    int32_t a_multiplier;
    int32_t b_multiplier;
    int32_t right_shift;
    uint8_t y_max;
    uint8_t y_min;
  } neon;
#endif
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    PYTORCH_QNNP_ALIGN(16) int32_t zero_point_product[4];
    PYTORCH_QNNP_ALIGN(16) uint16_t a_multiplier_lo[8];
    PYTORCH_QNNP_ALIGN(16) uint16_t a_multiplier_hi[8];
    PYTORCH_QNNP_ALIGN(16) uint16_t b_multiplier_lo[8];
    PYTORCH_QNNP_ALIGN(16) uint16_t b_multiplier_hi[8];
    PYTORCH_QNNP_ALIGN(16) int32_t remainder_mask[4];
    PYTORCH_QNNP_ALIGN(16) int32_t remainder_threshold[4];
    PYTORCH_QNNP_ALIGN(16) int16_t y_zero_point[8];
    PYTORCH_QNNP_ALIGN(16) uint8_t y_max[16];
    PYTORCH_QNNP_ALIGN(16) uint8_t y_min[16];
    uint32_t shift;
    uint32_t a_multiplier;
    uint32_t b_multiplier;
  } sse2;
#endif
};

union pytorch_qnnp_avgpool_quantization_params {
  struct {
    int32_t bias;
    int32_t multiplier;
    int64_t rounding;
    uint32_t right_shift;
    int32_t output_min_less_zero_point;
    int32_t output_max_less_zero_point;
    int32_t output_zero_point;
  } scalar;
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    int32_t bias;
    int32_t multiplier;
    int64_t left_shift;
    int16_t output_zero_point;
    uint8_t output_max;
    uint8_t output_min;
  } neon;
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    PYTORCH_QNNP_ALIGN(16) int32_t bias[4];
    PYTORCH_QNNP_ALIGN(16) uint32_t multiplier[4];
    PYTORCH_QNNP_ALIGN(16) uint64_t rounding[2];
    PYTORCH_QNNP_ALIGN(16) uint64_t right_shift[2];
    PYTORCH_QNNP_ALIGN(16) int16_t output_zero_point[8];
    PYTORCH_QNNP_ALIGN(16) uint8_t output_max[16];
    PYTORCH_QNNP_ALIGN(16) uint8_t output_min[16];
  } sse2;
#endif
};

union pytorch_qnnp_u8_clamping_params {
  struct {
    int32_t output_max;
    int32_t output_min;
  } scalar;
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    uint8_t output_max;
    uint8_t output_min;
  } neon;
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    PYTORCH_QNNP_ALIGN(16) uint8_t output_max[16];
    PYTORCH_QNNP_ALIGN(16) uint8_t output_min[16];
  } sse2;
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
};

typedef void (*q8gemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* a,
    size_t a_stride,
    const void* w,
    uint8_t* c,
    size_t c_stride,
    const union pytorch_qnnp_conv_quantization_params* quantization_params);

typedef void (*q8conv_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const uint8_t** a,
    const void* w,
    uint8_t* c,
    size_t c_stride,
    const union pytorch_qnnp_conv_quantization_params* quantization_params);

typedef void (*q8gemm_xzp_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* a,
    size_t a_stride,
    const int32_t* a_sum,
    const void* w,
    uint8_t* c,
    size_t c_stride,
    const union pytorch_qnnp_q31_requantization_params* requantization_params);

typedef void (*q8sum_rows_ukernel_function)(
    const uint8_t* a,
    size_t m,
    size_t k,
    size_t stride,
    int32_t multiplier,
    int32_t* sums);

typedef void (*xzipc_ukernel_function)(size_t n, const void* x, void* y);

typedef void (
    *xzipv_ukernel_function)(size_t n, size_t m, const void* x, void* y);

typedef void (*x8lut_ukernel_function)(
    size_t n,
    const uint8_t* x,
    const uint8_t* t,
    uint8_t* y);

typedef void (*sgemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t c_stride,
    const struct pytorch_qnnp_fp32_clamping_params* clamping_params);

typedef void (*sconv_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const float** a,
    const float* w,
    float* c,
    size_t c_stride,
    const struct pytorch_qnnp_fp32_clamping_params* clamping_params);

typedef void (*hgemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const void* a,
    size_t a_stride,
    const void* w,
    void* c,
    size_t c_stride,
    const struct pytorch_qnnp_fp16_clamping_params* clamping_params);

typedef void (*q8dwconv_up_ukernel_function)(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union pytorch_qnnp_conv_quantization_params* quantization_params);

typedef void (*q8dwconv_mp_ukernel_function)(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    int32_t* buffer,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union pytorch_qnnp_conv_quantization_params* quantization_params);

typedef void (*q8gavgpool_up_ukernel_function)(
    size_t m,
    size_t n,
    const uint8_t* x,
    size_t x_stride,
    const uint8_t* zero,
    uint8_t* y,
    const union pytorch_qnnp_avgpool_quantization_params* quantization_params);

typedef void (*q8gavgpool_mp_ukernel_function)(
    size_t m,
    size_t n,
    const uint8_t* x,
    size_t x_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* y,
    const union pytorch_qnnp_avgpool_quantization_params* quantization_params);

typedef void (*q8avgpool_up_ukernel_function)(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** x,
    const uint8_t* zero,
    uint8_t* y,
    size_t x_increment,
    size_t y_increment,
    const union pytorch_qnnp_avgpool_quantization_params* quantization_params);

typedef void (*q8avgpool_mp_ukernel_function)(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** x,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* y,
    size_t x_increment,
    size_t y_increment,
    const union pytorch_qnnp_avgpool_quantization_params* quantization_params);

typedef void (*u8maxpool_ukernel_function)(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** x,
    uint8_t* y,
    size_t x_increment,
    size_t y_increment,
    const union pytorch_qnnp_u8_clamping_params* params);

typedef void (*u8clamp_ukernel_function)(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union pytorch_qnnp_u8_clamping_params* params);

typedef uint8_t (*u8rmax_ukernel_function)(size_t n, const uint8_t* x);

typedef void (*u8lut32norm_ukernel_function)(
    size_t n,
    const uint8_t* x,
    const uint32_t* t,
    uint8_t* y);

typedef void (*q8vadd_ukernel_function)(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union pytorch_qnnp_add_quantization_params* quantization_params);

struct q8conv_parameters {
  q8gemm_ukernel_function gemm;
  q8conv_ukernel_function conv;
  uint8_t mr;
  uint8_t nr;
  uint8_t kr;
};

struct q8conv_xzp_parameters {
  q8gemm_xzp_ukernel_function gemm;
  /* no conv ukernel */
  uint8_t mr;
  uint8_t nr;
  uint8_t kr;
  uint8_t kc;
  size_t kthreshold;
};

struct q8dwconv_up_parameters {
  q8dwconv_up_ukernel_function updw;
  uint8_t cr;
};

struct q8dwconv_mp_parameters {
  q8dwconv_mp_ukernel_function mpdw;
  uint8_t cr;
};

struct q8sum_rows_parameters {
  q8sum_rows_ukernel_function sum_rows;
  uint32_t m;
};

struct q8gavgpool_parameters {
  q8gavgpool_up_ukernel_function ltnr;
  q8gavgpool_up_ukernel_function genr_lemr;
  q8gavgpool_mp_ukernel_function genr_gtmr;
  uint8_t mr;
  uint8_t nr;
};

struct q8avgpool_parameters {
  q8avgpool_up_ukernel_function ltkr;
  q8avgpool_up_ukernel_function gekr_lemr;
  q8avgpool_mp_ukernel_function gekr_gtmr;
  uint8_t mr;
  uint8_t qr;
  uint8_t kr;
};

struct u8maxpool_parameters {
  u8maxpool_ukernel_function ltkr;
  u8maxpool_ukernel_function gekr;
  uint8_t mr;
  uint8_t qr;
  uint8_t kr;
};

struct x8zip_parameters {
  xzipc_ukernel_function x2;
  xzipc_ukernel_function x3;
  xzipc_ukernel_function x4;
  xzipv_ukernel_function xm;
};

struct pytorch_qnnp_parameters {
  struct q8conv_parameters q8conv;
  struct q8conv_xzp_parameters q8conv_xzp;
  struct q8dwconv_up_parameters q8dw9;
  struct q8dwconv_mp_parameters q8dw25;
  struct q8sum_rows_parameters q8sum_rows;
  q8vadd_ukernel_function q8vadd;
  struct q8gavgpool_parameters q8gavgpool;
  struct q8avgpool_parameters q8avgpool;
  struct u8maxpool_parameters u8maxpool;
  u8lut32norm_ukernel_function u8lut32norm;
  u8clamp_ukernel_function u8clamp;
  u8rmax_ukernel_function u8rmax;
  struct x8zip_parameters x8zip;
  x8lut_ukernel_function x8lut;
  bool initialized;
};

extern PYTORCH_QNNP_INTERNAL struct pytorch_qnnp_parameters pytorch_qnnp_params;
