/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <fp16/bitcasts.h>

#include <qnnpack/params.h>
#include <qnnpack/scalar-utils.h>

static inline union pytorch_qnnp_q31_requantization_params
pytorch_qnnp_compute_scalar_requantization_params(
    float scale,
    uint8_t zero_point,
    uint8_t min,
    uint8_t max) {
  /* Compute requantization parameters */
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);
  const uint32_t scale_bits = fp32_to_bits(scale);

  /* Multiplier is in [0x40000000, 0x7FFFFF80] range */
  const int32_t multiplier = (int32_t)(
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  /* Shift is in [0, 31] range */
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  union pytorch_qnnp_q31_requantization_params params;
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  params.scalar.multiplier = multiplier;
  params.scalar.remainder_mask = (int32_t)remainder_mask;
  params.scalar.remainder_threshold = (int32_t)remainder_threshold;
  params.scalar.shift = (uint32_t)shift;
  params.scalar.min_less_zero_point =
      (int32_t)(uint32_t)min - (int32_t)(uint32_t)zero_point;
  params.scalar.max_less_zero_point =
      (int32_t)(uint32_t)max - (int32_t)(uint32_t)zero_point;
  params.scalar.zero_point = (int32_t)(uint32_t)zero_point;
  return params;
}

static inline union pytorch_qnnp_fp32_requantization_params
pytorch_qnnp_compute_scalar_fp32_requantization_params(
    float* scales,
    uint8_t zero_point,
    uint8_t min,
    uint8_t max) {

  union pytorch_qnnp_fp32_requantization_params params;
  params.scalar.scales = scales;
  params.scalar.output_zero_point = zero_point;
  params.scalar.output_max = max;
  params.scalar.output_min = min;
  params.scalar.min_less_zero_point = ((float)((int32_t)(uint32_t)min -
      (int32_t)(uint32_t)zero_point));
  params.scalar.max_less_zero_point = ((float)((int32_t)(uint32_t)max -
      (int32_t)(uint32_t)zero_point));
  params.scalar.magic = 12582912.0f;
  params.scalar.magic_less_zero_point = (INT32_C(0x4B400000) -
      (int32_t)(uint32_t)zero_point);
  return params;
}

static inline union pytorch_qnnp_q31_requantization_params
pytorch_qnnp_compute_requantization_params(
    float scale,
    uint8_t zero_point,
    uint8_t min,
    uint8_t max) {
  /* Compute requantization parameters */
  const uint32_t scale_bits = fp32_to_bits(scale);

  /* Multiplier is in [0x40000000, 0x7FFFFF80] range */
  const int32_t multiplier = (int32_t)(
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  /* Shift is in [0, 31] range */
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  union pytorch_qnnp_q31_requantization_params params;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  params.sse2.multiplier[0] = multiplier;
  params.sse2.multiplier[1] = multiplier;
  params.sse2.multiplier[2] = multiplier;
  params.sse2.multiplier[3] = multiplier;
  params.sse2.rounding[0] = UINT64_C(0x40000000);
  params.sse2.rounding[1] = UINT64_C(0x40000000);
  params.sse2.remainder_mask[0] = (int32_t)remainder_mask;
  params.sse2.remainder_mask[1] = (int32_t)remainder_mask;
  params.sse2.remainder_mask[2] = (int32_t)remainder_mask;
  params.sse2.remainder_mask[3] = (int32_t)remainder_mask;
  params.sse2.remainder_threshold[0] = (int32_t)remainder_threshold;
  params.sse2.remainder_threshold[1] = (int32_t)remainder_threshold;
  params.sse2.remainder_threshold[2] = (int32_t)remainder_threshold;
  params.sse2.remainder_threshold[3] = (int32_t)remainder_threshold;
  params.sse2.shift[0] = (uint64_t)(uint32_t)shift;
  params.sse2.shift[1] = (uint64_t)(uint32_t)shift;
  for (uint32_t i = 0; i < 8; i++) {
    params.sse2.zero_point[i] = (int16_t)(uint16_t)zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params.sse2.max[i] = max;
    params.sse2.min[i] = min;
  }
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  params.neon.multiplier = multiplier;
  params.neon.right_shift = -shift;
  params.neon.zero_point = (int16_t)(uint16_t)zero_point;
  params.neon.max = max;
  params.neon.min = min;
#else
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  params.scalar.multiplier = multiplier;
  params.scalar.remainder_mask = (int32_t)remainder_mask;
  params.scalar.remainder_threshold = (int32_t)remainder_threshold;
  params.scalar.shift = (uint32_t)shift;
  params.scalar.min_less_zero_point =
      (int32_t)(uint32_t)min - (int32_t)(uint32_t)zero_point;
  params.scalar.max_less_zero_point =
      (int32_t)(uint32_t)max - (int32_t)(uint32_t)zero_point;
  params.scalar.zero_point = (int32_t)(uint32_t)zero_point;
#endif
  return params;
}

static inline union pytorch_qnnp_conv_quantization_params
pytorch_qnnp_compute_conv_quantization_params(
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const float* requantization_scales,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max) {

  union pytorch_qnnp_conv_quantization_params params;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  params.sse2.kernel_zero_points = kernel_zero_points;
  for (uint32_t i = 0; i < 8; i++) {
    params.sse2.input_zero_point[i] = (int16_t)(uint16_t)input_zero_point;
  }
  params.sse2.requantization_scales = requantization_scales;
  for (uint32_t i = 0; i < 8; i++) {
    params.sse2.output_zero_point[i] = (int16_t)(uint16_t)output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params.sse2.output_max[i] = output_max;
    params.sse2.output_min[i] = output_min;
  }
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  params.neon.input_zero_point = (int16_t)(uint16_t)input_zero_point;
  params.neon.kernel_zero_points = kernel_zero_points;
  params.neon.requantization_scales = requantization_scales;
  params.neon.output_zero_point = (int16_t)(uint16_t)output_zero_point;
  params.neon.output_max = output_max;
  params.neon.output_min = output_min;
  params.neon.vfmin = ((float)((int32_t)(uint32_t)output_min -
      (int32_t)(uint32_t)output_zero_point));
  params.neon.vfmax = ((float)((int32_t)(uint32_t)output_max -
      (int32_t)(uint32_t)output_zero_point));
  params.neon.vfmagic = 12582912.0f;
  params.neon.vimagic = (INT32_C(0x4B400000) -
      (int32_t)(uint32_t)output_zero_point);
#else
  params.scalar.input_zero_point = (int32_t)(uint32_t)input_zero_point;
  params.scalar.kernel_zero_points = kernel_zero_points;
  params.scalar.requantization_scales = requantization_scales;
  params.scalar.output_min_less_zero_point =
      (int32_t)(uint32_t)output_min - (int32_t)(uint32_t)output_zero_point;
  params.scalar.output_max_less_zero_point =
      (int32_t)(uint32_t)output_max - (int32_t)(uint32_t)output_zero_point;
  params.scalar.output_zero_point = (int32_t)(uint32_t)output_zero_point;
#endif
  return params;
}

static inline union pytorch_qnnp_avgpool_quantization_params
pytorch_qnnp_compute_avgpool_quantization_params(
    int32_t bias,
    float scale,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max) {
  /* Compute requantization parameters */
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  union pytorch_qnnp_avgpool_quantization_params params;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  params.sse2.bias[0] = bias;
  params.sse2.bias[1] = bias;
  params.sse2.bias[2] = bias;
  params.sse2.bias[3] = bias;
  params.sse2.scale[0] = scale;
  params.sse2.scale[1] = scale;
  params.sse2.scale[2] = scale;
  params.sse2.scale[3] = scale;
  for (uint32_t i = 0; i < 8; i++) {
    params.sse2.output_zero_point[i] = (int16_t)(uint16_t)output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params.sse2.output_max[i] = output_max;
    params.sse2.output_min[i] = output_min;
  }
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  params.neon.bias = bias;
  params.neon.scale = scale;
  params.neon.output_zero_point = (int16_t)(uint16_t)output_zero_point;
  params.neon.output_max = output_max;
  params.neon.output_min = output_min;
  params.neon.vfmin = ((float)((int32_t)(uint32_t)output_min -
      (int32_t)(uint32_t)output_zero_point));
  params.neon.vfmax = ((float)((int32_t)(uint32_t)output_max -
      (int32_t)(uint32_t)output_zero_point));
  params.neon.vfmagic = 12582912.0f;
  params.neon.vimagic = (INT32_C(0x4B400000) -
      (int32_t)(uint32_t)output_zero_point);
#else
  params.scalar.bias = bias;
  params.scalar.scale = scale;
  params.scalar.output_zero_point = (int32_t)(uint32_t)output_zero_point;
  params.scalar.output_max = (int32_t)(uint32_t)output_max;
  params.scalar.output_min = (int32_t)(uint32_t)output_min;
#endif
  return params;
}

static inline union pytorch_qnnp_avgpool_quantization_params
pytorch_qnnp_compute_scalar_avgpool_quantization_params(
    int32_t bias,
    float scale,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max) {
  /* Compute requantization parameters */
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  union pytorch_qnnp_avgpool_quantization_params params;
  params.scalar.bias = bias;
  params.scalar.scale = scale;
  params.scalar.output_zero_point = (int32_t)(uint32_t)output_zero_point;
  params.scalar.output_max = (int32_t)(uint32_t)output_max;
  params.scalar.output_min = (int32_t)(uint32_t)output_min;
  return params;
}

static inline union pytorch_qnnp_u8_clamping_params
pytorch_qnnp_compute_u8_clamping_params(
    uint8_t output_min,
    uint8_t output_max) {
  assert(output_min <= output_max);

  union pytorch_qnnp_u8_clamping_params params;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  for (uint32_t i = 0; i < 16; i++) {
    params.sse2.output_max[i] = output_max;
    params.sse2.output_min[i] = output_min;
  }
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  params.neon.output_max = output_max;
  params.neon.output_min = output_min;
#else
  params.scalar.output_min = (int32_t)(uint32_t)output_min;
  params.scalar.output_max = (int32_t)(uint32_t)output_max;
#endif
  return params;
}

static inline union pytorch_qnnp_add_quantization_params
pytorch_qnnp_compute_add_quantization_params(
    uint8_t a_zero_point,
    uint8_t b_zero_point,
    uint8_t output_zero_point,
    float a_output_scale,
    float b_output_scale,
    uint8_t output_min,
    uint8_t output_max) {
  assert(a_output_scale >= 0x1.0p-14f);
  assert(b_output_scale >= 0x1.0p-14f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  /* Compute requantization parameters */
  const float max_output_scale =
      a_output_scale > b_output_scale ? a_output_scale : b_output_scale;
  assert(max_output_scale >= 0x1.0p-14f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t)(max_scale_bits >> 23) - 127;
  /* Shift is in [13, 31] range */
  const uint32_t shift = (uint32_t)(21 - max_scale_exponent);
  assert(shift < 32);
  assert(shift >= 13);

  const float scale_multiplier =
      fp32_from_bits((uint32_t)(21 - max_scale_exponent + 127) << 23);

  /* Multipliers are in [0, 2**22) range, largest multiplier is in [2**21,
   * 2**22) range */
  const uint32_t a_multiplier =
      (uint32_t)(int32_t)lrintf(a_output_scale * scale_multiplier);
  const uint32_t b_multiplier =
      (uint32_t)(int32_t)lrintf(b_output_scale * scale_multiplier);
  assert(
      (a_multiplier > b_multiplier ? a_multiplier : b_multiplier) >=
      UINT32_C(0x00200000));
  assert(a_multiplier < UINT32_C(0x00400000));
  assert(b_multiplier < UINT32_C(0x00400000));

  union pytorch_qnnp_add_quantization_params params;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  const int32_t zero_point_product = (int32_t) -
      (a_multiplier * (uint32_t)a_zero_point +
       b_multiplier * (uint32_t)b_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params.sse2.zero_point_product[i] = zero_point_product;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params.sse2.y_zero_point[i] = (int16_t)(uint16_t)output_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params.sse2.a_multiplier_lo[i] = (uint16_t)(uint32_t)a_multiplier;
    params.sse2.a_multiplier_hi[i] = (uint16_t)((uint32_t)a_multiplier >> 16);
    params.sse2.b_multiplier_lo[i] = (uint16_t)(uint32_t)b_multiplier;
    params.sse2.b_multiplier_hi[i] = (uint16_t)((uint32_t)b_multiplier >> 16);
  }
  params.sse2.a_multiplier = a_multiplier;
  params.sse2.b_multiplier = b_multiplier;
  for (uint32_t i = 0; i < 4; i++) {
    params.sse2.remainder_mask[i] = remainder_mask;
    params.sse2.remainder_threshold[i] = remainder_threshold;
  }
  params.sse2.shift = shift;
  for (uint32_t i = 0; i < 16; i++) {
    params.sse2.y_max[i] = output_max;
    params.sse2.y_min[i] = output_min;
  }
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  params.neon.a_zero_point = a_zero_point;
  params.neon.b_zero_point = b_zero_point;
  params.neon.y_zero_point = (int16_t)(uint16_t)output_zero_point;
  params.neon.a_multiplier = (int32_t)a_multiplier;
  params.neon.b_multiplier = (int32_t)b_multiplier;
  params.neon.right_shift = (int32_t)-shift;
  params.neon.y_max = output_max;
  params.neon.y_min = output_min;
#else
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  params.scalar.zero_point_product = (int32_t) -
      (a_multiplier * (uint32_t)a_zero_point +
       b_multiplier * (uint32_t)b_zero_point);
  params.scalar.a_multiplier = a_multiplier;
  params.scalar.b_multiplier = b_multiplier;
  params.scalar.remainder_mask = (int32_t)remainder_mask;
  params.scalar.remainder_threshold = (int32_t)remainder_threshold;
  params.scalar.shift = shift;
  params.scalar.y_zero_point = (int32_t)(uint32_t)output_zero_point;
  params.scalar.y_max = (int32_t)(uint32_t)output_max;
  params.scalar.y_min = (int32_t)(uint32_t)output_min;
#endif
  return params;
}

static inline union pytorch_qnnp_add_quantization_params
pytorch_qnnp_compute_scalar_add_quantization_params(
    uint8_t a_zero_point,
    uint8_t b_zero_point,
    uint8_t output_zero_point,
    float a_output_scale,
    float b_output_scale,
    uint8_t output_min,
    uint8_t output_max) {
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  /* Compute requantization parameters */
  const float max_output_scale =
      a_output_scale > b_output_scale ? a_output_scale : b_output_scale;
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t)(max_scale_bits >> 23) - 127;
  /* Shift is in [13, 31] range */
  const uint32_t shift = (uint32_t)(21 - max_scale_exponent);
  assert(shift < 32);
  assert(shift >= 13);

  /* Multipliers are in [0, 2**22) range, largest multiplier is in [2**21,
   * 2**22) range */
  const uint32_t a_multiplier = (uint32_t)(int32_t)lrintf(
      fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const uint32_t b_multiplier = (uint32_t)(int32_t)lrintf(
      fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(
      (a_multiplier > b_multiplier ? a_multiplier : b_multiplier) >=
      UINT32_C(0x00200000));
  assert(a_multiplier < UINT32_C(0x00400000));
  assert(b_multiplier < UINT32_C(0x00400000));

  union pytorch_qnnp_add_quantization_params params;
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  params.scalar.zero_point_product = (int32_t) -
      (a_multiplier * (uint32_t)a_zero_point +
       b_multiplier * (uint32_t)b_zero_point);
  params.scalar.a_multiplier = a_multiplier;
  params.scalar.b_multiplier = b_multiplier;
  params.scalar.remainder_mask = (int32_t)remainder_mask;
  params.scalar.remainder_threshold = (int32_t)remainder_threshold;
  params.scalar.shift = shift;
  params.scalar.y_zero_point = (int32_t)(uint32_t)output_zero_point;
  params.scalar.y_max = (int32_t)(uint32_t)output_max;
  params.scalar.y_min = (int32_t)(uint32_t)output_min;
  return params;
}

static inline uint8_t pytorch_qnnp_q31_requantize(
    int32_t n,
    union pytorch_qnnp_q31_requantization_params params) {
  const int64_t product = (int64_t)n * (int64_t)params.scalar.multiplier;
  const int32_t q31product =
      (int32_t)(uint32_t)((uint64_t)(product + INT64_C(0x40000000)) >> 31);
  const int32_t remainder =
      (q31product & params.scalar.remainder_mask) - (int32_t)(n < 0);
  n = asr_s32(q31product, params.scalar.shift) +
      (int32_t)(remainder > params.scalar.remainder_threshold);
  if (n < params.scalar.min_less_zero_point) {
    n = params.scalar.min_less_zero_point;
  }
  if (n > params.scalar.max_less_zero_point) {
    n = params.scalar.max_less_zero_point;
  }

  return (uint8_t)(n + params.scalar.zero_point);
}

static inline uint8_t pytorch_qnnp_fp32_requantize(
    int32_t n,
    union pytorch_qnnp_fp32_requantization_params params,
    int32_t output_channel_index) {

  const long lmin =
      (long)((int32_t)(uint32_t)params.scalar.output_min -
          (int32_t)(uint32_t)params.scalar.output_zero_point);
  const long lmax =
      (long)((int32_t)(uint32_t)params.scalar.output_max -
          (int32_t)(uint32_t)params.scalar.output_zero_point);

  const float n_scaled = (float)n * params.scalar.scales[output_channel_index];
  const long n_rounded = lrintf(n_scaled);
  const int32_t n_clamped = (int32_t)(
      n_rounded < lmin ? lmin : n_rounded > lmax ? lmax : n_rounded);
  const int32_t n_biased =
      n_clamped + (int32_t)(uint32_t)params.scalar.output_zero_point;

  return (uint8_t)n_biased;
}

static inline uint8_t pytorch_qnnp_fp32_requantize_magic(
    int32_t n,
    union pytorch_qnnp_fp32_requantization_params params,
    int32_t output_channel_index) {

  const float fmin = params.scalar.min_less_zero_point;
  const float fmax = params.scalar.max_less_zero_point;
  const float fmagic = params.scalar.magic;
  const int32_t imagic = params.scalar.magic_less_zero_point;

  const float n_scaled = (float)n * params.scalar.scales[output_channel_index];
  const float n_clamped =
      n_scaled < fmin ? fmin : n_scaled > fmax ? fmax : n_scaled;
  const int32_t n_biased = (int32_t)fp32_to_bits(n_clamped + fmagic) - imagic;

  return (uint8_t)n_biased;
}

static inline uint8_t pytorch_qnnp_avgpool_quantize(
    int32_t n,
    union pytorch_qnnp_avgpool_quantization_params params) {

  const float scaled_n = ((float)n)*params.scalar.scale;
  int32_t n_rounded = (int32_t)lrintf(scaled_n) + params.scalar.output_zero_point;

  const int32_t lmin =
      (int32_t)(uint32_t)params.scalar.output_min;
  const int32_t lmax =
      (int32_t)(uint32_t)params.scalar.output_max;

  n_rounded = (
      n_rounded < lmin ? lmin : n_rounded > lmax ? lmax : n_rounded);

  return (uint8_t)n_rounded;
}

static inline uint8_t pytorch_qnnp_add_quantize(
    uint8_t a,
    uint8_t b,
    union pytorch_qnnp_add_quantization_params params) {
  /* Multiply by factors and accumulate products */
  int32_t acc = params.scalar.zero_point_product +
      (int32_t)((uint32_t)a * params.scalar.a_multiplier) +
      (int32_t)((uint32_t)b * params.scalar.b_multiplier);

  /* Shift right and round */
  const int32_t rem = (acc & params.scalar.remainder_mask) - (int32_t)(acc < 0);
  acc = asr_s32(acc, params.scalar.shift) +
      (int32_t)(rem > params.scalar.remainder_threshold);

  /* Clamp and add output zero point */
  int32_t y = acc + params.scalar.y_zero_point;
  if (y >= params.scalar.y_max) {
    y = params.scalar.y_max;
  }
  if (y <= params.scalar.y_min) {
    y = params.scalar.y_min;
  }
  return (uint8_t)y;
}
