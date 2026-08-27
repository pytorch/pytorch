#pragma once

#if defined(CPU_CAPABILITY_SVE128) || defined(CPU_CAPABILITY_SVE256)

#include <arm_sve.h>

namespace at::vec::inline CPU_CAPABILITY {

// Implementation copied from Arm Optimized Routines:
// https://github.com/ARM-software/optimized-routines/blob/master/math/aarch64/sve/expf.c
static inline svfloat32_t exp_u20_fast_path(svfloat32_t values) {
  // For |x| >= 0x1.5d5e2ap+6f (~87.34), exp(x) may overflow or become
  // subnormal, which FEXPA does not handle accurately; callers should handle
  // those inputs separately.
  const svfloat32_t ln2_hi = svdup_n_f32(0x1.62e4p-1f);
  const svfloat32_t ln2_lo = svdup_n_f32(0x1.7f7d1cp-20f);
  const svfloat32_t c1 = svdup_n_f32(0.5f);
  const svfloat32_t inv_ln2 = svdup_n_f32(0x1.715476p+0f);

  const float shift = 0x1.803f8p17f;

  /* n = round(x/(ln2/N)).  */
  svfloat32_t z = svmad_x(svptrue_b32(), inv_ln2, values, shift);
  svfloat32_t n = svsub_x(svptrue_b32(), z, shift);

  /* r = x - n*ln2/N.  */
  svfloat32_t r = values;
  r = svmls_x(svptrue_b32(), r, n, ln2_hi);
  r = svmls_x(svptrue_b32(), r, n, ln2_lo);

  /* scale = 2^(n/N).  */
  svfloat32_t scale = svexpa(svreinterpret_u32(z));

  /* poly(r) = exp(r) - 1 ~= r + 0.5 r^2.  */
  svfloat32_t r2 = svmul_x(svptrue_b32(), r, r);
  svfloat32_t poly = svmla_x(svptrue_b32(), r, r2, c1);
  return svmla_x(svptrue_b32(), scale, scale, poly);
}

// Implementation from Arm Optimized Routines:
// https://github.com/ARM-software/optimized-routines/blob/v26.01/math/aarch64/experimental/sve/sv_expf_inline.h
static inline svfloat32_t fexp_u20(svfloat32_t values) {
  // fast exponential intended for cases where outputs will be downcasted to
  // FP16 / BF16 (e.g. attention softmax).
  // Accurate within 1 ULP for FP16
  // Accurate within 1 ULP for BF16 for inputs in [-87.346, max_float] &
  // clamps
  //  inputs < -87.346 to zero.
  // Implementation is similar to exp_u20, but:
  // - approximates exp(r) - 1 as r instead of r + 0.5 r^2
  // - does not split natural log (ln) into high / low parts
  // - avoids special case code by clamping exp(x) to 0 for x < -87.346 and
  // inf for x > 88.717

  constexpr float upper_bound = 0x1.62dea4p+6f;
  constexpr float lower_bound = -0x1.5d619ap+6f;

  const svfloat32_t ln2 = svdup_n_f32(0x1.62e43p-1f);
  const svfloat32_t inv_ln2 = svdup_n_f32(0x1.715476p+0f);

  constexpr float shift = 0x1.803f8p17f;

  // exp(x) = 2^n (1 + poly(r)), with 1 + poly(r) in [1/sqrt(2),sqrt(2)]
  // x = ln2*n + r, with r in [-ln2/2, ln2/2] and poly(r) ~= r.

  // n = round(x/(ln2/N))
  svfloat32_t z = svmad_x(svptrue_b32(), inv_ln2, values, shift);
  svfloat32_t n = svsub_x(svptrue_b32(), z, shift);

  // n = round(x/(ln2/N))
  svfloat32_t r = svmls_x(svptrue_b32(), values, n, ln2);

  // scale = 2^(n/N)
  svfloat32_t scale = svexpa(svreinterpret_u32(z));

  // poly(r) = exp(r) - 1 ~= r
  svfloat32_t y = svmla_x(svptrue_b32(), scale, scale, r);

  // clamp to 0, inf
  y = svsel_f32(
      svcmplt_f32(svptrue_b32(), values, svdup_n_f32(lower_bound)),
      svdup_n_f32(0.0f),
      y);
  y = svsel_f32(
      svcmpgt_f32(svptrue_b32(), values, svdup_n_f32(upper_bound)),
      svdup_n_f32(INFINITY),
      y);
  return y;
}

} // namespace CPU_CAPABILITY

#endif // defined(CPU_CAPABILITY_SVE128) || defined(CPU_CAPABILITY_SVE256)
