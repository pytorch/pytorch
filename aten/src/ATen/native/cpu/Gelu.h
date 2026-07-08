#pragma once

// On Windows, math.h needs to be included with _USE_MATH_DEFINES defined to
// access constants such as M_SQRT2 and M_2_SQRTPI.
#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#endif // _WIN32

#include <ATen/cpu/vec/vec.h>
#include <c10/util/BFloat16.h> // For c10::is_reduced_floating_point_v.

namespace at::native {
inline namespace CPU_CAPABILITY {
constexpr double kGeluBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
constexpr double kGeluKappa = 0.044715;

template <typename T>
using reduced_fp_to_float_t = std::conditional_t<c10::is_reduced_floating_point_v<T>, float, T>;

template <typename T, std::enable_if_t<c10::is_reduced_floating_point_v<T>, bool> = true>
float reduced_fp_to_float(T x) {
  return float(x);
}

template <typename T, std::enable_if_t<!c10::is_reduced_floating_point_v<T>, bool> = true>
T reduced_fp_to_float(T x) {
  return x;
}

template <typename T>
T scalar_gelu_approximated_with_tanh(T x) {
  using opmath_t = reduced_fp_to_float_t<T>;
  auto x_float = reduced_fp_to_float(x);
  auto x_cube = x_float * x_float * x_float;
  auto inner = opmath_t(kGeluBeta) * (x_float + opmath_t(kGeluKappa) * x_cube);
  return opmath_t(0.5) * x_float * (opmath_t(1) + std::tanh(inner));
}

template <typename T, std::enable_if_t<!c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu_approximated_with_tanh(vec::Vectorized<T> x) {
  const vec::Vectorized<T> kPointFiveVec(T(0.5));
  const vec::Vectorized<T> kOneVec(T(1));
  const vec::Vectorized<T> kGeluBetaVec((T(kGeluBeta)));
  const vec::Vectorized<T> kGeluKappaVec((T(kGeluKappa)));
  auto x_cube = x * x * x;
  vec::Vectorized<T> inner_vec = kGeluBetaVec * (x + kGeluKappaVec * x_cube);
  return kPointFiveVec * x * (kOneVec + inner_vec.tanh());
}

template <typename T, std::enable_if_t<c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu_approximated_with_tanh(vec::Vectorized<T> x) {
  auto [x0, x1] = at::vec::convert_to_float<T>(x);
  return at::vec::convert_from_float<T>(
      vectorized_gelu_approximated_with_tanh(x0),
      vectorized_gelu_approximated_with_tanh(x1));
}


template <typename T>
T scalar_gelu(T x) {
  using opmath_t = reduced_fp_to_float_t<T>;
  const auto kAlpha = opmath_t(M_SQRT1_2);
  // 1 + erf(x) = erfc(-x)
  return reduced_fp_to_float(x) * opmath_t(0.5) * std::erfc(-reduced_fp_to_float(x) * kAlpha);
}

// Standard normal CDF Phi(x) = 0.5 * erfc(-x / sqrt(2)) for float vectors.
// The erfc form does not cancel for x < 0, unlike 0.5 * (1 + erf(x/sqrt(2)))
// (gh-187806); Vectorized<float>::erfc() (SLEEF u15, double-precision
// internals) would be correct but is ~8x slower than erf(). Instead:
//   erfc(a) = (1 + p(q)) / (1 + 2a) * exp(-a * a),  q = (a - 4) / (a + 4)
// p: degree-10 Remez fit of (1 + 2a) * exp(a * a) * erfc(a) - 1, as in
// c10/metal/special_math.h but on all of a in [0, 10.5] (one extra degree)
// so the vector path stays branch-free. Max observed error ~5 ulp of erfc,
// below fp32 gelu's inherent argument-rounding error.
inline vec::Vectorized<float> vectorized_normal_cdf(vec::Vectorized<float> x) {
  using Vec = vec::Vectorized<float>;
  const auto a = x.abs() * Vec(float(M_SQRT1_2));
  // erfc(10.5) already underflows fp32; the clamp also handles +-infinity
  // (NaN passes through: vec::minimum propagates it)
  const auto t = vec::minimum(a, Vec(10.5f));
  const auto q = (t - Vec(4.0f)) / (t + Vec(4.0f));
  auto p = Vec(8.252649e-04f); // 0x1.b0ad2ep-11
  p = vec::fmadd(p, q, Vec(6.8451143e-03f)); // 0x1.c099f6p-8
  p = vec::fmadd(p, q, Vec(-1.606347e-02f)); // -0x1.072f14p-6
  p = vec::fmadd(p, q, Vec(3.6397815e-02f)); // 0x1.2a2bc0p-5
  p = vec::fmadd(p, q, Vec(-6.658963e-02f)); // -0x1.10c04ap-4
  p = vec::fmadd(p, q, Vec(9.3837336e-02f)); // 0x1.805b94p-4
  p = vec::fmadd(p, q, Vec(-1.0099377e-01f)); // -0x1.9daba4p-4
  p = vec::fmadd(p, q, Vec(6.809168e-02f)); // 0x1.16e74ep-4
  p = vec::fmadd(p, q, Vec(1.5377381e-02f)); // 0x1.f7e2d2p-7
  p = vec::fmadd(p, q, Vec(-1.3962102e-01f)); // -0x1.1df1a0p-3
  p = vec::fmadd(p, q, Vec(2.3299514e-01f)); // 0x1.dd2c8ep-3
  const auto s = t * t;
  auto e = s.neg().exp();
  // fold in the residual of the squaring: t * t = s + fma(t, t, -s) exactly
  e = vec::fmadd(e.neg(), vec::fmadd(t, t, s.neg()), e);
  const auto half_erfc = Vec(0.5f) * ((Vec(1.0f) + p) / vec::fmadd(Vec(2.0f), t, Vec(1.0f)) * e);
  // Phi(x) = 0.5 * erfc(a) for x < 0 and 1 - 0.5 * erfc(a) for x >= 0
  return Vec::blendv(Vec(1.0f) - half_erfc, half_erfc, x < Vec(0.0f));
}

inline vec::Vectorized<double> vectorized_normal_cdf(vec::Vectorized<double> x) {
  // the fp32 fit above is far from double accuracy; use full-precision erfc
  const vec::Vectorized<double> kMinusAlphaVec(-M_SQRT1_2);
  return vec::Vectorized<double>(0.5) * (x * kMinusAlphaVec).erfc();
}

template<typename T, std::enable_if_t<!c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu(vec::Vectorized<T> x) {
  return x * vectorized_normal_cdf(x);
}

template<typename T, std::enable_if_t<c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu(vec::Vectorized<T> x) {
  auto [x0, x1] = at::vec::convert_to_float<T>(x);
  return at::vec::convert_from_float<T>(vectorized_gelu(x0), vectorized_gelu(x1));
}

} // namespace CPU_CAPABILITY
} // namespace at::native
