#pragma once

#include <ATen/cpu/vec/vec.h>
#include <c10/util/BFloat16.h> // For std::is_reduced_floating_point_v.

namespace at::native {
constexpr double kGeluBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
constexpr double kGeluKappa = 0.044715;

template <typename T>
using reduced_fp_to_float_t = std::conditional_t<std::is_reduced_floating_point_v<T>, float, T>;

template <typename T, std::enable_if_t<std::is_reduced_floating_point_v<T>, bool> = true>
float reduced_fp_to_float(T x) {
  return float(x);
}

template <typename T, std::enable_if_t<!std::is_reduced_floating_point_v<T>, bool> = true>
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

template <typename T, std::enable_if_t<!std::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu_approximated_with_tanh(vec::Vectorized<T> x) {
  const vec::Vectorized<T> kPointFiveVec(T(0.5));
  const vec::Vectorized<T> kOneVec(T(1));
  const vec::Vectorized<T> kGeluBetaVec((T(kGeluBeta)));
  const vec::Vectorized<T> kGeluKappaVec((T(kGeluKappa)));
  auto x_cube = x * x * x;
  vec::Vectorized<T> inner_vec = kGeluBetaVec * (x + kGeluKappaVec * x_cube);
  return kPointFiveVec * x * (kOneVec + inner_vec.tanh());
}

template <typename T, std::enable_if_t<std::is_reduced_floating_point_v<T>, bool> = true>
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
  return reduced_fp_to_float(x) * opmath_t(0.5) * (opmath_t(1) + std::erf(reduced_fp_to_float(x) * kAlpha));
}

template<typename T, std::enable_if_t<!std::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu(vec::Vectorized<T> x) {
  const vec::Vectorized<T> kAlphaVec(T(M_SQRT1_2));
  const vec::Vectorized<T> kOneVec(T(1));
  const vec::Vectorized<T> kPointFiveVec(T(0.5));
  return x * kPointFiveVec * (kOneVec + (x * kAlphaVec).erf());
}

template<typename T, std::enable_if_t<std::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu(vec::Vectorized<T> x) {
  auto [x0, x1] = at::vec::convert_to_float<T>(x);
  return at::vec::convert_from_float<T>(vectorized_gelu(x0), vectorized_gelu(x1));
}


} // namespace
