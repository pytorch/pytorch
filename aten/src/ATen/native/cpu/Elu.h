#pragma once

// On Windows, math.h needs to be included with _USE_MATH_DEFINES defined to
// access constants such as M_SQRT2 and M_2_SQRTPI.
#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif // _WIN32

#include <ATen/cpu/vec/vec.h>
#include <c10/util/BFloat16.h> // For c10::is_reduced_floating_point_v.

namespace at::native {
inline namespace CPU_CAPABILITY {
/**
 * Return a function object that calculates ELU with the given
 * parameters on its input element.  ParamT is the type of the input
 * and output to the ELU, and MathT is the type (possibly
 * higher-precision, e.g. float if ParamT is reduced-precision float)
 * in which to do intermediate calculations.
 */
template <typename ParamT, typename MathT=ParamT>
auto get_scalar_elu_elementwise_func(MathT alpha, MathT scale, MathT input_scale) {
  const auto negcoef = alpha * scale;
  const auto poscoef = scale;
  const auto negiptcoef = input_scale;
  return [negcoef, negiptcoef, poscoef](ParamT a) -> ParamT {
    return MathT(a) < MathT(0)
      ? std::expm1(MathT(a) * negiptcoef) * negcoef
      : MathT(a) * poscoef;
  };
}

/**
 * Return a function object that calculates ELU with the given
 * parameters on its input element. The function object takes and
 * returns Vectorized<T>.
 */
template <typename T, std::enable_if_t<!c10::is_reduced_floating_point_v<T>, bool> = true>
auto get_vectorized_elu_elementwise_func(T alpha, T scale, T input_scale) {
  const vec::Vectorized<T> negcoef_vec(alpha * scale);
  const vec::Vectorized<T> poscoef_vec(scale);
  const vec::Vectorized<T> negiptcoef_vec(input_scale);
  const vec::Vectorized<T> zero_vec(static_cast<T>(0));
  return [negcoef_vec, poscoef_vec, negiptcoef_vec, zero_vec](vec::Vectorized<T> a) -> vec::Vectorized<T> {
    const auto cmp = a >= zero_vec;
    if (!cmp.zero_mask()) {
      return a * poscoef_vec;
    } else {
      return vec::Vectorized<T>::blendv((a * negiptcoef_vec).expm1() * negcoef_vec, a * poscoef_vec, cmp);
    }
  };
}

/**
 * Return a function object that calculates ELU with the given
 * parameters on its input element. The function object takes and
 * returns Vectorized<ParamT>, and Vectorized<MathT> is the type
 * (possibly higher-precision) in which to do intermediate
 * calculations.
 */
template <typename T, std::enable_if_t<c10::is_reduced_floating_point_v<T>, bool> = true>
auto get_vectorized_elu_elementwise_func(float alpha, float scale, float input_scale) {
  // Takes float->float.
  const auto float_func = get_vectorized_elu_elementwise_func<float>(alpha, scale, input_scale);
  return [float_func](vec::Vectorized<T> a) -> vec::Vectorized<T> {
    auto [a0, a1] = vec::convert_to_float<T>(a);
    auto res0 = float_func(a0);
    auto res1 = float_func(a1);
    return vec::convert_from_float<T>(res0, res1);
  };
}
} // namespace CPU_CAPABILITY
} // namespace at::native
