#pragma once

// Complex number math operations that act as no-ops for other dtypes.
#include <torch/headeronly/util/MathConstants.h>
#include <torch/headeronly/util/NumericUtils.h>
#include <torch/headeronly/util/complex.h>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly, native)
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

using torch::headeronly::_isnan;
using torch::headeronly::complex;
using torch::headeronly::Half;
using torch::headeronly::is_complex;
using torch::headeronly::pi;

template <typename SCALAR_TYPE, typename VALUE_TYPE = SCALAR_TYPE>
inline VALUE_TYPE zabs(SCALAR_TYPE z) {
  return z;
}

template <>
inline complex<float> zabs<complex<float>>(complex<float> z) {
  return complex<float>(std::abs(z));
}

template <>
inline float zabs<complex<float>, float>(complex<float> z) {
  return std::abs(z);
}

template <>
inline complex<double> zabs<complex<double>>(complex<double> z) {
  return complex<double>(std::abs(z));
}

template <>
inline double zabs<complex<double>, double>(complex<double> z) {
  return std::abs(z);
}

// This overload corresponds to non-complex dtypes.
// The function is consistent with its NumPy equivalent
// for non-complex dtypes where `pi` is returned for
// negative real numbers and `0` is returned for 0 or positive
// real numbers.
// Note: `nan` is propagated.
template <typename SCALAR_TYPE, typename VALUE_TYPE = SCALAR_TYPE>
inline VALUE_TYPE angle_impl(SCALAR_TYPE z) {
  if (_isnan(z)) {
    return z;
  }
  return z < 0 ? pi<double> : 0;
}

template <>
inline complex<float> angle_impl<complex<float>>(complex<float> z) {
  return complex<float>(std::arg(z), 0.0);
}

template <>
inline float angle_impl<complex<float>, float>(complex<float> z) {
  return std::arg(z);
}

template <>
inline complex<double> angle_impl<complex<double>>(complex<double> z) {
  return complex<double>(std::arg(z), 0.0);
}

template <>
inline double angle_impl<complex<double>, double>(complex<double> z) {
  return std::arg(z);
}

template <typename SCALAR_TYPE, typename VALUE_TYPE = SCALAR_TYPE>
constexpr VALUE_TYPE real_impl(SCALAR_TYPE z) {
  return z; // No-Op
}

template <>
constexpr complex<float> real_impl<complex<float>>(complex<float> z) {
  return complex<float>(z.real(), 0.0);
}

template <>
constexpr float real_impl<complex<float>, float>(complex<float> z) {
  return z.real();
}

template <>
constexpr complex<double> real_impl<complex<double>>(complex<double> z) {
  return complex<double>(z.real(), 0.0);
}

template <>
constexpr double real_impl<complex<double>, double>(complex<double> z) {
  return z.real();
}

template <typename SCALAR_TYPE, typename VALUE_TYPE = SCALAR_TYPE>
constexpr VALUE_TYPE imag_impl(SCALAR_TYPE /*z*/) {
  return 0;
}

template <>
constexpr complex<float> imag_impl<complex<float>>(complex<float> z) {
  return complex<float>(z.imag(), 0.0);
}

template <>
constexpr float imag_impl<complex<float>, float>(complex<float> z) {
  return z.imag();
}

template <>
constexpr complex<double> imag_impl<complex<double>>(complex<double> z) {
  return complex<double>(z.imag(), 0.0);
}

template <>
constexpr double imag_impl<complex<double>, double>(complex<double> z) {
  return z.imag();
}

template <typename TYPE>
inline TYPE conj_impl(TYPE z) {
  return z; // No-Op
}

template <>
inline complex<Half> conj_impl<complex<Half>>(complex<Half> z) {
  return complex<Half>{z.real(), -z.imag()};
}

template <>
inline complex<float> conj_impl<complex<float>>(complex<float> z) {
  return complex<float>(z.real(), -z.imag());
}

template <>
inline complex<double> conj_impl<complex<double>>(complex<double> z) {
  return complex<double>(z.real(), -z.imag());
}

template <typename TYPE>
inline TYPE ceil_impl(TYPE z) {
  return std::ceil(z);
}

template <>
inline complex<float> ceil_impl(complex<float> z) {
  return complex<float>(std::ceil(z.real()), std::ceil(z.imag()));
}

template <>
inline complex<double> ceil_impl(complex<double> z) {
  return complex<double>(std::ceil(z.real()), std::ceil(z.imag()));
}

template <typename T>
inline complex<T> sgn_impl(complex<T> z) {
  if (z == complex<T>(0, 0)) {
    return complex<T>(0, 0);
  } else {
    return z / zabs(z);
  }
}

template <typename TYPE>
inline TYPE floor_impl(TYPE z) {
  return std::floor(z);
}

template <>
inline complex<float> floor_impl(complex<float> z) {
  return complex<float>(std::floor(z.real()), std::floor(z.imag()));
}

template <>
inline complex<double> floor_impl(complex<double> z) {
  return complex<double>(std::floor(z.real()), std::floor(z.imag()));
}

template <typename TYPE>
inline TYPE round_impl(TYPE z) {
  return std::nearbyint(z);
}

template <>
inline complex<float> round_impl(complex<float> z) {
  return complex<float>(std::nearbyint(z.real()), std::nearbyint(z.imag()));
}

template <>
inline complex<double> round_impl(complex<double> z) {
  return complex<double>(std::nearbyint(z.real()), std::nearbyint(z.imag()));
}

template <typename TYPE>
inline TYPE trunc_impl(TYPE z) {
  return std::trunc(z);
}

template <>
inline complex<float> trunc_impl(complex<float> z) {
  return complex<float>(std::trunc(z.real()), std::trunc(z.imag()));
}

template <>
inline complex<double> trunc_impl(complex<double> z) {
  return complex<double>(std::trunc(z.real()), std::trunc(z.imag()));
}

template <typename TYPE, std::enable_if_t<!is_complex<TYPE>::value, int> = 0>
inline TYPE max_impl(TYPE a, TYPE b) {
  if (_isnan<TYPE>(a) || _isnan<TYPE>(b)) {
    return std::numeric_limits<TYPE>::quiet_NaN();
  } else {
    return std::max(a, b);
  }
}

template <typename TYPE, std::enable_if_t<is_complex<TYPE>::value, int> = 0>
inline TYPE max_impl(TYPE a, TYPE b) {
  if (_isnan<TYPE>(a)) {
    return a;
  } else if (_isnan<TYPE>(b)) {
    return b;
  } else {
    return std::abs(a) > std::abs(b) ? a : b;
  }
}

template <typename TYPE, std::enable_if_t<!is_complex<TYPE>::value, int> = 0>
inline TYPE min_impl(TYPE a, TYPE b) {
  if (_isnan<TYPE>(a) || _isnan<TYPE>(b)) {
    return std::numeric_limits<TYPE>::quiet_NaN();
  } else {
    return std::min(a, b);
  }
}

template <typename TYPE, std::enable_if_t<is_complex<TYPE>::value, int> = 0>
inline TYPE min_impl(TYPE a, TYPE b) {
  if (_isnan<TYPE>(a)) {
    return a;
  } else if (_isnan<TYPE>(b)) {
    return b;
  } else {
    return std::abs(a) < std::abs(b) ? a : b;
  }
}

} // namespace CPU_CAPABILITY
HIDDEN_NAMESPACE_END(torch, headeronly, native)

namespace at::native {
inline namespace CPU_CAPABILITY {
using torch::headeronly::native::angle_impl;
using torch::headeronly::native::ceil_impl;
using torch::headeronly::native::conj_impl;
using torch::headeronly::native::floor_impl;
using torch::headeronly::native::imag_impl;
using torch::headeronly::native::max_impl;
using torch::headeronly::native::min_impl;
using torch::headeronly::native::real_impl;
using torch::headeronly::native::round_impl;
using torch::headeronly::native::sgn_impl;
using torch::headeronly::native::trunc_impl;
using torch::headeronly::native::zabs;
} // namespace CPU_CAPABILITY
} // namespace at::native
