#pragma once

#include <complex>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <torch/headeronly/util/complex.h>

// std functions
//
// The implementation of these functions also follow the design of C++20

namespace std {

template <typename T>
constexpr T real(const c10::complex<T>& z) {
  return z.real();
}

template <typename T>
constexpr T imag(const c10::complex<T>& z) {
  return z.imag();
}

template <typename T>
C10_HOST_DEVICE T abs(const c10::complex<T>& z) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return thrust::abs(static_cast<thrust::complex<T>>(z));
#else
  return std::abs(static_cast<std::complex<T>>(z));
#endif
}

#if defined(USE_ROCM)
#define ROCm_Bug(x)
#else
#define ROCm_Bug(x) x
#endif

template <typename T>
C10_HOST_DEVICE T arg(const c10::complex<T>& z) {
  return ROCm_Bug(std)::atan2(std::imag(z), std::real(z));
}

#undef ROCm_Bug

template <typename T>
constexpr T norm(const c10::complex<T>& z) {
  return z.real() * z.real() + z.imag() * z.imag();
}

// For std::conj, there are other versions of it:
//   constexpr std::complex<float> conj( float z );
//   template< class DoubleOrInteger >
//   constexpr std::complex<double> conj( DoubleOrInteger z );
//   constexpr std::complex<long double> conj( long double z );
// These are not implemented
// TODO(@zasdfgbnm): implement them as c10::conj
template <typename T>
constexpr c10::complex<T> conj(const c10::complex<T>& z) {
  return c10::complex<T>(z.real(), -z.imag());
}

// Thrust does not have complex --> complex version of thrust::proj,
// so this function is not implemented at c10 right now.
// TODO(@zasdfgbnm): implement it by ourselves

// There is no c10 version of std::polar, because std::polar always
// returns std::complex. Use c10::polar instead;

} // namespace std

#define C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
// math functions are included in a separate file
#include <c10/util/complex_math.h> // IWYU pragma: keep
// utilities for complex types
#include <c10/util/complex_utils.h> // IWYU pragma: keep
#undef C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
