#pragma once

// Complex number math operations that act as no-ops for other dtypes.
#include <complex>
#include <c10/util/math_compat.h>
#include<ATen/NumericUtils.h>

namespace at { namespace native {
namespace {

template <typename TYPE>
struct ztype {
  using value_t = TYPE;
};

template <>
struct ztype<std::complex<double>> {
  using value_t = double;
};

template <>
struct ztype<std::complex<float>> {
  using value_t = float;
};

template<>
inline std::complex<float> zabs <std::complex<float>> (std::complex<float> z) {
  return std::complex<float>(std::abs(z));
}

template<>
inline float zabs <std::complex<float>, float> (std::complex<float> z) {
  return std::abs(z);
}

template<>
inline std::complex<double> zabs <std::complex<double>> (std::complex<double> z) {
  return std::complex<double>(std::abs(z));
}

template<>
inline double zabs <std::complex<double>, double> (std::complex<double> z) {
  return std::abs(z);
}

template<>
inline std::complex<float> angle_impl <std::complex<float>> (std::complex<float> z) {
  return std::complex<float>(std::arg(z), 0.0);
}

template<>
inline float angle_impl <std::complex<float>, float> (std::complex<float> z) {
  return std::arg(z);
}

template<>
inline std::complex<double> angle_impl <std::complex<double>> (std::complex<double> z) {
  return std::complex<double>(std::arg(z), 0.0);
}

template<>
inline double angle_impl <std::complex<double>, double> (std::complex<double> z) {
  return std::arg(z);
}

template<>
constexpr std::complex<float> real_impl <std::complex<float>> (std::complex<float> z) {
  return std::complex<float>(z.real(), 0.0);
}

template<>
constexpr float real_impl <std::complex<float>, float> (std::complex<float> z) {
  return z.real();
}

template<>
constexpr std::complex<double> real_impl <std::complex<double>> (std::complex<double> z) {
  return std::complex<double>(z.real(), 0.0);
}

template<>
constexpr double real_impl <std::complex<double>, double> (std::complex<double> z) {
  return z.real();
}

template<>
constexpr std::complex<float> imag_impl <std::complex<float>> (std::complex<float> z) {
  return std::complex<float>(z.imag(), 0.0);
}

template<>
constexpr float imag_impl <std::complex<float>, float> (std::complex<float> z) {
  return z.imag();
}

template<>
constexpr std::complex<double> imag_impl <std::complex<double>> (std::complex<double> z) {
  return std::complex<double>(z.imag(), 0.0);
}

template<>
constexpr double imag_impl <std::complex<double>, double> (std::complex<double> z) {
  return z.imag();
}

template<>
inline std::complex<float> conj_impl <std::complex<float>> (std::complex<float> z) {
  return std::complex<float>(z.real(), -z.imag());
}

template<>
inline std::complex<double> conj_impl <std::complex<double>> (std::complex<double> z) {
  return std::complex<double>(z.real(), -z.imag());
}

template <>
inline std::complex<float> ceil_impl (std::complex<float> z) {
  return std::complex<float>(std::ceil(z.real()), std::ceil(z.imag()));
}

template <>
inline std::complex<double> ceil_impl (std::complex<double> z) {
  return std::complex<double>(std::ceil(z.real()), std::ceil(z.imag()));
}

template <>
inline std::complex<float> floor_impl (std::complex<float> z) {
  return std::complex<float>(std::floor(z.real()), std::floor(z.imag()));
}

template <>
inline std::complex<double> floor_impl (std::complex<double> z) {
  return std::complex<double>(std::floor(z.real()), std::floor(z.imag()));
}

template <>
inline std::complex<float> round_impl (std::complex<float> z) {
  return std::complex<float>(std::nearbyint(z.real()), std::nearbyint(z.imag()));
}

template <>
inline std::complex<double> round_impl (std::complex<double> z) {
  return std::complex<double>(std::nearbyint(z.real()), std::nearbyint(z.imag()));
}

template <>
inline std::complex<float> trunc_impl (std::complex<float> z) {
  return std::complex<float>(std::trunc(z.real()), std::trunc(z.imag()));
}

template <>
inline std::complex<double> trunc_impl (std::complex<double> z) {
  return std::complex<double>(std::trunc(z.real()), std::trunc(z.imag()));
}

} // end namespace
}} //end at::native
