#pragma once

// Complex number math operations that act as no-ops for other dtypes.
#include <complex.h>

namespace at { namespace native {
namespace {

template <typename TYPE>
struct ztype {
  using type = TYPE;
};

template <>
struct ztype<std::complex<double>> {
  using type = double;
};

template <>
struct ztype<std::complex<float>> {
  using type = float;
};

template <typename TYPE>
inline TYPE real_impl (TYPE z) {
  return z; //No-Op
}

template<>
inline std::complex<float> real_impl <std::complex<float>> (std::complex<float> z) {
  return std::complex<float>(std::real(z), 0.0);
}

template<>
inline std::complex<double> real_impl <std::complex<double>> (std::complex<double> z) {
  return std::complex<double>(std::real(z), 0.0);
}

template <typename TYPE>
inline TYPE imag_impl (TYPE z) {
  return 0;
}

template<>
inline std::complex<float> imag_impl <std::complex<float>> (std::complex<float> z) {
  return std::complex<float>(0.0, std::imag(z));
}

template<>
inline std::complex<double> imag_impl <std::complex<double>> (std::complex<double> z) {
  return std::complex<double>(0.0, std::imag(z));
}

} // end namespace
}} //end at::native
