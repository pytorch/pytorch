#pragma once

#include <complex>

namespace std {
template<typename Tp>
inline bool
isnan(const std::complex<Tp> &z) {
  return std::isnan(std::real(z)) || std::isnan(std::imag(z));
}

template<typename Tp>
inline bool
isinf(const std::complex<Tp> &z) {
  return isinf(std::real(z)) || isinf(std::imag(z));
}
}
