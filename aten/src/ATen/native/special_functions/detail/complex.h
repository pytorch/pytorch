#pragma once

#include <c10/util/complex.h>

namespace std {
template<typename T1>
inline bool
isnan(const c10::complex<T1> &z) {
  return std::isnan(std::real(z)) || std::isnan(std::imag(z));
}

template<typename T1>
inline bool
isinf(const c10::complex<T1> &z) {
  return isinf(std::real(z)) || isinf(std::imag(z));
}
}
