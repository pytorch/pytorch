#pragma once

#include <ATen/native/special_functions/detail/is_zero.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
bool
is_real(const T1)
noexcept {
  return true;
}

template<typename T1>
inline constexpr
bool
is_real(const std::complex<T1> &z, const T1 m = T1(1))
noexcept {
  return is_zero(std::imag(z), m);
}
}
}
}
}
