#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
bool
is_imag(const std::complex<T1> &w, const T1 m = T1(1))
noexcept {
  return is_zero(std::real(w), m);
}

template<typename T1>
inline constexpr
bool
is_imag(const T1)
noexcept {
  return false;
}
}
}
}
}
