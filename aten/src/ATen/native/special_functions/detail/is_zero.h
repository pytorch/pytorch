#pragma once

#include <cmath>
#include <complex>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
bool
is_zero(T1 x, T1 m = T1(1))
noexcept {
  if (std::isnan(x) || std::isnan(m)) {
    return false;
  } else {
    return x == T1(0) || std::abs(x) < m * std::numeric_limits<T1>::epsilon();
  }
}

template<typename T1>
inline constexpr
bool
is_zero(const std::complex<T1> &z, T1 m = T1(1))
noexcept {
  return is_zero(std::abs(z), m);
}
}
}
}
}
