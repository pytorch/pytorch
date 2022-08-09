#pragma once

#include <ATen/native/special_functions/detail/is_real.h>
#include <ATen/native/special_functions/detail/max_abs.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
bool
is_equal(T1 a, T1 b, T1 m = T1(1))
noexcept {
  if (std::isnan(a) || std::isnan(b) || std::isnan(m)) {
    return false;
  } else {
    if ((a != T1(0)) || (b != T1(0))) {
      const auto epsilon = std::numeric_limits<T1>::epsilon();

      return (std::abs(a - b) < max_abs(a, b) * (m * epsilon));
    } else {
      return true;
    }
  }
}

template<typename T1>
inline constexpr
bool
is_equal(const std::complex<T1> &a, const std::complex<T1> &b, T1 m = T1(1))
noexcept {
  if (!is_zero(std::abs(a), m) || !is_zero(std::abs(b), m)) {
    const auto epsilon = std::numeric_limits<T1>::epsilon();

    return (std::abs(a - b) < max_abs(a, b) * (m * epsilon));
  } else {
    return true;
  }
}

template<typename T1>
inline constexpr
bool
is_equal(const std::complex<T1> &a, T1 b, T1 m = T1(1))
noexcept {
  return is_real(a, m) && is_equal(std::real(a), b, m);
}

template<typename T1>
inline constexpr
bool
is_equal(const T1 a, std::complex<T1> &b, T1 m = T1(1))
noexcept {
  return is_real(b, m) && is_equal(a, std::real(b), m);
}
}
}
}
}
