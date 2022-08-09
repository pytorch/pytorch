#pragma once

#include <ATen/native/special_functions/detail/is_integer.h>
#include <ATen/native/special_functions/detail/is_real.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
is_integer_t
is_even_integer(T1 x, T1 m = T1(1))
noexcept {
  if (std::isnan(x) || std::isnan(m)) {
    return {false, 0};
  } else {
    const auto is_integer_x = is_integer(x, m);

    const auto integer_x = is_integer_x();

    return {is_integer_x && !(integer_x & 1), integer_x};
  }
}

template<typename T1>
inline constexpr
is_integer_t
is_even_integer(const std::complex<T1> &z, T1 m = T1(1))
noexcept {
  if (is_real(z, m)) {
    const auto integer = is_integer(std::real(z), m);

    return {integer && !(integer() & 1), integer()};
  } else {
    return {false, 0};
  }
}
}
}
}
}
