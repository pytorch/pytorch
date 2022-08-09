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
is_half_odd_integer(T1 x, T1 m = T1(1))
noexcept {
  if (std::isnan(x) || std::isnan(m)) {
    return {false, 0};
  } else {
    const auto nearby = static_cast<int>(std::nearbyint(T1(2) * x));

    return {
        ((nearby & 1) == 1) && is_equal(T1(2) * x, T1(nearby), m),
        (nearby - 1) / 2,
    };
  }
}

template<typename T1>
inline constexpr
is_integer_t
is_half_odd_integer(const std::complex<T1> &z, T1 m = T1(1))
noexcept {
  if (is_real(z, m)) {
    return is_half_odd_integer(std::real(z), m);
  } else {
    return {false, 0};
  }
}
}
}
}
}
