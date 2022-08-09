#pragma once

#include <ATen/native/special_functions/detail/is_equal.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
struct is_integer_t {
  bool fp_is_integral = false;

  int value = 0;

  constexpr explicit operator bool() const noexcept {
    return this->fp_is_integral;
  }

  constexpr int operator()() const noexcept {
    return this->value;
  }
};

template<typename T1>
inline constexpr
is_integer_t
is_integer(T1 a, T1 m = T1(1))
noexcept {
  if (std::isnan(a) || std::isnan(m)) {
    return is_integer_t{false, 0};
  } else {
    const auto nearby_integer_a = static_cast<int>(std::nearbyint(a));

    return is_integer_t{
        is_equal(a, T1(nearby_integer_a), m),
        nearby_integer_a,
    };
  }
}

template<typename T1>
inline constexpr
is_integer_t
is_integer(const std::complex<T1> &a, T1 m = T1(1))
noexcept {
  if (is_real(a, m)) {
    return is_integer(std::real(a), m);
  } else {
    return is_integer_t{false, 0};
  }
}
}
}
}
}
