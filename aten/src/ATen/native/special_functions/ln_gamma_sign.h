#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
T1
ln_gamma_sign(T1 a) {
  if (a >= T1{0}) {
    return T1{1};
  } else if (a == std::nearbyint(a)) {
    return T1{0};
  } else {
    if (int(-a) % 2 == 0) {
      return -T1{1};
    } else {
      return +T1{1};
    }
  }
}

template<typename T1>
inline constexpr
std::complex<T1>
ln_gamma_sign(std::complex<T1> z) {
  return std::complex<T1>{1};
}
} // namespace special_functions
} // namespace native
} // namespace at
