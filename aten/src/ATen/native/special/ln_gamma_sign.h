#pragma once

#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
T1
ln_gamma_sign(T1 a) {
  if (a >= T1(0)) {
    return T1(1);
  } else if (a == std::nearbyint(a)) {
    return T1(0);
  } else {
    if (int(-a) % 2 == 0) {
      return -T1(1);
    } else {
      return +T1(1);
    }
  }
}

template<typename T1>
C10_HOST_DEVICE
inline constexpr
c10::complex<T1>
ln_gamma_sign(c10::complex<T1> z) {
  return c10::complex<T1>(1);
}
} // namespace special
} // namespace native
} // namespace at
