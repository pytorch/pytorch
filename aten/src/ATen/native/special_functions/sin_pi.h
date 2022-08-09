#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include "detail/sin_pi.h"

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
sin_pi(T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::sin_pi<T2>(x);
}

template<typename T1>
inline constexpr
std::complex<T1>
sin_pi(std::complex<T1> z) {
  return detail::sin_pi(z);
}
} // namespace special_functions
} // namespace native
} // namespace at
