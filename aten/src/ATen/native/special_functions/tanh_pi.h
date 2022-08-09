#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
tanh_pi(T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::tanh_pi<T2>(x);
}

template<typename T1>
std::complex<T1>
tanh_pi(std::complex<T1> z) {
  return detail::tanh_pi(z);
}
} // namespace special_functions
} // namespace native
} // namespace at
