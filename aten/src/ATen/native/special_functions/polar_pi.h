#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/sin_cos_t.h>
#include <ATen/native/special_functions/detail/sin_cos_pi.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
std::complex<T1>
polar_pi(T1 rho, T1 phi_pi) {
  detail::sin_cos_t<T1> sin_cos = detail::sin_cos_pi(phi_pi);

  return std::complex<T1>{rho * sin_cos.cos_v, rho * sin_cos.sin_v};
}

template<typename T1>
inline constexpr
std::complex<T1>
polar_pi(T1 rho, const std::complex<T1> &phi_pi) {
  detail::sin_cos_t<T1> sin_cos = detail::sin_cos_pi(phi_pi);

  return std::complex<T1>{rho * sin_cos.cos_v, rho * sin_cos.sin_v};
}
} // namespace special_functions
} // namespace native
} // namespace at
