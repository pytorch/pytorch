#pragma once

#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/detail/sin_cos_t.h>
#include <ATen/native/special/detail/sin_cos_pi.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
inline constexpr
c10::complex<T1>
polar_pi(T1 rho, T1 phi_pi) {
  const auto sin_cos = detail::sin_cos_pi(phi_pi);

  return {
    rho * sin_cos.cos_v,
    rho * sin_cos.sin_v,
  };
}

template<typename T1>
inline constexpr
c10::complex<T1>
polar_pi(T1 rho, const c10::complex<T1> &phi_pi) {
  detail::sin_cos_t<T1> sin_cos = detail::sin_cos_pi(phi_pi);

  return {rho * sin_cos.cos_v, rho * sin_cos.sin_v};
}
} // namespace special
} // namespace native
} // namespace at
