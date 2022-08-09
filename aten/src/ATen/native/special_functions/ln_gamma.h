#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/ln_gamma.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
ln_gamma(T1 a) {
  using T2 = detail::promote_t<T1>;

  return detail::ln_gamma<T2>(a);
}

template<typename T1>
inline constexpr
std::complex<detail::promote_t<T1>>
ln_gamma(std::complex<T1> a) {
  using T2 = std::complex<detail::promote_t<T1>>;

  return detail::ln_gamma<T2>(a);
}
} // namespace special_functions
} // namespace native
} // namespace at
