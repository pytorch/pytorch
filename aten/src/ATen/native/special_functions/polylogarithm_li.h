#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
polylogarithm_li(T1 s, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::polylog<T3>(s, x);
}

template<typename T1, typename T2>
inline constexpr
std::complex<detail::promote_t<T1, T2>>
polylogarithm_li(T1 s, std::complex<T1> z) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::polylog<T3>(s, z);
}
} // namespace special_functions
} // namespace native
} // namespace at
