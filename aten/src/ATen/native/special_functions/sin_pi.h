#pragma once

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
sin_pi(T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::sin_pi<T2>(x);
} // detail::promote_t<T1> sin_pi(T1 x)

template<typename T1>
inline constexpr
std::complex<T1>
sin_pi(std::complex<T1> z) {
  return detail::sin_pi(z);
} // std::complex<T1> sin_pi(std::complex<T1> z)
} // namespace special_functions
} // namespace native
} // namespace at
