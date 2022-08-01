#pragma once

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
sinhc_pi(T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::sinhc_pi<T2>(x);
} // detail::promote_t<T1> sinhc_pi(T1 x)
} // namespace special_functions
} // namespace native
} // namespace at
