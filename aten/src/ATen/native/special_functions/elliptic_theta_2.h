#pragma once

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr C10_HOST_DEVICE
detail::promote_t<T1, T2>
elliptic_theta_2(T1 n, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::elliptic_theta_2<T3>(n, x);
} // detail::promote_t<T1, T2> elliptic_theta_2(T1 n, T2 x)
} // namespace special_functions
} // namespace native
} // namespace at
