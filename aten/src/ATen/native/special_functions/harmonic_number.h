#pragma once

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr C10_HOST_DEVICE
detail::promote_t<T1>
harmonic_number(unsigned int n) {
  using T2 = detail::promote_t<T1>;

  return detail::harmonic_number<T2>(n);
} // detail::promote_t<T1> harmonic_number(unsigned int n)
} // namespace special_functions
} // namespace native
} // namespace at
