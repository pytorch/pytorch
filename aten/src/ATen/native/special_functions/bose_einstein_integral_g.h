#pragma once

#include <ATen/native/special_functions/detail/bose_einstein_integral_g.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
bose_einstein_integral_g(T1 s, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::bose_einstein_integral_g<T3>(s, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
