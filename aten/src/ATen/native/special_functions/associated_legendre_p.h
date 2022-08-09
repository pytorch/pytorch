#pragma once

#include <ATen/native/special_functions/detail/associated_legendre_p.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
associated_legendre_p(unsigned int l, unsigned int m, T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::associated_legendre_p<T2>(l, m, x).P_lm;
}
} // namespace special_functions
} // namespace native
} // namespace at
