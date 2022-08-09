#pragma once

#include <ATen/native/special_functions/detail/gauss_hypergeometric_2_f_1.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2, typename T3, typename T4>
inline constexpr
typename detail::promote_t<T1, T2, T3, T4>
gauss_hypergeometric_2_f_1(T1 a, T2 b, T3 c, T4 x) {
  using T5 = detail::promote_t<T1, T2, T3, T4>;

  return detail::gauss_hypergeometric_2_f_1<T5>(a, b, c, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
