#pragma once

#include <ATen/native/special_functions/detail/kummer_confluent_hypergeometric_1_f_1.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2, typename T3>
inline constexpr
detail::promote_t<T1, T2, T3>
kummer_confluent_hypergeometric_1_f_1(T1 a, T2 c, T3 x) {
  using T4 = detail::promote_t<T1, T2, T3>;

  return detail::kummer_confluent_hypergeometric_1_f_1<T4>(a, c, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
