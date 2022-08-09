#pragma once

#include <ATen/native/special_functions/detail/confluent_hypergeometric_0_f_1.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr typename detail::promote_t<T1, T2>
confluent_hypergeometric_0_f_1(T1 c, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::confluent_hypergeometric_0_f_1<T3>(c, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
