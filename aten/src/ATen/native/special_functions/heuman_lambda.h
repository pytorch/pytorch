#pragma once

#include <ATen/native/special_functions/detail/heuman_lambda.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
heuman_lambda(T1 k, T2 phi) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::heuman_lambda<T3>(k, phi);
}
} // namespace special_functions
} // namespace native
} // namespace at
