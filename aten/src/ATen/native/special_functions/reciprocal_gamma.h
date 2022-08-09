#pragma once

#include <ATen/native/special_functions/detail/reciprocal_gamma.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
reciprocal_gamma(T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::reciprocal_gamma<T2>(x);
}
} // namespace special_functions
} // namespace native
} // namespace at
