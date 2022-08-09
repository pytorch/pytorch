#pragma once

#include <ATen/native/special_functions/detail/cosine_integral_ci.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
cosine_integral_ci(T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::cosine_integral_ci<T2>(z);
}
} // namespace special_functions
} // namespace native
} // namespace at
