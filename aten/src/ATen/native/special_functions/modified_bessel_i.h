#pragma once

#include <ATen/native/special_functions/detail/modified_bessel_i.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
modified_bessel_i(T1 n, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::modified_bessel_i<T3>(n, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
