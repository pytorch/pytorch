#pragma once

#include <ATen/native/special_functions/detail/harmonic_number.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
harmonic_number(unsigned int n) {
  using T2 = detail::promote_t<T1>;

  return detail::harmonic_number<T2>(n);
}
} // namespace special_functions
} // namespace native
} // namespace at
