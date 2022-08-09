#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/bernoulli_number.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
bernoulli_number(unsigned int n) {
  using T2 = detail::promote_t<T1>;

  return detail::bernoulli_number<T2>(n);
}
} // namespace special_functions
} // namespace native
} // namespace at
