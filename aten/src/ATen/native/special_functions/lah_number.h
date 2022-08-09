#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr T1
lah_number(unsigned int n, unsigned int k) {
  return detail::lah_number<T1>(n, k);
}
} // namespace special_functions
} // namespace native
} // namespace at
