#pragma once

#include <ATen/native/special_functions/detail/stirling_number_2.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
T1
stirling_number_2(unsigned int n, unsigned int m) {
  return detail::stirling_number_2<T1>(n, m);
}
} // namespace special_functions
} // namespace native
} // namespace at
