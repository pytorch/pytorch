#pragma once

#include <ATen/native/special_functions/detail/airy.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
airy_bi(T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::airy<T2>(x).bi;
}
} // namespace special_functions
} // namespace native
} // namespace at
