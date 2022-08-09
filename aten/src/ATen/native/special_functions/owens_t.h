#pragma once

#include <ATen/native/special_functions/detail/owens_t.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
owens_t(T1 h, T1 a) {
  using T2 = detail::promote_t<T1>;

  return detail::owens_t<T2>(h, a);
}
} // namespace special_functions
} // namespace native
} // namespace at
