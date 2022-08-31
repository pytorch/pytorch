#pragma once

#include <ATen/native/special_functions/detail/promote.h>
#include <ATen/native/special_functions/detail/sinhc.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
sinhc(T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::sinhc<T2>(z);
}
} // namespace special_functions
} // namespace native
} // namespace at
