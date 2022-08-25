#pragma once

#include <ATen/native/special/detail/airy.h>
#include <ATen/native/special/detail/promote_t.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
exp_airy_ai(T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::airy<T2>(z, true).ai;
}
} // namespace special
} // namespace native
} // namespace at
