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
airy_bi(T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::airy<T2>(z).bi;
}
} // namespace special
} // namespace native
} // namespace at
