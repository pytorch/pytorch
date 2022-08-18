#pragma once

#include <ATen/native/special/detail/carlson_elliptic_r_d.h>
#include <ATen/native/special/detail/complete_carlson_elliptic_r_g.h>
#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/numeric_t.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
carlson_elliptic_r_g(T1 x, T1 y, T1 z) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (z == T1(0x00)) {
    return complete_carlson_elliptic_r_g(x, y);
  } else if (x == T1(0x00)) {
    return complete_carlson_elliptic_r_g(y, z);
  } else if (y == T1(0x00)) {
    return complete_carlson_elliptic_r_g(z, x);
  } else {
    return (x * (y + z) * carlson_elliptic_r_d(y, z, x) + y * (z + x) * carlson_elliptic_r_d(z, x, y) + z * (x + y) * carlson_elliptic_r_d(x, y, z)) / T2(0x06);
  }
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
