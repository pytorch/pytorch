#pragma once

#include <ATen/native/special_functions/detail/complex.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
complete_carlson_elliptic_r_f(T1 x, T1 y) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x) || std::isnan(y)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else {
    auto a = std::sqrt(x);
    auto b = std::sqrt(y);

    while (true) {
      const auto c = a;

      a = (a + b) / T2(0x02);

      b = std::sqrt(c) * std::sqrt(b);

      if (std::abs(a - b) < T2(2.7L) * std::sqrt(std::numeric_limits<T2>::epsilon()) * std::abs(x)) {
        return c10::numbers::pi_v<T2> / (a + b);
      }
    }
  }
}
} // namespace detail
} // namespace special_functions
} // namespace native
} // namespace at
