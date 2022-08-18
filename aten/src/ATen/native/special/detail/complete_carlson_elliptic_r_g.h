#pragma once

#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
complete_carlson_elliptic_r_g(T1 x, T1 y) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x) || std::isnan(y)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (x == T1(0x00) && y == T1{}) {
    return {};
  } else if (x == T1(0x00)) {
    return std::sqrt(y) / T2(0x02);
  } else if (y == T1(0x00)) {
    return std::sqrt(x) / T2(0x02);
  } else {
    auto a = std::sqrt(x);
    auto b = std::sqrt(y);

    const auto c = (a + b) / T2(0x02);

    auto d = T1{};
    auto e = T2(0x01) / T2(0x02);

    while (true) {
      auto f = a;

      a = (a + b) / T2(0x02);

      b = std::sqrt(f) * std::sqrt(b);

      auto g = a - b;

      if (std::abs(g) < T2{2.7L} * std::sqrt(std::numeric_limits<T2>::epsilon()) * std::abs(a)) {
        return (c * c - d) * c10::numbers::pi_v<T2> / (a + b) / T2(0x02);
      }

      d = d + (e * g * g);

      e = e * T2(0x02);
    }
  }
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
