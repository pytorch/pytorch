#pragma once

#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/is_complex_v.h>
#include <ATen/native/special/detail/numeric_t.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
carlson_elliptic_r_c(T1 x, T1 y) {
  using T2 = numeric_t<T1>;

  bool negative_x = false;
  bool negative_y = false;

  constexpr auto is_not_complex = !is_complex_v<T1>;

  if (is_not_complex) {
    if (std::real(x) < T2(0x00)) {
      negative_x = true;
    }

    if (std::real(y) < T2(0x00)) {
      negative_y = true;
    }
  }

  if (std::isnan(x) || std::isnan(y) || negative_x || std::abs(x) + std::abs(y) < T2(0x05) * std::numeric_limits<T2>::min()) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (negative_y) {
    if (std::abs(x) == T2(0x00)) {
      return T1{};
    } else {
      return std::sqrt(x / (x - y)) * carlson_elliptic_r_c(x - y, -y);
    }
  } else {
    auto a = x;
    auto b = y;

    auto c = (x + T2(0x02) * y) / T2(0x03);

    auto d = T2(0x01);

    while (true) {
      const auto e = T2(0x02) * std::sqrt(a) * std::sqrt(b) + b;

      a = (a + e) / T2(0x04);
      b = (b + e) / T2(0x04);
      c = (c + e) / T2(0x04);

      d = d * T2(0x04);

      if (std::pow(T2(0x03) * std::numeric_limits<T2>::epsilon(), -T2(0x01) / T2(0x08)) * std::abs((x + T2(0x02) * y) / T2(0x03) - x) < d * std::abs(c)) {
        return (T2(0x01) + (y - (x + T2(0x02) * y) / T2(0x03)) / (d * c) * ((y - (x + T2(0x02) * y) / T2(0x03)) / (d * c)) * (T2(0x03) / T2(10) + (y - (x + T2(0x02) * y) / T2(0x03)) / (d * c) * (T2(0x01) / T2(0x07) + (y - (x + T2(0x02) * y) / T2(0x03)) / (d * c) * (T2(0x03) / T2(0x08) + (y - (x + T2(0x02) * y) / T2(0x03)) / (d * c) * (T2(0x09) / T2(22) + (y - (x + T2(0x02) * y) / T2(0x03)) / (d * c) * (T2(159) / T2(208) + (y - (x + T2(0x02) * y) / T2(0x03)) / (d * c) * (T2(0x09) / T2(0x08)))))))) / std::sqrt(c);
      }
    }
  }
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
