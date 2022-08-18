#pragma once

#include <ATen/native/special/detail/carlson_elliptic_r_c.h>
#include <ATen/native/special/detail/complete_carlson_elliptic_r_f.h>
#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/numeric_t.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
carlson_elliptic_r_f(T1 x, T1 y, T1 z) {
  using T2 = numeric_t<T1>;

  bool negative = false;

  constexpr auto is_not_complex = !is_complex_v<T1>;

  if (is_not_complex && (std::real(x) < T2(0x00) || std::real(y) < T2(0x00) || std::real(z) < T2(0x00))) {
    negative = true;
  }

  if (std::isnan(x) || std::isnan(y) || std::isnan(z) || negative || std::abs(x) + std::abs(y) < T2(0x05) * std::numeric_limits<T2>::min() || std::abs(x) + std::abs(z) < T2(0x05) * std::numeric_limits<T2>::min() || std::abs(y) + std::abs(z) < T2(0x05) * std::numeric_limits<T2>::min()) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (std::abs(z) < std::numeric_limits<T2>::epsilon()) {
    return complete_carlson_elliptic_r_f(x, y);
  } else if (std::abs(z - y) < std::numeric_limits<T2>::epsilon()) {
    return carlson_elliptic_r_c(x, y);
  } else {
    auto a = x;
    auto b = y;
    auto c = z;
    auto d = (x + y + z) / T2(0x03);
    auto e = T2(0x01);

    while (true) {
      const auto f = std::sqrt(a) * std::sqrt(b) + std::sqrt(b) * std::sqrt(c) + std::sqrt(c) * std::sqrt(a);

      a = (a + f) / T2(0x04);
      b = (b + f) / T2(0x04);
      c = (c + f) / T2(0x04);
      d = (d + f) / T2(0x04);

      e = e * T2(0x04);

      if (std::pow(T2(0x03) * std::numeric_limits<T2>::epsilon(), -T2(0x01) / T2(0x06)) * std::max(std::abs((x + y + z) / T2(0x03) - z), std::max(std::abs((x + y + z) / T2(0x03) - x), std::abs((x + y + z) / T2(0x03) - y))) < e * std::abs(d)) {
        return (T2(0x01) - (((x + y + z) / T2(0x03) - x) / (e * d) * (((x + y + z) / T2(0x03) - y) / (e * d)) - -(((x + y + z) / T2(0x03) - x) / (e * d) + ((x + y + z) / T2(0x03) - y) / (e * d)) * -(((x + y + z) / T2(0x03) - x) / (e * d) + ((x + y + z) / T2(0x03) - y) / (e * d))) / T2(10) + ((x + y + z) / T2(0x03) - x) / (e * d) * (((x + y + z) / T2(0x03) - y) / (e * d)) * -(((x + y + z) / T2(0x03) - x) / (e * d) + ((x + y + z) / T2(0x03) - y) / (e * d)) / T2(14) + (((x + y + z) / T2(0x03) - x) / (e * d) * (((x + y + z) / T2(0x03) - y) / (e * d)) - -(((x + y + z) / T2(0x03) - x) / (e * d) + ((x + y + z) / T2(0x03) - y) / (e * d)) * -(((x + y + z) / T2(0x03) - x) / (e * d) + ((x + y + z) / T2(0x03) - y) / (e * d))) * (((x + y + z) / T2(0x03) - x) / (e * d) * (((x + y + z) / T2(0x03) - y) / (e * d)) - -(((x + y + z) / T2(0x03) - x) / (e * d) + ((x + y + z) / T2(0x03) - y) / (e * d)) * -(((x + y + z) / T2(0x03) - x) / (e * d) + ((x + y + z) / T2(0x03) - y) / (e * d))) / T2(24) - T2(0x03) * (((x + y + z) / T2(0x03) - x) / (e * d) * (((x + y + z) / T2(0x03) - y) / (e * d)) - -(((x + y + z) / T2(0x03) - x) / (e * d) + ((x + y + z) / T2(0x03) - y) / (e * d)) * -(((x + y + z) / T2(0x03) - x) / (e * d) + ((x + y + z) / T2(0x03) - y) / (e * d))) * (((x + y + z) / T2(0x03) - x) / (e * d) * (((x + y + z) / T2(0x03) - y) / (e * d)) * -(((x + y + z) / T2(0x03) - x) / (e * d) + ((x + y + z) / T2(0x03) - y) / (e * d))) / T2(44)) / std::sqrt(d);
      }
    }
  }
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
