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
carlson_elliptic_r_d(T1 x, T1 y, T1 z) {
  using T2 = numeric_t<T1>;

  bool negative = false;

  constexpr auto is_not_complex = !is_complex_v<T1>;

  if (is_not_complex && (std::real(x) < T2(0x00) || std::real(y) < T2(0x00) || std::real(z) < T2(0x00))) {
    negative = true;
  }

  if (std::isnan(x) || std::isnan(y) || std::isnan(z) || negative || std::abs(x) + std::abs(y) < T2(0x05) * std::numeric_limits<T2>::min() || std::abs(z) < T2(0x05) * std::numeric_limits<T2>::min()) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else {
    auto a = x;
    auto b = y;
    auto c = z;

    auto d = (x + y + T2(0x03) * z) / T2(0x05);

    auto e = T2(0x01);
    auto f = T1(0x00);

    while (true) {
      const auto g = std::sqrt(a) * std::sqrt(b) + std::sqrt(b) * std::sqrt(c) + std::sqrt(c) * std::sqrt(a);

      f = f + (T2(0x01) / e / std::sqrt(c) / (c + g));

      a = (a + g) / T2(0x04);
      b = (b + g) / T2(0x04);
      c = (c + g) / T2(0x04);
      d = (d + g) / T2(0x04);

      e = e * T2(0x04);

      if (std::pow(std::numeric_limits<T2>::epsilon() / T2(0x04), -T2(0x01) / T2(0x06)) * std::max(std::abs((x + y + T2(0x03) * z) / T2(0x05) - z), std::max(std::abs((x + y + T2(0x03) * z) / T2(0x05) - x), std::abs((x + y + T2(0x03) * z) / T2(0x05) - y))) < e * std::abs(d)) {
        return (T2(0x01) - T2(0x03) * (((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) * (((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) - T2(0x06) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03)))) / T2(14) + (T2(0x03) * (((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) * (((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d))) - T2(0x08) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03)))) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03)) / T2(0x06) + T2(0x09) * (((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) * (((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) - T2(0x06) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03)))) * (((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) * (((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) - T2(0x06) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03)))) / T2(88) - T2(0x03) * (T2(0x03) * (((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) * (((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) - -(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03))) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03)))) / T2(22) - T2(0x09) * (((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) * (((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) - T2(0x06) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03)))) * ((T2(0x03) * (((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) * (((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d))) - T2(0x08) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03)))) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03))) / T2(52) + T2(0x03) * (((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) * (((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03)) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03) * (-(((x + y + T2(0x03) * z) / T2(0x05) - x) / (e * d) + ((x + y + T2(0x03) * z) / T2(0x05) - y) / (e * d)) / T2(0x03)))) / T2(26)) / e / d / std::sqrt(d) + T2(0x03) * f;
      }
    }
  }
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
