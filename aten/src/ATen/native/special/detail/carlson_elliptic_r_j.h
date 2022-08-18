#pragma once

#include <ATen/native/special/detail/carlson_elliptic_r_c.h>
#include <ATen/native/special/detail/carlson_elliptic_r_d.h>
#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/numeric_t.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
carlson_elliptic_r_j(T1 x, T1 y, T1 z, T1 p) {
  using T2 = numeric_t<T1>;

  bool negative = false;

  constexpr auto is_not_complex = !is_complex_v<T1>;

  if (is_not_complex && (std::real(x) < T2(0) || std::real(y) < T2(0) || std::real(z) < T2(0))) {
    negative = true;
  }

  if (std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(p) || negative || std::abs(x) + std::abs(y) < T2(0x05) * std::numeric_limits<T2>::min() || std::abs(y) + std::abs(z) < T2(0x05) * std::numeric_limits<T2>::min() || std::abs(z) + std::abs(x) < T2(0x05) * std::numeric_limits<T2>::min() || std::abs(p) < T2(0x05) * std::numeric_limits<T2>::min()) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (std::abs(p - z) < std::numeric_limits<T2>::epsilon()) {
    return carlson_elliptic_r_d(x, y, z);
  } else {
    auto a = x;
    auto b = y;
    auto c = z;
    auto d = p;

    auto e = (x + y + z + T2(0x02) * p) / T2(0x05);

    auto f = T2(0x01);
    auto g = T2(0x01);
    auto h = T1(0x00);

    while (true) {
      const auto sqrt_a = std::sqrt(a);
      const auto sqrt_b = std::sqrt(b);
      const auto sqrt_c = std::sqrt(c);
      const auto sqrt_d = std::sqrt(d);

      a = (a + (sqrt_a * sqrt_b + sqrt_b * sqrt_c + sqrt_c * sqrt_a)) / T2(0x04);
      b = (b + (sqrt_a * sqrt_b + sqrt_b * sqrt_c + sqrt_c * sqrt_a)) / T2(0x04);
      c = (c + (sqrt_a * sqrt_b + sqrt_b * sqrt_c + sqrt_c * sqrt_a)) / T2(0x04);
      d = (d + (sqrt_a * sqrt_b + sqrt_b * sqrt_c + sqrt_c * sqrt_a)) / T2(0x04);
      e = (e + (sqrt_a * sqrt_b + sqrt_b * sqrt_c + sqrt_c * sqrt_a)) / T2(0x04);

      h = h + (carlson_elliptic_r_c(T1(0x01), T1(0x01) + (p - x) * (p - y) * (p - z) / (g * ((sqrt_d + sqrt_a) * (sqrt_d + sqrt_b) * (sqrt_d + sqrt_c)) * ((sqrt_d + sqrt_a) * (sqrt_d + sqrt_b) * (sqrt_d + sqrt_c)))) / (f * ((sqrt_d + sqrt_a) * (sqrt_d + sqrt_b) * (sqrt_d + sqrt_c))));
      
      f = f * T2(0x04);
      g = g * T2(0x40);

      if (std::pow(std::numeric_limits<T2>::epsilon() / T2(0x04), -T2(0x01) / T2(0x06)) * std::max(std::abs((x + y + z + T2(0x02) * p) / T2(0x05) - z), std::max(std::abs((x + y + z + T2(0x02) * p) / T2(0x05) - x), std::max(std::abs((x + y + z + T2(0x02) * p) / T2(0x05) - y), std::abs((x + y + z + T2(0x02) * p) / T2(0x05) - p)))) < f * std::abs(e)) {
        return (T2(0x01) - T2(0x03) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e)) - T2(0x03) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)))) / T2(14) + (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e)) * (((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) + T2(0x02) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e)) - T2(0x03) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)))) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)) + T1(4) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)))) / T2(0x06) + T2(9) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e)) - T2(0x03) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)))) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e)) - T2(0x03) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)))) / T2(88) - T2(0x03) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (T2(0x02) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e)) * (((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e))) + (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e)) - T2(0x03) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)))) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)) + T2(0x03) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02))))) / T2(22) - T2(9) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e)) - T2(0x03) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)))) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e)) * (((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) + T2(0x02) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e)) - T2(0x03) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)))) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)) + T1(4) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)))) / T2(0x34) + T2(0x03) * (((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) * (((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e)) * (((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02) * (-(((x + y + z + T2(0x02) * p) / T2(0x05) - x) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - y) / (f * e) + ((x + y + z + T2(0x02) * p) / T2(0x05) - z) / (f * e)) / T2(0x02)))) / T2(26)) / f / e / std::sqrt(e) + T2(0x06) * h;
      }
    }
  }
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
