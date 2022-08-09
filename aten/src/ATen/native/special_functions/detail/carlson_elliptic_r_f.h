#pragma once

#include "carlson_elliptic_r_c.h"
#include "complete_carlson_elliptic_r_f.h"

namespace at::native::special_functions::detail {
template<typename T1>
T1
carlson_elliptic_r_f(T1 x, T1 y, T1 z) {
  using T2 = numeric_t<T1>;

  bool neg_arg = false;
  if constexpr (!is_complex_v<T1>)
    if (std::real(x) < T2(0)
        || std::real(y) < T2(0)
        || std::real(z) < T2(0))
      neg_arg = true;

  if (std::isnan(x) || std::isnan(y) || std::isnan(z))
    return std::numeric_limits<T2>::quiet_NaN();
  else if (neg_arg)
    throw std::domain_error("carlson_elliptic_r_f: argument less than zero");
  else if (std::abs(x) + std::abs(y) < T2(5) * std::numeric_limits<T2>::min()
      || std::abs(x) + std::abs(z) < T2(5) * std::numeric_limits<T2>::min()
      || std::abs(y) + std::abs(z) < T2(5) * std::numeric_limits<T2>::min())
    throw std::domain_error("carlson_elliptic_r_f: argument too small");

  if (std::abs(z) < std::numeric_limits<T2>::epsilon())
    return complete_carlson_elliptic_r_f(x, y);
  else if (std::abs(z - y) < std::numeric_limits<T2>::epsilon()) { return carlson_elliptic_r_c(x, y); }
  else {
    auto xt = x;
    auto yt = y;
    auto zt = z;
    auto A0 = (x + y + z) / T2(3);
    auto Q = std::pow(T2(3) * std::numeric_limits<T2>::epsilon(), -T2(1) / T2(6))
        * std::max(std::abs(A0 - z), std::max(std::abs(A0 - x), std::abs(A0 - y)));
    auto A = A0;
    auto f = T2(1);

    while (true) {
      auto lambda = std::sqrt(xt) * std::sqrt(yt) + std::sqrt(yt) * std::sqrt(zt) + std::sqrt(zt) * std::sqrt(xt);
      A = (A + lambda) / T2(4);
      xt = (xt + lambda) / T2(4);
      yt = (yt + lambda) / T2(4);
      zt = (zt + lambda) / T2(4);
      f *= T2(4);
      if (Q < f * std::abs(A)) {
        auto Xi = (A0 - x) / (f * A);
        auto Yi = (A0 - y) / (f * A);
        auto Zi = -(Xi + Yi);
        auto E2 = Xi * Yi - Zi * Zi;
        auto E3 = Xi * Yi * Zi;
        return (T2(1) - E2 / T2(10) + E3 / T2(14) + E2 * E2 / T2(24) - T2(3) * E2 * E3 / T2(44)) / std::sqrt(A);
      }
    }

    return T1(0);
  }
}
}
