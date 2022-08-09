#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/is_complex_v.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
carlson_elliptic_r_d(T1 x, T1 y, T1 z) {
  using T2 = numeric_t<T1>;

  bool neg_arg = false;
  if constexpr (!is_complex_v < T1 >)
    if (std::real(x) < T2(0)
        || std::real(y) < T2(0)
        || std::real(z) < T2(0))
      neg_arg = true;

  if (std::isnan(x) || std::isnan(y) || std::isnan(z))
    return std::numeric_limits<T2>::quiet_NaN();
  if (neg_arg)
    throw std::domain_error("carlson_elliptic_r_d: argument less than zero");
  else if (std::abs(x) + std::abs(y) < T2(5) * std::numeric_limits<T2>::min()
      || std::abs(z) < T2(5) * std::numeric_limits<T2>::min())
    throw std::domain_error("carlson_elliptic_r_d: arguments too small");
  else {
    auto xt = x;
    auto yt = y;
    auto zt = z;
    auto A0 = (x + y + T2(3) * z) / T2(5);
    auto Q = std::pow(std::numeric_limits<T2>::epsilon() / T2(4), -T2(1) / T2(6))
        * std::max(std::abs(A0 - z),
                   std::max(std::abs(A0 - x),
                            std::abs(A0 - y)));
    auto A = A0;
    auto f = T2(1);
    auto sum = T1(0);

    while (true) {
      auto lambda = std::sqrt(xt) * std::sqrt(yt)
          + std::sqrt(yt) * std::sqrt(zt)
          + std::sqrt(zt) * std::sqrt(xt);
      sum += T2(1) / f / std::sqrt(zt) / (zt + lambda);
      A = (A + lambda) / T2(4);
      xt = (xt + lambda) / T2(4);
      yt = (yt + lambda) / T2(4);
      zt = (zt + lambda) / T2(4);
      f *= T2(4);
      if (Q < f * std::abs(A)) {
        auto Xi = (A0 - x) / (f * A);
        auto Yi = (A0 - y) / (f * A);
        auto Zi = -(Xi + Yi) / T2(3);
        auto ZZ = Zi * Zi;
        auto XY = Xi * Yi;
        auto E2 = XY - T2(6) * ZZ;
        auto E3 = (T2(3) * XY - T2(8) * ZZ) * Zi;
        auto E4 = T2(3) * (XY - ZZ) * ZZ;
        auto E5 = XY * Zi * ZZ;
        return (T2(1)
            - T2(3) * E2 / T2(14)
            + E3 / T2(6)
            + T2(9) * E2 * E2 / T2(88)
            - T2(3) * E4 / T2(22)
            - T2(9) * E2 * E3 / T2(52)
            + T2(3) * E5 / T2(26)) / f / A / std::sqrt(A)
            + T2(3) * sum;
      }
    }

    return T1(0);
  }
}
}
