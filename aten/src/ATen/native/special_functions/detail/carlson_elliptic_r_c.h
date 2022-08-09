#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/is_complex_v.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
carlson_elliptic_r_c(T1 x, T1 y) {
  using T2 = numeric_t<T1>;

  bool negative_x = false;
  bool negative_y = false;

  if constexpr (!is_complex_v<T1>) {
    if (std::real(x) < T2(0)) {
      negative_x = true;
    }

    if (std::real(y) < T2(0)) {
      negative_y = true;
    }
  }

  if (std::isnan(x) || std::isnan(y)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (negative_x) {
    throw std::domain_error("carlson_elliptic_r_c: argument less than zero");
  } else if (std::abs(x) + std::abs(y) < T2(5) * std::numeric_limits<T2>::min()) {
    throw std::domain_error("carlson_elliptic_r_c: arguments too small");
  } else if (negative_y) {
    if (std::abs(x) == T2(0)) {
      return T1{};
    } else {
      return std::sqrt(x / (x - y)) * carlson_elliptic_r_c(x - y, -y);
    }
  } else {
    auto xt = x;
    auto yt = y;
    auto A0 = (x + T2(2) * y) / T2(3);
    auto Q = std::pow(T2(3) * std::numeric_limits<T2>::epsilon(), -T2(1) / T2(8)) * std::abs(A0 - x);
    auto A = A0;
    auto f = T2(1);

    while (true) {
      auto lambda = T2(2) * std::sqrt(xt) * std::sqrt(yt) + yt;
      A = (A + lambda) / T2(4);
      xt = (xt + lambda) / T2(4);
      yt = (yt + lambda) / T2(4);
      f *= T2(4);

      if (Q < f * std::abs(A)) {
        return (T2(1) + (y - A0) / (f * A) * ((y - A0) / (f * A)) * (T2(3) / T2(10) + (y - A0) / (f * A)
            * (T2(1) / T2(7) + (y - A0) / (f * A) * (T2(3) / T2(8) + (y - A0) / (f * A)
                * (T2(9) / T2(22) + (y - A0) / (f * A) * (T2(159) / T2(208) + (y - A0) / (f * A) * (T2(9) / T2(8))))))))
            / std::sqrt(A);
      }
    }

    return T1(0);
  }
}
}
