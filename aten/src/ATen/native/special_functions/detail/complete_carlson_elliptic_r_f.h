#pragma once

#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
complete_carlson_elliptic_r_f(T1 x, T1 y) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x) || std::isnan(y)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else {
    x = std::sqrt(x);
    y = std::sqrt(y);

    while (true) {
      const auto previous_x = x;

      x = (x + y) / T2(2);

      y = std::sqrt(previous_x) * std::sqrt(y);

      if (std::abs(x - y) < T2(2.7L) * std::sqrt(std::numeric_limits<T2>::epsilon()) * std::abs(x)) {
        return c10::numbers::pi_v<T2> / (x + y);
      }
    }
  }
}
}
