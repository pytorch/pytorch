#pragma once

#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
complete_carlson_elliptic_r_g(T1 x, T1 y) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x) || std::isnan(y)) { return std::numeric_limits<T2>::quiet_NaN(); }
  else if (x == T1(0) && y == T1{}) { return T1{}; }
  else if (x == T1(0)) { return std::sqrt(y) / T2(2); }
  else if (y == T1(0)) { return std::sqrt(x) / T2(2); }
  else {
    auto xt = std::sqrt(x);
    auto yt = std::sqrt(y);
    const auto A = (xt + yt) / T2(2);
    auto sum = T1{};
    auto sf = T2(1) / T2(2);

    while (true) {
      auto xtt = xt;
      xt = (xt + yt) / T2(2);
      yt = std::sqrt(xtt) * std::sqrt(yt);
      auto del = xt - yt;
      if (std::abs(del) < T2{2.7L} * std::sqrt(std::numeric_limits<T2>::epsilon()) * std::abs(xt))
        return (A * A - sum) * c10::numbers::pi_v<T2> / (xt + yt) / T2(2);
      sum += sf * del * del;
      sf *= T2(2);
    }
  }
}
}
}
}
}
