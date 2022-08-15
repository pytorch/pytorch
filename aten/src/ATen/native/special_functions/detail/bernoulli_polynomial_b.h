#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
bernoulli_polynomial_b(unsigned int n, T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    auto p = bernoulli_number<T1>(0);
    auto q = T1(1);

    for (auto j = 1; j <= n; j++) {
      q = q * (T1(n + 1 - j) / T1(j));
      p = x * p + q * bernoulli_number<T1>(j);
    }

    return p;
  }
}
}
}
}
}
