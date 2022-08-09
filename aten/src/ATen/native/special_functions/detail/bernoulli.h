#pragma once

#include <cmath>

#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
constexpr T1
bernoulli_series(unsigned int n) {
  constexpr T1 series[12] = {
      T1{1ULL} / T1{1ULL},
      T1{1ULL} / T1{6ULL},
      -T1{1ULL} / T1{30ULL},
      T1{1ULL} / T1{42ULL},
      -T1{1ULL} / T1{30ULL},
      T1{5ULL} / T1{66ULL},
      -T1{691ULL} / T1{2730ULL},
      T1{7ULL} / T1{6ULL},
      -T1{3617ULL} / T1{510ULL},
      T1{43867ULL} / T1{798ULL},
      -T1{174611ULL} / T1{330ULL},
      T1{854513ULL} / T1{138ULL}
  };

  if (n == 0) {
    return T1(1);
  } else if (n == 1) {
    return -T1(1) / T1(2);
  } else if (n % 2 == 1) {
    return T1(0);
  } else if (n / 2 < 12) {
    return series[n / 2];
  } else {
    auto p = T1(1);

    if ((n / 2) % 2 == 0) {
      p = p * -T1(1);
    }

    for (unsigned int j = 1; j <= n; j++) {
      p = p * (j / c10::numbers::tau_v<T1>);
    }

    p = p * T1(2);

    auto q = T1(0);

    for (unsigned int k = 2; k < 1000; k++) {
      auto r = std::pow(T1(k), -T1(n));

      q = q + r;

      if (r < std::numeric_limits<T1>::epsilon() * q) {
        break;
      }
    }

    return p + p * q;
  }
}
}
