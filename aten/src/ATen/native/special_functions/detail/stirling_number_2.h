#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
stirling_number_2(unsigned int n, unsigned int m) {
  if (m > n) {
    return T1(0);
  } else if (m == n) {
    return T1(1);
  } else if (m == 0 && n >= 1) {
    return T1(0);
  } else if (n == 0) {
    return T1(m == 0);
  } else if (m == 0) {
    return T1(n == 0);
  } else {
    std::vector<T1> previous_series(m + 1);
    std::vector<T1> series(m + 1);

    previous_series[1] = T1(1);

    if (n == 1) {
      return previous_series[m];
    }

    for (auto j = 1u; j <= n; j++) {
      series[1] = previous_series[1];

      for (auto k = 2u; k <= m; k++) {
        series[k] = k * previous_series[k] + previous_series[k - 1];
      }

      std::swap(previous_series, series);
    }

    return series[m];
  }
}
}
}
}
}
