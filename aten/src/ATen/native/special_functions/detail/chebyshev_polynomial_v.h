#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
struct chebyshev_polynomial_v_t {
  unsigned int n;
  T1 x;
  T1 V_n;
  T1 V_nm1;
  T1 V_nm2;

  constexpr T1
  deriv() const noexcept {
    auto apbp2k = T1(2 * n);
    return (n * (T1{1} - apbp2k * x) * V_nm1
        + T1(2 * (n + 0.5L) * (n + -0.5L)) * V_nm2)
        / (apbp2k * (T1{1} - x * x));
  }
};

template<typename T1>
chebyshev_polynomial_v_t<T1>
chebyshev_polynomial_v(unsigned int n, T1 x) {
  if (n == 0) {
    return {n, x, T1(1), T1(0), T1(0)};
  } else if (n == 1) {
    return {n, x, T1(2) * x - T1(1), T1(1), T1(0)};
  } else {
    auto p = T1(1);
    auto q = T1(2) * x - T1(1);
    auto r = T1(2) * x * q - p;

    for (unsigned int j = 2; j < n; j++) {
      p = q;
      q = r;
      r = T1(2) * x * q - p;
    }

    return {n, x, r, q, p};
  }
}
}
}
}
}
