#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
struct chebyshev_polynomial_u_t {
  unsigned int n;
  T1 x;
  T1 U_n;
  T1 U_nm1;
  T1 U_nm2;

  constexpr T1
  deriv() const noexcept {
    return (T1(n + 1) * U_nm1 - T1(n) * x * U_n)
        / (T1{1} - x * x);
  }
};

template<typename T1>
chebyshev_polynomial_u_t<T1>
chebyshev_polynomial_u(unsigned int n, T1 x) {
  if (n == 0) {
    return {n, x, T1(1), T1(0), T1(0)};
  } else if (n == 1) {
    return {n, x, T1(2) * x, T1(1), T1(0)};
  } else {
    auto p = T1(1);
    auto q = T1(2) * x;
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
