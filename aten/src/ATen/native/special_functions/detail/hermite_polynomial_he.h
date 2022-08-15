#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename Tp>
struct hermite_he_t {
  unsigned int n;
  Tp x;
  Tp He_n;
  Tp He_nm1;
  Tp He_nm2;

  constexpr Tp
  deriv() const noexcept { return Tp(n) * He_nm1; }

  constexpr Tp
  deriv2() const noexcept { return Tp(n * (n - 1)) * He_nm2; }
};

template<typename T1>
hermite_he_t<T1>
hermite_polynomial_he(unsigned int n, T1 x) {
  if (n == 0) {
    return {n, x, T1(1), T1(0), T1(0)};
  } else if (n == 1) {
    return {n, x, x, T1(1), T1(0)};
  } else {
    auto p = x;
    auto q = T1(1);
    auto r = x * p - q;

    for (unsigned int j = 3; j <= n; j++) {
      q = p;
      p = r;
      r = x * p - T1(j - 1) * q;
    }

    return {n, x, r, p, q};
  }
}
}
}
}
}
