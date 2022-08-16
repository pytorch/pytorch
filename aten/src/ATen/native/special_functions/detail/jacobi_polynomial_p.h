#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename Tp>
struct jacobi_t {
  unsigned int n;
  Tp alpha1;
  Tp beta1;
  Tp x;
  Tp P_n;
  Tp P_nm1;
  Tp P_nm2;

  constexpr Tp
  deriv() const noexcept {
    auto apbp2k = alpha1 + beta1 + Tp(2 * n);
    return (n * (alpha1 - beta1 - apbp2k * x) * P_nm1
        + Tp{2} * (n + alpha1) * (n + beta1) * P_nm2)
        / (apbp2k * (Tp{1} - x * x));
  }
};

template<typename T1>
jacobi_t<T1>
jacobi_polynomial_p(unsigned int n, T1 alpha1, T1 beta1, T1 x) {
  if (std::isnan(alpha1) || std::isnan(beta1) || std::isnan(x)) {
    return {n, alpha1, beta1, x, std::numeric_limits<T1>::quiet_NaN(), std::numeric_limits<T1>::quiet_NaN(),
            std::numeric_limits<T1>::quiet_NaN()};
  } else if (n == 0) {
    return {n, alpha1, beta1, x, T1(1), T1(0), T1(0)};
  } else if (n == 1) {
    return {n, alpha1, beta1, x, (alpha1 - beta1 + (alpha1 + beta1 + T1(2)) * x) / T1(2), T1(1), T1(0)};
  } else {
    auto p = (alpha1 - beta1 + (alpha1 + beta1 + T1(2)) * x) / T1(2);
    auto q = T1(1);
    auto r = ((((((alpha1 + beta1 + T1(2)) + T1(2)) - T1(1)) * ((alpha1 - beta1) * (alpha1 + beta1)))
        + (((((alpha1 + beta1 + T1(2)) + T1(2)) - T1(1)) - T1(1)) * (((alpha1 + beta1 + T1(2)) + T1(2)) - T1(1))
            * ((alpha1 + beta1 + T1(2)) + T1(2))) * x) * ((alpha1 - beta1 + (alpha1 + beta1 + T1(2)) * x) / T1(2))
        - (T1(2) * (alpha1 + T1(1)) * (beta1 + T1(1)) * ((alpha1 + beta1 + T1(2)) + T1(2))) * T1(1))
        / (T1(4) * (alpha1 + beta1 + T1(2)) * ((((alpha1 + beta1 + T1(2)) + T1(2)) - T1(1)) - T1(1)));

    for (unsigned int j = 3; j <= n; j++) {
      if (T1(2) * j * (alpha1 + beta1 + T1(j)) * (alpha1 + beta1 + T1(j) + T1(j) - T1(1) - T1(1)) == T1(0)) {
        throw std::runtime_error("jacobi_polynomial_p: Failure in recursion");
      }

      q = p;
      p = r;
      r = (((alpha1 + beta1 + T1(j) + T1(j) - T1(1)) * ((alpha1 - beta1) * (alpha1 + beta1))
          + (alpha1 + beta1 + T1(j) + T1(j) - T1(1) - T1(1)) * (alpha1 + beta1 + T1(j) + T1(j) - T1(1))
              * (alpha1 + beta1 + T1(j) + T1(j)) * x) * p
          - T1(2) * (alpha1 + T1(j - 1)) * (beta1 + T1(j - 1)) * (alpha1 + beta1 + T1(j) + T1(j)) * q)
          / (T1(2) * j * (alpha1 + beta1 + T1(j)) * (alpha1 + beta1 + T1(j) + T1(j) - T1(1) - T1(1)));
    }

    return {n, alpha1, beta1, x, r, p, q};
  }
}
}
}
}
}
