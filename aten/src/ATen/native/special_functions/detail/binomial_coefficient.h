#pragma once

#include <ATen/native/special_functions/detail/ln_binomial_coefficient.h>
#include <ATen/native/special_functions/detail/gamma.h>
#include <ATen/native/special_functions/detail/factorial.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
binomial_coefficient(unsigned int n, unsigned int k) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (k > n) {
    return T1(0);
  } else if (k == 0 || k == n) {
    return T1(1);
  } else {
    const auto m = n - k;

    const auto size = c10::numbers::factorials_size<T3>;

    if (k < size && m < size && n < size) {
      const auto factorial_k = at::native::special_functions::detail::factorial<T1>(k);
      const auto factorial_m = at::native::special_functions::detail::factorial<T1>(m);
      const auto factorial_n = at::native::special_functions::detail::factorial<T1>(n);

      return factorial_n / factorial_k / factorial_m;
    } else {
      const auto ln_binomial_coefficient = at::native::special_functions::detail::ln_binomial_coefficient<T2>(n, k);

      if (std::abs(ln_binomial_coefficient) > std::numeric_limits<T3>::max_exponent10 * std::log(T3(10)) - T3(1)) {
        return std::numeric_limits<T1>::infinity();
      } else {
        return std::exp(ln_binomial_coefficient);
      }
    }
  }
}

template<typename T1>
T1
binomial_coefficient(T1 n, unsigned int k) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(n)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    const auto nearbyint_n = int(std::nearbyint(n));

    const auto size = c10::numbers::factorials_size<T3>;

    if (nearbyint_n == n && nearbyint_n >= 0 && nearbyint_n < size) {
      return at::native::special_functions::detail::binomial_coefficient<T1>(static_cast<unsigned int>(nearbyint_n),
                                                                               k);
    } else if (std::abs(n) < size && k < size) {
      const auto m = n - T1(1) - k;

      const auto gamma_k = at::native::special_functions::detail::gamma(k + T1(1));
      const auto gamma_m = at::native::special_functions::detail::gamma(m);
      const auto gamma_n = at::native::special_functions::detail::gamma(n + T1(1));

      return gamma_n / gamma_k / gamma_m;
    } else {
      const auto ln_binomial_coefficient = at::native::special_functions::detail::ln_binomial_coefficient(n, k);

      const auto q = log_binomial_coefficient_sign(n, k);

      if (ln_binomial_coefficient > std::numeric_limits<T1>::max_exponent10() * std::log(T1(10)) - T1(1)) {
        return std::numeric_limits<T1>::infinity() * q;
      } else {
        return std::exp(ln_binomial_coefficient) * q;
      }
    }
  }
}
}
}
}
}
