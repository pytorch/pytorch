#pragma once

#include <ATen/native/special/detail/factorial.h>
#include <ATen/native/special/detail/gamma.h>
#include <ATen/native/special/detail/ln_binomial_coefficient.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
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
    if (k < c10::numbers::factorials_size<T3>() && n - k < c10::numbers::factorials_size<T3>() && n < c10::numbers::factorials_size<T3>()) {
      return factorial<T1>(n) / factorial<T1>(k) / factorial<T1>(n - k);
    } else {
      if (std::abs(ln_binomial_coefficient<T2>(n, k)) > std::numeric_limits<T3>::max_exponent10 * std::log(T3(10)) - T3(1)) {
        return std::numeric_limits<T1>::infinity();
      } else {
        return std::exp(ln_binomial_coefficient<T2>(n, k));
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
    if (int(std::nearbyint(n)) == n && int(std::nearbyint(n)) >= 0 && int(std::nearbyint(n)) < c10::numbers::factorials_size<T3>()) {
      return binomial_coefficient<T1>(static_cast<unsigned int>(int(std::nearbyint(n))), k);
    } else if (std::abs(n) < c10::numbers::factorials_size<T3>() && k < c10::numbers::factorials_size<T3>()) {
      return gamma(n + T1(1)) / gamma(k + T1(1)) / gamma(n - T1(1) - k);
    } else {
      if (ln_binomial_coefficient(n, k) > std::numeric_limits<T1>::max_exponent10() * std::log(T1(10)) - T1(1)) {
        return std::numeric_limits<T1>::infinity() * log_binomial_coefficient_sign(n, k);
      } else {
        return std::exp(ln_binomial_coefficient(n, k)) * log_binomial_coefficient_sign(n, k);
      }
    }
  }
}
}
}
}
}
