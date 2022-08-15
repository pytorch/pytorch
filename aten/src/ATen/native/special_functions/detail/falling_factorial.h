#pragma once

#include <ATen/native/special_functions/detail/ln_gamma_sign.h>
#include <ATen/native/special_functions/detail/ln_factorial.h>
#include <ATen/native/special_functions/detail/is_integer.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/factorial.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
falling_factorial(T1 a, int n) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  const auto is_integer_a = is_integer(a);

  if (std::isnan(a))
    return std::numeric_limits<T3>::quiet_NaN();
  else if (n == 0)
    return T1{1};
  else if (is_integer_a) {
    auto na = is_integer_a();
    if (na < n)
      return T1{0};
    else if (na < static_cast<int>(c10::numbers::factorials_size<T3>)
        && na - n < static_cast<int>(c10::numbers::factorials_size<T3>))
      return factorial<T3>(na) / factorial<T3>(na - n);
    else
      return std::exp(ln_factorial<T3>(na)
                          - ln_factorial<T3>(na - n));
  } else if (std::abs(a) < c10::numbers::factorials_size<T3>
      && std::abs(a - n) < c10::numbers::factorials_size<T3>) {
    auto prod = a;
    for (int k = 1; k < n; ++k)
      prod *= (a - k);
    return prod;
  } else {
    auto logpoch = ln_gamma(a + T1{1})
        - ln_gamma(a - n + T1{1});
    auto sign = log_gamma_sign(a + T1{1})
        * log_gamma_sign(a - n + T1{1});
    if (logpoch < std::log(std::numeric_limits<T1>::max()))
      return sign * std::exp(logpoch);
    else
      return sign * std::numeric_limits<T1>::infinity();
  }
}

template<typename T1>
T1
falling_factorial(T1 a, T1 nu) {
  using T2 = T1;
  using T3 = numeric_t<T2>;
  const auto is_integer_n = is_integer(nu);
  const auto is_integer_a = is_integer(a);

  if (std::isnan(nu) || std::isnan(a))
    return std::numeric_limits<T3>::quiet_NaN();
  else if (nu == T1{0})
    return T1{1};
  else if (is_integer_n) {
    const auto integer_a = is_integer_a();
    const auto integer_n = is_integer_n();

    if (is_integer_a && integer_a < integer_n)
      return T1{0};
    else
      return falling_factorial(a, integer_n);
  } else {
    auto logpoch = ln_gamma(a + T1{1})
        - ln_gamma(a - nu + T1{1});
    auto sign = log_gamma_sign(a + T1{1})
        * log_gamma_sign(a - nu + T1{1});
    if (logpoch < std::log(std::numeric_limits<T1>::max()))
      return sign * std::exp(logpoch);
    else
      return sign * std::numeric_limits<T1>::infinity();
  }
}
}
}
}
}
