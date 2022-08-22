#pragma once

#include <ATen/native/special/detail/ln_factorial.h>
#include <ATen/native/special/ln_gamma_sign.h>
#include <ATen/native/special/detail/is_integer.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <ATen/native/special/detail/factorial.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
falling_factorial(T1 x, int n) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(x)) {
    return std::numeric_limits<T3>::quiet_NaN();
  } else if (n == 0) {
    return T1{1};
  } else if (is_integer(x)) {
    if (is_integer(x)() < n) {
      return T1{0};
    } else if (is_integer(x)() < static_cast<int>(c10::numbers::factorials_size<T3>()) && is_integer(x)() - n < static_cast<int>(c10::numbers::factorials_size<T3>())) {
      return factorial<T3>(is_integer(x)()) / factorial<T3>(is_integer(x)() - n);
    } else {
      return std::exp(ln_factorial<T3>(is_integer(x)()) - ln_factorial<T3>(is_integer(x)() - n));
    }
  } else if (std::abs(x) < c10::numbers::factorials_size<T3>() && std::abs(x - n) < c10::numbers::factorials_size<T3>()) {
    auto p = x;

    for (auto j = 1; j < n; j++) {
      p = p * (x - j);
    }

    return p;
  } else if (ln_gamma(x + T1{1}) - ln_gamma(x - n + T1{1}) < std::log(std::numeric_limits<T1>::max())) {
    return at::native::special::ln_gamma_sign(x + T1{1}) * at::native::special::ln_gamma_sign(x - n + T1{1}) * std::exp(ln_gamma(x + T1{1}) - ln_gamma(x - n + T1{1}));
  } else {
    return at::native::special::ln_gamma_sign(x + T1{1}) * at::native::special::ln_gamma_sign(x - n + T1{1}) * std::numeric_limits<T1>::infinity();
  }
}

template<typename T1>
T1
falling_factorial(T1 x, T1 n) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(n) || std::isnan(x)) {
    return std::numeric_limits<T3>::quiet_NaN();
  } else if (n == T1{0}) {
    return T1{1};
  } else if (is_integer(n)) {
    if (is_integer(x) && is_integer(x)() < is_integer(n)()) {
      return T1{0};
    } else {
      return falling_factorial(x, is_integer(n)());
    }
  } else if (ln_gamma(x + T1{1}) - ln_gamma(x - n + T1{1}) < std::log(std::numeric_limits<T1>::max())) {
    return at::native::special::ln_gamma_sign(x + T1{1}) * at::native::special::ln_gamma_sign(x - n + T1{1}) * std::exp(ln_gamma(x + T1{1}) - ln_gamma(x - n + T1{1}));
  } else {
    return at::native::special::ln_gamma_sign(x + T1{1}) * at::native::special::ln_gamma_sign(x - n + T1{1}) * std::numeric_limits<T1>::infinity();
  }
}
}
}
}
}
